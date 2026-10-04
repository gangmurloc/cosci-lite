import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "bin" / "cosci-status"
LAUNCHER = ROOT / "bin" / "cosci-bg"


def test_run_log_lines_reach_a_redirected_file_while_the_process_is_still_running(tmp_path):
    out = tmp_path / "log.txt"
    code = "from cosci.cli import _utf8_stdout; import time; _utf8_stdout(); print('step 1'); time.sleep(30)"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONUNBUFFERED"}
    env["PYTHONPATH"] = str(ROOT / "src")
    with open(out, "w") as f:
        p = subprocess.Popen([sys.executable, "-c", code], stdout=f, stderr=subprocess.STDOUT, env=env)
    try:
        deadline = time.time() + 5
        while time.time() < deadline and "step 1" not in out.read_text():
            time.sleep(0.1)
        assert p.poll() is None
        assert "step 1" in out.read_text()
    finally:
        p.kill()
        p.wait()


def _run_dir(tmp_path, calls):
    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "state.json").write_text(json.dumps({
        "goal": "g", "created": "2026-10-04T04:32:46+09:00", "steps_done": ["parse"],
        "config": {"pipeline": {"rounds": 1, "meta_feedback": True}}, "hypotheses": {}, "papers": {}, "matches": []}))
    (run / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    return run


LIMIT = "claude -p error (success): You've hit your session limit · resets 5:30am (Asia/Seoul)"


def test_status_does_not_count_limit_waits_as_calls_and_shows_when_the_wait_ends(tmp_path):
    run = _run_dir(tmp_path, [
        {"t": "2026-10-04T05:10:00+09:00", "role": "parse", "backend": "claude", "ok": True, "latency": 40.0},
        {"t": "2026-10-04T05:12:00+09:00", "role": "evolve", "backend": "local", "ok": False, "latency": 2.0,
         "error": "http://127.0.0.1:11434/v1: connection error: refused"},
        {"t": "2026-10-04T05:14:56+09:00", "role": "evolve", "backend": "claude", "wait": 964,
         "until": "2026-10-04T05:31:00+09:00", "reason": "usage_limit"},
    ])
    out = subprocess.run([sys.executable, str(STATUS), str(run)], capture_output=True, text=True, check=True).stdout
    calls, failed = re.search(r"\[LLM 호출\] (\d+)건 \(실패 (\d+)건", out).groups()
    assert (calls, failed) == ("2", "1")
    assert "05:31" in out


def test_status_names_the_cause_of_failures(tmp_path):
    fail = {"role": "evolve", "backend": "claude", "ok": False, "latency": 2.0, "error": LIMIT}
    run = _run_dir(tmp_path, [dict(fail, t="2026-10-04T05:14:59+09:00"), dict(fail, t="2026-10-04T05:15:03+09:00"),
                              {"t": "2026-10-04T05:16:00+09:00", "role": "review", "backend": "local", "ok": False,
                               "latency": 25.0, "error": "http://127.0.0.1:11434/v1: connection error: refused"}])
    out = subprocess.run([sys.executable, str(STATUS), str(run)], capture_output=True, text=True, check=True).stdout
    assert re.search(r"사용량 한도 2건", out) and re.search(r"연결 실패 1건", out)


def _launch(args, **env):
    return subprocess.run(["bash", str(LAUNCHER), *args], capture_output=True, text=True,
                          env={**os.environ, "COSCI_BIN": "/bin/true", **env})


def test_launcher_refuses_a_goal_file_that_still_has_template_placeholders(tmp_path):
    goal = tmp_path / "goal.md"
    goal.write_text("# 연구 목표\n\n주제를 찾고 싶다.\n\n## 제약\n- 컴퓨트: (예: RTX A5000 24GB × 2)\n", encoding="utf-8")
    r = _launch(["--dry-run", "t-placeholder", "--goal-file", str(goal)])
    assert r.returncode != 0
    assert "(예:" in r.stdout + r.stderr          # shows the offending line


def test_launcher_dry_run_shows_the_goal_and_the_log_it_would_write(tmp_path):
    goal = tmp_path / "goal.md"
    goal.write_text("# 연구 목표\n\nwrite-read 상호작용 주제를 찾고 싶다.\n\n## 제약\n- 컴퓨트: A5000 2장\n", encoding="utf-8")
    r = _launch(["--dry-run", "t-ok", "--goal-file", str(goal), "--preset", "mock"])
    assert r.returncode == 0, r.stderr
    assert "write-read 상호작용 주제를 찾고 싶다." in r.stdout
    assert "logs/t-ok.log" in r.stdout and "--preset mock" in r.stdout
    assert not (ROOT / "logs" / "t-ok.log").exists()   # dry run starts nothing


def test_launcher_will_not_overwrite_an_existing_log(tmp_path):
    goal = tmp_path / "goal.md"
    goal.write_text("# 연구 목표\n\n주제를 찾고 싶다.\n", encoding="utf-8")
    existing = ROOT / "logs" / "t-existing.log"
    existing.parent.mkdir(exist_ok=True)
    existing.write_text("previous run\n")
    try:
        r = _launch(["--dry-run", "t-existing", "--goal-file", str(goal)])
        assert r.returncode != 0
        assert "t-existing.log" in r.stdout + r.stderr   # says which log is in the way
        assert existing.read_text() == "previous run\n"
    finally:
        existing.unlink()


def _tmux():
    import shutil
    return shutil.which("tmux") or (str(Path.home() / ".local/bin/tmux") if (Path.home() / ".local/bin/tmux").exists() else None)


def test_launcher_starts_the_run_and_its_output_lands_in_the_log(tmp_path):
    import pytest
    if not _tmux():
        pytest.skip("needs tmux")
    fake = tmp_path / "cosci"
    fake.write_text("#!/bin/sh\nfor a in \"$@\"; do printf '[%s]' \"$a\"; done; echo\n")
    fake.chmod(0o755)
    name = f"t-launch-{os.getpid()}"
    log = ROOT / "logs" / f"{name}.log"
    try:
        r = _launch([name, "--preset", "mock", "--goal", "two words"], COSCI_BIN=str(fake), COSCI_NO_NOTIFY="1")
        assert r.returncode == 0, r.stdout + r.stderr
        deadline = time.time() + 10
        while time.time() < deadline and not (log.exists() and "[run]" in log.read_text()):
            time.sleep(0.2)
        assert log.exists(), "the launcher reported success but nothing ran"
        assert "[run][--preset][mock][--goal][two words]" in log.read_text()
    finally:
        subprocess.run([_tmux(), "kill-session", "-t", f"cosci-{name}"], capture_output=True)
        for p in (ROOT / "logs").glob(f"{name}.*"):
            p.unlink()
