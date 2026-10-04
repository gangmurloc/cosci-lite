"""Command-line interface: cosci init | check | run | resume | continue | add-idea | report | save-idea | list"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, report
from .config import EXAMPLE_YAML, expand_path, load_config
from .literature import LiteratureSearch
from .llm import Router
from .models import Hypothesis
from .state import RunState
from .supervisor import Supervisor, new_run_dir

GOAL_TEMPLATE = """# 연구 목표

(무엇을 알고 싶은지 자연어로 적으세요. 예: LLM agent의 long-term memory에서, 학부 연구로 3~6개월 안에 할 수 있는
정량 실험 기반 연구 주제를 찾고 싶다.)

## 선호/관심
- 분야:
- 기여 유형: (empirical finding / evaluation methodology / method / benchmark ...)

## 제약
- 컴퓨트: (예: RTX A5000 24GB × 2)
- 기간: (예: 3~6개월)
- API 예산: (예: 월 $50 이하)

## 피하고 싶은 것
-
"""


def _utf8_stdout() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            # line_buffering: a redirected log (nohup, tee, > file) must show progress while the run is going
            s.reconfigure(encoding="utf-8", line_buffering=True)  # type: ignore[attr-defined]
        except Exception:
            pass


def _cfg(args: argparse.Namespace, extra: dict | None = None) -> dict:
    overrides: dict = {}
    if getattr(args, "rounds", None) is not None:
        overrides.setdefault("pipeline", {})["rounds"] = args.rounds
    if getattr(args, "n_initial", None) is not None:
        overrides.setdefault("pipeline", {})["n_initial"] = args.n_initial
    if getattr(args, "out", None):
        overrides.setdefault("output", {})["runs_dir"] = args.out
    if getattr(args, "ideas_dir", None):
        overrides.setdefault("output", {})["ideas_dir"] = args.ideas_dir
    if getattr(args, "no_save", False):
        overrides.setdefault("output", {})["auto_save_top_k"] = 0
    if extra:
        overrides.update(extra)
    config_path = getattr(args, "config", None)
    if not config_path and Path("cosci.yaml").exists():
        config_path = "cosci.yaml"
    return load_config(config_path, preset=getattr(args, "preset", None), overrides=overrides)


def cmd_init(args: argparse.Namespace) -> int:
    cfg_path = Path(args.path)
    if cfg_path.exists() and not args.force:
        print(f"{cfg_path} 이미 존재 (덮어쓰려면 --force)")
    else:
        cfg_path.write_text(EXAMPLE_YAML, encoding="utf-8")
        print(f"설정 파일 생성: {cfg_path}")
    goal = Path("goal.md")
    if not goal.exists():
        goal.write_text(GOAL_TEMPLATE, encoding="utf-8")
        print(f"연구 목표 템플릿 생성: {goal}")
    print("다음: goal.md 작성 → `cosci check` → `cosci run --goal-file goal.md`")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    ok = True
    used = sorted(set(cfg["roles"].values()))
    router = Router(cfg, None)
    for name in used:
        role = next(r for r, b in cfg["roles"].items() if b == name)
        try:
            res = router.backend_for(role).ping()
            print(f"[backend] {name:8s} ({cfg['backends'][name]['type']}): {res}")
        except Exception as e:
            ok = False
            print(f"[backend] {name:8s} FAILED: {type(e).__name__}: {str(e)[:300]}")
    for src, status in LiteratureSearch(cfg["literature"]).check().items():
        print(f"[search ] {src:16s}: {status}")
        ok = ok and status.startswith("ok")
    return 0 if ok else 1


def _read_goal(args: argparse.Namespace) -> str:
    if args.goal_file:
        return Path(args.goal_file).read_text(encoding="utf-8")
    if args.goal:
        return args.goal
    print("--goal 또는 --goal-file 이 필요합니다", file=sys.stderr)
    sys.exit(2)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    goal = _read_goal(args)
    run_dir = new_run_dir(cfg, goal)
    st = RunState(run_dir=run_dir, goal=goal)
    print(f"실행 폴더: {run_dir}")
    Supervisor(cfg, st).run()
    print(f"\n리포트: {run_dir / 'report.md'}")
    _print_rank(st, 5)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    st = RunState.load(args.run_dir)
    cfg = load_config(None, overrides=st.config) if st.config else _cfg(args)
    Supervisor(cfg, st).run()
    print(f"리포트: {st.run_dir / 'report.md'}")
    _print_rank(st, 5)
    return 0


def cmd_continue(args: argparse.Namespace) -> int:
    st = RunState.load(args.run_dir)
    cfg = load_config(None, overrides=st.config)
    cfg["pipeline"]["rounds"] = int(cfg["pipeline"]["rounds"]) + int(args.rounds)
    for k, v in (("claude", args.extra_claude_calls), ("local", args.extra_local_calls)):
        if v and k in cfg["budget"]["max_calls"]:
            cfg["budget"]["max_calls"][k] += v
    if "overview" in st.steps_done:
        st.steps_done.remove("overview")
    # the last round had no feedback step; allow it now
    Supervisor(cfg, st).run()
    print(f"리포트: {st.run_dir / 'report.md'}")
    _print_rank(st, 5)
    return 0


def cmd_add_idea(args: argparse.Namespace) -> int:
    st = RunState.load(args.run_dir)
    fields = {"title": args.title, "one_liner": args.statement, "statement": args.statement,
              "falsification": args.falsification or "", "min_experiment": args.min_experiment or "",
              "rationale": args.rationale or ""}
    if args.file:
        body = Path(args.file).read_text(encoding="utf-8")
        fields["idea"] = body
    h = Hypothesis(hid=st.next_hid(), fields=fields, origin="user", strategy="researcher")
    st.add_hypothesis(h)
    st.save()
    print(f"{h.hid} 추가됨. `cosci continue {args.run_dir} --rounds 1` 로 리뷰·토너먼트에 참여시키세요.")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    st = RunState.load(args.run_dir)
    path = report.write_report(st)
    print(f"리포트: {path}")
    return 0


def cmd_save_idea(args: argparse.Namespace) -> int:
    st = RunState.load(args.run_dir)
    ideas_dir = expand_path(args.ideas_dir or (st.config.get("output") or {}).get("ideas_dir"))
    if not ideas_dir:
        print("--ideas-dir 를 지정하세요", file=sys.stderr)
        return 2
    for hid in args.hids:
        if hid not in st.hypotheses:
            print(f"{hid}: 없음", file=sys.stderr)
            continue
        print(f"아이디어 저장: {report.save_idea(st, hid, ideas_dir)}")
    st.save()
    return 0


def _print_rank(st: RunState, n: int) -> None:
    ranked = st.ranked()
    if not ranked:
        return
    print("\n순위  ID       Elo   승-패-무  Novelty  제목")
    for i, h in enumerate(ranked[:n], 1):
        print(f"{i:>3}  {h.hid:7s} {h.elo:5.0f}  {h.wins}-{h.losses}-{h.draws:<4}  {h.novelty:7s}  {h.title[:60]}")


def cmd_list(args: argparse.Namespace) -> int:
    st = RunState.load(args.run_dir)
    _print_rank(st, args.n)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cosci", description="Co-Scientist-style research topic discovery (core loop)")
    ap.add_argument("--version", action="version", version=f"cosci-lite {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", "-c", help="YAML config (default: ./cosci.yaml if present)")
        p.add_argument("--preset", choices=["hybrid", "claude-only", "local-only", "mock"])

    p = sub.add_parser("init", help="create cosci.yaml and goal.md templates")
    p.add_argument("--path", default="cosci.yaml")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("check", help="ping backends and search sources")
    common(p)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("run", help="run the full core loop")
    common(p)
    p.add_argument("--goal", help="research goal text")
    p.add_argument("--goal-file", help="markdown file with the research goal")
    p.add_argument("--rounds", type=int)
    p.add_argument("--n-initial", type=int, dest="n_initial")
    p.add_argument("--out", help="runs directory")
    p.add_argument("--ideas-dir", dest="ideas_dir", help="Research-Ideas folder for auto-saved idea files")
    p.add_argument("--no-save", action="store_true", help="do not auto-save ideas")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("resume", help="resume an interrupted run")
    p.add_argument("run_dir")
    p.set_defaults(fn=cmd_resume)

    p = sub.add_parser("continue", help="run more evolution/tournament rounds on a finished run")
    p.add_argument("run_dir")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--extra-claude-calls", type=int, default=40)
    p.add_argument("--extra-local-calls", type=int, default=300)
    p.set_defaults(fn=cmd_continue)

    p = sub.add_parser("add-idea", help="add your own hypothesis to a run (scientist-in-the-loop)")
    p.add_argument("run_dir")
    p.add_argument("--title", required=True)
    p.add_argument("--statement", required=True)
    p.add_argument("--rationale")
    p.add_argument("--falsification")
    p.add_argument("--min-experiment", dest="min_experiment")
    p.add_argument("--file", help="markdown file with more detail")
    p.set_defaults(fn=cmd_add_idea)

    p = sub.add_parser("report", help="re-render report.md")
    p.add_argument("run_dir")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("save-idea", help="save hypotheses as Research-Ideas files + INDEX.md")
    p.add_argument("run_dir")
    p.add_argument("hids", nargs="+")
    p.add_argument("--ideas-dir", dest="ideas_dir")
    p.set_defaults(fn=cmd_save_idea)

    p = sub.add_parser("list", help="print the ranking")
    p.add_argument("run_dir")
    p.add_argument("-n", type=int, default=20)
    p.set_defaults(fn=cmd_list)
    return ap


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
