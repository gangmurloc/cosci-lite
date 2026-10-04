import json
import os
import stat
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from cosci.llm.base import CallBudget, CallLogger
from cosci.llm.claude_code import ClaudeCodeBackend
from cosci.llm.openai_compat import OpenAICompatBackend

SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX shell script as fake claude")
def test_claude_code_backend_with_fake_cli(tmp_path):
    fake = tmp_path / "claude"
    log = tmp_path / "argv.json"
    fake.write_text(
        "#!" + sys.executable + "\n"
        "import sys, json\n"
        f"json.dump(sys.argv[1:], open({str(log)!r}, 'w'))\n"
        "prompt = sys.stdin.read()\n"
        "assert 'OUTPUT FORMAT' in prompt\n"
        "print(json.dumps({'type':'result','is_error':False,'result':'{\"answer\": 4}',"
        "'structured_output':{'answer':4},'total_cost_usd':0.01,'num_turns':2}))\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    be = ClaudeCodeBackend("claude", {"type": "claude_code", "executable": str(fake), "model": "sonnet"},
                           budget=CallBudget({"claude": 5}), logger=CallLogger(tmp_path / "calls.jsonl"),
                           workdir=tmp_path / "cwd")
    out = be.complete_json("What is 2+2?", "sys", SCHEMA, role="parse")
    assert out == {"answer": 4}
    argv = json.load(open(log))
    assert "-p" in argv and "--bare" not in argv
    assert argv[argv.index("--output-format") + 1] == "json"
    assert "--json-schema" in argv and argv[argv.index("--tools") + 1] == ""
    assert be.cost_usd == pytest.approx(0.01)


class _Handler(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.calls.append(body)
        if body.get("response_format", {}).get("type") == "json_schema":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "response_format json_schema not supported"}')
            return
        content = '<think>reasoning...</think>```json\n{"answer": 7}\n```'
        resp = {"choices": [{"message": {"role": "assistant", "content": content}}], "usage": {}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_openai_compat_degrades_json_mode_and_strips_think():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        be = OpenAICompatBackend("local", {"type": "openai_compatible",
                                           "base_url": f"http://127.0.0.1:{srv.server_port}/v1",
                                           "model": "m", "json_mode": "schema",
                                           "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
        out = be.complete_json("q", "sys", SCHEMA, role="generate")
        assert out == {"answer": 7}
        assert be.json_mode == "object"
        assert _Handler.calls[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    finally:
        srv.shutdown()


# --------------------------------------------------------------------------- helpers for the tests below
def _serve(handler):
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _reply(h, content, finish="stop", **message_extra):
    data = json.dumps({"choices": [{"message": {"role": "assistant", "content": content, **message_extra},
                                    "finish_reason": finish}], "usage": {}}).encode()
    h.send_response(200)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(data)))
    h.end_headers()
    h.wfile.write(data)


def _local(srv, **cfg):
    return OpenAICompatBackend("local", {"type": "openai_compatible", "base_url": f"http://127.0.0.1:{srv.server_port}/v1",
                                         "model": "m", **cfg}, budget=CallBudget({"local": 5}))


def _required_only(schema):
    """The smallest answer a schema-constrained server may legally give: required keys only."""
    if "enum" in schema:
        return schema["enum"][0]
    typ = schema.get("type")
    if typ == "object":
        return {k: _required_only(schema["properties"][k]) for k in schema.get("required", [])}
    if typ == "array":
        return [_required_only(schema.get("items", {"type": "string"}))]
    return {"integer": 1, "number": 1.0, "boolean": True}.get(typ, "x")


class _RequiredOnlyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _reply(self, json.dumps(_required_only(body["response_format"]["json_schema"]["schema"])))

    def log_message(self, *a):
        pass


def test_schema_constrained_server_cannot_drop_the_fields_judges_and_reports_read():
    from cosci.agents import prompts as P

    srv = _serve(_RequiredOnlyHandler)
    try:
        out = _local(srv, json_mode="schema").complete_json("q", "sys", P.GEN_SCHEMA, role="generate")
    finally:
        srv.shutdown()
    hyp = out["hypotheses"][0]
    missing = [k for k in ("rationale", "novelty_claim", "predictions", "prior_work_ids", "idea",
                           "datasets", "baselines", "metrics", "risks") if not hyp.get(k)]
    assert missing == []


class _ThinkingHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("reasoning_effort") == "none":
            _reply(self, '{"answer": 3}')
        else:  # Ollama with a thinking model: the token budget is spent on reasoning, content stays empty
            _reply(self, "", finish="length", reasoning="let me think about this for a long time ...")

    def log_message(self, *a):
        pass


def test_openai_compat_turns_thinking_off_when_reasoning_consumes_the_answer():
    srv = _serve(_ThinkingHandler)
    try:
        be = _local(srv, json_mode="object", extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        out = be.complete_json("q", "sys", SCHEMA, role="generate", retries=0)
    finally:
        srv.shutdown()
    assert out == {"answer": 3}
    assert be.budget.used["local"] == 1   # recovered inside one call, not by burning retries


class _OkResponse:
    status_code = 200
    text = ""

    def json(self):
        return {"choices": [{"message": {"content": '{"answer": 5}'}, "finish_reason": "stop"}]}


def _down_then_up(monkeypatch, failures):
    import requests
    import time as _time

    state = {"posts": 0, "slept": []}

    def fake_post(url, **kw):
        state["posts"] += 1
        if state["posts"] <= failures:
            raise requests.ConnectionError("connection refused")
        return _OkResponse()

    monkeypatch.setattr("cosci.llm.openai_compat.requests.post", fake_post)
    monkeypatch.setattr(_time, "sleep", state["slept"].append)
    return state


def test_openai_compat_survives_a_server_restart(monkeypatch):
    state = _down_then_up(monkeypatch, failures=6)
    be = OpenAICompatBackend("local", {"type": "openai_compatible", "base_url": "http://127.0.0.1:9/v1", "model": "m"},
                             budget=CallBudget({"local": 5}))
    assert be.complete_json("q", "sys", SCHEMA, role="generate", retries=0) == {"answer": 5}
    assert state["posts"] == 7
    assert be.budget.used["local"] == 1


def test_openai_compat_gives_up_once_the_server_has_been_down_for_connect_wait(monkeypatch):
    from cosci.llm.base import LLMError

    state = _down_then_up(monkeypatch, failures=10**6)
    be = OpenAICompatBackend("local", {"type": "openai_compatible", "base_url": "http://127.0.0.1:9/v1", "model": "m",
                                       "connect_wait": 30}, budget=CallBudget({"local": 5}))
    with pytest.raises(LLMError):
        be.complete_json("q", "sys", SCHEMA, role="generate", retries=0)
    assert 30 <= sum(state["slept"]) < 90


# --------------------------------------------------------------------------- Claude usage limit
LIMIT_MSG = "You've hit your session limit · resets 5:30am (Asia/Seoul)"


def _fake_claude(tmp_path, limit_hits):
    """A fake `claude` that reports the subscription limit `limit_hits` times, then answers."""
    fake = tmp_path / "claude"
    counter = tmp_path / "invocations"
    fake.write_text(
        "#!" + sys.executable + "\n"
        "import sys, json, os\n"
        f"p = {str(counter)!r}\n"
        "n = int(open(p).read()) if os.path.exists(p) else 0\n"
        "open(p, 'w').write(str(n + 1))\n"
        "sys.stdin.read()\n"
        f"if n < {limit_hits}:\n"
        f"    print(json.dumps({{'type': 'result', 'is_error': True, 'subtype': 'success', 'result': {LIMIT_MSG!r}}}))\n"
        "else:\n"
        "    print(json.dumps({'type': 'result', 'is_error': False, 'result': '{\"answer\": 4}',"
        " 'structured_output': {'answer': 4}, 'total_cost_usd': 0.01}))\n"
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    return fake


def _claude(tmp_path, fake, **cfg):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    be = ClaudeCodeBackend("claude", {"type": "claude_code", "executable": str(fake), "model": "sonnet", **cfg},
                           budget=CallBudget({"claude": 5}), logger=CallLogger(tmp_path / "calls.jsonl"),
                           workdir=tmp_path / "cwd")
    slept = []
    be._sleep = slept.append
    be._now = lambda: datetime(2026, 10, 4, 5, 14, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    return be, slept


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX shell script as fake claude")
def test_claude_backend_waits_for_the_usage_limit_to_reset_instead_of_failing(tmp_path):
    be, slept = _claude(tmp_path, _fake_claude(tmp_path, limit_hits=1))
    out = be.complete_json("What is 2+2?", "sys", SCHEMA, role="evolve", retries=0)
    assert out == {"answer": 4}
    assert len(slept) == 1 and 16 * 60 <= slept[0] <= 18 * 60   # 05:14 -> 05:30 plus a small margin
    assert be.budget.used["claude"] == 1                         # the refused call is not charged


@pytest.mark.skipif(os.name == "nt", reason="uses a POSIX shell script as fake claude")
def test_claude_backend_stops_waiting_after_limit_wait_max(tmp_path):
    from cosci.llm.base import LLMError

    be, slept = _claude(tmp_path, _fake_claude(tmp_path, limit_hits=10**6), limit_wait_max=3600)
    with pytest.raises(LLMError):
        be.complete_json("What is 2+2?", "sys", SCHEMA, role="evolve", retries=0)
    assert 0 < sum(slept) <= 3600


# --------------------------------------------------------------------------- unloading an Ollama model
class _OllamaLikeHandler(BaseHTTPRequestHandler):
    """/v1/chat/completions answers with required keys only; /api/ps and /api/generate behave like Ollama's."""
    loaded: list = []
    unloads: list = []
    requests_seen: list = []

    @classmethod
    def reset(cls, loaded):
        cls.loaded, cls.unloads, cls.requests_seen = list(loaded), [], []

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        type(self).requests_seen.append(("GET", self.path))
        self._json({"models": [{"name": m, "model": m} for m in type(self).loaded]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests_seen.append(("POST", self.path))
        if self.path == "/api/generate":
            type(self).unloads.append(body)
            self._json({"done": True, "done_reason": "unload"})
        else:
            _reply(self, json.dumps(_required_only(body["response_format"]["json_schema"]["schema"])))

    def log_message(self, *a):
        pass


def _close_with(loaded, **cfg):
    _OllamaLikeHandler.reset(loaded)
    srv = _serve(_OllamaLikeHandler)
    try:
        _local(srv, **cfg).close()
    finally:
        srv.shutdown()
    return _OllamaLikeHandler


def test_close_unloads_the_model_from_an_ollama_server_when_asked_to():
    seen = _close_with(["m", "other"], unload_on_exit=True)
    assert seen.unloads == [{"model": "m", "keep_alive": 0}]


def test_close_does_not_touch_a_model_that_is_not_loaded():
    # an unload request for a model that is not in memory must not be sent: it could load 18 GB just to drop it
    seen = _close_with(["other"], unload_on_exit=True)
    assert seen.unloads == []


def test_close_leaves_the_server_alone_by_default():
    seen = _close_with(["m"])
    assert seen.requests_seen == []


def test_duplicate_judge_uses_the_compare_backend_unless_a_dedupe_role_is_configured():
    from cosci.config import load_config
    from cosci.llm import Router

    cfg = load_config(None, preset="mock", overrides={"backends": {"judge": {"type": "mock"}, "other": {"type": "mock"}},
                                                      "roles": {"compare": "judge"}})
    assert Router(cfg, None).backend_for("dedupe").name == "judge"
    cfg["roles"]["dedupe"] = "other"
    assert Router(cfg, None).backend_for("dedupe").name == "other"
