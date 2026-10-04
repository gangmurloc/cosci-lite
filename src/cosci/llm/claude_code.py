"""Claude Code CLI backend (`claude -p`), runs on the user's logged-in Claude account.

Notes (verified against Claude Code 2.1.x `claude --help` and a live call):
- the prompt is read from stdin when using -p; --output-format json returns
  {"result", "structured_output", "is_error", "subtype", "total_cost_usd", "num_turns", ...}
- --json-schema gives schema-validated output in "structured_output"
- do NOT use --bare: it only accepts ANTHROPIC_API_KEY auth (skips subscription/OAuth login)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .base import LLMBackend, LLMError, RateLimited

WEB_ROLES = {"novelty", "evolve", "digest"}

# e.g. "You've hit your session limit · resets 5:30am (Asia/Seoul)", "... weekly limit · resets Oct 9, 10am (...)"
_LIMIT_RE = re.compile(r"hit your [\w\s-]*limit|usage limit reached|rate limit", re.IGNORECASE)
_RESET_RE = re.compile(r"resets?(?:\s+at)?\s+(?:(?P<mon>[A-Za-z]{3,9})\s+(?P<day>\d{1,2}),?\s+)?"
                       r"(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>am|pm)\s*(?:\((?P<tz>[^)]+)\))?", re.IGNORECASE)
_MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}


def parse_limit_reset(message: str, now: datetime) -> datetime | None:
    """When a usage-limit message says the limit resets, as the next such moment after `now` (None if unstated)."""
    m = _RESET_RE.search(message or "")
    if not m:
        return None
    tz = now.tzinfo
    if m["tz"]:
        try:
            tz = ZoneInfo(m["tz"].strip())
        except Exception:
            pass
    local = now.astimezone(tz)
    hour = int(m["h"]) % 12 + (12 if m["ap"].lower() == "pm" else 0)
    reset = local.replace(hour=hour, minute=int(m["m"] or 0), second=0, microsecond=0)
    if m["mon"]:
        month = _MONTHS.get(m["mon"][:3].lower())
        if not month:
            return None
        reset = reset.replace(month=month, day=int(m["day"]))
        return reset if reset > local else reset.replace(year=reset.year + 1)
    return reset if reset > local else reset + timedelta(days=1)


class ClaudeCodeBackend(LLMBackend):
    type_name = "claude_code"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        exe = self.cfg.get("executable", "claude")
        resolved = shutil.which(exe) or (shutil.which(exe + ".exe") if os.name == "nt" else None) \
            or (shutil.which(exe + ".cmd") if os.name == "nt" else None)
        self.exe = resolved or exe
        self.is_shim = self.exe.lower().endswith((".cmd", ".bat"))
        flag = str(self.cfg.get("json_schema_flag", "auto")).lower()
        self.use_schema_flag = (flag == "on") or (flag == "auto" and not self.is_shim)

    def _cmd(self, schema: dict[str, Any] | None, role: str) -> list[str]:
        cmd = [self.exe, "-p", "--output-format", "json", "--no-session-persistence",
               "--strict-mcp-config", "--permission-mode", "dontAsk"]
        model = self.cfg.get("model")
        if model:
            cmd += ["--model", str(model)]
        use_web = bool(self.cfg.get("allow_web")) and role in WEB_ROLES
        if use_web:
            cmd += ["--tools", "WebSearch,WebFetch", "--allowedTools", "WebSearch,WebFetch"]
        elif not self.is_shim:
            cmd += ["--tools", ""]   # no tools needed; empty arg is unsafe through .cmd shims
        cmd += ["--max-turns", str(int(self.cfg.get("max_turns", 6)))]
        if schema is not None and self.use_schema_flag:
            cmd += ["--json-schema", json.dumps(schema, ensure_ascii=False)]
        return cmd

    def _call(self, prompt: str, system: str, schema: dict[str, Any] | None, role: str):
        stdin_text = (f"<system_instructions>\n{system}\n</system_instructions>\n\n" if system else "") + prompt
        cmd = self._cmd(schema, role)
        cwd = str(self.workdir) if self.workdir else None
        if cwd:
            os.makedirs(cwd, exist_ok=True)
        try:
            proc = subprocess.run(cmd, input=stdin_text.encode("utf-8"), capture_output=True,
                                  timeout=int(self.cfg.get("timeout", 900)), cwd=cwd)
        except FileNotFoundError as e:
            raise LLMError(f"Claude Code executable not found: {self.exe!r}. Install Claude Code or set backends.{self.name}.executable") from e
        except subprocess.TimeoutExpired as e:
            raise LLMError(f"claude -p timed out after {self.cfg.get('timeout', 900)}s") from e
        out = proc.stdout.decode("utf-8", errors="replace")
        err = proc.stderr.decode("utf-8", errors="replace")
        if proc.returncode != 0 and not out.strip():
            self._raise_if_limited(err)
            raise LLMError(f"claude -p exit {proc.returncode}: {err[:400]}")
        try:
            d = json.loads(out)
        except json.JSONDecodeError:
            # fall back: maybe plain text output
            return out, None, {}
        if d.get("is_error"):
            self._raise_if_limited(str(d.get("result")))
            raise LLMError(f"claude -p error ({d.get('subtype')}): {str(d.get('result'))[:400]}")
        meta = {"cost_usd": d.get("total_cost_usd", 0.0), "num_turns": d.get("num_turns"),
                "session_id": d.get("session_id")}
        return str(d.get("result") or ""), d.get("structured_output"), meta

    def _raise_if_limited(self, message: str) -> None:
        if _LIMIT_RE.search(message or ""):
            raise RateLimited(f"claude -p: {message[:300]}", parse_limit_reset(message, self._now()))
