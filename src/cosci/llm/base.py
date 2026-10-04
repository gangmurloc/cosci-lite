"""Common LLM backend interface: JSON completion with validation, retries, budget, logging."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..utils import JSONExtractError, extract_json, now_iso


class LLMError(RuntimeError):
    pass


class RateLimited(LLMError):
    """The backend refused the call because a usage limit is active. `reset_at` is when it lifts, if known."""

    def __init__(self, message: str, reset_at: datetime | None = None):
        super().__init__(message)
        self.reset_at = reset_at


class BudgetExceeded(RuntimeError):
    pass


class CallBudget:
    """Thread-safe per-backend call counter with limits."""

    def __init__(self, limits: dict[str, int] | None = None, used: dict[str, int] | None = None):
        self.limits = dict(limits or {})
        self.used = dict(used or {})
        self._lock = threading.Lock()

    def take(self, backend: str) -> None:
        with self._lock:
            limit = self.limits.get(backend)
            n = self.used.get(backend, 0)
            if limit is not None and n >= limit:
                raise BudgetExceeded(f"call budget exhausted for backend {backend!r} ({n}/{limit})")
            self.used[backend] = n + 1

    def refund(self, backend: str) -> None:
        """Give back a call that the backend refused before doing any work (usage limit)."""
        with self._lock:
            self.used[backend] = max(0, self.used.get(backend, 0) - 1)


class CallLogger:
    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()

    def log(self, **rec: Any) -> None:
        if not self.path:
            return
        rec = {"t": now_iso(), **rec}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- schema helpers
_DEFAULTS = {"string": "", "array": [], "object": {}, "integer": 0, "number": 0.0, "boolean": False}


def coerce(obj: Any, schema: dict[str, Any], path: str = "$") -> tuple[Any, list[str]]:
    """Lenient validation: fixes small issues (missing nested fields, enum case) and reports hard errors.

    Hard errors (returned) = wrong top-level type or missing *top-level* required keys.
    """
    errors: list[str] = []
    typ = schema.get("type")
    if typ == "object":
        if not isinstance(obj, dict):
            return obj, [f"{path}: expected object"]
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in obj or obj[key] is None:
                sub_t = props.get(key, {}).get("type", "string")
                if path == "$":
                    errors.append(f"missing required key {key!r}")
                else:
                    obj[key] = _DEFAULTS.get(sub_t, "")
        for key, sub in props.items():
            if key in obj and obj[key] is not None:
                obj[key], sub_err = coerce(obj[key], sub, f"{path}.{key}")
                if path == "$":
                    errors.extend(sub_err)
        return obj, errors
    if typ == "array":
        if isinstance(obj, (str, dict)):
            obj = [obj]
        if not isinstance(obj, list):
            return [], ([f"{path}: expected array"] if path.count(".") <= 1 else [])
        items = schema.get("items", {})
        out = []
        for i, it in enumerate(obj):
            v, sub_err = coerce(it, items, f"{path}[{i}]")
            out.append(v)
        return out, []
    if "enum" in schema:
        allowed = schema["enum"]
        if obj not in allowed:
            s = str(obj).strip()
            for a in allowed:
                if s.upper() == str(a).upper() or s.upper().startswith(str(a).upper()):
                    return a, []
            return allowed[0], []
        return obj, []
    if typ == "integer":
        try:
            return int(round(float(obj))), []
        except Exception:
            return 0, []
    if typ == "number":
        try:
            return float(obj), []
        except Exception:
            return 0.0, []
    if typ == "boolean":
        if isinstance(obj, str):
            return obj.strip().lower() in ("true", "yes", "1"), []
        return bool(obj), []
    if typ == "string" and not isinstance(obj, str):
        if isinstance(obj, (list, dict)):
            return json.dumps(obj, ensure_ascii=False), []
        return str(obj), []
    return obj, []


def schema_hint(schema: dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False)


class LLMBackend:
    """Subclasses implement _call(prompt, system, schema, role) -> (text, structured_or_None, meta)."""

    type_name = "base"

    def __init__(self, name: str, cfg: dict[str, Any], budget: CallBudget | None = None,
                 logger: CallLogger | None = None, workdir: Path | None = None):
        self.name = name
        self.cfg = cfg
        self.budget = budget or CallBudget()
        self.logger = logger or CallLogger(None)
        self.workdir = workdir
        self.max_workers = int(cfg.get("max_workers", 1))
        self.cost_usd = 0.0
        self._cost_lock = threading.Lock()
        self._sleep = time.sleep
        self._now = lambda: datetime.now().astimezone()

    def _call(self, prompt: str, system: str, schema: dict[str, Any] | None, role: str) -> tuple[str, Any, dict[str, Any]]:
        raise NotImplementedError

    def complete_json(self, prompt: str, system: str, schema: dict[str, Any], role: str,
                      retries: int = 2) -> dict[str, Any]:
        """Return a dict conforming (leniently) to `schema`."""
        full_prompt = (
            prompt
            + "\n\n# OUTPUT FORMAT\nRespond with ONLY one JSON object (no prose, no code fences) matching this JSON Schema:\n"
            + schema_hint(schema)
        )
        last_err = ""
        attempt = 0
        limit_waited = 0.0
        while True:
            self.budget.take(self.name)
            p = full_prompt if not last_err else (
                full_prompt + f"\n\n# PREVIOUS ATTEMPT FAILED\n{last_err}\nReturn valid JSON only.")
            t0 = time.time()
            text, structured, meta = "", None, {}
            try:
                text, structured, meta = self._call(p, system, schema, role)
                obj = structured if isinstance(structured, dict) else extract_json(text)
                if isinstance(obj, list) and schema.get("type") == "object":
                    # some models return the array directly; wrap if schema has a single array field
                    arr_keys = [k for k, v in schema.get("properties", {}).items() if v.get("type") == "array"]
                    if len(arr_keys) == 1:
                        obj = {arr_keys[0]: obj}
                obj, errors = coerce(obj, schema)
                if errors:
                    raise JSONExtractError("; ".join(errors[:5]))
                cost = float(meta.get("cost_usd", 0.0) or 0.0)
                with self._cost_lock:
                    self.cost_usd += cost
                self.logger.log(role=role, backend=self.name, attempt=attempt, ok=True,
                                latency=round(time.time() - t0, 2), prompt_chars=len(p),
                                response_chars=len(text or ""), cost_usd=cost)
                return obj
            except RateLimited as e:
                # A usage limit is not a failed attempt: give the call back, wait for the limit to lift and go on.
                # Failing here instead silently dropped whole evolve/novelty jobs while the limit was active.
                self.budget.refund(self.name)
                wait = self._limit_wait(e)
                if limit_waited + wait > float(self.cfg.get("limit_wait_max", 6 * 3600)):
                    self.logger.log(role=role, backend=self.name, attempt=attempt, ok=False,
                                    latency=round(time.time() - t0, 2), prompt_chars=len(p),
                                    response_chars=0, error=str(e)[:500])
                    raise LLMError(f"[{self.name}/{role}] usage limit did not lift within limit_wait_max: {e}") from e
                until = self._now() + timedelta(seconds=wait)
                self.logger.log(role=role, backend=self.name, wait=round(wait),
                                until=until.isoformat(timespec="seconds"), reason="usage_limit")
                print(f"      ! {self.name} 사용량 한도 — {until:%H:%M}까지 기다렸다가 이어서 진행 ({wait / 60:.0f}분)",
                      flush=True)
                self._sleep(wait)
                limit_waited += wait
            except (JSONExtractError, LLMError, ValueError) as e:
                last_err = str(e)[:500]
                self.logger.log(role=role, backend=self.name, attempt=attempt, ok=False,
                                latency=round(time.time() - t0, 2), prompt_chars=len(p),
                                response_chars=len(text or ""), error=last_err)
                if attempt >= retries:
                    raise LLMError(f"[{self.name}/{role}] failed after {retries + 1} attempts: {last_err}") from e
                time.sleep(min(2 ** attempt, 8))
                attempt += 1

    def _limit_wait(self, e: RateLimited) -> float:
        """Seconds to wait before retrying a rate-limited call."""
        if e.reset_at is None:
            return float(self.cfg.get("limit_poll", 300))   # reset time unknown: check again later
        return max(0.0, (e.reset_at - self._now()).total_seconds()) + 60   # a minute of margin past the reset

    def ping(self) -> str:
        obj = self.complete_json("Reply with ok=true.", "You are a test endpoint.",
                                 {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
                                 role="ping", retries=0)
        return "ok" if obj.get("ok") else f"unexpected: {obj}"
