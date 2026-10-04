"""LLM backend factory and role router."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BudgetExceeded, CallBudget, CallLogger, LLMBackend, LLMError
from .claude_code import ClaudeCodeBackend
from .mock import MockBackend
from .openai_compat import OpenAICompatBackend

_TYPES = {
    "claude_code": ClaudeCodeBackend,
    "openai_compatible": OpenAICompatBackend,
    "mock": MockBackend,
}


def make_backend(name: str, cfg: dict[str, Any], budget: CallBudget, logger: CallLogger,
                 workdir: Path | None) -> LLMBackend:
    typ = cfg.get("type")
    if typ not in _TYPES:
        raise ValueError(f"backend {name!r}: unknown type {typ!r} (choose {list(_TYPES)})")
    return _TYPES[typ](name, cfg, budget=budget, logger=logger, workdir=workdir)


class Router:
    """Maps roles -> backend instances (instantiated lazily, only for backends in use)."""

    def __init__(self, config: dict[str, Any], run_dir: Path | None, used_calls: dict[str, int] | None = None):
        self.config = config
        limits = (config.get("budget") or {}).get("max_calls") or {}
        self.budget = CallBudget(limits, used_calls)
        self.logger = CallLogger(run_dir / "calls.jsonl" if run_dir else None)
        self.workdir = (run_dir / "_claude_cwd") if run_dir else None
        self._backends: dict[str, LLMBackend] = {}

    def backend_for(self, role: str) -> LLMBackend:
        name = self.config["roles"][role]
        if name not in self._backends:
            self._backends[name] = make_backend(name, self.config["backends"][name], self.budget,
                                                self.logger, self.workdir)
        return self._backends[name]

    def complete(self, role: str, prompt: str, system: str, schema: dict[str, Any], retries: int = 2) -> dict[str, Any]:
        lang = self.config.get("language", "ko")
        if lang == "ko":
            prompt += ("\n\n(Reminder: write ALL free-text JSON values in Korean, keeping technical terms, "
                       "model/dataset names and paper titles in English; search_queries and title_en stay in English.)")
        return self.backend_for(role).complete_json(prompt, system, schema, role=role, retries=retries)

    def max_workers(self, role: str) -> int:
        return max(1, self.backend_for(role).max_workers)

    @property
    def cost_usd(self) -> float:
        return sum(b.cost_usd for b in self._backends.values())


__all__ = ["Router", "LLMBackend", "LLMError", "BudgetExceeded", "CallBudget", "CallLogger", "make_backend"]
