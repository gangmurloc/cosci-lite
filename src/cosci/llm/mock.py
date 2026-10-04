"""Deterministic mock backend: fabricates schema-conforming JSON (for tests / dry runs)."""
from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any

from .base import LLMBackend

_WORDS = ("memory retrieval agent consolidation reader budget update conflict evaluation tournament "
          "compression latency benchmark planner critic curriculum routing sparse cache graph "
          "episodic procedural temporal contamination scaling distillation verifier tool").split()


class MockBackend(LLMBackend):
    type_name = "mock"

    def _call(self, prompt: str, system: str, schema: dict[str, Any] | None, role: str):
        seed = int(hashlib.sha256((role + prompt).encode("utf-8")).hexdigest()[:12], 16)
        rng = random.Random(seed)
        pids = sorted(set(re.findall(r"\[(P\d+)\]", prompt)), key=lambda s: int(s[1:]))
        self._last_prompt = prompt
        obj = self._fake(schema or {"type": "object"}, "root", rng, pids, role)
        return json.dumps(obj, ensure_ascii=False), None, {"cost_usd": 0.0}

    def _fake(self, schema: dict[str, Any], key: str, rng: random.Random, pids: list[str], role: str) -> Any:
        typ = schema.get("type")
        if "enum" in schema:
            choices = schema["enum"]
            if key == "verdict":
                return "reject" if rng.random() < 0.12 else "keep"
            return rng.choice(choices)
        if typ == "object":
            return {k: self._fake(v, k, rng, pids, role) for k, v in schema.get("properties", {}).items()}
        if typ == "array":
            if key.endswith("_ids") or key in ("paper_ids",):
                return rng.sample(pids, k=min(2, len(pids))) if pids else []
            n = schema.get("minItems", 2)
            if key == "hypotheses":
                m = re.search(r"Generate exactly (\d+)", self._last_prompt)
                n = int(m.group(1)) if m else 3
            return [self._fake(schema.get("items", {"type": "string"}), key[:-1] if key.endswith("s") else key,
                               rng, pids, role) for _ in range(n)]
        if typ == "integer":
            lo, hi = schema.get("minimum", 1), schema.get("maximum", 5)
            return rng.randint(int(lo), int(hi))
        if typ == "number":
            lo, hi = schema.get("minimum", 0.0), schema.get("maximum", 1.0)
            return round(rng.uniform(float(lo), float(hi)), 2)
        if typ == "boolean":
            return rng.random() < 0.5
        # string
        if key == "paper_id":
            return rng.choice(pids) if pids else ""
        if key in ("title", "name"):
            return " ".join(rng.sample(_WORDS, 4)).title()
        if key.endswith("queries") or key == "query":
            return " ".join(rng.sample(_WORDS, 3))
        return f"[mock] {key} " + " ".join(rng.sample(_WORDS, 5))
