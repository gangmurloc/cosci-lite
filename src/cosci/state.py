"""Persistent run state (the system's 'context memory')."""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Hypothesis, Match, Paper
from .utils import normalize_title, now_iso


@dataclass
class RunState:
    run_dir: Path
    goal: str = ""
    created: str = field(default_factory=now_iso)
    config: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    digest: dict[str, Any] = field(default_factory=dict)
    papers: dict[str, Paper] = field(default_factory=dict)
    hypotheses: dict[str, Hypothesis] = field(default_factory=dict)
    matches: list[Match] = field(default_factory=list)
    feedback: list[dict[str, Any]] = field(default_factory=list)
    overview: dict[str, Any] = field(default_factory=dict)
    steps_done: list[str] = field(default_factory=list)
    calls: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    saved_ideas: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ papers
    def add_paper(self, data: dict[str, Any], query: str = "") -> Paper:
        """Register a paper (dedupe by normalized title / arXiv id). Returns the stored Paper."""
        with self._lock:
            key = normalize_title(data.get("title", ""))
            arx = (data.get("arxiv_id") or "").strip()
            for p in self.papers.values():
                if (arx and p.arxiv_id == arx) or (key and normalize_title(p.title) == key):
                    if query and query not in p.queries:
                        p.queries.append(query)
                    # fill missing fields
                    for k in ("abstract", "arxiv_id", "doi", "url", "venue"):
                        if not getattr(p, k) and data.get(k):
                            setattr(p, k, data[k])
                    if p.year is None and data.get("year"):
                        p.year = data["year"]
                    return p
            pid = f"P{len(self.papers) + 1}"
            paper = Paper(pid=pid, **{k: v for k, v in data.items() if k in Paper.__dataclass_fields__ and k != "pid"})
            if query:
                paper.queries.append(query)
            self.papers[pid] = paper
            return paper

    # -------------------------------------------------------------- hypotheses
    def next_hid(self) -> str:
        with self._lock:
            nums = [int(m.group(1)) for h in self.hypotheses for m in [re.match(r"H(\d+)", h)] if m]
            return f"H{(max(nums) if nums else 0) + 1}"

    def next_version_id(self, parent: str) -> str:
        with self._lock:
            base = parent.split("-v")[0]
            versions = [1]
            for h in self.hypotheses:
                if h == base:
                    continue
                m = re.match(rf"^{re.escape(base)}-v(\d+)$", h)
                if m:
                    versions.append(int(m.group(1)))
            return f"{base}-v{max(versions) + 1}"

    def add_hypothesis(self, h: Hypothesis) -> Hypothesis:
        with self._lock:
            if h.hid in self.hypotheses:
                raise ValueError(f"duplicate hypothesis id {h.hid}")
            self.hypotheses[h.hid] = h
            return h

    def active(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses.values() if h.status == "active"]

    def ranked(self, include_inactive: bool = False) -> list[Hypothesis]:
        hs = list(self.hypotheses.values()) if include_inactive else self.active()
        return sorted(hs, key=lambda h: (-h.elo, -h.wins, h.hid))

    # ------------------------------------------------------------------ events
    def log_event(self, kind: str, **data: Any) -> None:
        with self._lock:
            self.events.append({"t": now_iso(), "kind": kind, **data})

    def mark_done(self, step: str) -> None:
        with self._lock:
            if step not in self.steps_done:
                self.steps_done.append(step)

    def is_done(self, step: str) -> bool:
        return step in self.steps_done

    # ------------------------------------------------------------------ io
    @property
    def path(self) -> Path:
        return self.run_dir / "state.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "created": self.created,
            "config": self.config,
            "plan": self.plan,
            "digest": self.digest,
            "papers": {k: p.to_dict() for k, p in self.papers.items()},
            "hypotheses": {k: h.to_dict() for k, h in self.hypotheses.items()},
            "matches": [m.to_dict() for m in self.matches],
            "feedback": self.feedback,
            "overview": self.overview,
            "steps_done": self.steps_done,
            "calls": self.calls,
            "cost_usd": self.cost_usd,
            "saved_ideas": self.saved_ideas,
            "events": self.events,
        }

    def save(self) -> None:
        with self._lock:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)

    @classmethod
    def load(cls, run_dir: str | os.PathLike) -> "RunState":
        run_dir = Path(run_dir)
        with open(run_dir / "state.json", "r", encoding="utf-8") as f:
            d = json.load(f)
        st = cls(run_dir=run_dir)
        st.goal = d.get("goal", "")
        st.created = d.get("created", st.created)
        st.config = d.get("config", {})
        st.plan = d.get("plan", {})
        st.digest = d.get("digest", {})
        st.papers = {k: Paper.from_dict(v) for k, v in d.get("papers", {}).items()}
        st.hypotheses = {k: Hypothesis.from_dict(v) for k, v in d.get("hypotheses", {}).items()}
        st.matches = [Match.from_dict(m) for m in d.get("matches", [])]
        st.feedback = d.get("feedback", [])
        st.overview = d.get("overview", {})
        st.steps_done = d.get("steps_done", [])
        st.calls = d.get("calls", {})
        st.cost_usd = d.get("cost_usd", 0.0)
        st.saved_ideas = d.get("saved_ideas", [])
        st.events = d.get("events", [])
        return st
