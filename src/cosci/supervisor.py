"""Supervisor: orchestrates the core loop with resumable steps."""
from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path
from typing import Any, Callable

from . import report
from .agents import steps as S
from .config import expand_path
from .literature import LiteratureSearch
from .llm import BudgetExceeded, Router
from .state import RunState
from .utils import slugify


def new_run_dir(cfg: dict[str, Any], goal: str) -> Path:
    base = expand_path(cfg["output"].get("runs_dir") or "runs") or Path("runs")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return base / f"{stamp}-{slugify(goal[:60], 32)}"


class Supervisor:
    def __init__(self, cfg: dict[str, Any], state: RunState, log: Callable[[str], None] = print):
        self.cfg = cfg
        self.state = state
        self.log = log
        state.config = copy.deepcopy(cfg)
        self._base_cost = float(state.cost_usd or 0.0)
        self.router = Router(cfg, state.run_dir, used_calls=state.calls)
        self.search = LiteratureSearch(cfg["literature"])
        self.ctx = S.Context(state=state, router=self.router, search=self.search, cfg=cfg, log=log)

    # ------------------------------------------------------------------ plan
    def plan_steps(self) -> list[tuple[str, Callable[[], Any]]]:
        p = self.cfg["pipeline"]
        ctx = self.ctx
        seq: list[tuple[str, Callable[[], Any]]] = [
            ("parse", lambda: S.parse_goal(ctx)),
            ("literature", lambda: S.explore_literature(ctx)),
            ("generate:0", lambda: S.generate(ctx, int(p["n_initial"]), 0)),
            ("review:0", lambda: S.review_new(ctx, 0)),
            ("tournament:0", lambda: S.run_tournament(ctx, 0)),
        ]
        if p.get("meta_feedback", True):
            seq.append(("feedback:0", lambda: S.meta_feedback(ctx, 0)))
        for r in range(1, int(p["rounds"]) + 1):
            seq += [
                (f"evolve:{r}", (lambda r=r: S.evolve(ctx, r))),
                (f"review:{r}", (lambda r=r: S.review_new(ctx, r))),
                (f"tournament:{r}", (lambda r=r: S.run_tournament(ctx, r))),
            ]
            if p.get("meta_feedback", True) and r < int(p["rounds"]):
                seq.append((f"feedback:{r}", (lambda r=r: S.meta_feedback(ctx, r))))
        seq.append(("overview", lambda: S.overview(ctx)))
        return seq

    # ------------------------------------------------------------------ run
    def run(self) -> RunState:
        try:
            st = self.state
            st.save()
            stopped = None
            for name, fn in self.plan_steps():
                if st.is_done(name):
                    continue
                try:
                    fn()
                except BudgetExceeded as e:
                    stopped = str(e)
                    self.log(f"! 예산 소진으로 중단: {e}")
                    st.log_event("budget_stop", step=name, error=str(e))
                    break
                except KeyboardInterrupt:
                    self.log("! 사용자 중단 — 상태를 저장합니다. `cosci resume <run_dir>`로 이어서 실행 가능")
                    self._sync()
                    st.save()
                    raise
                st.mark_done(name)
                self._sync()
                st.save()
            if stopped and not st.overview:
                # try to still write an overview with whatever budget remains on its backend
                try:
                    S.overview(self.ctx)
                except BudgetExceeded:
                    pass
            self._sync()
            st.save()
            report.write_report(st)
            self._autosave()
            st.save()
            return st
        finally:
            self.router.close()   # e.g. take the local model off the GPU

    def _sync(self) -> None:
        self.state.calls = dict(self.router.budget.used)
        self.state.cost_usd = round(self._base_cost + self.router.cost_usd, 4)

    def _autosave(self) -> None:
        out = self.cfg["output"]
        ideas_dir = expand_path(out.get("ideas_dir"))
        k = int(out.get("auto_save_top_k", 0) or 0)
        if not ideas_dir or k <= 0:
            return
        saved = report.autosave_ideas(self.state, ideas_dir, k, out.get("auto_save_min_novelty", "N2"))
        for path in saved:
            self.log(f"아이디어 저장: {path}")
