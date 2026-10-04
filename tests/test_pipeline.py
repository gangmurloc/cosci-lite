from pathlib import Path

from cosci import report
from cosci.config import load_config
from cosci.state import RunState
from cosci.supervisor import Supervisor

FIXTURE = Path(__file__).parent / "fixtures" / "papers.json"


def _cfg(tmp_path, **pipeline):
    p = {"n_initial": 6, "gen_batch": 3, "rounds": 2, "evolve_top_k": 3, "matches_per_hyp": 2}
    p.update(pipeline)
    return load_config(None, preset="mock", overrides={
        "literature": {"sources": ["fixture"], "fixture_path": str(FIXTURE), "cache_dir": str(tmp_path / "cache")},
        "pipeline": p,
        "output": {"runs_dir": str(tmp_path / "runs"), "ideas_dir": str(tmp_path / "ideas"),
                   "auto_save_top_k": 2, "auto_save_min_novelty": "N0"},
    })


def test_end_to_end_mock(tmp_path):
    cfg = _cfg(tmp_path)
    st = RunState(run_dir=tmp_path / "runs" / "r1", goal="LLM agent memory research topics for an undergrad")
    Supervisor(cfg, st, log=lambda m: None).run()

    assert st.plan.get("search_queries")
    assert len(st.papers) > 0 and all(p.pid.startswith("P") for p in st.papers.values())
    gen = [h for h in st.hypotheses.values() if h.origin == "generated"]
    evo = [h for h in st.hypotheses.values() if h.origin == "evolved"]
    assert len(gen) == 6
    assert evo and all(h.parents for h in evo)
    assert any("-v" in h.hid for h in evo)
    assert len(st.matches) > 0
    assert "overview" in st.steps_done and "tournament:2" in st.steps_done
    # novelty reviews only cite papers that were given
    for h in st.hypotheses.values():
        nr = h.novelty_review or {}
        for c in nr.get("closest_prior", []):
            assert c["paper_id"] in nr["papers_considered"]
    # report and ideas
    rep = (st.run_dir / "report.md").read_text(encoding="utf-8")
    assert "순위 (Elo 토너먼트)" in rep and "문헌" in rep
    ideas = tmp_path / "ideas"
    idx = (ideas / "INDEX.md").read_text(encoding="utf-8")
    files = [p for p in ideas.glob("*.md") if p.name != "INDEX.md"]
    assert len(files) == 2
    for f in files:
        assert f"]({f.name})" in idx
        text = f.read_text(encoding="utf-8")
        for sec in ("## 한 줄 요약", "## 관련 연구", "## 검증 계획", "## 변경 기록"):
            assert sec in text

    # resume is a no-op for completed steps; elo state persists
    st2 = RunState.load(st.run_dir)
    n_matches = len(st2.matches)
    Supervisor(load_config(None, overrides=st2.config), st2, log=lambda m: None).run()
    assert len(st2.matches) == n_matches


def test_budget_stop_still_reports(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["budget"]["max_calls"]["mock"] = 12
    st = RunState(run_dir=tmp_path / "runs" / "r2", goal="tiny budget")
    Supervisor(cfg, st, log=lambda m: None).run()
    assert (st.run_dir / "report.md").exists()
    assert any(e["kind"] == "budget_stop" for e in st.events)


def test_index_update_is_idempotent(tmp_path):
    report.update_index(tmp_path, "a.md", "Idea A", "summary", "field")
    report.update_index(tmp_path, "b.md", "Idea B", "summary", "field")
    report.update_index(tmp_path, "a.md", "Idea A", "new summary", "field", status="검증 중")
    text = (tmp_path / "INDEX.md").read_text(encoding="utf-8")
    assert text.count("](a.md)") == 1 and text.count("](b.md)") == 1
    assert "new summary" in text and "검증 중" in text


# --------------------------------------------------------------------------- semantic duplicate check
class _ScriptedRouter:
    """Stands in for the LLM boundary: generation returns fresh hypotheses, the duplicate judge returns `verdicts`."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.roles_called = []
        self.n = 0

    def complete(self, role, prompt, system, schema, retries=2):
        import re
        self.roles_called.append(role)
        if role == "generate":
            k = int(re.search(r"Generate exactly (\d+)", prompt).group(1))
            out = []
            for _ in range(k):
                self.n += 1
                t = f"topic{self.n}"     # no shared words, so the TF-IDF proximity check stays out of the way
                out.append({"title": f"{t}a", "one_liner": f"{t}b", "statement": f"{t}c {t}d"})
            return {"hypotheses": out}
        if role == "dedupe":
            return {"groups": self.verdicts.pop(0) if self.verdicts else []}
        raise AssertionError(f"unexpected role {role}")

    def max_workers(self, role):
        return 1


def _ctx(tmp_path, router, hyps=()):
    from cosci.agents import steps as S
    from cosci.models import Hypothesis

    cfg = _cfg(tmp_path)
    st = RunState(run_dir=tmp_path / "runs" / "d", goal="g")
    st.plan = {"title": "t", "core_question": "q"}
    for hid, origin, parents in hyps:
        st.add_hypothesis(Hypothesis(hid=hid, fields={"title": hid, "statement": f"claim of {hid}"}, origin=origin,
                                     parents=list(parents)))
    return S.Context(state=st, router=router, search=None, cfg=cfg, log=lambda m: None)


def test_hypotheses_judged_to_make_the_same_claim_leave_only_the_earliest_active(tmp_path):
    from cosci.agents import steps as S

    router = _ScriptedRouter([[{"ids": ["H7", "H4", "H5"], "shared_claim": "raw turns beat extraction for small readers"},
                               {"ids": ["H2", "H99"], "shared_claim": "refers to an id that does not exist"},
                               {"ids": ["H1"], "shared_claim": "a group of one is not a duplicate"}]])
    ctx = _ctx(tmp_path, router, [(f"H{i}", "generated", []) for i in range(1, 9)])
    S.llm_dedupe(ctx, list(ctx.state.hypotheses.values()))
    status = {h.hid: h.status for h in ctx.state.hypotheses.values()}
    assert status == {"H1": "active", "H2": "active", "H3": "active", "H4": "active", "H5": "duplicate",
                      "H6": "active", "H7": "duplicate", "H8": "active"}
    assert "H4" in ctx.state.hypotheses["H7"].status_reason


def test_duplicate_check_never_removes_older_hypotheses_or_evolved_children(tmp_path):
    from cosci.agents import steps as S

    router = _ScriptedRouter([[{"ids": ["H1", "H2", "H1-v2"], "shared_claim": "same"}]])
    ctx = _ctx(tmp_path, router, [("H1", "generated", []), ("H2", "generated", []), ("H1-v2", "evolved", ["H1"]),
                                  ("H3", "generated", [])])
    S.llm_dedupe(ctx, [ctx.state.hypotheses["H3"]])        # only H3 is new this time
    assert {h.hid: h.status for h in ctx.state.hypotheses.values()} == \
        {"H1": "active", "H2": "active", "H1-v2": "active", "H3": "active"}


def test_generation_replaces_duplicates_so_the_requested_number_stays_distinct(tmp_path):
    from cosci.agents import steps as S

    router = _ScriptedRouter([[{"ids": ["H1", "H4"], "shared_claim": "same"}, {"ids": ["H2", "H6"], "shared_claim": "same"}],
                              []])
    ctx = _ctx(tmp_path, router)
    S.generate(ctx, 6, 0)
    hyps = list(ctx.state.hypotheses.values())
    assert sorted(h.hid for h in hyps if h.status == "duplicate") == ["H4", "H6"]
    assert len([h for h in hyps if h.status == "active"]) == 6
    assert router.roles_called.count("dedupe") == 2        # the replacements are checked too, then it stops


def test_the_local_model_is_unloaded_when_a_run_finishes(tmp_path):
    from test_backends import _OllamaLikeHandler, _serve

    _OllamaLikeHandler.reset(loaded=["m"])
    srv = _serve(_OllamaLikeHandler)
    try:
        cfg = _cfg(tmp_path, rounds=1)
        cfg["backends"]["local"] = {"type": "openai_compatible", "base_url": f"http://127.0.0.1:{srv.server_port}/v1",
                                    "model": "m", "json_mode": "schema", "unload_on_exit": True}
        cfg["roles"]["feedback"] = "local"
        cfg["budget"]["max_calls"]["local"] = 50
        st = RunState(run_dir=tmp_path / "runs" / "u", goal="unload at the end")
        Supervisor(cfg, st, log=lambda m: None).run()
    finally:
        srv.shutdown()
    assert _OllamaLikeHandler.unloads == [{"model": "m", "keep_alive": 0}]
