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
