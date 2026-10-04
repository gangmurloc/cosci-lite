import json
import random

import pytest

from cosci import tournament
from cosci.llm.base import coerce
from cosci.utils import JSONExtractError, extract_json, slugify


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": [1, 2]}\n```\nbye') == {"a": [1, 2]}
    assert extract_json('<think>hmm {"x": 0}</think>{"a": "}"}') == {"a": "}"}
    assert extract_json('prefix {"a": {"b": "c"}} suffix') == {"a": {"b": "c"}}
    with pytest.raises(JSONExtractError):
        extract_json("no json here")


def test_slugify():
    assert slugify("Write-time Memory × Reader Scaling!") == "write-time-memory-reader-scaling"
    s = slugify("한국어 제목")
    assert s.startswith("idea-") and len(s) > 6


def test_elo_update_zero_sum_and_direction():
    a, b = tournament.update(1200, 1200, 1.0, 32)
    assert a == pytest.approx(1216) and b == pytest.approx(1184)
    a2, b2 = tournament.update(1300, 1100, 0.5, 32)
    assert a2 < 1300 and b2 > 1100
    assert (a2 + b2) == pytest.approx(2400)


def test_schedule_counts_and_no_duplicates():
    ids = [f"H{i}" for i in range(1, 9)]
    elo = {i: 1200 + 10 * n for n, i in enumerate(ids)}
    pairs = tournament.schedule(ids, elo, {}, [], 3, random.Random(0), priority=["H8"])
    keys = [frozenset(p) for p in pairs]
    assert len(keys) == len(set(keys))
    counts = {i: 0 for i in ids}
    for a, b in pairs:
        counts[a] += 1
        counts[b] += 1
    assert min(counts.values()) >= 2


def test_coerce_enum_and_required():
    schema = {"type": "object", "required": ["winner", "items"],
              "properties": {"winner": {"type": "string", "enum": ["A", "B"]},
                             "items": {"type": "array", "items": {"type": "object", "required": ["t"],
                                                                  "properties": {"t": {"type": "string"}}}}}}
    obj, err = coerce({"winner": "b", "items": [{}]}, schema)
    assert not err and obj["winner"] == "B" and obj["items"][0]["t"] == ""
    _, err = coerce({"items": []}, schema)
    assert err and "winner" in err[0]


def test_limit_reset_time_is_read_in_its_own_timezone():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from cosci.llm.claude_code import parse_limit_reset

    seoul = ZoneInfo("Asia/Seoul")
    before = datetime(2026, 10, 4, 5, 14, tzinfo=seoul)
    msg = "You've hit your session limit · resets 5:30am (Asia/Seoul)"
    assert parse_limit_reset(msg, before) == datetime(2026, 10, 4, 5, 30, tzinfo=seoul)
    # the stated time has already passed today -> it means tomorrow
    assert parse_limit_reset(msg, datetime(2026, 10, 4, 6, 0, tzinfo=seoul)) == datetime(2026, 10, 5, 5, 30, tzinfo=seoul)
    assert parse_limit_reset("You've hit your session limit · resets 3pm (Asia/Seoul)", before) == \
        datetime(2026, 10, 4, 15, 0, tzinfo=seoul)
    assert parse_limit_reset("You've hit your weekly limit · resets Oct 9, 10am (Asia/Seoul)", before) == \
        datetime(2026, 10, 9, 10, 0, tzinfo=seoul)
    # a caller in another timezone still gets the same instant
    utc_now = datetime(2026, 10, 3, 20, 14, tzinfo=ZoneInfo("UTC"))
    assert parse_limit_reset(msg, utc_now) == datetime(2026, 10, 4, 5, 30, tzinfo=seoul)
    assert parse_limit_reset("Credit balance is too low", before) is None
    assert parse_limit_reset("You've hit your session limit", before) is None


def _state_with(tmp_path, specs):
    """specs: (hid, elo, parents)"""
    from cosci.models import Hypothesis
    from cosci.state import RunState

    st = RunState(run_dir=tmp_path, goal="g")
    for hid, elo, parents in specs:
        st.add_hypothesis(Hypothesis(hid=hid, fields={"title": f"title of {hid}"}, parents=list(parents), elo=elo,
                                     origin="evolved" if parents else "generated"))
    return st


RUN2 = [("H7-v2", 1271, ["H7"]), ("H7-v4", 1261, ["H7-v2"]), ("H8-v3", 1260, ["H8-v2"]), ("H7", 1255, []),
        ("H8-v2", 1234, ["H8"]), ("H10", 1220, ["H7", "H7-v2"]), ("H9", 1201, ["H7"]), ("H8", 1183, []),
        ("H3", 1075, [])]


def test_top_ranks_within_one_game_of_the_leader_are_reported_as_tied(tmp_path):
    from cosci import report

    st = _state_with(tmp_path, RUN2)
    assert report.tied_with_leader(st.ranked(), elo_k=32) == ["H7-v2", "H7-v4", "H8-v3", "H7"]
    clear = _state_with(tmp_path, [("H1", 1300, []), ("H2", 1200, [])])
    assert report.tied_with_leader(clear.ranked(), elo_k=32) == ["H1"]


def test_hypotheses_are_grouped_by_the_original_idea_they_descend_from(tmp_path):
    from cosci import report

    st = _state_with(tmp_path, RUN2)
    assert report.lineage_root(st, "H7-v4") == "H7"      # grandchild
    assert report.lineage_root(st, "H10") == "H7"        # combination: first parent's family
    assert report.lineage_root(st, "H8-v3") == "H8"
    assert report.lineage_root(st, "H3") == "H3"
    assert report.families(st, st.ranked()) == {"H7": ["H7-v2", "H7-v4", "H7", "H10", "H9"],
                                                "H8": ["H8-v3", "H8-v2", "H8"], "H3": ["H3"]}


def test_report_says_which_top_ranks_are_indistinguishable_and_which_family_each_belongs_to(tmp_path):
    from cosci import report

    st = _state_with(tmp_path, RUN2)
    st.config = {"pipeline": {"elo_k": 32}}
    rep = report.render_report(st)
    tie_lines = [line for line in rep.splitlines() if "H7-v2, H7-v4, H8-v3, H7" in line]
    assert len(tie_lines) == 1 and "H8-v2" not in tie_lines[0]
    row = next(line for line in rep.splitlines() if line.startswith("| 2 | H7-v4 |"))
    assert row.rstrip().endswith("| H7 |")
