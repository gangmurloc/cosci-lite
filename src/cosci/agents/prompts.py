"""Prompt templates and JSON schemas for every agent role."""
from __future__ import annotations

import json
from typing import Any, Iterable

from ..models import Hypothesis, Paper
from ..utils import truncate

# ============================================================================ system
_LANG = {
    "ko": ("Language: write every free-text value in Korean. Keep technical terms, model/dataset/benchmark "
           "names and paper titles in English. JSON keys stay in English. search_queries must be in English."),
    "en": "Language: English.",
}

SYSTEM_TEMPLATE = """You are one agent inside "cosci-lite", an AI co-scientist system that helps a researcher
discover defensible research topics in Computer Science / AI (LLMs, agents, LLM memory, RAG, ML systems,
software engineering). Optimise for scientific defensibility and information gain per unit of researcher time,
not for sounding innovative.

Rules:
1. Never fabricate papers, authors, numbers, benchmarks, datasets or results. Refer to literature ONLY through
   the PAPERS list given in the prompt, using ids like [P3]. If nothing in PAPERS supports a claim, say so.
2. Keep evidence levels distinct: [VERIFIED] (directly supported by a listed paper), [SUPPORTED INFERENCE],
   [HYPOTHESIS], [SPECULATION].
3. Novelty is an empirical question. Never call an idea novel just because you have not seen it.
4. Prefer falsifiable hypotheses testable with quantitative experiments under the stated constraints.
5. Consider compute / token cost, baseline fairness, data leakage, benchmark contamination, and whether a gain
   could come from more compute, a stronger model, more context, retrieval budget or prompt tuning.
6. Do not praise. Evaluate.
{lang}"""


def system_prompt(language: str = "ko") -> str:
    return SYSTEM_TEMPLATE.format(lang=_LANG.get(language, _LANG["en"]))


# ============================================================================ schemas
def _s(desc: str = "") -> dict[str, Any]:
    return {"type": "string", "description": desc} if desc else {"type": "string"}


def _arr(items: dict[str, Any] | None = None, desc: str = "") -> dict[str, Any]:
    d: dict[str, Any] = {"type": "array", "items": items or {"type": "string"}}
    if desc:
        d["description"] = desc
    return d


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required}


PLAN_SCHEMA = _obj({
    "title": _s("short working title"),
    "core_question": _s(),
    "motivation": _s(),
    "target_contribution": _s("e.g. empirical finding / evaluation methodology / method / benchmark"),
    "scope_in": _arr(),
    "scope_out": _arr(),
    "constraints": _arr(desc="compute, data, time, budget constraints (explicit or provisional)"),
    "evaluation_criteria": _arr(),
    "key_terms": _arr(),
    "search_queries": _arr(desc="6-10 short English queries (3-7 words) for arXiv / Semantic Scholar"),
    "likely_benchmarks": _arr(),
    "likely_baselines": _arr(),
    "open_questions": _arr(),
}, ["title", "core_question", "search_queries"])

_PID_LIST = _arr(desc="ids from PAPERS, e.g. P3")

DIGEST_SCHEMA = _obj({
    "known_facts": _arr(_obj({"claim": _s(), "paper_ids": _PID_LIST,
                              "label": {"type": "string", "enum": ["VERIFIED", "SUPPORTED INFERENCE"]}},
                             ["claim", "paper_ids"])),
    "limitations": _arr(_obj({"text": _s(), "paper_ids": _PID_LIST}, ["text"])),
    "contradictions": _arr(_obj({"text": _s(), "paper_ids": _PID_LIST}, ["text"])),
    "gaps": _arr(_obj({
        "gap_id": _s("G1, G2, ..."),
        "gap_type": _s("e.g. contradictory findings, untested assumption, missing ablation, evaluation weakness"),
        "known": _s(), "unknown": _s(), "why_it_matters": _s(), "how_to_test": _s(),
        "paper_ids": _PID_LIST,
        "novelty_risk": _s("risk that the gap is already solved, and why"),
    }, ["gap_id", "known", "unknown", "how_to_test"])),
}, ["known_facts", "gaps"])

FEASIBILITY = {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}

HYP_SCHEMA = _obj({
    "title": _s("specific, <= 15 words"),
    "title_en": _s("short English title (used for file names)"),
    "one_liner": _s("one-sentence summary"),
    "strategy": _s(),
    "gap_ids": _arr(),
    "statement": _s("the falsifiable hypothesis itself"),
    "rationale": _s("scientific rationale / mechanism"),
    "prior_work_ids": _PID_LIST,
    "novelty_claim": _s("what exactly would be new relative to the listed papers"),
    "assumptions": _arr(desc="critical assumptions"),
    "predictions": _arr(desc="testable predictions"),
    "falsification": _s("result that would show the hypothesis is wrong"),
    "problem": _s("problem & motivation: which gap, why it matters now"),
    "idea": _s("core idea / method intuition, how it differs from existing methods"),
    "contributions": _arr(desc="expected paper contributions (2-3)"),
    "contribution_type": _s("new method / empirical finding / benchmark / evaluation methodology / ..."),
    "datasets": _arr(), "baselines": _arr(desc="include simplest, strongest and compute-matched baselines"),
    "metrics": _arr(),
    "min_experiment": _s("cheapest informative experiment (1-2 days)"),
    "resources": _s("GPU / API / time estimate under the constraints; mark estimates as estimates"),
    "feasibility": FEASIBILITY,
    "risks": _arr(desc="likely reviewer attacks and responses"),
    "next_steps": _arr(),
}, [
    # Everything the reviewer, the tournament judge (fmt_hyp) and the report read is required. A server that
    # enforces the schema emits required keys only when it likes, so optional fields came back missing for
    # about half of the locally generated hypotheses and those were then judged on a fraction of the content.
    "title", "title_en", "one_liner", "statement", "rationale", "prior_work_ids", "novelty_claim", "assumptions",
    "predictions", "falsification", "problem", "idea", "contributions", "datasets", "baselines", "metrics",
    "min_experiment", "resources", "feasibility", "risks", "next_steps",
])

GEN_SCHEMA = _obj({"hypotheses": _arr(HYP_SCHEMA)}, ["hypotheses"])

SEVERITY = {"type": "string", "enum": ["CRITICAL", "MAJOR", "MINOR"]}
_SCORE = {"type": "integer", "minimum": 1, "maximum": 5}

REVIEW_SCHEMA = _obj({
    "verdict": {"type": "string", "enum": ["keep", "revise", "reject"]},
    "scores": _obj({"alignment": _SCORE, "plausibility": _SCORE, "testability": _SCORE,
                    "feasibility": _SCORE, "impact": _SCORE},
                   ["alignment", "plausibility", "testability", "feasibility", "impact"]),
    "criticisms": _arr(_obj({"severity": SEVERITY, "text": _s()}, ["severity", "text"])),
    "most_fragile_assumption": _s(),
    "baseline_that_could_kill_it": _s(),
    "confound_risks": _arr(),
    "cheapest_falsification": _s(),
    "search_queries": _arr(desc="3-5 short English queries to find the closest prior work, incl. alternative terminology"),
}, ["verdict", "criticisms", "search_queries"])

NOVELTY_SCHEMA = _obj({
    "novelty": {"type": "string", "enum": ["N0", "N1", "N2", "N3", "N4"]},
    "closest_prior": _arr(_obj({"paper_id": _s("id from PAPERS"), "overlap": _s(), "difference": _s()},
                               ["paper_id", "overlap", "difference"])),
    "substantive_difference": {"type": "boolean"},
    "rationale": _s(),
    "remaining_uncertainty": _s(),
    "suggested_reformulation": _s("a sharper version that would survive the novelty threat, or empty"),
    "extra_refs": _arr(_obj({"title": _s(), "url": _s(), "note": _s()}, ["title"]),
                       desc="ONLY if you used web search tools: papers not in PAPERS (will be marked unverified)"),
}, ["novelty", "closest_prior", "rationale"])

COMPARE_SCHEMA = _obj({
    "strengths_a": _arr(), "weaknesses_a": _arr(),
    "strengths_b": _arr(), "weaknesses_b": _arr(),
    "deciding_factor": _s(),
    "winner": {"type": "string", "enum": ["A", "B"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
}, ["deciding_factor", "winner"])

DEBATE_SCHEMA = _obj({
    "debate": _arr(_obj({"speaker": _s("advocate_A | advocate_B | skeptic"), "point": _s()}, ["speaker", "point"])),
    "key_assumption": _s("the assumption that decides the comparison"),
    "deciding_factor": _s(),
    "winner": {"type": "string", "enum": ["A", "B"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
}, ["deciding_factor", "winner"])

EVOLVE_SCHEMA = _obj({
    "hypothesis": HYP_SCHEMA,
    "changes": _arr(desc="what changed vs the parent(s)"),
    "addressed_critiques": _arr(),
    "new_risks": _arr(),
}, ["hypothesis", "changes"])

FEEDBACK_SCHEMA = _obj({
    "recurring_weaknesses": _arr(),
    "overused_assumptions": _arr(),
    "underexplored_directions": _arr(),
    "guidance_generation": _arr(desc="short imperative guidance for the next generation/evolution round"),
    "guidance_review": _arr(),
}, ["recurring_weaknesses", "guidance_generation"])

OVERVIEW_SCHEMA = _obj({
    "summary": _s("current state of knowledge and where the strongest opportunities are"),
    "directions": _arr(_obj({"name": _s(), "why": _s(), "hypothesis_ids": _arr(), "first_experiment": _s()},
                            ["name", "why"])),
    "key_uncertainties": _arr(),
    "next_literature": _arr(desc="what to read/search next (topics or listed paper ids)"),
    "recommended_next_action": _s(),
}, ["summary", "directions"])

# ============================================================================ strategies
GEN_STRATEGIES = {
    "limitation_extension": "Extend a limitation explicitly stated in a listed paper.",
    "assumption_challenge": "Challenge an assumption shared by several existing methods.",
    "cross_subfield_combination": "Combine mechanisms from different research subfields.",
    "failure_explanation": "Explain a known failure mode or contradictory finding.",
    "efficiency": "Reduce compute, latency, memory or data requirements while preserving quality.",
    "adaptive": "Replace fixed behaviour with adaptive behaviour.",
    "cross_domain_transfer": "Transfer a mechanism from another domain.",
    "measurement": "Design a better measurement or evaluation methodology.",
    "counterintuitive": "Derive a counter-intuitive but testable prediction.",
    "simplification": "Simplify a complex method while preserving performance.",
    "why_it_works": "Investigate why an existing method works (mechanism).",
    "where_it_breaks": "Investigate where an existing method stops working (boundary conditions).",
}

EVOLVE_STRATEGIES = {
    "grounding": "Strengthen the hypothesis with literature evidence; fill reasoning gaps; fix claims that "
                 "the listed papers contradict.",
    "feasibility": "Make it practical under the constraints: cheaper experiment, smaller models, public data, "
                   "clear baselines; repair invalid assumptions.",
    "combination": "Combine the strongest elements of the two parent hypotheses into one coherent hypothesis.",
    "simplification": "Simplify for easier verification: fewer moving parts, one sharp falsifiable claim.",
    "out_of_box": "Move away from the parent: propose a divergent hypothesis addressing the same goal from a "
                  "different angle (different mechanism or evaluation).",
}


# ============================================================================ formatting
def fmt_papers(papers: Iterable[Paper], abstract_chars: int = 450) -> str:
    lines = []
    for p in papers:
        meta = ", ".join(x for x in [str(p.year or ""), f"arXiv:{p.arxiv_id}" if p.arxiv_id else p.venue] if x)
        lines.append(f"[{p.pid}] {p.title} ({meta})\n    {truncate(p.abstract, abstract_chars)}")
    return "\n".join(lines) if lines else "(no papers retrieved)"


def fmt_plan(plan: dict[str, Any]) -> str:
    keys = ["title", "core_question", "motivation", "target_contribution", "scope_in", "scope_out",
            "constraints", "evaluation_criteria"]
    return json.dumps({k: plan.get(k) for k in keys if plan.get(k)}, ensure_ascii=False, indent=1)


def fmt_hyp(h: Hypothesis, detail: str = "full") -> str:
    f = h.fields
    if detail == "title":
        return f"{h.hid}: {h.title}"
    d: dict[str, Any] = {"id": h.hid, "title": f.get("title"), "statement": f.get("statement"),
                         "rationale": f.get("rationale"), "novelty_claim": f.get("novelty_claim"),
                         "assumptions": f.get("assumptions"), "predictions": f.get("predictions"),
                         "falsification": f.get("falsification"), "min_experiment": f.get("min_experiment"),
                         "feasibility": f.get("feasibility"), "resources": f.get("resources"),
                         "prior_work_ids": f.get("prior_work_ids")}
    if detail == "full":
        d.update({"idea": f.get("idea"), "baselines": f.get("baselines"), "datasets": f.get("datasets"),
                  "metrics": f.get("metrics"), "risks": f.get("risks")})
    if h.initial_review:
        crit = h.initial_review.get("criticisms") or []
        d["review_criticisms"] = [f"[{c.get('severity')}] {c.get('text')}" for c in crit[:4]]
    if h.novelty_review:
        nr = h.novelty_review
        d["novelty_review"] = {"label": nr.get("novelty"), "rationale": truncate(nr.get("rationale"), 400),
                               "closest_prior": [c.get("paper_id") for c in nr.get("closest_prior") or []][:3]}
    return json.dumps(d, ensure_ascii=False, indent=1)


def _feedback_block(feedback: list[str]) -> str:
    if not feedback:
        return ""
    return "\n# META-REVIEW FEEDBACK FROM PREVIOUS ROUNDS (use selectively, do not overfit)\n- " + "\n- ".join(feedback)


# ============================================================================ prompts
def parse_prompt(goal: str) -> str:
    return f"""# TASK: Research goal parsing (Research Plan Configuration)
Convert the researcher's goal into a precise research specification. Do NOT propose solutions yet.
If information is missing, make provisional assumptions and state them in constraints / open_questions.
search_queries: 6-10 short English queries covering the core topic, alternative terminology, competing
approaches, known limitations / negative results, and benchmarks.

# RESEARCH GOAL (from the researcher)
{goal}
"""


def digest_prompt(plan: dict[str, Any], papers: list[Paper]) -> str:
    return f"""# TASK: Literature digest
From the PAPERS below, extract what is known, stated limitations, contradictions, and 3-7 research gaps that are
relevant to the research plan. Every fact must cite paper ids. Rank gaps by novelty potential, scientific value,
feasibility and experimental clarity (most promising first). Do not confuse "not mentioned in these papers"
with "not studied by the field": state the novelty_risk for each gap.

# RESEARCH PLAN
{fmt_plan(plan)}

# PAPERS
{fmt_papers(papers)}
"""


def generate_prompt(plan: dict[str, Any], digest: dict[str, Any], papers: list[Paper], n: int,
                    strategies: list[str], existing: list[Hypothesis], feedback: list[str]) -> str:
    strat = "\n".join(f"- {s}: {GEN_STRATEGIES[s]}" for s in strategies)
    exist = "\n".join(f"- {fmt_hyp(h, 'title')}" for h in existing) or "(none yet)"
    gaps = json.dumps(digest.get("gaps", []), ensure_ascii=False, indent=1)
    return f"""# TASK: Hypothesis generation
Generate exactly {n} competing research hypotheses for the research plan. Use one strategy per hypothesis from
the list (assign them in order; set the `strategy` field). Hypotheses must be genuinely different (not cosmetic
variants) and different from the EXISTING list. Each must be falsifiable, testable under the constraints, and
grounded in the PAPERS (cite ids in prior_work_ids). State novelty claims relative to specific papers.
Do not invent effect sizes: if a prediction or falsification needs a threshold, mark it as a provisional
threshold to be fixed after a pilot, rather than presenting it as expected from prior work.

# STRATEGIES (use in this order)
{strat}

# RESEARCH PLAN
{fmt_plan(plan)}

# KNOWLEDGE GAPS (from the literature digest)
{gaps}

# EXISTING HYPOTHESES (avoid duplicates)
{exist}
{_feedback_block(feedback)}

# PAPERS
{fmt_papers(papers)}
"""


def review_prompt(plan: dict[str, Any], h: Hypothesis, papers: list[Paper]) -> str:
    return f"""# TASK: Initial review (skeptical peer reviewer)
Find reasons this hypothesis might be wrong or not worth testing. Classify each criticism as CRITICAL (may
invalidate the hypothesis or contribution), MAJOR (needs substantial revision) or MINOR. Check: factual
correctness, consistency with the PAPERS, hidden assumptions and confounders, baseline fairness, whether a gain
could come from more compute / stronger model / more context / prompt tuning, leakage, statistical validity,
feasibility under the constraints. verdict = reject only for CRITICAL problems that cannot be repaired.
Also write 3-5 English search queries (with alternative terminology) to find the closest prior work.

# RESEARCH PLAN
{fmt_plan(plan)}

# HYPOTHESIS
{fmt_hyp(h)}

# PAPERS (context)
{fmt_papers(papers, 250)}
"""


def novelty_prompt(plan: dict[str, Any], h: Hypothesis, papers: list[Paper], allow_web: bool) -> str:
    web = ("You MAY use web search to find additional closest prior work; list such papers in extra_refs with "
           "a URL (they will be marked unverified). Still cite PAPERS ids where possible."
           if allow_web else "Do not invent papers; extra_refs must be an empty list.")
    return f"""# TASK: Novelty review (search-grounded)
Decide how novel the hypothesis is relative to the retrieved PAPERS, which were found with targeted searches.
Steps: (1) restate the exact claimed contribution; (2) find the closest prior work in PAPERS; (3) compare
mechanism, experimental setup and evaluation; (4) decide whether the remaining difference is substantive.
Labels: N0 known/established, N1 minor variation, N2 combination with uncertain novelty, N3 potentially novel,
N4 strong evidence of novelty. Be conservative: absence of evidence in a small retrieved set justifies at most N3.
closest_prior must use paper ids from PAPERS. {web}

# RESEARCH PLAN
{fmt_plan(plan)}

# HYPOTHESIS
{fmt_hyp(h, 'core')}

# PAPERS (retrieved for this hypothesis)
{fmt_papers(papers, 600)}
"""


def compare_prompt(plan: dict[str, Any], a: Hypothesis, b: Hypothesis) -> str:
    return f"""# TASK: Pairwise comparison (tournament match)
Compare hypothesis A and hypothesis B for the research plan on: relevance, plausibility, novelty (use the
novelty reviews), falsifiability/testability, expected scientific value, feasibility under the constraints,
experimental clarity. Do not prefer a hypothesis because it sounds more sophisticated. Pick a winner.

# RESEARCH PLAN
{fmt_plan(plan)}

# HYPOTHESIS A
{fmt_hyp(a, 'core')}

# HYPOTHESIS B
{fmt_hyp(b, 'core')}
"""


def debate_prompt(plan: dict[str, Any], a: Hypothesis, b: Hypothesis) -> str:
    return f"""# TASK: Scientific debate (tournament match between top-ranked hypotheses)
Simulate a short structured debate between three experts: advocate_A, advocate_B and a skeptic (3-6 turns in
total). The skeptic attacks the most fragile assumption of each. Then decide which hypothesis is more worth
the researcher's time, considering novelty (use the novelty reviews), correctness, testability, expected
contribution and feasibility. Identify the key assumption that determines the comparison.

# RESEARCH PLAN
{fmt_plan(plan)}

# HYPOTHESIS A
{fmt_hyp(a)}

# HYPOTHESIS B
{fmt_hyp(b)}
"""


def evolve_prompt(plan: dict[str, Any], strategy: str, parents: list[Hypothesis], papers: list[Paper],
                  match_notes: list[str], feedback: list[str]) -> str:
    par = "\n\n".join(f"## PARENT {p.hid}\n{fmt_hyp(p)}" for p in parents)
    notes = "\n- ".join(match_notes[:8]) or "(none)"
    return f"""# TASK: Evolution ({strategy})
Create ONE new improved hypothesis from the parent(s). Strategy: {EVOLVE_STRATEGIES[strategy]}
Address the CRITICAL/MAJOR criticisms and the novelty threats where possible. Do not just rephrase. The new
hypothesis must still be falsifiable and testable under the constraints. List what changed and why.

# RESEARCH PLAN
{fmt_plan(plan)}

{par}

# TOURNAMENT NOTES ABOUT THE PARENT(S) (deciding factors from lost/won matches)
- {notes}
{_feedback_block(feedback)}

# PAPERS
{fmt_papers(papers, 300)}
"""


def feedback_prompt(plan: dict[str, Any], reviews: list[str], deciding: list[str]) -> str:
    rv = "\n- ".join(reviews[:40]) or "(none)"
    dc = "\n- ".join(deciding[:40]) or "(none)"
    return f"""# TASK: Meta-review of this round
Synthesise recurring patterns across reviews and tournament deciding factors: repeated weaknesses, assumptions
shared by too many hypotheses, underexplored directions. Produce short guidance for the next generation /
evolution round and for reviewers.

# RESEARCH PLAN
{fmt_plan(plan)}

# REVIEW CRITICISMS (sample)
- {rv}

# TOURNAMENT DECIDING FACTORS (sample)
- {dc}
"""


def overview_prompt(plan: dict[str, Any], top: list[Hypothesis], digest: dict[str, Any]) -> str:
    hs = "\n\n".join(f"(rank {i + 1}, Elo {h.elo:.0f}) {fmt_hyp(h, 'core')}" for i, h in enumerate(top))
    gaps = json.dumps(digest.get("gaps", [])[:7], ensure_ascii=False)
    return f"""# TASK: Research overview (final meta-review)
Summarise the state of this research exploration for the researcher: the strongest directions (grouping
related hypotheses), why they matter, the first experiment for each, key uncertainties, and what literature to
check next. Be explicit about uncertainty and novelty risk. Do not overstate.

# RESEARCH PLAN
{fmt_plan(plan)}

# GAPS
{gaps}

# TOP HYPOTHESES (by Elo)
{hs}
"""
