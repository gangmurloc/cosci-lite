"""Agent steps: parse, literature, generation, reflection, ranking, evolution, meta-review, overview."""
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import similarity, tournament
from ..literature import LiteratureSearch
from ..llm import BudgetExceeded, LLMError, Router
from ..models import Hypothesis, Match, Paper
from ..state import RunState
from . import prompts as P


@dataclass
class Context:
    state: RunState
    router: Router
    search: LiteratureSearch
    cfg: dict[str, Any]
    log: Callable[[str], None] = print
    system: str = field(default="")

    def __post_init__(self) -> None:
        if not self.system:
            self.system = P.system_prompt(self.cfg.get("language", "ko"))

    @property
    def pcfg(self) -> dict[str, Any]:
        return self.cfg["pipeline"]

    @property
    def lcfg(self) -> dict[str, Any]:
        return self.cfg["literature"]


def _pmap(ctx: Context, role: str, fn: Callable[[Any], Any], items: list[Any]) -> list[Any]:
    """Run fn over items in parallel (bounded by the role's backend workers). Budget errors propagate."""
    if not items:
        return []
    workers = min(ctx.router.max_workers(role), len(items))
    if workers <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fn, x) for x in items]
        results = []
        budget_err: BudgetExceeded | None = None
        for fut in futures:
            try:
                results.append(fut.result())
            except BudgetExceeded as e:
                budget_err = e
                results.append(None)
        if budget_err:
            raise budget_err
        return results


# ============================================================================ parse
def parse_goal(ctx: Context) -> None:
    ctx.log("[1/6] 연구 목표 파싱 (Research Plan Configuration)")
    plan = ctx.router.complete("parse", P.parse_prompt(ctx.state.goal), ctx.system, P.PLAN_SCHEMA)
    ctx.state.plan = plan
    ctx.log(f"      제목: {plan.get('title')}  | 검색어 {len(plan.get('search_queries', []))}개")


# ============================================================================ literature
def explore_literature(ctx: Context) -> None:
    ctx.log("[2/6] 문헌 탐색")
    st = ctx.state
    queries = list(dict.fromkeys(st.plan.get("search_queries") or []))
    max_papers = int(ctx.lcfg.get("max_papers", 60))
    for q in queries:
        if len(st.papers) >= max_papers:
            break
        for r in ctx.search.search(q):
            if len(st.papers) >= max_papers:
                break
            st.add_paper(r, query=q)
    ctx.log(f"      논문 {len(st.papers)}편 수집" + (f" (검색 오류 {len(ctx.search.errors)}건)" if ctx.search.errors else ""))
    for e in ctx.search.errors[-3:]:
        ctx.log(f"      ! {e}")
    papers = select_papers(st, int(ctx.lcfg.get("digest_papers", 30)))
    if not papers:
        ctx.log("      ! 수집된 논문이 없어 digest를 건너뜁니다 (검색 소스/네트워크 확인 필요)")
        st.digest = {"known_facts": [], "limitations": [], "contradictions": [], "gaps": []}
        return
    digest = ctx.router.complete("digest", P.digest_prompt(st.plan, papers), ctx.system, P.DIGEST_SCHEMA)
    digest = _clean_pid_refs(digest, set(st.papers))
    st.digest = digest
    ctx.log(f"      knowledge gap {len(digest.get('gaps', []))}개 도출")


def select_papers(st: RunState, n: int, prefer: list[str] | None = None) -> list[Paper]:
    """Pick papers for prompts: preferred ids first, then by #queries hit and citations."""
    prefer = [p for p in (prefer or []) if p in st.papers]
    rest = sorted((p for p in st.papers.values() if p.pid not in prefer),
                  key=lambda p: (-len(p.queries), -(p.citations or 0), int(p.pid[1:])))
    return [st.papers[p] for p in prefer][:n] + rest[: max(0, n - len(prefer))]


def _clean_pid_refs(obj: Any, valid: set[str]) -> Any:
    """Recursively drop paper ids that are not in the registry (anti-hallucination)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k.endswith("paper_ids") or k == "prior_work_ids":
                out[k] = [x for x in (v or []) if isinstance(x, str) and x.strip("[] ") in valid]
                out[k] = [x.strip("[] ") for x in out[k]]
            else:
                out[k] = _clean_pid_refs(v, valid)
        return out
    if isinstance(obj, list):
        return [_clean_pid_refs(x, valid) for x in obj]
    return obj


# ============================================================================ generation
def generate(ctx: Context, n: int, round_no: int) -> list[Hypothesis]:
    ctx.log(f"[3/6] 가설 생성 ({n}개)")
    st = ctx.state
    already = [h for h in st.hypotheses.values() if h.origin == "generated" and h.round == round_no]
    created = _generate_batches(ctx, max(0, n - len(already)), round_no, len(already))  # resume: only what is missing
    dedupe(ctx, created)
    llm_dedupe(ctx, created)
    dropped = [h for h in created if h.status == "duplicate"]
    if dropped and ctx.pcfg.get("replace_duplicates", True):
        # one replacement pass, so duplicates do not shrink the pool the tournament starts from
        more = _generate_batches(ctx, len(dropped), round_no, len(already) + len(created))
        dedupe(ctx, more)
        llm_dedupe(ctx, more)
        created += more
    ctx.log("      " + ", ".join(f"{h.hid}" + ("(중복)" if h.status == "duplicate" else "") for h in created))
    return created


def _generate_batches(ctx: Context, n: int, round_no: int, offset: int) -> list[Hypothesis]:
    st = ctx.state
    strategies = list(P.GEN_STRATEGIES)
    rng = random.Random(int(ctx.pcfg.get("seed", 42)) + round_no)
    rng.shuffle(strategies)
    batch = max(1, int(ctx.pcfg.get("gen_batch", 4)))
    papers = select_papers(st, int(ctx.lcfg.get("digest_papers", 30)))
    feedback = _latest_guidance(st)
    created: list[Hypothesis] = []
    while len(created) < n:
        k = min(batch, n - len(created))
        strat = [strategies[(offset + i) % len(strategies)] for i in range(k)]
        offset += k
        existing = [h for h in st.hypotheses.values()]
        prompt = P.generate_prompt(st.plan, st.digest, papers, k, strat, existing, feedback)
        try:
            out = ctx.router.complete("generate", prompt, ctx.system, P.GEN_SCHEMA)
        except LLMError as e:
            ctx.log(f"      ! 생성 실패: {e}")
            break
        for i, f in enumerate((out.get("hypotheses") or [])[:k]):
            if not isinstance(f, dict) or not f.get("title"):
                continue
            f = _clean_pid_refs(f, set(st.papers))
            h = Hypothesis(hid=st.next_hid(), fields=f, origin="generated",
                           strategy=strat[min(i, len(strat) - 1)], round=round_no)
            st.add_hypothesis(h)
            created.append(h)
        st.save()
    return created


def llm_dedupe(ctx: Context, new: list[Hypothesis]) -> None:
    """Semantic duplicate check among generated hypotheses.

    TF-IDF proximity cannot see paraphrases: two generated hypotheses stating the same claim scored 0.15,
    inside the 0.12-0.28 range of unrelated pairs and far below the 0.85 threshold. One judge call groups hypotheses that make the same
    claim; in each group the earliest stays active and later ones from `new` are marked duplicate (kept in
    the report, left out of the tournament). Evolved hypotheses are refinements by design and are not checked.
    """
    if not ctx.pcfg.get("llm_dedupe", True):
        return
    st = ctx.state
    pool = [h for h in st.hypotheses.values() if h.origin == "generated" and h.status == "active"]
    fresh = {h.hid for h in new if h.origin == "generated" and h.status == "active"}
    if len(pool) < 2 or not fresh:
        return
    try:
        out = ctx.router.complete("dedupe", P.dedupe_prompt(pool), ctx.system, P.DEDUPE_SCHEMA)
    except LLMError as e:
        st.log_event("dedupe_failed", error=str(e)[:200])
        return
    order = {h.hid: i for i, h in enumerate(pool)}
    for group in out.get("groups") or []:
        ids = sorted({str(x).strip("[] ") for x in group.get("ids") or []} & set(order), key=lambda hid: order[hid])
        for hid in ids[1:]:
            h = st.hypotheses[hid]
            if hid in fresh and h.status == "active":
                h.status = "duplicate"
                h.status_reason = f"{ids[0]}와 같은 주장 (LLM 판정): {str(group.get('shared_claim', ''))[:200]}"
    st.save()


def dedupe(ctx: Context, new: list[Hypothesis]) -> None:
    """Proximity check: mark near-duplicates of earlier hypotheses (kept, not deleted)."""
    thr = float(ctx.pcfg.get("dedupe_threshold", 0.85))
    st = ctx.state
    texts = {h.hid: h.text_for_similarity() for h in st.hypotheses.values() if h.status != "duplicate"}
    sims = similarity.pairwise(texts)
    order = list(st.hypotheses)
    for h in new:
        if h.status != "active":
            continue
        for other in order:
            if other == h.hid:
                break
            o = st.hypotheses[other]
            if o.status == "duplicate" or h.hid in o.parents or other in h.parents:
                continue
            s = similarity.get(sims, h.hid, other)
            if s >= thr:
                h.status = "duplicate"
                h.status_reason = f"{other}와 유사도 {s:.2f} (Proximity)"
                break


def _latest_guidance(st: RunState) -> list[str]:
    if not st.feedback:
        return []
    fb = st.feedback[-1]
    return list(fb.get("guidance_generation") or [])[:6]


# ============================================================================ reflection
def review_new(ctx: Context, round_no: int) -> None:
    """Initial review for unreviewed hypotheses, then novelty review for active ones without one.

    State is saved after every completed review, so an interrupted run resumes without redoing work.
    """
    st = ctx.state
    todo = [h for h in st.hypotheses.values() if h.status == "active" and h.initial_review is None]
    pending_novelty = [h for h in st.hypotheses.values()
                       if h.status == "active" and h.initial_review is not None and h.novelty_review is None]
    if not todo and not pending_novelty:
        return
    ctx.log(f"[4/6] 리뷰: 초기 리뷰 {len(todo)}개 + novelty 리뷰")
    ctx_papers = select_papers(st, 12)

    def _initial(h: Hypothesis) -> None:
        try:
            rv = ctx.router.complete("review", P.review_prompt(st.plan, h, ctx_papers), ctx.system, P.REVIEW_SCHEMA)
        except LLMError as e:
            st.log_event("review_failed", hid=h.hid, error=str(e)[:300])
            h.initial_review = {"verdict": "keep", "criticisms": [], "search_queries": [h.title], "error": str(e)[:200]}
            return
        h.initial_review = rv
        if rv.get("verdict") == "reject":
            crit = [c.get("text", "") for c in rv.get("criticisms", []) if c.get("severity") == "CRITICAL"]
            h.status = "rejected"
            h.status_reason = "초기 리뷰 기각: " + (crit[0] if crit else "CRITICAL 문제")
        st.save()

    _pmap(ctx, "review", _initial, todo)
    survivors = [h for h in st.hypotheses.values()
                 if h.status == "active" and h.initial_review is not None and h.novelty_review is None]
    nq = int(ctx.lcfg.get("novelty_queries", 3))
    npap = int(ctx.lcfg.get("novelty_papers", 10))
    allow_web = bool(ctx.cfg["backends"][ctx.cfg["roles"]["novelty"]].get("allow_web"))

    # targeted searches (sequential: rate limits)
    targeted: dict[str, list[str]] = {}
    for h in survivors:
        pids: list[str] = []
        for q in (h.initial_review or {}).get("search_queries", [])[:nq]:
            for r in ctx.search.search(q, k=max(4, npap // 2)):
                pids.append(st.add_paper(r, query=q).pid)
        prior = [p for p in h.fields.get("prior_work_ids", []) if p in st.papers]
        targeted[h.hid] = list(dict.fromkeys(prior + pids))

    def _novelty(h: Hypothesis) -> None:
        papers = select_papers(st, npap, prefer=targeted.get(h.hid, []))
        valid = {p.pid for p in papers}
        try:
            nr = ctx.router.complete("novelty", P.novelty_prompt(st.plan, h, papers, allow_web),
                                     ctx.system, P.NOVELTY_SCHEMA)
        except LLMError as e:
            st.log_event("novelty_failed", hid=h.hid, error=str(e)[:300])
            return
        kept, dropped = [], []
        for c in nr.get("closest_prior") or []:
            pid = str(c.get("paper_id", "")).strip("[] ")
            if pid in valid:
                c["paper_id"] = pid
                kept.append(c)
            else:
                dropped.append(pid)
        nr["closest_prior"] = kept
        nr["dropped_invalid_ids"] = dropped
        nr["papers_considered"] = sorted(valid, key=lambda s: int(s[1:]))
        if not allow_web:
            nr["extra_refs"] = []
        h.novelty_review = nr
        st.save()

    _pmap(ctx, "novelty", _novelty, survivors)
    rej = sum(1 for h in todo if h.status == "rejected")
    ctx.log(f"      기각 {rej}개 | novelty: " + ", ".join(f"{h.hid}={h.novelty}" for h in survivors))


# ============================================================================ ranking
def run_tournament(ctx: Context, round_no: int) -> None:
    st = ctx.state
    active = st.ranked()
    if len(active) < 2:
        return
    ids = [h.hid for h in active]
    elo = {h.hid: h.elo for h in active}
    sims = similarity.pairwise({h.hid: h.text_for_similarity() for h in active})
    played = [(m.a, m.b) for m in st.matches]
    new_ids = [h.hid for h in active if h.games == 0]
    rng = random.Random(int(ctx.pcfg.get("seed", 42)) * 31 + round_no)
    pairs = tournament.schedule(ids, elo, sims, played, int(ctx.pcfg.get("matches_per_hyp", 3)), rng,
                                priority=new_ids)
    top_k = set(ids[: int(ctx.pcfg.get("debate_top_k", 4))])
    swap_compare = bool(ctx.pcfg.get("swap_check", True))
    swap_debate = bool(ctx.pcfg.get("swap_check_debate", False))
    ctx.log(f"[5/6] 토너먼트 라운드 {round_no}: 경기 {len(pairs)}개 (debate = 상위 {len(top_k)}개끼리)")

    def _judge(args: tuple[str, str, str]) -> dict[str, Any] | None:
        a, b, mode = args
        ha, hb = st.hypotheses[a], st.hypotheses[b]
        prompt = P.debate_prompt(st.plan, ha, hb) if mode == "debate" else P.compare_prompt(st.plan, ha, hb)
        schema = P.DEBATE_SCHEMA if mode == "debate" else P.COMPARE_SCHEMA
        try:
            return ctx.router.complete(mode, prompt, ctx.system, schema)
        except LLMError as e:
            st.log_event("match_failed", a=a, b=b, error=str(e)[:200])
            return None

    def _mode(a: str, b: str) -> str:
        return "debate" if (a in top_k and b in top_k) else "compare"

    def _swap(mode: str) -> bool:
        return swap_debate if mode == "debate" else swap_compare

    jobs: list[tuple[str, str, str]] = []
    for a, b in pairs:
        mode = _mode(a, b)
        jobs.append((a, b, mode))
        if _swap(mode):
            jobs.append((b, a, mode))
    # group by role for parallelism
    results: dict[tuple[str, str, str], Any] = {}
    for mode in ("compare", "debate"):
        sub = [j for j in jobs if j[2] == mode]
        for j, r in zip(sub, _pmap(ctx, mode, _judge, sub)):
            results[j] = r

    k = float(ctx.pcfg.get("elo_k", 32))
    for a, b in pairs:
        mode = _mode(a, b)
        r1 = results.get((a, b, mode))
        r2 = results.get((b, a, mode)) if _swap(mode) else None
        s1 = None if r1 is None else (1.0 if r1.get("winner") == "A" else 0.0)
        s2 = None if r2 is None else (0.0 if r2.get("winner") == "A" else 1.0)  # swapped: A there is b
        scores = [s for s in (s1, s2) if s is not None]
        if not scores:
            continue
        score_a = 0.5 if (len(scores) == 2 and scores[0] != scores[1]) else scores[0]
        ha, hb = st.hypotheses[a], st.hypotheses[b]
        ha.elo, hb.elo = tournament.update(ha.elo, hb.elo, score_a, k)
        if score_a == 1.0:
            ha.wins += 1; hb.losses += 1
        elif score_a == 0.0:
            ha.losses += 1; hb.wins += 1
        else:
            ha.draws += 1; hb.draws += 1
        detail = {"deciding_factor": (r1 or r2 or {}).get("deciding_factor", ""),
                  "deciding_factor_swapped": (r2 or {}).get("deciding_factor", "") if r2 else "",
                  "swap_checked": len(scores) == 2,
                  "consistent": len(scores) < 2 or scores[0] == scores[1],
                  "key_assumption": (r1 or {}).get("key_assumption", "")}
        st.matches.append(Match(a=a, b=b, round=round_no, mode=mode, score_a=score_a, detail=detail))
    top = st.ranked()[:3]
    ctx.log("      상위: " + " > ".join(f"{h.hid}({h.elo:.0f})" for h in top))


# ============================================================================ evolution
def evolve(ctx: Context, round_no: int) -> list[Hypothesis]:
    st = ctx.state
    ranked = st.ranked()
    k = int(ctx.pcfg.get("evolve_top_k", 3))
    top = ranked[:k]
    if not top:
        return []
    jobs: list[tuple[str, list[Hypothesis]]] = []
    single = ["grounding", "feasibility", "simplification"]
    for i, h in enumerate(top):
        jobs.append((single[(i + round_no) % len(single)], [h]))
    if len(top) >= 2:
        jobs.append(("combination", [top[0], top[1]]))
    jobs.append(("out_of_box", [top[0]]))
    # resume support: skip jobs whose child already exists for this round
    done = {(h.strategy, tuple(h.parents)) for h in st.hypotheses.values()
            if h.origin == "evolved" and h.round == round_no}
    jobs = [(s, ps) for s, ps in jobs if (s, tuple(p.hid for p in ps)) not in done]
    if not jobs:
        return []
    ctx.log(f"[진화 {round_no}] 새 가설 {len(jobs)}개 (" + ", ".join(s for s, _ in jobs) + ")")
    feedback = _latest_guidance(st)

    created: list[Hypothesis] = []

    def _one(job: tuple[str, list[Hypothesis]]) -> None:
        strategy, parents = job
        prefer: list[str] = []
        for p in parents:
            prefer += (p.novelty_review or {}).get("papers_considered", [])
            prefer += p.fields.get("prior_work_ids", [])
        papers = select_papers(st, 14, prefer=list(dict.fromkeys(prefer)))
        notes = _match_notes(st, [p.hid for p in parents])
        try:
            out = ctx.router.complete("evolve", P.evolve_prompt(st.plan, strategy, parents, papers, notes, feedback),
                                      ctx.system, P.EVOLVE_SCHEMA)
        except LLMError as e:
            st.log_event("evolve_failed", strategy=strategy, parents=[p.hid for p in parents], error=str(e)[:200])
            return
        if not isinstance(out.get("hypothesis"), dict) or not out["hypothesis"].get("title"):
            st.log_event("evolve_failed", strategy=strategy, parents=[p.hid for p in parents], error="empty hypothesis")
            return
        f = _clean_pid_refs(out["hypothesis"], set(st.papers))
        f["evolution_changes"] = out.get("changes", [])
        f["addressed_critiques"] = out.get("addressed_critiques", [])
        with st._lock:  # atomic id allocation + insert
            hid = st.next_version_id(parents[0].hid) if strategy in single else st.next_hid()
            h = Hypothesis(hid=hid, fields=f, origin="evolved", strategy=strategy,
                           parents=[p.hid for p in parents], round=round_no)
            st.add_hypothesis(h)
            created.append(h)
        st.save()

    _pmap(ctx, "evolve", _one, jobs)
    dedupe(ctx, created)
    ctx.log("      " + ", ".join(f"{h.hid}←{'+'.join(h.parents)}" for h in created))
    return created


def _match_notes(st: RunState, hids: list[str]) -> list[str]:
    notes = []
    for m in st.matches:
        if m.a in hids or m.b in hids:
            me = m.a if m.a in hids else m.b
            won = (m.score_a == 1.0 and me == m.a) or (m.score_a == 0.0 and me == m.b)
            res = "draw" if m.score_a == 0.5 else ("won" if won else "lost")
            other = m.b if me == m.a else m.a
            notes.append(f"{me} {res} vs {other}: {m.detail.get('deciding_factor', '')}")
    return notes[-10:]


# ============================================================================ meta-review
def meta_feedback(ctx: Context, round_no: int) -> None:
    st = ctx.state
    reviews = []
    for h in st.hypotheses.values():
        for c in (h.initial_review or {}).get("criticisms", [])[:3]:
            reviews.append(f"{h.hid} [{c.get('severity')}] {c.get('text')}")
    deciding = [f"{m.a} vs {m.b}: {m.detail.get('deciding_factor', '')}" for m in st.matches if m.round == round_no]
    try:
        fb = ctx.router.complete("feedback", P.feedback_prompt(st.plan, reviews, deciding), ctx.system, P.FEEDBACK_SCHEMA)
    except LLMError as e:
        st.log_event("feedback_failed", error=str(e)[:200])
        return
    fb["round"] = round_no
    st.feedback.append(fb)
    ctx.log(f"      메타리뷰 피드백 {len(fb.get('guidance_generation', []))}개")


def overview(ctx: Context) -> None:
    st = ctx.state
    top = st.ranked()[: int(ctx.cfg["output"].get("report_top_k", 5))]
    if not top:
        return
    ctx.log("[6/6] 연구 개요(research overview) 작성")
    try:
        st.overview = ctx.router.complete("overview", P.overview_prompt(st.plan, top, st.digest), ctx.system,
                                          P.OVERVIEW_SCHEMA)
    except LLMError as e:
        st.log_event("overview_failed", error=str(e)[:200])
