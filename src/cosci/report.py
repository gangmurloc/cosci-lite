"""Markdown outputs: run report, per-idea files (Research-Ideas template) and INDEX.md maintenance."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .models import NOVELTY_DESC, Hypothesis, novelty_rank
from .state import RunState
from .utils import slugify, today, truncate


def _cell(text: Any, n: int = 120) -> str:
    return truncate(str(text or ""), n).replace("|", "\\|")


def _paper_link(st: RunState, pid: str) -> str:
    p = st.papers.get(pid)
    if not p:
        return pid
    year = f", {p.year}" if p.year else ""
    return f"[{p.title}]({p.link}){year}" if p.link else f"{p.title}{year}"


def _bullets(items: Any, prefix: str = "- ") -> str:
    items = [i for i in (items or []) if str(i).strip()]
    return "\n".join(f"{prefix}{i}" for i in items) if items else "- (없음)"


def _run_id(st: RunState) -> str:
    return st.run_dir.name


# ============================================================================ run report
def tied_with_leader(ranked: list[Hypothesis], elo_k: float = 32.0) -> list[str]:
    """Ids at the top whose Elo is within one game's largest swing (K) of the leader.

    A single result could reorder them, so their relative rank carries no information.
    """
    out: list[str] = []
    for h in ranked:
        if ranked[0].elo - h.elo > elo_k:
            break
        out.append(h.hid)
    return out


def lineage_root(st: RunState, hid: str) -> str:
    """The originally generated hypothesis this one descends from (following the first parent)."""
    seen: set[str] = set()
    while hid in st.hypotheses and st.hypotheses[hid].parents and hid not in seen:
        seen.add(hid)
        hid = st.hypotheses[hid].parents[0]
    return hid


def families(st: RunState, ranked: list[Hypothesis]) -> dict[str, list[str]]:
    """root id -> its family members in rank order; families ordered by their best-ranked member."""
    out: dict[str, list[str]] = {}
    for h in ranked:
        out.setdefault(lineage_root(st, h.hid), []).append(h.hid)
    return out


def render_report(st: RunState) -> str:
    plan = st.plan or {}
    cfg = st.config or {}
    roles = cfg.get("roles", {})
    backends = cfg.get("backends", {})
    used = sorted(set(roles.values()))
    be = ", ".join(f"{b}={backends.get(b, {}).get('model', backends.get(b, {}).get('type'))}" for b in used)
    ranked = st.ranked()
    top_k = int((cfg.get("output") or {}).get("report_top_k", 5))
    L: list[str] = []
    L.append(f"# {plan.get('title') or 'cosci-lite run'} — 연구 주제 탐색 리포트\n")
    L.append(f"- 실행: `{_run_id(st)}` · 생성 {st.created}")
    L.append(f"- 백엔드: {be}")
    L.append(f"- LLM 호출: {st.calls} · 추정 비용(claude 보고값): ${st.cost_usd:.2f}")
    L.append(f"- 가설 {len(st.hypotheses)}개 (활성 {len(ranked)}) · 경기 {len(st.matches)}개 · 논문 {len(st.papers)}편")
    swapped = [m for m in st.matches if m.detail.get("swap_checked")]
    if swapped:
        cons = sum(1 for m in swapped if m.detail.get("consistent")) / len(swapped)
        L.append(f"- 순서 바꿔 재판정한 경기 {len(swapped)}개 중 판정 일치 {cons:.0%} "
                 "(낮으면 심판 모델의 position bias/불안정성이 큼 → 무승부 처리됨)")
    L.append(f"- 완료 단계: {', '.join(st.steps_done)}\n")
    L.append("## 연구 목표\n")
    L.append("> " + "\n> ".join(st.goal.strip().splitlines()) + "\n")
    if plan:
        L.append(f"**핵심 질문**: {plan.get('core_question', '')}\n")
        if plan.get("constraints"):
            L.append("**제약/가정**\n" + _bullets(plan.get("constraints")) + "\n")

    ov = st.overview or {}
    if ov:
        L.append("## 요약 (Research overview)\n")
        L.append(str(ov.get("summary", "")) + "\n")
        if ov.get("directions"):
            L.append("| 방향 | 이유 | 관련 가설 | 첫 실험 |\n|---|---|---|---|")
            for d in ov["directions"]:
                L.append(f"| {_cell(d.get('name'), 60)} | {_cell(d.get('why'), 160)} | "
                         f"{', '.join(d.get('hypothesis_ids') or [])} | {_cell(d.get('first_experiment'), 140)} |")
            L.append("")
        if ov.get("key_uncertainties"):
            L.append("**핵심 불확실성**\n" + _bullets(ov["key_uncertainties"]) + "\n")
        if ov.get("recommended_next_action"):
            L.append(f"**추천 다음 행동**: {ov['recommended_next_action']}\n")

    L.append("## 순위 (Elo 토너먼트)\n")
    L.append("| 순위 | ID | 제목 | Elo | 승-패-무 | Novelty | Feasibility | 출처 | 계열 |\n"
             "|---|---|---|---|---|---|---|---|---|")
    for i, h in enumerate(ranked, 1):
        src = h.origin + (f"({h.strategy})" if h.strategy else "")
        if h.parents:
            src += " ← " + "+".join(h.parents)
        L.append(f"| {i} | {h.hid} | {_cell(h.title, 70)} | {h.elo:.0f} | {h.wins}-{h.losses}-{h.draws} | "
                 f"{h.novelty} | {h.fields.get('feasibility', '?')} | {_cell(src, 50)} | {lineage_root(st, h.hid)} |")
    L.append("")
    elo_k = float((cfg.get("pipeline") or {}).get("elo_k", 32))
    tied = tied_with_leader(ranked, elo_k)
    if len(tied) > 1:
        L.append(f"> 상위 {len(tied)}개({', '.join(tied)})는 1위와의 Elo 차이가 {elo_k:.0f}점 이내입니다. "
                 "한 경기 결과로 뒤바뀔 수 있는 차이라, 이들 사이의 순위는 구분되지 않는 것으로 보아야 합니다.\n")
    fams = families(st, ranked)
    if any(len(members) > 1 for members in fams.values()):
        rank_of = {h.hid: i for i, h in enumerate(ranked, 1)}
        L.append("> 계열(같은 원본 가설에서 나온 묶음)별 최고 순위: "
                 + " · ".join(f"{root} 계열 {len(m)}개(최고 {rank_of[m[0]]}위 {m[0]})" for root, m in fams.items()) + "\n")
    L.append("> Elo는 LLM 심판의 상대 평가이며 정답(ground truth)이 아닙니다. Novelty는 검색된 논문 범위 안에서의 판단입니다.\n")

    L.append(f"## 상위 {min(top_k, len(ranked))}개 가설 상세\n")
    for h in ranked[:top_k]:
        L.append(render_card(st, h))

    gaps = (st.digest or {}).get("gaps") or []
    if gaps:
        L.append("## 지식 공백 (literature digest)\n")
        for g in gaps:
            refs = ", ".join(g.get("paper_ids") or [])
            L.append(f"- **{g.get('gap_id', '')}** ({g.get('gap_type', '')}) — 모르는 것: {g.get('unknown', '')} "
                     f"· 검증: {g.get('how_to_test', '')} · 이미 풀렸을 위험: {g.get('novelty_risk', '')}"
                     + (f" · 근거: {refs}" if refs else ""))
        L.append("")

    inactive = [h for h in st.hypotheses.values() if h.status != "active"]
    if inactive:
        L.append("## 기각·중복 가설 (기록 보존)\n")
        for h in inactive:
            L.append(f"- {h.hid} [{h.status}] {h.title} — {h.status_reason}")
        L.append("")

    evolved = [h for h in st.hypotheses.values() if h.parents]
    if evolved:
        L.append("## 진화 계보\n")
        for h in evolved:
            ch = h.fields.get("evolution_changes") or []
            L.append(f"- {'+'.join(h.parents)} → **{h.hid}** ({h.strategy}): " + "; ".join(str(c) for c in ch[:3]))
        L.append("")

    if st.feedback:
        L.append("## 라운드별 메타리뷰 피드백\n")
        for fb in st.feedback:
            L.append(f"**Round {fb.get('round')}** — 반복 약점: " + "; ".join(fb.get("recurring_weaknesses") or []))
            L.append("  - 다음 생성 지침: " + "; ".join(fb.get("guidance_generation") or []))
        L.append("")

    L.append("## 문헌 (검색으로 수집된 논문만)\n")
    L.append("| ID | 제목 | 연도 | 출처 |\n|---|---|---|---|")
    for p in st.papers.values():
        title = f"[{_cell(p.title, 90)}]({p.link})" if p.link else _cell(p.title, 90)
        L.append(f"| {p.pid} | {title} | {p.year or ''} | {p.source} |")
    L.append("")
    errs = [e for e in st.events if e.get("kind", "").endswith("_failed") or e.get("kind") == "budget_stop"]
    if errs:
        L.append("## 실행 중 오류/중단\n")
        for e in errs[-15:]:
            L.append(f"- {e.get('kind')}: {truncate(str({k: v for k, v in e.items() if k not in ('t', 'kind')}), 200)}")
    return "\n".join(L) + "\n"


def render_card(st: RunState, h: Hypothesis) -> str:
    f = h.fields
    L = [f"### {h.hid}. {h.title}\n",
         f"*Elo {h.elo:.0f} · {h.wins}승 {h.losses}패 {h.draws}무 · Novelty {h.novelty} · "
         f"Feasibility {f.get('feasibility', '?')} · 전략 {h.strategy or '-'}*\n",
         f"**가설**: {f.get('statement', '')}\n",
         f"**근거**: {f.get('rationale', '')}\n",
         f"**반증 조건**: {f.get('falsification', '')}\n",
         f"**최소 실험**: {f.get('min_experiment', '')}\n"]
    if f.get("predictions"):
        L.append("**예측**\n" + _bullets(f.get("predictions")) + "\n")
    nr = h.novelty_review or {}
    if nr:
        L.append(f"**Novelty 리뷰 ({nr.get('novelty')}: {NOVELTY_DESC.get(str(nr.get('novelty')), '')})**: "
                 f"{nr.get('rationale', '')}\n")
        for c in nr.get("closest_prior") or []:
            L.append(f"- 가장 가까운 선행연구 {_paper_link(st, c.get('paper_id', ''))} — 겹침: {c.get('overlap', '')} / "
                     f"차이: {c.get('difference', '')}")
        for x in nr.get("extra_refs") or []:
            L.append(f"- (미확인) {x.get('title', '')} {x.get('url', '')}")
        if nr.get("suggested_reformulation"):
            L.append(f"- 개선 제안: {nr['suggested_reformulation']}")
        L.append("")
    ir = h.initial_review or {}
    crit = [c for c in ir.get("criticisms") or [] if c.get("severity") in ("CRITICAL", "MAJOR")]
    if crit:
        L.append("**주요 비판**\n" + "\n".join(f"- [{c.get('severity')}] {c.get('text')}" for c in crit[:5]) + "\n")
    return "\n".join(L) + "\n"


def write_report(st: RunState) -> Path:
    path = st.run_dir / "report.md"
    path.write_text(render_report(st), encoding="utf-8")
    return path


# ============================================================================ idea files
def render_idea(st: RunState, h: Hypothesis) -> str:
    f = h.fields
    plan = st.plan or {}
    tags = [str(t) for t in (plan.get("key_terms") or [])][:6]
    field_name = plan.get("title") or "AI research"
    nr = h.novelty_review or {}
    rel: list[str] = []
    for c in nr.get("closest_prior") or []:
        rel.append(f"- {_paper_link(st, c.get('paper_id', ''))} — {c.get('difference', '')}")
    cited = {c.get("paper_id") for c in nr.get("closest_prior") or []}
    for pid in f.get("prior_work_ids") or []:
        if pid not in cited and pid in st.papers:
            rel.append(f"- {_paper_link(st, pid)} — 가설 생성 시 참고")
    for x in nr.get("extra_refs") or []:
        rel.append(f"- (미확인) {x.get('title', '')} {x.get('url', '')}".rstrip())
    contribs = f.get("contributions") or []
    risks = list(f.get("risks") or [])
    for c in (h.initial_review or {}).get("criticisms") or []:
        if c.get("severity") in ("CRITICAL", "MAJOR"):
            risks.append(f"[{c.get('severity')}] {c.get('text')}")
    nexts = f.get("next_steps") or []
    return f"""---
title: {f.get('title', h.hid)}
date: {today()}
status: 후보
field: {field_name}
tags: [{', '.join(tags)}]
source: cosci-lite {_run_id(st)} / {h.hid}
novelty: {h.novelty}
elo: {h.elo:.0f}
---

# {f.get('title', h.hid)}

## 한 줄 요약
{f.get('one_liner', '')}

## 문제와 동기
{f.get('problem', '')}

## 핵심 아이디어
{f.get('idea', '')}

**가설**: {f.get('statement', '')}

**반증 조건**: {f.get('falsification', '')}

## 관련 연구
{chr(10).join(rel) if rel else '- (검색으로 확인된 관련 논문 없음)'}

Novelty 판정: {h.novelty} — {nr.get('rationale', '(리뷰 없음)')}

## 예상 기여
{chr(10).join(f'{i}. {c}' for i, c in enumerate(contribs, 1)) if contribs else '1. '}

## 검증 계획
- 데이터셋: {', '.join(map(str, f.get('datasets') or []))}
- 베이스라인: {', '.join(map(str, f.get('baselines') or []))}
- 평가 지표: {', '.join(map(str, f.get('metrics') or []))}
- 최소 실험 (1~2일 안에 끝나는 sanity check): {f.get('min_experiment', '')}
- 필요 자원: {f.get('resources', '')}

## 리스크와 예상 반론
{_bullets(risks)}

## 다음 단계
{_bullets(nexts, '- [ ] ')}

## 변경 기록
- {today()}: 최초 저장 (cosci-lite 자동 저장, Elo {h.elo:.0f}, {h.wins}승 {h.losses}패, novelty {h.novelty})
"""


def idea_filename(h: Hypothesis) -> str:
    base = h.fields.get("title_en") or h.title
    return f"{today()}-{slugify(str(base))}.md"


INDEX_HEADER = """# 연구 아이디어 인덱스

아이디어 채팅에서 좋은 아이디어가 나오면 Claude가 이 폴더에 `YYYY-MM-DD-제목.md` 파일로 저장하고, 아래 표에 한 줄씩 추가해요.

상태: 후보 → 검증 중 → 구현 중 → 완료 / 보류 / 폐기

| 날짜 | 아이디어 | 한 줄 요약 | 분야 | 상태 | 파일 |
|---|---|---|---|---|---|
"""


def update_index(ideas_dir: Path, filename: str, title: str, summary: str, field_name: str,
                 status: str = "후보") -> None:
    """Append a row for `filename`, or update the existing row's status/summary (re-reads the latest file)."""
    idx = ideas_dir / "INDEX.md"
    text = idx.read_text(encoding="utf-8") if idx.exists() else INDEX_HEADER
    row = (f"| {today()} | {_cell(title, 80)} | {_cell(summary, 140)} | {_cell(field_name, 40)} | {status} | "
           f"[파일]({filename}) |")
    lines = text.rstrip("\n").split("\n")
    for i, line in enumerate(lines):
        if f"]({filename})" in line:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            date = cells[0] if cells else today()
            lines[i] = row.replace(f"| {today()} |", f"| {date} |", 1)
            break
    else:
        last_table = max((i for i, l in enumerate(lines) if l.strip().startswith("|")), default=-1)
        if last_table == -1:
            lines += ["", "| 날짜 | 아이디어 | 한 줄 요약 | 분야 | 상태 | 파일 |", "|---|---|---|---|---|---|"]
            last_table = len(lines) - 1
        lines.insert(last_table + 1, row)
    idx.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_idea(st: RunState, hid: str, ideas_dir: Path) -> Path:
    h = st.hypotheses[hid]
    ideas_dir.mkdir(parents=True, exist_ok=True)
    name = idea_filename(h)
    path = ideas_dir / name
    marker = f"source: cosci-lite {_run_id(st)} / {h.hid}"
    if path.exists() and marker not in path.read_text(encoding="utf-8", errors="ignore"):
        stem = path.stem
        n = 2
        while (ideas_dir / f"{stem}-{n}.md").exists():
            n += 1
        path = ideas_dir / f"{stem}-{n}.md"
    path.write_text(render_idea(st, h), encoding="utf-8")
    update_index(ideas_dir, path.name, h.title, str(h.fields.get("one_liner", "")),
                 str((st.plan or {}).get("title", "")))
    if str(path) not in st.saved_ideas:
        st.saved_ideas.append(str(path))
    return path


def autosave_ideas(st: RunState, ideas_dir: Path, k: int, min_novelty: str = "N2") -> list[Path]:
    thr = novelty_rank(min_novelty)
    out = []
    for h in st.ranked()[:k]:
        if novelty_rank(h.novelty) >= thr:
            out.append(save_idea(st, h.hid, ideas_dir))
    return out


_ID_RE = re.compile(r"^H\d+(-v\d+)?$")
