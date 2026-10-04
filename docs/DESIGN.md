# cosci-lite design notes (MVP: core loop)

This is a lightweight personal version of the Co-Scientist paper (Gottweis et al., Nature 2026): a research-topic discovery agent.
The MVP covers only the core loop the paper describes: **literature search → hypothesis generation → novelty review → Elo tournament → evolution**.

## 1. Pipeline

```
research goal (text/md)
  └─ [parse]        Research Plan Configuration (criteria, constraints, search queries)
  └─ [literature]   arXiv / Semantic Scholar / OpenAlex search → paper registry (P1..Pn) → digest (facts, limitations, gaps)
  └─ [generate]     N hypotheses from several strategies (H1..HN)
  └─ [review]       initial review (keep/revise/reject + search queries)
                    → targeted search per hypothesis → novelty review (N0–N4, closest prior work restricted to P-ids)
  └─ [tournament]   Elo (1200 start, K=32), pairwise comparisons, debate between top-k, order swapped to reduce position bias
  └─ repeat R rounds:
       [evolve]     new versions of top-k hypotheses (grounding / feasibility / combination / simplification / out-of-box)
       [review]     review the new hypotheses
       [tournament] new and existing hypotheses compete
       [feedback]   distil recurring weaknesses → appended to next round's generate/evolve prompts (a lite Meta-review)
  └─ [overview]     research overview (directions, uncertainties, next literature to read)
  └─ [report]       report.md for the run plus Research-Ideas idea files and INDEX.md (top-k auto-saved)
```

## 2. Elements kept from the paper

| Paper | cosci-lite |
|---|---|
| Research plan configuration | `agents/steps.py: parse_goal` |
| Generation agent: literature exploration, multiple strategies | `agents/steps.py: generate` (strategy list, avoids duplicates) |
| Reflection agent: initial review → full review with search | `agents/steps.py: review_new` (initial review without search; novelty review with targeted search) |
| Ranking agent: Elo; debate for top hypotheses, single comparison below | `agents/steps.py: run_tournament`, `tournament.py` |
| Proximity agent: similarity used to pick opponents and dedupe | `similarity.py` (TF-IDF or Jaccard) |
| Evolution agent: creates new hypotheses, never overwrites the original | `agents/steps.py: evolve` (H3 → H3-v2, parent links) |
| Meta-review: feedback appended to prompts | `agents/steps.py: meta_feedback / overview` (round feedback + final overview) |
| Context memory: persistent state, restartable | `state.py` (state.json, saved after each step, `resume`) |
| Scientist-in-the-loop | `cosci add-idea` (add your own hypothesis to the tournament), report review |

Left out for now: asynchronous worker queue, a full Supervisor with resource allocation, observation/simulation reviews, multi-turn debate (approximated by a single-prompt debate).

## 3. LLM backends and role routing

- `claude_code`: `claude -p` subprocess. The prompt is sent on stdin with `--output-format json`; `--json-schema` is used only when the executable is `.exe` or on a non-Windows OS (`.cmd` shims can break JSON quoting).
  - Does not use `--bare`, because `--bare` reads only API-key auth and would bypass the subscription login.
  - Tools are off by default (`--tools ""`). With `allow_web: true`, WebSearch/WebFetch are allowed.
  - Runs with cwd set to the run folder, so no project CLAUDE.md is loaded, plus `--strict-mcp-config` and `--no-session-persistence`.
- `openai_compatible`: vLLM / Ollama / LM Studio (`/v1/chat/completions`). `<think>` blocks are stripped. json_mode can be `schema`, `object` or `none`.
- `mock`: fake responses generated from the schema, for tests.
- Routing: `roles:` in the config maps each role (parse, digest, generate, review, novelty, compare, debate, evolve, feedback, overview) to a backend.

## 4. Anti-hallucination rules

- Prompts may cite only papers from the provided PAPERS list, by P-id. The code removes any P-id that is not in the registry.
- References extra to that list (when allow_web is on) are tagged `(미확인)` ("unverified") in the report.
- Novelty is labelled N0–N4 and is always reported together with the closest prior work.
- Hypotheses that are rejected or found to be duplicates are kept in state with their reason (the protocol says nothing is silently discarded).

## 5. Elo tournament

- Expected score E_A = 1 / (1 + 10^((R_B − R_A)/400)); R_A ← R_A + K(S_A − E_A); a draw scores 0.5.
- Pairing:
  - Each hypothesis gets m matches per round. Opponent score = 0.5·similarity + 0.5·(1 − |ΔElo|/400); pairs that have already met are penalized.
  - When both hypotheses are in the top-k, the `debate` role is used; otherwise `compare`.
  - With `swap_check`, each match is judged in both orders (A,B) and (B,A). If the two verdicts disagree, the match is recorded as a draw.

## 6. Output

- `runs/<timestamp>-<slug>/state.json`, `calls.jsonl` (log of every LLM call), `report.md`
- `ideas_dir/<date>-<slug>.md` uses the Research-Ideas template. INDEX.md gains a new row, or the existing row is updated.
