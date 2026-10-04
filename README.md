# cosci-lite: my research-topic discovery agent

A lightweight, personal re-implementation of the structure from the Co-Scientist paper
([Gottweis et al., *Nature* 2026](https://doi.org/10.1038/s41586-026-10644-y)).
Give it a research goal and it runs this core loop:

**literature search → hypothesis generation → novelty review → Elo tournament → evolution**

Results come out as a Markdown report. The top ideas are also saved automatically into your
`Research-Ideas` folder, and `INDEX.md` is updated.

```
goal.md ─▶ [parse] ─▶ [literature: arXiv · Semantic Scholar · OpenAlex] ─▶ [generate N]
        ─▶ [review: initial review + novelty review grounded in targeted search]
        ─▶ [Elo tournament: compare / debate, order swapped and re-judged]
        ─▶ ( [evolve top-k] ─▶ [review] ─▶ [tournament] ─▶ [meta feedback] ) × R
        ─▶ [overview] ─▶ report.md + Research-Ideas/*.md + INDEX.md
```

## 1. Install (Windows)

1. Python 3.10 or newer.
2. Install this package:
   ```
   pip install -e .          # basic
   pip install -e .[sim]     # adds scikit-learn for TF-IDF similarity (recommended)
   ```
3. **Claude Code**: install it, run `claude` once in a terminal, and log in.
   - Check that `claude -p "hi"` prints a reply.
   - On Windows the executable may be `claude.exe` (native installer) or `claude.cmd` (npm). With `.cmd`, the `--json-schema` flag is switched off automatically and JSON output is enforced through the prompt instead.
4. **Local model server** (optional, for the hybrid or local-only setup).
   - **vLLM** runs on Linux or WSL2. Examples for RTX A5000s:
     ```
     # one GPU: 14B (4-bit AWQ)
     vllm serve Qwen/Qwen3-14B-AWQ --port 8000 --max-model-len 32768 --gpu-memory-utilization 0.90
     # two GPUs: 32B (4-bit AWQ)
     vllm serve Qwen/Qwen3-32B-AWQ --port 8000 --tensor-parallel-size 2 --max-model-len 32768
     ```
     A server inside WSL2 is normally reachable from Windows at `localhost:8000`.
   - **Ollama** (Windows or Linux). In `cosci.yaml`, set `base_url: http://localhost:11434/v1` and set `model` to an Ollama model name. Checked against Ollama 0.35.1 with `qwen3.5:27b`:
     - `json_mode: schema` works (older servers that reject it are downgraded to `object` automatically).
     - Ollama ignores vLLM's `chat_template_kwargs`, so a thinking model spends the whole answer on reasoning and returns empty content. Set `extra_body: {reasoning_effort: none}`. If you forget, the backend notices the empty answer and switches reasoning off itself after the first call.
     - Set `max_workers: 1`. Ollama answers one request at a time, so parallel requests only queue and time out.
     - `extra_body: {keep_alive: 30s}` controls how long the model stays on the GPU after the last call.
5. (Recommended) Set the `S2_API_KEY` environment variable to a Semantic Scholar API key. Without a key you will often hit 429 errors.

## 2. Quick start

```
cosci init                    # creates cosci.yaml and goal.md
# open goal.md and write your research goal and constraints (compute, timeline, budget)
cosci check                   # ping the LLM backends and literature search sources
cosci run --goal-file goal.md
```

- Set `output.ideas_dir` in `cosci.yaml` to your ideas folder (for example `~/Research-Ideas`). The top ideas (default 2, novelty N2 or higher) are then saved there in the Research-Ideas template, and `INDEX.md` is updated.
- To test the flow without spending any LLM calls: `cosci run --preset mock --goal "test" --no-save` (search still runs for real).

## 3. Backend presets

| preset | Roles (generation, comparison, etc.) | When to use |
|---|---|---|
| `hybrid` (default) | Bulk work (generate, review, compare, feedback) runs on **local**. Quality-critical work (parse, digest, novelty, debate, evolve, overview) runs on **Claude Code**. | Local GPU plus subscription |
| `claude-only` | All roles on Claude Code | No local server (slower, uses more of your subscription quota) |
| `local-only` | All roles on the local model | Uses zero subscription quota (quality depends on the model) |

To pick a backend per role, edit the `roles:` section of `cosci.yaml`.

## 4. Researcher involvement (scientist-in-the-loop)

```
cosci list runs/<run>                         # print the current ranking
cosci add-idea runs/<run> --title "..." --statement "..."   # add your own hypothesis to the tournament
cosci continue runs/<run> --rounds 1          # run one more round of evolution + tournament
cosci save-idea runs/<run> H3 H5-v2           # save the hypotheses you want into Research-Ideas
cosci resume runs/<run>                        # resume from the point of interruption (state is saved per step and per review)
cosci report runs/<run>                        # regenerate report.md
```

## 5. Running on a shared GPU server

Two helper scripts in `bin/` (Linux only; they assume a conda env named `cosci` and use tmux when it is installed):

```
bin/cosci-bg run3 --goal-file goal.md      # start in a tmux session; log in logs/run3.log; ntfy when it ends
bin/cosci-bg --dry-run run3 --goal-file goal.md   # show the goal and the command without starting anything
bin/cosci-status -w                         # live progress of the latest run (steps, ranking, calls, GPU)
```

`cosci.ollama.example.yaml` is the configuration used there (Ollama on one 24 GB GPU plus Claude Code). `bin/claude` is a
wrapper for machines where the only Claude Code install is the one bundled with the VS Code extension.

- `cosci-bg` prints the goal file it is about to use and refuses to start if template placeholders or empty items are still in it (an unsaved editor buffer is the usual cause). It also refuses to overwrite an existing log, warns when the Ollama server is down or not pinned to one GPU, and records the exact command in `logs/<name>.run.sh`.
- **Subscription usage limit**: when `claude -p` reports "You've hit your session limit · resets 5:30am (...)", the run waits until that time and then continues, instead of failing the calls. The wait is capped by `backends.claude.limit_wait_max` (default 6 hours) and shows up in `cosci-status`.
- **Server restarts**: if the local model server is down, calls keep retrying for `backends.local.connect_wait` seconds (default 180).

## 6. Outputs

- `runs/<timestamp>-<slug>/report.md` contains:
  - overview
  - ranking table
  - detailed cards for the top hypotheses
  - knowledge gaps
  - rejected and duplicate hypotheses, with reasons
  - evolution lineage
  - per-round meta-review feedback
  - the list of papers collected
- `state.json`: all state (context memory). `calls.jsonl`: a log of every LLM call (role, latency, failure reason).
- `Research-Ideas/<date>-<slug>.md`: one file per idea, in the Research-Ideas template. `INDEX.md` gets a new row, or the existing row is updated.

## 7. Anti-hallucination and evaluation discipline

- Literature can only be cited by IDs (P1…Pn) from papers the system actually retrieved. Any ID outside that set is removed in code.
- If `allow_web: true`, papers Claude finds on its own are marked "(unverified)".
- Novelty is labelled N0–N4 and is always shown together with the closest prior work. Rejected and duplicate hypotheses are not deleted; they stay on record with the reason.
- In tournaments, each comparison is judged a second time with the order swapped. If the two verdicts disagree, the match counts as a draw. The report shows the agreement rate.

## 8. Measured test run (reference)

- Setup:
  - `claude-only` preset, `model: haiku`
  - Literature from the **offline fixture**: 12 fake papers, so the novelty judgments are meaningless
  - 3 initial hypotheses, 1 round, run in the development sandbox
- Result:
  - 30 LLM calls, 0 failures
  - About 35 minutes of wall-clock time; one `claude -p` call took 25–220 seconds
  - Cost reported by Claude: $2.97 (at list prices; on a subscription this counts against usage limits)
- Outputs: `examples/sample_report_claude_haiku_fixture.md` and `examples/sample_idea_file.md`
- Measured on a shared GPU server (2026-10-04, live arXiv + OpenAlex search, `qwen3.5:27b` on one RTX A5000 through Ollama):
  - `local-only`, 3 hypotheses, 1 round: 57 local calls, 40 minutes.
  - `hybrid` (Claude `sonnet`), 8 hypotheses, 2 rounds: 46 Claude calls and 124 local calls, 1 hour 51 minutes, $6.25 at list prices. This run hit the subscription session limit once; 15 Claude calls failed because the version at the time did not wait for the reset.
- With the default settings (8 hypotheses, 2 rounds), the hybrid preset is expected to make roughly 40–70 Claude calls and 100–200 local calls *(an estimate; the measured run above made 46 and 124)*. Turning on live search adds time spent waiting on API rate limits.

## 9. Limitations (please read)

- **Elo is a relative rating from an LLM judge, not ground truth.** The paper itself makes the same point. With 3–4 games per hypothesis the top ranks are usually within one game of each other; the report now says which top ranks are indistinguishable and groups hypotheses by the original idea they descend from.
- **Duplicate detection is weak.** Proximity uses TF-IDF similarity with a 0.85 threshold. In the measured hybrid run, three generated hypotheses that made the same claim scored 0.12–0.28 against each other, the same range as unrelated pairs, so paraphrased duplicates are not caught. Read the ranking by family, not by row.
- Novelty judgments only cover the papers that were retrieved. Even an N3 or higher should be checked by searching again yourself.
- Free-tier rate limits on the search APIs can slow runs down. arXiv calls are spaced 3 seconds apart.
- Not yet implemented: the asynchronous Supervisor and worker queue, observation and simulation reviews, and true multi-turn debates (approximated with a single-prompt debate).
- `claude -p` runs on your logged-in account and **consumes your subscription quota**. I could not find official documentation on usage policy or billing for automated `claude -p` runs, so check this yourself.

## 10. Development

```
pytest                                 # if pytest is installed
python tests/run_without_pytest.py     # in environments without pytest
```

Structure: `src/cosci/`
- `llm/`: backends (claude_code, openai_compatible, mock) and the role router
- `literature/`: search
- `agents/`: prompts and steps
- `tournament.py`: Elo
- `similarity.py`: Proximity
- `supervisor.py`: orchestration
- `report.py`: md output and INDEX
- `cli.py`

Design notes are in `docs/DESIGN.md`.

## License

MIT. See `LICENSE`.
