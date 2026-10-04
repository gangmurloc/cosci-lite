"""Configuration: defaults, YAML loading, presets."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROLES = [
    "parse",     # research goal -> plan configuration
    "digest",    # literature digest (facts, limitations, gaps)
    "generate",  # initial hypothesis generation
    "review",    # initial review (no search) + search queries
    "novelty",   # search-grounded novelty review
    "compare",   # single-turn pairwise comparison
    "debate",    # simulated debate for top-k pairs
    "evolve",    # evolution of top hypotheses
    "feedback",  # per-round meta-review feedback
    "overview",  # final research overview
]

DEFAULTS: dict[str, Any] = {
    "language": "ko",
    "backends": {
        "claude": {
            "type": "claude_code",
            "executable": "claude",
            "model": "sonnet",
            "timeout": 900,
            "allow_web": False,
            "max_turns": 6,
            "json_schema_flag": "auto",   # auto | on | off
            "max_workers": 2,
            "limit_wait_max": 21600,      # seconds to wait in total for a subscription usage limit to reset (0 = fail)
            "limit_poll": 300,            # wait this long when the limit message does not say when it resets
        },
        "local": {
            "type": "openai_compatible",
            "base_url": "http://localhost:8000/v1",
            "model": "Qwen/Qwen3-14B-AWQ",
            "api_key": "EMPTY",
            "temperature": 0.7,
            "max_tokens": 8000,
            "json_mode": "schema",        # schema | object | none
            "timeout": 600,
            "max_workers": 4,
            "connect_wait": 180,          # seconds to keep retrying while the server is down / restarting
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        "mock": {"type": "mock", "max_workers": 1},
    },
    "roles": {
        "parse": "claude",
        "digest": "claude",
        "generate": "local",
        "review": "local",
        "novelty": "claude",
        "compare": "local",
        "debate": "claude",
        "evolve": "claude",
        "feedback": "local",
        "overview": "claude",
    },
    "literature": {
        "sources": ["arxiv", "semantic_scholar", "openalex"],
        "per_query": 8,
        "max_papers": 60,
        "digest_papers": 30,
        "novelty_queries": 3,
        "novelty_papers": 10,
        "cache_dir": "~/.cosci/cache",
        "mailto": "",
        "s2_api_key_env": "S2_API_KEY",
        "fixture_path": "",
        "min_year": None,
    },
    "pipeline": {
        "n_initial": 8,
        "gen_batch": 3,
        "rounds": 2,
        "evolve_top_k": 3,
        "matches_per_hyp": 3,
        "debate_top_k": 4,
        "swap_check": True,          # compare matches judged in both orders
        "swap_check_debate": False,  # debates are already order-robust and costlier
        "elo_k": 32,
        "dedupe_threshold": 0.85,
        "meta_feedback": True,
        "seed": 42,
    },
    "budget": {"max_calls": {"claude": 80, "local": 600, "mock": 100000}},
    "output": {
        "runs_dir": "runs",
        "ideas_dir": None,
        "auto_save_top_k": 2,
        "auto_save_min_novelty": "N2",
        "report_top_k": 5,
    },
}

PRESETS: dict[str, dict[str, str]] = {
    "hybrid": dict(DEFAULTS["roles"]),
    "claude-only": {r: "claude" for r in ROLES},
    "local-only": {r: "local" for r in ROLES},
    "mock": {r: "mock" for r in ROLES},
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | os.PathLike | None = None, preset: str | None = None,
                overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, user)
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {list(PRESETS)}")
        cfg["roles"] = dict(PRESETS[preset])
    if overrides:
        cfg = deep_merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    missing = [r for r in ROLES if r not in cfg["roles"]]
    if missing:
        raise ValueError(f"roles missing in config: {missing}")
    for role, backend in cfg["roles"].items():
        if backend not in cfg["backends"]:
            raise ValueError(f"role {role!r} uses undefined backend {backend!r}")


def expand_path(p: str | None) -> Path | None:
    if not p:
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


EXAMPLE_YAML = """# cosci-lite config. Only include values you want to change; the rest use defaults.
language: ko            # ko: free text in Korean (technical terms in English), en: English

backends:
  claude:               # Claude Code CLI (claude -p). Runs on your logged-in Claude subscription.
    type: claude_code
    executable: claude  # on Windows this may be claude.cmd or claude.exe
    model: sonnet       # sonnet | opus | full model name
    allow_web: false    # true: Claude may also use WebSearch/WebFetch during novelty review (more calls/time)
    timeout: 900
  local:                # OpenAI-compatible server (vLLM, Ollama, LM Studio)
    type: openai_compatible
    base_url: http://localhost:8000/v1
    model: Qwen/Qwen3-14B-AWQ
    json_mode: schema   # vLLM and recent Ollama: schema; older servers: object or none (auto-degrades on HTTP 400)
    extra_body:
      chat_template_kwargs: {enable_thinking: false}   # vLLM: thinking off
    # Ollama instead of vLLM:
    #   base_url: http://127.0.0.1:11434/v1
    #   model: qwen3.5:27b
    #   max_workers: 1      # Ollama answers one request at a time; parallel requests just queue and time out
    #   extra_body:
    #     reasoning_effort: none   # thinking off (Ollama ignores chat_template_kwargs)
    #     keep_alive: 30s          # how long the model stays on the GPU after the last call

# Which backend each role uses (hybrid default). Use --preset claude-only | local-only to switch everything.
roles:
  parse: claude
  digest: claude
  generate: local
  review: local
  novelty: claude
  compare: local
  debate: claude
  evolve: claude
  feedback: local
  overview: claude

literature:
  sources: [arxiv, semantic_scholar, openalex]
  per_query: 8
  max_papers: 60
  mailto: ""            # entering an email puts you in OpenAlex's polite pool
  min_year: 2023        # ignore papers older than this year (null = no limit)

pipeline:
  n_initial: 8          # number of initial hypotheses
  rounds: 2             # evolution + tournament rounds
  evolve_top_k: 3
  matches_per_hyp: 3
  debate_top_k: 4
  swap_check: true      # judge both (A,B) and (B,A) to reduce position bias

budget:
  max_calls: {claude: 80, local: 600}

output:
  runs_dir: runs
  ideas_dir: "~/Research-Ideas"   # auto-save ideas + update INDEX.md (use null to turn auto-save off)
  auto_save_top_k: 2
  auto_save_min_novelty: N2
"""
