"""Small shared helpers: JSON extraction, slugs, hashing, time."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import unicodedata
from typing import Any

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class JSONExtractError(ValueError):
    pass


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks (Qwen3 / reasoning models)."""
    text = _THINK_RE.sub("", text)
    # unterminated think block: drop everything up to the closing tag if present
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str) -> Any:
    """Parse the first JSON object/array found in an LLM response.

    Handles code fences, leading prose and <think> blocks.
    """
    if text is None:
        raise JSONExtractError("empty response")
    text = strip_think(text)
    candidates = [text]
    candidates += [m.group(1) for m in _FENCE_RE.finditer(text)]
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except Exception:
            pass
        obj = _scan_balanced(cand)
        if obj is not None:
            return obj
    raise JSONExtractError(f"could not find JSON in response: {text[:200]!r}")


def _scan_balanced(text: str) -> Any | None:
    """Find the first balanced {...} or [...] block that parses as JSON."""
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        chunk = text[start : i + 1]
                        try:
                            return json.loads(chunk)
                        except Exception:
                            break
            start = text.find(opener, start + 1)
    return None


def slugify(text: str, max_len: int = 48) -> str:
    """ASCII lowercase-hyphen slug. Falls back to a hash for non-latin titles."""
    norm = unicodedata.normalize("NFKD", text)
    ascii_text = norm.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    if len(slug) < 3:
        slug = "idea-" + short_hash(text, 8)
    return slug[:max_len].rstrip("-")


def short_hash(obj: Any, n: int = 12) -> str:
    if not isinstance(obj, str):
        obj = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(obj.encode("utf-8")).hexdigest()[:n]


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def today() -> str:
    return _dt.date.today().isoformat()


def truncate(text: str | None, n: int) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()
