"""Literature search over free scholarly APIs (no keys required), with disk cache and rate limiting.

Sources:
- arxiv            https://export.arxiv.org/api/query   (Atom; ~1 request / 3 s etiquette)
- semantic_scholar https://api.semanticscholar.org/graph/v1/paper/search  (optional S2_API_KEY)
- openalex         https://api.openalex.org/works        (optional mailto for the polite pool)
- fixture          local JSON file (tests / offline)
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests

from ..utils import normalize_title, short_hash

_STOP = set("a an the of for and or in on with to from by via using towards toward is are be as at "
            "into over under than that this these those what which how can do does llm llms".split())

USER_AGENT = "cosci-lite/0.1 (personal research tool)"


class SearchError(RuntimeError):
    pass


class LiteratureSearch:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.sources = list(cfg.get("sources") or ["arxiv"])
        cache = cfg.get("cache_dir") or "~/.cosci/cache"
        self.cache_dir = Path(os.path.expanduser(cache)) / "search"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.per_query = int(cfg.get("per_query", 8))
        self.min_year = cfg.get("min_year")
        self._last_call: dict[str, float] = {}
        self._lock = threading.Lock()
        self.errors: list[str] = []
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    # ------------------------------------------------------------------ public
    def search(self, query: str, k: int | None = None) -> list[dict[str, Any]]:
        """Search all configured sources; returns de-duplicated paper dicts."""
        k = k or self.per_query
        results: list[dict[str, Any]] = []
        for src in self.sources:
            try:
                results.extend(self._cached(src, query, k))
            except Exception as e:  # never let one source kill the run
                msg = f"{src}: {type(e).__name__}: {str(e)[:160]}"
                self.errors.append(msg)
        out, seen = [], set()
        for r in results:
            key = r.get("arxiv_id") or normalize_title(r.get("title", ""))
            if not key or key in seen:
                continue
            if self.min_year and r.get("year") and int(r["year"]) < int(self.min_year):
                continue
            seen.add(key)
            out.append(r)
        return out

    def check(self) -> dict[str, str]:
        status = {}
        for src in self.sources:
            try:
                n = len(self._fetch(src, "retrieval augmented generation", 2))
                status[src] = f"ok ({n} results)"
            except Exception as e:
                status[src] = f"FAILED: {type(e).__name__}: {str(e)[:120]}"
        return status

    # ------------------------------------------------------------------ cache
    def _cached(self, src: str, query: str, k: int) -> list[dict[str, Any]]:
        path = self.cache_dir / f"{src}-{short_hash([query, k], 16)}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        res = self._fetch(src, query, k)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
        return res

    def _throttle(self, src: str, min_interval: float) -> None:
        with self._lock:
            last = self._last_call.get(src, 0.0)
            wait = min_interval - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            self._last_call[src] = time.time()

    def _get(self, src: str, url: str, params: dict[str, Any], headers: dict[str, str] | None = None,
             min_interval: float = 1.0, tries: int = 4) -> requests.Response:
        last = ""
        for i in range(tries):
            self._throttle(src, min_interval)
            try:
                r = self.session.get(url, params=params, headers=headers or {}, timeout=30)
            except requests.RequestException as e:
                last = str(e)
                time.sleep(2 ** i)
                continue
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(3 * (2 ** i), 30))
                continue
            break
        raise SearchError(f"{src} request failed: {last}")

    # ------------------------------------------------------------------ sources
    def _fetch(self, src: str, query: str, k: int) -> list[dict[str, Any]]:
        if src == "arxiv":
            return self._arxiv(query, k)
        if src == "semantic_scholar":
            return self._s2(query, k)
        if src == "openalex":
            return self._openalex(query, k)
        if src == "fixture":
            return self._fixture(query, k)
        raise SearchError(f"unknown source {src!r}")

    def _arxiv(self, query: str, k: int) -> list[dict[str, Any]]:
        terms = [t for t in re.findall(r"[A-Za-z0-9\-]+", query) if t.lower() not in _STOP][:6]
        if not terms:
            return []
        out: list[dict[str, Any]] = []
        for joiner in (" AND ", " OR "):   # strict first, relax if empty
            q = joiner.join(f"all:{t}" for t in terms)
            r = self._get("arxiv", "https://export.arxiv.org/api/query",
                          {"search_query": q, "start": 0, "max_results": k, "sortBy": "relevance"},
                          min_interval=3.1)
            out = _parse_arxiv_atom(r.text)
            if out:
                break
        return out

    def _s2(self, query: str, k: int) -> list[dict[str, Any]]:
        headers = {}
        key = os.environ.get(self.cfg.get("s2_api_key_env") or "S2_API_KEY", "")
        if key:
            headers["x-api-key"] = key
        fields = "title,abstract,year,venue,externalIds,url,citationCount,authors"
        r = self._get("semantic_scholar", "https://api.semanticscholar.org/graph/v1/paper/search",
                      {"query": query, "limit": k, "fields": fields}, headers=headers,
                      min_interval=1.1 if key else 3.0, tries=4 if key else 2)
        out = []
        for p in (r.json().get("data") or []):
            ext = p.get("externalIds") or {}
            out.append({
                "title": (p.get("title") or "").strip(),
                "year": p.get("year"),
                "abstract": p.get("abstract") or "",
                "authors": [a.get("name", "") for a in (p.get("authors") or [])][:8],
                "venue": p.get("venue") or "",
                "url": p.get("url") or "",
                "arxiv_id": ext.get("ArXiv") or "",
                "doi": ext.get("DOI") or "",
                "citations": p.get("citationCount"),
                "source": "semantic_scholar",
            })
        return out

    def _openalex(self, query: str, k: int) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"search": query, "per-page": k}
        if self.cfg.get("mailto"):
            params["mailto"] = self.cfg["mailto"]
        r = self._get("openalex", "https://api.openalex.org/works", params, min_interval=0.2)
        out = []
        for w in (r.json().get("results") or []):
            doi = (w.get("doi") or "").replace("https://doi.org/", "")
            arxiv_id = ""
            m = re.search(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", doi, re.I)
            if m:
                arxiv_id = m.group(1)
            loc = w.get("primary_location") or {}
            src = (loc.get("source") or {}) if isinstance(loc, dict) else {}
            out.append({
                "title": (w.get("title") or w.get("display_name") or "").strip(),
                "year": w.get("publication_year"),
                "abstract": _openalex_abstract(w.get("abstract_inverted_index")),
                "authors": [a.get("author", {}).get("display_name", "") for a in (w.get("authorships") or [])][:8],
                "venue": src.get("display_name") or "",
                "url": loc.get("landing_page_url") or w.get("id") or "",
                "arxiv_id": arxiv_id,
                "doi": doi,
                "citations": w.get("cited_by_count"),
                "source": "openalex",
            })
        return out

    def _fixture(self, query: str, k: int) -> list[dict[str, Any]]:
        path = self.cfg.get("fixture_path")
        if not path:
            raise SearchError("fixture source requires literature.fixture_path")
        with open(path, "r", encoding="utf-8") as f:
            papers = json.load(f)
        q = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", query) if t.lower() not in _STOP)

        def score(p: dict[str, Any]) -> int:
            text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
            return sum(1 for t in q if t in text)

        ranked = sorted(papers, key=lambda p: -score(p))
        return [dict(p, source="fixture") for p in ranked[:k] if score(p) > 0]


def _parse_arxiv_atom(xml_text: str) -> list[dict[str, Any]]:
    ns = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out = []
    for e in root.findall("a:entry", ns):
        idurl = (e.findtext("a:id", default="", namespaces=ns) or "").strip()
        m = re.search(r"abs/([^v\s]+)(v\d+)?$", idurl)
        arxiv_id = m.group(1) if m else ""
        published = e.findtext("a:published", default="", namespaces=ns) or ""
        title = " ".join((e.findtext("a:title", default="", namespaces=ns) or "").split())
        if not title or title.lower() == "error":
            continue
        out.append({
            "title": title,
            "year": int(published[:4]) if published[:4].isdigit() else None,
            "abstract": " ".join((e.findtext("a:summary", default="", namespaces=ns) or "").split()),
            "authors": [a.findtext("a:name", default="", namespaces=ns) for a in e.findall("a:author", ns)][:8],
            "venue": "arXiv",
            "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else idurl,
            "arxiv_id": arxiv_id,
            "doi": e.findtext("arxiv:doi", default="", namespaces=ns) or "",
            "citations": None,
            "source": "arxiv",
        })
    return out


def _openalex_abstract(inv: dict[str, list[int]] | None) -> str:
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))
