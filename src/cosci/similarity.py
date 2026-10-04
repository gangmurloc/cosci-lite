"""Proximity: pairwise text similarity between hypotheses (TF-IDF if scikit-learn is present, else Jaccard)."""
from __future__ import annotations

import re


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9가-힣]+", text.lower()) if len(t) > 2}


def pairwise(texts: dict[str, str]) -> dict[tuple[str, str], float]:
    ids = list(texts)
    if len(ids) < 2:
        return {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        mat = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", sublinear_tf=True).fit_transform(
            [texts[i] for i in ids])
        sims = cosine_similarity(mat)
        return {(a, b): float(sims[i, j]) for i, a in enumerate(ids) for j, b in enumerate(ids) if i < j}
    except Exception:
        toks = {i: _tokens(texts[i]) for i in ids}
        out = {}
        for x, a in enumerate(ids):
            for b in ids[x + 1:]:
                u = toks[a] | toks[b]
                out[(a, b)] = len(toks[a] & toks[b]) / len(u) if u else 0.0
        return out


def get(sim: dict[tuple[str, str], float], a: str, b: str) -> float:
    return sim.get((a, b), sim.get((b, a), 0.0))
