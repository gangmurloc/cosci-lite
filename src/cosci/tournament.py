"""Elo rating and match scheduling."""
from __future__ import annotations

import random
from typing import Iterable


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def update(ra: float, rb: float, score_a: float, k: float = 32.0) -> tuple[float, float]:
    """Return new (ra, rb) after a game where A scored score_a (1, 0.5 or 0)."""
    ea = expected(ra, rb)
    ra2 = ra + k * (score_a - ea)
    rb2 = rb + k * ((1.0 - score_a) - (1.0 - ea))
    return ra2, rb2


def schedule(ids: list[str], elo: dict[str, float], sim: dict[tuple[str, str], float],
             played: Iterable[tuple[str, str]], matches_per: int, rng: random.Random,
             priority: Iterable[str] = ()) -> list[tuple[str, str]]:
    """Choose pairs for one round.

    Each id gets ~matches_per games. Opponent score = 0.5*similarity + 0.5*Elo closeness;
    already-played pairs are strongly penalised; `priority` ids (new / top) are scheduled first.
    """
    played_set = {frozenset(p) for p in played}
    count = {i: 0 for i in ids}
    pairs: list[tuple[str, str]] = []
    used: set[frozenset] = set()
    prio = [i for i in priority if i in count]
    order = prio + [i for i in rng.sample(ids, len(ids)) if i not in prio]
    for _ in range(matches_per):
        for a in order:
            if count[a] >= matches_per:
                continue
            best, best_score = None, -1e9
            for b in ids:
                if b == a or count[b] >= matches_per + 1:
                    continue
                key = frozenset((a, b))
                if key in used:
                    continue
                s = sim.get((a, b), sim.get((b, a), 0.0))
                closeness = 1.0 - min(abs(elo.get(a, 1200) - elo.get(b, 1200)) / 400.0, 1.0)
                score = 0.5 * s + 0.5 * closeness + rng.random() * 0.05
                if key in played_set:
                    score -= 1.0
                if score > best_score:
                    best, best_score = b, score
            if best is not None:
                used.add(frozenset((a, best)))
                pairs.append((a, best))
                count[a] += 1
                count[best] += 1
    return pairs
