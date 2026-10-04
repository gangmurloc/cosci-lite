"""Data records stored in the run state."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

NOVELTY_LEVELS = ["N0", "N1", "N2", "N3", "N4"]
NOVELTY_DESC = {
    "N0": "Known / already established",
    "N1": "Minor variation of known work",
    "N2": "Combination of existing ideas, novelty uncertain",
    "N3": "Potentially novel",
    "N4": "Strong evidence of novelty after literature check",
}


def novelty_rank(label: str | None) -> int:
    try:
        return NOVELTY_LEVELS.index((label or "").strip().upper()[:2])
    except ValueError:
        return -1


@dataclass
class Paper:
    pid: str                 # local id, e.g. "P3"
    title: str
    year: int | None = None
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    venue: str = ""
    url: str = ""
    arxiv_id: str = ""
    doi: str = ""
    citations: int | None = None
    source: str = ""         # arxiv | semantic_scholar | openalex | fixture
    queries: list[str] = field(default_factory=list)

    @property
    def link(self) -> str:
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        if self.url:
            return self.url
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Paper":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Hypothesis:
    hid: str                               # H1, H3-v2, ...
    fields: dict[str, Any]                 # LLM-produced content (title, statement, ...)
    origin: str = "generated"              # generated | evolved | user
    strategy: str = ""
    parents: list[str] = field(default_factory=list)
    round: int = 0
    status: str = "active"                 # active | rejected | duplicate
    status_reason: str = ""
    initial_review: dict[str, Any] | None = None
    novelty_review: dict[str, Any] | None = None
    elo: float = 1200.0
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def title(self) -> str:
        return str(self.fields.get("title", self.hid))

    @property
    def novelty(self) -> str:
        if self.novelty_review:
            return str(self.novelty_review.get("novelty", "?"))
        return "?"

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws

    def text_for_similarity(self) -> str:
        f = self.fields
        return " ".join(str(f.get(k, "")) for k in ("title", "one_liner", "statement", "idea"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Hypothesis":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Match:
    a: str
    b: str
    round: int
    mode: str                 # compare | debate
    score_a: float            # 1, 0, 0.5
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Match":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
