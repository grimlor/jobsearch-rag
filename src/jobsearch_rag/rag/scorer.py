"""Semantic scoring and LLM disqualifier classification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoreResult:
    """Component scores for a single job listing."""

    fit_score: float
    archetype_score: float
    history_score: float
    disqualified: bool
    disqualifier_reason: str | None = None

    @property
    def is_valid(self) -> bool:
        """All component scores are in [0.0, 1.0]."""
        return all(
            0.0 <= s <= 1.0 for s in (self.fit_score, self.archetype_score, self.history_score)
        )


class Scorer:
    """Computes semantic similarity scores and runs the LLM disqualifier."""

    async def score(self, jd_text: str) -> ScoreResult:
        """Score a job description against resume, archetypes, and decision history."""
        raise NotImplementedError

    async def disqualify(self, jd_text: str) -> tuple[bool, str | None]:
        """Run the LLM disqualifier prompt. Returns (disqualified, reason)."""
        raise NotImplementedError
