"""Score fusion, deduplication, and final ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jobsearch_rag.adapters.base import JobListing
    from jobsearch_rag.rag.scorer import ScoreResult


@dataclass
class RankedListing:
    """A listing enriched with its computed scores and final rank."""

    listing: JobListing
    scores: ScoreResult
    final_score: float
    duplicate_boards: list[str] | None = None


class Ranker:
    """Fuses weighted component scores, deduplicates across boards,
    and produces the final ranked shortlist.
    """

    def __init__(
        self,
        archetype_weight: float,
        fit_weight: float,
        history_weight: float,
        min_score_threshold: float,
    ) -> None:
        self.archetype_weight = archetype_weight
        self.fit_weight = fit_weight
        self.history_weight = history_weight
        self.min_score_threshold = min_score_threshold

    def rank(
        self,
        listings: list[tuple[JobListing, ScoreResult]],
    ) -> list[RankedListing]:
        """Apply score fusion, deduplication, and threshold filtering."""
        raise NotImplementedError

    def compute_final_score(self, scores: ScoreResult) -> float:
        """Weighted sum of component scores, zeroed if disqualified."""
        if scores.disqualified:
            return 0.0
        return (
            self.archetype_weight * scores.archetype_score
            + self.fit_weight * scores.fit_score
            + self.history_weight * scores.history_score
        )
