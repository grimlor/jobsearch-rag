"""Pipeline runner — orchestrates adapters → RAG → rank."""

from __future__ import annotations


class PipelineRunner:
    """Top-level orchestrator: loads adapters, runs browser sessions,
    feeds results through the RAG scorer, and hands off to the ranker.
    """

    async def run(
        self,
        boards: list[str] | None = None,
        *,
        overnight: bool = False,
    ) -> None:
        """Execute a full search-score-rank-export pipeline.

        Args:
            boards: Specific board names to search.  ``None`` = all enabled.
            overnight: If ``True``, enforce extended throttling for stealth boards.
        """
        raise NotImplementedError
