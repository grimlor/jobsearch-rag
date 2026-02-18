"""Pipeline runner — orchestrates adapters → RAG → rank.

The PipelineRunner is the top-level orchestrator that ties the entire
system together:

1. Load and validate settings
2. Health-check Ollama (fail fast before browser work)
3. For each enabled board: authenticate → search → extract details
4. Score each listing through the RAG pipeline
5. Rank, deduplicate, and filter
6. Export results

The runner owns the control flow but delegates all domain logic to
specialized components (adapters, scorer, ranker, exporters).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobsearch_rag.adapters import AdapterRegistry
from jobsearch_rag.adapters.session import SessionConfig, SessionManager, throttle
from jobsearch_rag.errors import ActionableError
from jobsearch_rag.pipeline.ranker import RankedListing, Ranker, RankSummary
from jobsearch_rag.rag.comp_parser import compute_comp_score, parse_compensation
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.indexer import Indexer
from jobsearch_rag.rag.scorer import Scorer
from jobsearch_rag.rag.store import VectorStore

if TYPE_CHECKING:
    from jobsearch_rag.adapters.base import JobListing
    from jobsearch_rag.config import Settings
    from jobsearch_rag.rag.scorer import ScoreResult

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Results from a pipeline run, consumed by exporters and CLI."""

    ranked_listings: list[RankedListing] = field(default_factory=list)
    summary: RankSummary = field(default_factory=RankSummary)
    failed_listings: int = 0
    boards_searched: list[str] = field(default_factory=list)


class PipelineRunner:
    """Top-level orchestrator: loads adapters, runs browser sessions,
    feeds results through the RAG scorer, and hands off to the ranker.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._embedder = Embedder(
            base_url=settings.ollama.base_url,
            embed_model=settings.ollama.embed_model,
            llm_model=settings.ollama.llm_model,
        )
        self._store = VectorStore(persist_dir=settings.chroma.persist_dir)
        self._scorer = Scorer(
            store=self._store,
            embedder=self._embedder,
            disqualify_on_llm_flag=settings.scoring.disqualify_on_llm_flag,
        )
        self._ranker = Ranker(
            archetype_weight=settings.scoring.archetype_weight,
            fit_weight=settings.scoring.fit_weight,
            history_weight=settings.scoring.history_weight,
            comp_weight=settings.scoring.comp_weight,
            min_score_threshold=settings.scoring.min_score_threshold,
        )
        self._base_salary = settings.scoring.base_salary

    async def run(
        self,
        boards: list[str] | None = None,
        *,
        overnight: bool = False,
    ) -> RunResult:
        """Execute a full search-score-rank pipeline.

        Args:
            boards: Specific board names to search.  ``None`` = all enabled.
            overnight: If ``True``, enforce extended throttling for stealth boards.

        Returns:
            A :class:`RunResult` with ranked listings and summary statistics.
        """
        # Step 1: Health check Ollama before any browser work
        await self._embedder.health_check()
        logger.info("Ollama health check passed")

        # Step 1b: Auto-index if collections are empty (first run or post-reset)
        await self._ensure_indexed()

        # Step 2: Determine which boards to search
        board_names = boards or list(self._settings.enabled_boards)
        if overnight:
            # In overnight mode, also include overnight-only boards
            for b in self._settings.overnight_boards:
                if b not in board_names:
                    board_names.append(b)

        logger.info("Searching boards: %s", ", ".join(board_names))

        # Step 3: Collect listings from all boards (concurrently)
        all_listings: list[JobListing] = []
        failed_count = 0

        async def _search_one(board_name: str) -> tuple[str, list[JobListing], int]:
            """Search a single board, returning (name, listings, failures).

            Board-level errors are caught here so one failure
            doesn't cancel the other concurrent searches.
            """
            try:
                board_listings, board_failures = await self._search_board(
                    board_name, overnight=overnight
                )
                logger.info(
                    "Board '%s': %d listings collected, %d failures",
                    board_name,
                    len(board_listings),
                    board_failures,
                )
                return board_name, board_listings, board_failures
            except ActionableError as exc:
                logger.error("Board '%s' failed entirely: %s", board_name, exc.error)
                return board_name, [], 0

        results = await asyncio.gather(*[_search_one(b) for b in board_names])

        for _name, board_listings, board_failures in results:
            all_listings.extend(board_listings)
            failed_count += board_failures

        if not all_listings:
            logger.warning("No listings collected from any board")
            return RunResult(
                boards_searched=board_names,
                failed_listings=failed_count,
            )

        # Step 4: Score each listing
        scored: list[tuple[JobListing, ScoreResult]] = []
        embeddings: dict[str, list[float]] = {}

        for listing in all_listings:
            try:
                score_result = await self._scorer.score(listing.full_text)

                # Parse compensation from JD text and compute comp_score
                comp = parse_compensation(listing.full_text)
                if comp is not None:
                    listing.comp_min = comp.comp_min
                    listing.comp_max = comp.comp_max
                    listing.comp_source = comp.comp_source
                    listing.comp_text = comp.comp_text
                score_result.comp_score = compute_comp_score(
                    listing.comp_max, self._base_salary
                )

                scored.append((listing, score_result))
                # Cache the embedding for deduplication
                embedding = await self._embedder.embed(listing.full_text)
                embeddings[listing.url] = embedding
            except ActionableError as exc:
                logger.warning(
                    "Scoring failed for %s (%s): %s",
                    listing.title,
                    listing.url,
                    exc.error,
                )
                failed_count += 1

        # Step 5: Rank, deduplicate, filter
        ranked, summary = self._ranker.rank(scored, embeddings)

        return RunResult(
            ranked_listings=ranked,
            summary=summary,
            failed_listings=failed_count,
            boards_searched=board_names,
        )

    async def _ensure_indexed(self) -> None:
        """Auto-index resume and archetypes if collections are empty.

        After a ``reset`` or on first run, the ``resume`` and
        ``role_archetypes`` collections will be empty.  Rather than
        failing every scoring call, detect this and run the indexer
        automatically — it's the only sensible recovery.
        """
        needs_resume = self._collection_empty("resume")
        needs_archetypes = self._collection_empty("role_archetypes")

        if not needs_resume and not needs_archetypes:
            return

        logger.info("Empty collections detected — auto-indexing before scoring")
        indexer = Indexer(store=self._store, embedder=self._embedder)

        if needs_archetypes:
            n = await indexer.index_archetypes(self._settings.archetypes_path)
            logger.info("Auto-indexed %d archetypes", n)

        if needs_resume:
            n = await indexer.index_resume(self._settings.resume_path)
            logger.info("Auto-indexed %d resume chunks", n)

    def _collection_empty(self, name: str) -> bool:
        """Return True if the named collection is missing or has zero documents."""
        try:
            return self._store.collection_count(name) == 0
        except ActionableError:
            return True

    async def _search_board(
        self,
        board_name: str,
        *,
        overnight: bool = False,
    ) -> tuple[list[JobListing], int]:
        """Search a single board and return (listings, failure_count).

        Manages the browser session lifecycle for this board.
        """
        adapter = AdapterRegistry.get(board_name)
        board_cfg = self._settings.boards.get(board_name)

        if board_cfg is None:
            logger.warning("No config section for board '%s' — skipping", board_name)
            return [], 0

        is_overnight = overnight or board_name in self._settings.overnight_boards
        config = SessionConfig(
            board_name=board_name,
            headless=board_cfg.headless,
            stealth=board_name == "linkedin",
            overnight=is_overnight,
            browser_channel=board_cfg.browser_channel,
        )

        listings: list[JobListing] = []
        failed = 0

        async with SessionManager(config) as session:
            page = await session.new_page()

            # Authenticate
            await adapter.authenticate(page)
            await session.save_storage_state()

            # Search each configured URL
            for search_url in board_cfg.searches:
                await throttle(adapter)
                try:
                    results = await adapter.search(
                        page, search_url, max_pages=board_cfg.max_pages
                    )
                except ActionableError as exc:
                    logger.warning(
                        "Search failed for %s @ %s: %s",
                        board_name,
                        search_url,
                        exc.error,
                    )
                    continue

                # Extract details for each listing (skip if already enriched)
                for listing in results:
                    if listing.full_text.strip():
                        # Already enriched during search (e.g. click-through)
                        listings.append(listing)
                        continue

                    await throttle(adapter)
                    try:
                        enriched = await adapter.extract_detail(page, listing)
                        if enriched.full_text.strip():
                            listings.append(enriched)
                        else:
                            logger.warning(
                                "Empty JD text for %s — skipping",
                                listing.url,
                            )
                            failed += 1
                    except ActionableError as exc:
                        logger.warning(
                            "Detail extraction failed for %s: %s",
                            listing.url,
                            exc.error,
                        )
                        failed += 1
                    except Exception:
                        logger.exception(
                            "Unexpected error extracting %s",
                            listing.url,
                        )
                        failed += 1

        return listings, failed
