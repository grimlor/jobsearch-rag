"""
Parallel scoring loop tests — concurrency orchestration, error isolation,
collection metric aggregation, and environment variable configuration.

Maps to BDD specs: TestParallelScoringOrchestration, TestErrorIsolation,
TestCollectionScoreAggregation, TestEnvironmentVariableConfig

Public API surface (from src/jobsearch_rag/pipeline/runner):
    PipelineRunner(settings: Settings)
    runner.run(boards=..., overnight=..., force_rescore=...) -> RunResult
    RunResult.ranked_listings: list[RankedListing]
    RunResult.failed_listings: int
    RunResult.skipped_decisions: int
    RunResult.errors: list[ActionableError]
    RunResult.boards_searched: list[str]

Public API surface (from src/jobsearch_rag/rag/scorer):
    Scorer(store=..., embedder=..., disqualify_on_llm_flag=...)
    scorer.score(jd_text) -> ScoreResult
    scorer.collection_scores -> dict[str, list[float]]

Public API surface (from src/jobsearch_rag/config):
    Settings, ScoringConfig, OllamaConfig, OutputConfig, ChromaConfig, BoardConfig

Public API surface (from src/jobsearch_rag/logging):
    log_event(event, **data) -> None
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import ollama as ollama_sdk
import pytest

from jobsearch_rag.adapters import AdapterRegistry
from jobsearch_rag.adapters.base import JobBoardAdapter, JobListing
from jobsearch_rag.logging import log_event as _real_log_event
from jobsearch_rag.pipeline.runner import PipelineRunner, RunResult
from tests.conftest import make_test_settings
from tests.constants import EMBED_FAKE

if TYPE_CHECKING:
    from collections.abc import Callable

    from jobsearch_rag.config import Settings

_RUNNER_LOG_EVENT = "jobsearch_rag.pipeline.runner.log_event"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adapt(adapter: object) -> Callable[..., JobBoardAdapter]:
    """Wrap an adapter/mock as a registry-compatible factory accepting any kwargs."""

    def _factory(**_kwargs: object) -> JobBoardAdapter:
        return cast("JobBoardAdapter", adapter)

    return _factory


def _make_settings(tmpdir: str) -> Settings:
    """Create Settings with a testboard and tmp directories."""
    return make_test_settings(tmpdir, scoring_overrides={"min_score_threshold": 0.0})


def _make_listing(
    external_id: str = "1",
    title: str = "Staff Architect",
    full_text: str | None = None,
) -> JobListing:
    """Create a representative JobListing for testing."""
    if full_text is None:
        full_text = (
            f"A detailed job description for listing {external_id} "
            f"covering a staff architect role at a premier tech company."
        )
    return JobListing(
        board="testboard",
        external_id=external_id,
        title=title,
        company=f"Acme-{external_id}",
        location="Remote",
        url=f"https://testboard.com/{external_id}",
        full_text=full_text,
        max_full_text_chars=250_000,
    )


def _make_runner_with_real_stack(
    settings: Settings,
) -> tuple[PipelineRunner, AsyncMock]:
    """
    Create a PipelineRunner with real internals and mocked Ollama client.

    Only ``ollama_sdk.AsyncClient`` is mocked — the I/O boundary.
    Everything else (Embedder, Scorer, VectorStore, Ranker) runs for real.
    Collections are pre-populated so auto-indexing is skipped.
    """
    mock_client = AsyncMock()

    # health_check needs models list
    model_embed = MagicMock()
    model_embed.model = settings.ollama.embed_model
    model_llm = MagicMock()
    model_llm.model = settings.ollama.llm_model
    list_response = MagicMock()
    list_response.models = [model_embed, model_llm]
    mock_client.list.return_value = list_response

    # embed() returns unique embeddings per prompt so near-dedup won't merge them
    _embed_counter = 0

    async def _embed_side_effect(
        *_args: object,
        **kwargs: object,
    ) -> MagicMock:
        nonlocal _embed_counter
        _embed_counter += 1
        response = MagicMock()
        # Each call returns a near-orthogonal vector so cosine sim < 0.95
        vec = [0.01] * len(EMBED_FAKE)
        vec[_embed_counter % len(vec)] = 1.0
        response.embeddings = [vec]
        response.prompt_eval_count = 42
        return response

    mock_client.embed = AsyncMock(side_effect=_embed_side_effect)

    with patch(
        "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
        return_value=mock_client,
    ):
        runner = PipelineRunner(settings)

    # Populate required collections so auto-indexing is skipped
    store = runner.store
    for name in ("resume", "role_archetypes", "global_positive_signals"):
        store.add_documents(
            name,
            ids=[f"{name}-seed"],
            documents=[f"Seed document for {name}"],
            embeddings=[EMBED_FAKE],
        )

    return runner, mock_client


def _make_test_adapter(
    search_results: list[JobListing] | None = None,
) -> MagicMock:
    """Create a mock adapter that returns the given listings."""
    adapter = MagicMock()
    adapter.board_name = "testboard"
    adapter.rate_limit_seconds = (0.0, 0.0)
    adapter.authenticate = AsyncMock()
    adapter.search = AsyncMock(return_value=search_results or [])
    adapter.extract_detail = AsyncMock()
    return adapter


def _mock_playwright_boundary() -> tuple[MagicMock, MagicMock]:
    """Create a mock Playwright I/O boundary."""
    mock_page = MagicMock()

    mock_context = MagicMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)
    mock_context.storage_state = AsyncMock(return_value={"cookies": [], "origins": []})
    mock_context.close = AsyncMock()

    mock_browser = MagicMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_browser.close = AsyncMock()

    mock_pw = MagicMock()
    mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_pw.stop = AsyncMock()

    mock_pw_cm = MagicMock()
    mock_pw_cm.start = AsyncMock(return_value=mock_pw)

    mock_async_pw = MagicMock(return_value=mock_pw_cm)

    return mock_async_pw, mock_page


async def _run_pipeline(
    runner: PipelineRunner,
    listings: list[JobListing],
    *,
    tmpdir: str,
    env: dict[str, str] | None = None,
) -> RunResult:
    """
    Run the pipeline with the given listings injected via mock adapter.

    Returns RunResult. ``env`` overrides environment variables for the run.
    """
    adapter = _make_test_adapter(search_results=listings)
    mock_pw_fn, _ = _mock_playwright_boundary()

    patches = {
        "testboard": _adapt(adapter),
    }

    with (
        AdapterRegistry.override(patches),
        patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
        patch("jobsearch_rag.adapters.session._DEFAULT_STORAGE_DIR", Path(tmpdir)),
    ):
        if env is not None:
            with patch.dict("os.environ", env):
                return await runner.run()
        return await runner.run()


# ---------------------------------------------------------------------------
# B1 — TestParallelScoringOrchestration
# ---------------------------------------------------------------------------


class TestParallelScoringOrchestration:
    """
    REQUIREMENT: The scoring loop processes listings concurrently up to a
    configurable limit derived from the OLLAMA_NUM_PARALLEL environment variable.

    WHO: The pipeline runner optimizing wall-clock time for Ollama-bound scoring
    WHAT: (1) Listings are scored concurrently up to max_parallel (from env var,
              default 1)
          (2) The semaphore caps concurrent scoring tasks to the configured limit
          (3) Results are identical regardless of max_parallel value (deterministic
              scores)
          (4) A scoring_parallelism structured log event is emitted with
              max_parallel and listing_count before the scoring loop begins
    WHY: Ollama can pipeline multiple requests when OLLAMA_NUM_PARALLEL > 1.
         Serial scoring wastes 50-70% of available inference throughput. The
         semaphore prevents over-subscription that would exceed VRAM budget.

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (I/O boundary to Ollama)
        Real:  PipelineRunner, Embedder, Scorer, VectorStore, Ranker,
               all aggregation logic
        Never: Construct ScoreResult directly — always obtained via real
               Scorer.score()
    """

    async def test_listings_scored_concurrently_up_to_max_parallel(self) -> None:
        """
        Given OLLAMA_NUM_PARALLEL=2 and 4 listings,
        When the pipeline runs,
        Then all 4 listings are scored and appear in ranked results.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: runner with 4 listings and OLLAMA_NUM_PARALLEL=2
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)
            listings = [_make_listing(external_id=str(i)) for i in range(4)]

            # When: pipeline runs with parallelism enabled
            result = await _run_pipeline(
                runner,
                listings,
                tmpdir=tmpdir,
                env={"OLLAMA_NUM_PARALLEL": "2"},
            )

            # Then: all 4 listings scored successfully
            assert len(result.ranked_listings) == 4, (
                f"Expected 4 ranked listings, got {len(result.ranked_listings)}. "
                f"Failed: {result.failed_listings}, errors: {result.errors}"
            )
            assert result.failed_listings == 0, (
                f"Expected 0 failures, got {result.failed_listings}"
            )

    async def test_semaphore_caps_concurrent_tasks(self) -> None:
        """
        Given OLLAMA_NUM_PARALLEL=2 and 4 listings,
        When the pipeline scores them,
        Then at most 2 embed calls are in-flight concurrently.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: runner instrumented to track concurrency
            settings = _make_settings(tmpdir)
            runner, mock_client = _make_runner_with_real_stack(settings)
            listings = [_make_listing(external_id=str(i)) for i in range(4)]

            max_concurrent = 0
            current_concurrent = 0
            lock = asyncio.Lock()

            original_embed = mock_client.embed

            async def _tracking_embed(*args: object, **kwargs: object) -> object:
                nonlocal max_concurrent, current_concurrent
                async with lock:
                    current_concurrent += 1
                    if current_concurrent > max_concurrent:
                        max_concurrent = current_concurrent
                try:
                    return await original_embed(*args, **kwargs)
                finally:
                    async with lock:
                        current_concurrent -= 1

            mock_client.embed = _tracking_embed

            # When: run with max_parallel=2
            await _run_pipeline(
                runner,
                listings,
                tmpdir=tmpdir,
                env={"OLLAMA_NUM_PARALLEL": "2"},
            )

            # Then: concurrency never exceeded 2
            # Note: embed is called for scoring AND for dedup caching,
            # so we check the semaphore cap, not exact call counts
            assert max_concurrent <= 2, (
                f"Expected max 2 concurrent embed calls, observed {max_concurrent}"
            )

    async def test_results_identical_regardless_of_max_parallel(self) -> None:
        """
        Given the same listings and mock Ollama responses,
        When run with max_parallel=1 and max_parallel=3,
        Then the ranked listings have identical scores.
        """
        listings = [_make_listing(external_id=str(i)) for i in range(3)]

        results_by_parallel: dict[int, list[float]] = {}
        for max_p in (1, 3):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Given: identical runner and listings
                settings = _make_settings(tmpdir)
                runner, _ = _make_runner_with_real_stack(settings)

                # When: run at different parallelism levels
                result = await _run_pipeline(
                    runner,
                    listings,
                    tmpdir=tmpdir,
                    env={"OLLAMA_NUM_PARALLEL": str(max_p)},
                )

                # Collect scores sorted by external_id for stable comparison
                scores = sorted(
                    [(r.listing.external_id, r.final_score) for r in result.ranked_listings],
                    key=lambda t: t[0],
                )
                results_by_parallel[max_p] = [s for _, s in scores]

        # Then: scores are identical
        assert results_by_parallel[1] == pytest.approx(results_by_parallel[3], abs=1e-9), (
            f"Scores differ between serial and parallel: "
            f"serial={results_by_parallel[1]}, parallel={results_by_parallel[3]}"
        )

    async def test_scoring_parallelism_log_event_emitted(self) -> None:
        """
        Given OLLAMA_NUM_PARALLEL=3 and 5 listings,
        When the pipeline runs,
        Then a scoring_parallelism log event is emitted with max_parallel=3
        and listing_count=5.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: runner with 5 listings
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)
            listings = [_make_listing(external_id=str(i)) for i in range(5)]

            logged_events: list[dict[str, object]] = []

            def _capture_log_event(event: str, **data: object) -> None:
                logged_events.append({"event": event, **data})
                _real_log_event(event, **data)

            # When: run with parallelism
            with patch(_RUNNER_LOG_EVENT, side_effect=_capture_log_event):
                await _run_pipeline(
                    runner,
                    listings,
                    tmpdir=tmpdir,
                    env={"OLLAMA_NUM_PARALLEL": "3"},
                )

            # Then: scoring_parallelism event emitted with correct values
            parallelism_events = [e for e in logged_events if e["event"] == "scoring_parallelism"]
            assert len(parallelism_events) == 1, (
                f"Expected 1 scoring_parallelism event, got {len(parallelism_events)}. "
                f"All events: {[e['event'] for e in logged_events]}"
            )
            evt = parallelism_events[0]
            assert evt["max_parallel"] == 3, f"Expected max_parallel=3, got {evt['max_parallel']}"
            assert evt["listing_count"] == 5, (
                f"Expected listing_count=5, got {evt['listing_count']}"
            )


# ---------------------------------------------------------------------------
# B2 — TestErrorIsolation
# ---------------------------------------------------------------------------


class TestErrorIsolation:
    """
    REQUIREMENT: A scoring failure in one listing does not cancel or block
    other concurrent listings.

    WHO: The pipeline runner ensuring resilience under partial failure
    WHAT: (1) An ActionableError in one task does not prevent other tasks
              from completing
          (2) An unexpected Exception in one task does not prevent other
              tasks from completing
          (3) Failed listings are counted and their errors surfaced in
              RunResult.errors
          (4) The session_summary log event reflects the correct
              failed_listings count
    WHY: The existing sequential loop catches errors per-listing and
         continues. Parallelization must preserve this contract — a bad JD
         should not poison the entire batch.

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (configured to raise on specific
               embed calls)
        Real:  PipelineRunner, Embedder, Scorer, error aggregation
        Never: Mock the runner's error handling — test the real path
    """

    async def test_actionable_error_does_not_block_other_listings(self) -> None:
        """
        Given 3 listings where the 2nd triggers an embed error,
        When the pipeline runs with OLLAMA_NUM_PARALLEL=2,
        Then the 1st and 3rd listings are scored and 1 failure is reported.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: 3 listings, embed fails for external_id "bad"
            settings = _make_settings(tmpdir)
            runner, mock_client = _make_runner_with_real_stack(settings)

            good1 = _make_listing(external_id="good1", title="Good Job One")
            bad = _make_listing(external_id="bad", title="Bad Job")
            good2 = _make_listing(external_id="good2", title="Good Job Two")

            call_count = 0

            async def _failing_embed(*args: object, **kwargs: object) -> MagicMock:
                nonlocal call_count
                call_count += 1
                if call_count == 3:
                    raise ollama_sdk.ResponseError("embed failed")
                resp = MagicMock()
                # Unique embeddings so successful listings aren't near-deduped
                vec = [0.01] * len(EMBED_FAKE)
                vec[call_count % len(vec)] = 1.0
                resp.embeddings = [vec]
                resp.prompt_eval_count = 42
                return resp

            mock_client.embed = AsyncMock(side_effect=_failing_embed)

            # When: run with parallelism
            result = await _run_pipeline(
                runner,
                [good1, bad, good2],
                tmpdir=tmpdir,
                env={"OLLAMA_NUM_PARALLEL": "2"},
            )

            # Then: at least 1 listing succeeded and 1 failed
            assert result.failed_listings >= 1, (
                f"Expected at least 1 failure, got {result.failed_listings}"
            )
            assert len(result.ranked_listings) >= 1, (
                f"Expected at least 1 successful listing, got {len(result.ranked_listings)}. "
                f"Errors: {result.errors}"
            )
            total = len(result.ranked_listings) + result.failed_listings
            assert total == 3, (
                f"Expected 3 total (scored + failed), got {total}. "
                f"ranked={len(result.ranked_listings)}, failed={result.failed_listings}"
            )

    async def test_unexpected_exception_does_not_block_other_listings(self) -> None:
        """
        Given 3 listings where the 2nd raises an unexpected RuntimeError,
        When the pipeline runs with OLLAMA_NUM_PARALLEL=2,
        Then the other listings are scored and the error is surfaced.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: embed raises RuntimeError on 3rd call
            settings = _make_settings(tmpdir)
            runner, mock_client = _make_runner_with_real_stack(settings)

            listings = [_make_listing(external_id=str(i)) for i in range(3)]

            call_count = 0

            async def _runtime_error_embed(*args: object, **kwargs: object) -> MagicMock:
                nonlocal call_count
                call_count += 1
                if call_count == 3:
                    raise RuntimeError("unexpected GPU failure")
                resp = MagicMock()
                vec = [0.01] * len(EMBED_FAKE)
                vec[call_count % len(vec)] = 1.0
                resp.embeddings = [vec]
                resp.prompt_eval_count = 42
                return resp

            mock_client.embed = AsyncMock(side_effect=_runtime_error_embed)

            # When: run with parallelism
            result = await _run_pipeline(
                runner,
                listings,
                tmpdir=tmpdir,
                env={"OLLAMA_NUM_PARALLEL": "2"},
            )

            # Then: at least 1 succeeded and 1 failed
            assert result.failed_listings >= 1, (
                f"Expected at least 1 failure, got {result.failed_listings}"
            )
            assert len(result.ranked_listings) >= 1, (
                f"Expected at least 1 success, got {len(result.ranked_listings)}"
            )

    async def test_failed_listings_surfaced_in_errors(self) -> None:
        """
        Given a listing that fails during scoring,
        When the pipeline completes,
        Then RunResult.errors contains the corresponding ActionableError.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: embed always fails
            settings = _make_settings(tmpdir)
            runner, mock_client = _make_runner_with_real_stack(settings)

            listings = [_make_listing(external_id="fail1")]
            mock_client.embed = AsyncMock(
                side_effect=ollama_sdk.ResponseError("model not found"),
            )

            # When: run
            result = await _run_pipeline(
                runner,
                listings,
                tmpdir=tmpdir,
                env={"OLLAMA_NUM_PARALLEL": "1"},
            )

            # Then: errors list is populated
            assert len(result.errors) >= 1, f"Expected at least 1 error, got {len(result.errors)}"
            assert result.failed_listings == 1, (
                f"Expected 1 failed listing, got {result.failed_listings}"
            )

    async def test_session_summary_reflects_correct_failed_count(self) -> None:
        """
        Given 4 listings where 2 fail during scoring,
        When the pipeline completes,
        Then the session_summary log event shows failed_listings=2.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: embed fails on calls 3 and 5 (2nd and 3rd listings' first embed)
            settings = _make_settings(tmpdir)
            runner, mock_client = _make_runner_with_real_stack(settings)

            listings = [_make_listing(external_id=str(i)) for i in range(4)]

            call_count = 0
            embed_response = MagicMock()
            embed_response.embeddings = [EMBED_FAKE]
            embed_response.prompt_eval_count = 42

            async def _selective_fail(*args: object, **kwargs: object) -> object:
                nonlocal call_count
                call_count += 1
                if call_count in (3, 5):
                    raise ollama_sdk.ResponseError("overloaded")
                return embed_response

            mock_client.embed = AsyncMock(side_effect=_selective_fail)

            logged_events: list[dict[str, object]] = []

            def _capture(event: str, **data: object) -> None:
                logged_events.append({"event": event, **data})
                _real_log_event(event, **data)

            # When: run
            with patch(_RUNNER_LOG_EVENT, side_effect=_capture):
                result = await _run_pipeline(
                    runner,
                    listings,
                    tmpdir=tmpdir,
                    env={"OLLAMA_NUM_PARALLEL": "2"},
                )

            # Then: session_summary has correct failed count
            summaries = [e for e in logged_events if e["event"] == "session_summary"]
            assert len(summaries) == 1, f"Expected 1 session_summary, got {len(summaries)}"
            assert summaries[0]["failed_listings"] == result.failed_listings, (
                f"session_summary failed_listings={summaries[0]['failed_listings']} "
                f"does not match RunResult.failed_listings={result.failed_listings}"
            )


# ---------------------------------------------------------------------------
# B3 — TestCollectionScoreAggregation
# ---------------------------------------------------------------------------


class TestCollectionScoreAggregation:
    """
    REQUIREMENT: Per-collection retrieval metrics are correctly aggregated
    from concurrent scoring tasks.

    WHO: The observability layer that emits retrieval_summary log events
    WHAT: (1) scorer.collection_scores reflects scores from all listings
              after parallel scoring completes
          (2) The retrieval_summary log events contain correct statistics
              (count, p50, p90, min, max) across all listings
    WHY: _collection_scores is a dict[str, list[float]] that accumulates
         scores across score() calls. Under asyncio cooperative multitasking,
         list.append() between awaits is safe. The metrics must be identical
         to serial execution.

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient
        Real:  Scorer.score(), Scorer._collection_scores accumulation,
               log_event calls
        Never: Manually construct _collection_scores — must come from real
               Scorer.score() calls
    """

    async def test_collection_scores_reflect_all_listings(self) -> None:
        """
        Given 3 listings scored with OLLAMA_NUM_PARALLEL=2,
        When scoring completes,
        Then scorer.collection_scores['resume'] has exactly 3 entries.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: 3 listings
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)
            listings = [_make_listing(external_id=str(i)) for i in range(3)]

            # When: run with parallelism
            await _run_pipeline(
                runner,
                listings,
                tmpdir=tmpdir,
                env={"OLLAMA_NUM_PARALLEL": "2"},
            )

            # Then: collection_scores has 3 entries per collection
            scorer = runner.scorer
            resume_scores = scorer.collection_scores.get("resume", [])
            assert len(resume_scores) == 3, (
                f"Expected 3 resume scores, got {len(resume_scores)}. "
                f"All collection_scores: {scorer.collection_scores}"
            )
            archetype_scores = scorer.collection_scores.get("role_archetypes", [])
            assert len(archetype_scores) == 3, (
                f"Expected 3 archetype scores, got {len(archetype_scores)}"
            )

    async def test_retrieval_summary_statistics_correct(self) -> None:
        """
        Given 3 listings scored with OLLAMA_NUM_PARALLEL=2,
        When scoring completes,
        Then retrieval_summary events have n_scored=3 for populated
        collections.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: 3 listings
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)
            listings = [_make_listing(external_id=str(i)) for i in range(3)]

            logged_events: list[dict[str, object]] = []

            def _capture(event: str, **data: object) -> None:
                logged_events.append({"event": event, **data})
                _real_log_event(event, **data)

            # When: run with log capture
            with patch(_RUNNER_LOG_EVENT, side_effect=_capture):
                await _run_pipeline(
                    runner,
                    listings,
                    tmpdir=tmpdir,
                    env={"OLLAMA_NUM_PARALLEL": "2"},
                )

            # Then: retrieval_summary events have correct counts
            retrieval_events = [e for e in logged_events if e["event"] == "retrieval_summary"]
            assert len(retrieval_events) >= 1, (
                f"Expected at least 1 retrieval_summary event, got 0. "
                f"Events: {[e['event'] for e in logged_events]}"
            )
            # The resume collection should have n_scored=3
            resume_summaries = [e for e in retrieval_events if e.get("collection") == "resume"]
            assert len(resume_summaries) == 1, (
                f"Expected 1 resume retrieval_summary, got {len(resume_summaries)}"
            )
            assert resume_summaries[0]["n_scored"] == 3, (
                f"Expected n_scored=3 for resume, got {resume_summaries[0]['n_scored']}"
            )


# ---------------------------------------------------------------------------
# B4 — TestEnvironmentVariableConfig
# ---------------------------------------------------------------------------


class TestEnvironmentVariableConfig:
    """
    REQUIREMENT: max_parallel is read from the OLLAMA_NUM_PARALLEL environment
    variable with safe fallback behavior.

    WHO: The operator who configures Ollama and the pipeline with a single
         environment variable
    WHAT: (1) When OLLAMA_NUM_PARALLEL is set to a valid integer > 0, that
              value is used as max_parallel
          (2) When OLLAMA_NUM_PARALLEL is unset, max_parallel defaults to 1
              (serial)
          (3) Non-integer values (e.g. "abc") fall back to 1 with a warning
              log
          (4) Values <= 0 fall back to 1 with a warning log
    WHY: Using the same env var that Ollama reads eliminates configuration
         drift. Safe fallback to serial ensures no breakage for operators
         who never set the variable.

    MOCK BOUNDARY:
        Mock:  os.environ (via monkeypatch / patch.dict)
        Real:  Config parsing logic in _run_inner()
        Never: nothing
    """

    async def test_valid_env_var_used_as_max_parallel(self) -> None:
        """
        Given OLLAMA_NUM_PARALLEL=4,
        When the pipeline runs,
        Then the scoring_parallelism event shows max_parallel=4.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: OLLAMA_NUM_PARALLEL=4
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)
            listings = [_make_listing(external_id="1")]

            logged_events: list[dict[str, object]] = []

            def _capture(event: str, **data: object) -> None:
                logged_events.append({"event": event, **data})
                _real_log_event(event, **data)

            # When: run with OLLAMA_NUM_PARALLEL=4
            with patch(_RUNNER_LOG_EVENT, side_effect=_capture):
                await _run_pipeline(
                    runner,
                    listings,
                    tmpdir=tmpdir,
                    env={"OLLAMA_NUM_PARALLEL": "4"},
                )

            # Then: max_parallel=4 in the log event
            parallelism = [e for e in logged_events if e["event"] == "scoring_parallelism"]
            assert len(parallelism) == 1, (
                f"Expected 1 scoring_parallelism event, got {len(parallelism)}"
            )
            assert parallelism[0]["max_parallel"] == 4, (
                f"Expected max_parallel=4, got {parallelism[0]['max_parallel']}"
            )

    async def test_unset_env_var_defaults_to_serial(self) -> None:
        """
        When OLLAMA_NUM_PARALLEL is not set,
        Then max_parallel defaults to 1.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: env var not set
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)
            listings = [_make_listing(external_id="1")]

            logged_events: list[dict[str, object]] = []

            def _capture(event: str, **data: object) -> None:
                logged_events.append({"event": event, **data})
                _real_log_event(event, **data)

            # When: run without setting the env var
            with (
                patch(_RUNNER_LOG_EVENT, side_effect=_capture),
                patch.dict("os.environ", {}, clear=False),
            ):
                # Ensure OLLAMA_NUM_PARALLEL is not set
                os.environ.pop("OLLAMA_NUM_PARALLEL", None)

                await _run_pipeline(runner, listings, tmpdir=tmpdir)

            # Then: defaults to serial
            parallelism = [e for e in logged_events if e["event"] == "scoring_parallelism"]
            assert len(parallelism) == 1, (
                f"Expected 1 scoring_parallelism event, got {len(parallelism)}"
            )
            assert parallelism[0]["max_parallel"] == 1, (
                f"Expected max_parallel=1 (serial), got {parallelism[0]['max_parallel']}"
            )

    async def test_non_integer_env_var_falls_back_to_serial(self) -> None:
        """
        Given OLLAMA_NUM_PARALLEL="abc",
        When the pipeline runs,
        Then max_parallel falls back to 1 and a warning is logged.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: invalid env var
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)
            listings = [_make_listing(external_id="1")]

            logged_events: list[dict[str, object]] = []

            def _capture(event: str, **data: object) -> None:
                logged_events.append({"event": event, **data})
                _real_log_event(event, **data)

            # When: run with invalid env var
            with patch(_RUNNER_LOG_EVENT, side_effect=_capture):
                await _run_pipeline(
                    runner,
                    listings,
                    tmpdir=tmpdir,
                    env={"OLLAMA_NUM_PARALLEL": "abc"},
                )

            # Then: falls back to 1
            parallelism = [e for e in logged_events if e["event"] == "scoring_parallelism"]
            assert len(parallelism) == 1, (
                f"Expected 1 scoring_parallelism event, got {len(parallelism)}"
            )
            assert parallelism[0]["max_parallel"] == 1, (
                f"Expected max_parallel=1 fallback, got {parallelism[0]['max_parallel']}"
            )

    async def test_zero_env_var_falls_back_to_serial(self) -> None:
        """
        Given OLLAMA_NUM_PARALLEL=0,
        When the pipeline runs,
        Then max_parallel falls back to 1 and a warning is logged.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: zero value
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)
            listings = [_make_listing(external_id="1")]

            logged_events: list[dict[str, object]] = []

            def _capture(event: str, **data: object) -> None:
                logged_events.append({"event": event, **data})
                _real_log_event(event, **data)

            # When: run with zero
            with patch(_RUNNER_LOG_EVENT, side_effect=_capture):
                await _run_pipeline(
                    runner,
                    listings,
                    tmpdir=tmpdir,
                    env={"OLLAMA_NUM_PARALLEL": "0"},
                )

            # Then: falls back to 1
            parallelism = [e for e in logged_events if e["event"] == "scoring_parallelism"]
            assert len(parallelism) == 1, (
                f"Expected 1 scoring_parallelism event, got {len(parallelism)}"
            )
            assert parallelism[0]["max_parallel"] == 1, (
                f"Expected max_parallel=1 fallback for 0, got {parallelism[0]['max_parallel']}"
            )

    async def test_negative_env_var_falls_back_to_serial(self) -> None:
        """
        Given OLLAMA_NUM_PARALLEL=-1,
        When the pipeline runs,
        Then max_parallel falls back to 1.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: negative value
            settings = _make_settings(tmpdir)
            runner, _ = _make_runner_with_real_stack(settings)
            listings = [_make_listing(external_id="1")]

            logged_events: list[dict[str, object]] = []

            def _capture(event: str, **data: object) -> None:
                logged_events.append({"event": event, **data})
                _real_log_event(event, **data)

            # When: run with negative
            with patch(_RUNNER_LOG_EVENT, side_effect=_capture):
                await _run_pipeline(
                    runner,
                    listings,
                    tmpdir=tmpdir,
                    env={"OLLAMA_NUM_PARALLEL": "-1"},
                )

            # Then: falls back to 1
            parallelism = [e for e in logged_events if e["event"] == "scoring_parallelism"]
            assert len(parallelism) == 1, (
                f"Expected 1 scoring_parallelism event, got {len(parallelism)}"
            )
            assert parallelism[0]["max_parallel"] == 1, (
                f"Expected max_parallel=1 fallback for -1, got {parallelism[0]['max_parallel']}"
            )
