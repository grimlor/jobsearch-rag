"""Cross-run deduplication tests — skip previously-decided listings.

Maps to BDD spec: TestCrossRunDedup

Tests verify that the runner excludes already-decided listings from
scoring, reflects the exclusion count in the run summary, and supports
a --force-rescore override flag.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobsearch_rag.adapters.base import JobListing
from jobsearch_rag.config import (
    BoardConfig,
    ChromaConfig,
    OllamaConfig,
    OutputConfig,
    ScoringConfig,
    Settings,
)
from jobsearch_rag.pipeline.runner import PipelineRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBED_FAKE = [0.1, 0.2, 0.3, 0.4, 0.5]


def _make_settings(tmpdir: str) -> Settings:
    return Settings(
        enabled_boards=["testboard"],
        overnight_boards=[],
        boards={
            "testboard": BoardConfig(
                name="testboard",
                searches=["https://testboard.com/search"],
                max_pages=1,
                headless=True,
            ),
        },
        scoring=ScoringConfig(),
        ollama=OllamaConfig(),
        output=OutputConfig(output_dir=str(Path(tmpdir) / "output")),
        chroma=ChromaConfig(persist_dir=tmpdir),
    )


def _make_listing(
    external_id: str = "job-1",
    title: str = "Architect",
    full_text: str = "Full JD text " * 20,
) -> JobListing:
    return JobListing(
        board="testboard",
        external_id=external_id,
        title=title,
        company="TestCo",
        location="Remote",
        url=f"https://testboard.com/jobs/{external_id}",
        full_text=full_text,
    )


def _mock_recorder(decided_ids: set[str] | None = None) -> MagicMock:
    """Create a mock DecisionRecorder that knows about decided_ids."""
    recorder = MagicMock()
    decided = decided_ids or set()
    recorder.get_decision = MagicMock(
        side_effect=lambda jid: {"verdict": "yes"} if jid in decided else None
    )
    return recorder


class TestCrossRunDedup:
    """Re-searching does not re-process listings that already
    have a recorded decision."""

    @pytest.mark.asyncio
    async def test_listing_with_existing_decision_is_excluded_from_scoring(
        self,
    ) -> None:
        """A listing whose job_id is already in the decisions collection
        is skipped entirely — no Ollama compute wasted on re-scoring."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner = PipelineRunner(settings)
            runner._decision_recorder = _mock_recorder({"already-decided"})

            decided_listing = _make_listing(external_id="already-decided")
            new_listing = _make_listing(external_id="brand-new")

            with (
                patch.object(runner, "_search_board", new_callable=AsyncMock) as mock_search,
                patch.object(runner._embedder, "health_check", new_callable=AsyncMock),
                patch.object(runner, "_ensure_indexed", new_callable=AsyncMock),
                patch.object(runner._scorer, "score", new_callable=AsyncMock) as mock_score,
                patch.object(runner._embedder, "embed", new_callable=AsyncMock) as mock_embed,
            ):
                mock_search.return_value = ([decided_listing, new_listing], 0)
                mock_score.return_value = MagicMock(
                    fit_score=0.8, archetype_score=0.7, history_score=0.5,
                    comp_score=0.5, disqualified=False, disqualifier_reason=None,
                )
                mock_embed.return_value = EMBED_FAKE

                await runner.run()

            # Score should only have been called for the new listing
            assert mock_score.call_count == 1

    @pytest.mark.asyncio
    async def test_excluded_listing_does_not_appear_in_export(self) -> None:
        """Excluded listings are not passed to the ranker, so they won't
        appear in results.md, results.csv, or the JD files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner = PipelineRunner(settings)
            runner._decision_recorder = _mock_recorder({"decided-1"})

            decided = _make_listing(external_id="decided-1", title="Old Role")
            new = _make_listing(external_id="new-1", title="New Role")

            with (
                patch.object(runner, "_search_board", new_callable=AsyncMock) as mock_search,
                patch.object(runner._embedder, "health_check", new_callable=AsyncMock),
                patch.object(runner, "_ensure_indexed", new_callable=AsyncMock),
                patch.object(runner._scorer, "score", new_callable=AsyncMock) as mock_score,
                patch.object(runner._embedder, "embed", new_callable=AsyncMock) as mock_embed,
            ):
                mock_search.return_value = ([decided, new], 0)
                mock_score.return_value = MagicMock(
                    fit_score=0.8, archetype_score=0.7, history_score=0.5,
                    comp_score=0.5, disqualified=False, disqualifier_reason=None,
                )
                mock_embed.return_value = EMBED_FAKE

                result = await runner.run()

            # Only the new listing should appear in ranked results
            titles = [r.listing.title for r in result.ranked_listings]
            assert "Old Role" not in titles
            if result.ranked_listings:
                assert "New Role" in titles

    @pytest.mark.asyncio
    async def test_exclusion_count_appears_in_run_summary(self) -> None:
        """The run summary includes how many listings were skipped due
        to existing decisions, so the operator sees the dedup effect."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner = PipelineRunner(settings)
            runner._decision_recorder = _mock_recorder({"decided-1"})

            decided = _make_listing(external_id="decided-1")
            new = _make_listing(external_id="new-1")

            with (
                patch.object(runner, "_search_board", new_callable=AsyncMock) as mock_search,
                patch.object(runner._embedder, "health_check", new_callable=AsyncMock),
                patch.object(runner, "_ensure_indexed", new_callable=AsyncMock),
                patch.object(runner._scorer, "score", new_callable=AsyncMock) as mock_score,
                patch.object(runner._embedder, "embed", new_callable=AsyncMock) as mock_embed,
            ):
                mock_search.return_value = ([decided, new], 0)
                mock_score.return_value = MagicMock(
                    fit_score=0.8, archetype_score=0.7, history_score=0.5,
                    comp_score=0.5, disqualified=False, disqualifier_reason=None,
                )
                mock_embed.return_value = EMBED_FAKE

                result = await runner.run()

            assert result.skipped_decisions >= 1

    @pytest.mark.asyncio
    async def test_listing_with_no_decision_is_scored_normally(self) -> None:
        """Listings that have never been decided upon proceed through
        the full scoring pipeline as before."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner = PipelineRunner(settings)
            runner._decision_recorder = _mock_recorder()  # empty — no decisions

            new_listing = _make_listing(external_id="never-seen")

            with (
                patch.object(runner, "_search_board", new_callable=AsyncMock) as mock_search,
                patch.object(runner._embedder, "health_check", new_callable=AsyncMock),
                patch.object(runner, "_ensure_indexed", new_callable=AsyncMock),
                patch.object(runner._scorer, "score", new_callable=AsyncMock) as mock_score,
                patch.object(runner._embedder, "embed", new_callable=AsyncMock) as mock_embed,
            ):
                mock_search.return_value = ([new_listing], 0)
                mock_score.return_value = MagicMock(
                    fit_score=0.8, archetype_score=0.7, history_score=0.5,
                    comp_score=0.5, disqualified=False, disqualifier_reason=None,
                )
                mock_embed.return_value = EMBED_FAKE

                await runner.run()

            assert mock_score.call_count == 1

    @pytest.mark.asyncio
    async def test_force_rescore_flag_overrides_exclusion(self) -> None:
        """When --force-rescore is passed, even previously-decided listings
        are sent through the scoring pipeline again."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner = PipelineRunner(settings)
            recorder = _mock_recorder({"decided-1"})
            runner._decision_recorder = recorder

            decided = _make_listing(external_id="decided-1")

            with (
                patch.object(runner, "_search_board", new_callable=AsyncMock) as mock_search,
                patch.object(runner._embedder, "health_check", new_callable=AsyncMock),
                patch.object(runner, "_ensure_indexed", new_callable=AsyncMock),
                patch.object(runner._scorer, "score", new_callable=AsyncMock) as mock_score,
                patch.object(runner._embedder, "embed", new_callable=AsyncMock) as mock_embed,
            ):
                mock_search.return_value = ([decided], 0)
                mock_score.return_value = MagicMock(
                    fit_score=0.8, archetype_score=0.7, history_score=0.5,
                    comp_score=0.5, disqualified=False, disqualifier_reason=None,
                )
                mock_embed.return_value = EMBED_FAKE

                await runner.run(force_rescore=True)

            # Score should be called even though there's a decision
            assert mock_score.call_count == 1

    @pytest.mark.asyncio
    async def test_decision_lookup_uses_job_id_not_url(self) -> None:
        """Cross-run dedup uses external_id (job_id) for lookup, not the
        URL — the same listing can appear at different URLs across runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _make_settings(tmpdir)
            runner = PipelineRunner(settings)
            recorder = _mock_recorder()  # empty
            runner._decision_recorder = recorder

            listing = _make_listing(external_id="canonical-id-123")

            with (
                patch.object(runner, "_search_board", new_callable=AsyncMock) as mock_search,
                patch.object(runner._embedder, "health_check", new_callable=AsyncMock),
                patch.object(runner, "_ensure_indexed", new_callable=AsyncMock),
                patch.object(runner._scorer, "score", new_callable=AsyncMock) as mock_score,
                patch.object(runner._embedder, "embed", new_callable=AsyncMock) as mock_embed,
            ):
                mock_search.return_value = ([listing], 0)
                mock_score.return_value = MagicMock(
                    fit_score=0.8, archetype_score=0.7, history_score=0.5,
                    comp_score=0.5, disqualified=False, disqualifier_reason=None,
                )
                mock_embed.return_value = EMBED_FAKE

                await runner.run()

            # Verify get_decision was called with external_id, not URL
            recorder.get_decision.assert_called_with("canonical-id-123")
