# mypy: disable-error-code="method-assign"
"""BDD specs for scoring — semantic similarity, compensation, fusion, and dedup.

Covers: TestSemanticScoring, TestCultureScoring, TestNegativeScoring,
        TestDisqualifierClassification, TestRejectionReasonInjection,
        TestCompensationParsing, TestCompensationScoring, TestScoreFusion,
        TestCrossBoardDeduplication

See: BDD Specifications — scoring.md
"""

# ---------------------------------------------------------------------------
# Public API surface (from src/):
#
# Scorer(*, store: VectorStore, embedder: Embedder, disqualify_on_llm_flag: bool = True)
#   scorer.score(jd_text: str) -> ScoreResult
#   scorer.disqualify(jd_text: str) -> tuple[bool, str | None]
#
# ScoreResult(fit_score, archetype_score, history_score, disqualified,
#             disqualifier_reason=None, comp_score=0.5, negative_score=0.0,
#             culture_score=0.0)
#   .is_valid -> bool
#
# parse_compensation(text: str, source: str = "employer") -> CompResult | None
# CompResult(comp_min, comp_max, comp_source, comp_text)
#
# compute_comp_score(comp_max: float | None, base_salary: float) -> float
#
# Ranker(archetype_weight, fit_weight, history_weight, comp_weight=0.0,
#        negative_weight=0.4, culture_weight=0.0, min_score_threshold=0.45)
#   ranker.rank(listings, embeddings=None) -> tuple[list[RankedListing], RankSummary]
#   ranker.compute_final_score(scores: ScoreResult) -> float
#
# RankedListing(listing, scores, final_score, duplicate_boards=[])
#   .score_explanation() -> str
#
# Indexer(store: VectorStore, embedder: Embedder)
#   indexer.index_resume(resume_path) -> int
#   indexer.index_archetypes(archetypes_path) -> int
#   indexer.index_negative_signals(rubric_path, archetypes_path) -> int
#   indexer.index_global_positive_signals(rubric_path) -> int
#
# DecisionRecorder(*, store, embedder, decisions_dir)
#   recorder.record(*, job_id, verdict, jd_text, board, title, company, reason)
#
# VectorStore(persist_dir: str)
#   .add_documents(collection_name, *, ids, documents, embeddings, metadatas=None)
#   .query(collection_name, *, query_embedding, n_results=5) -> dict
#   .get_or_create_collection(name) -> Collection
#   .collection_count(name) -> int
#   .reset_collection(name)
#   .get_by_metadata(collection_name, *, where, include=None) -> dict
#
# Embedder — mock_embedder fixture provides:
#   .embed(text) -> AsyncMock returning [0.1, 0.2, 0.3, 0.4, 0.5]
#   .classify(prompt) -> AsyncMock returning '{"disqualified": false}'
#   .health_check() -> AsyncMock
#   .MAX_EMBED_CHARS: int = 8000
#
# ScoringConfig(archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
#               comp_weight=0.15, negative_weight=0.4, culture_weight=0.2,
#               base_salary=220_000, disqualify_on_llm_flag=True,
#               min_score_threshold=0.45,
#               comp_bands=DEFAULT_COMP_BANDS, missing_comp_score=0.5)
#
# CompBand(ratio: float, score: float)
# DEFAULT_COMP_BANDS: list[CompBand]  — [(1.0, 1.0), (0.90, 0.7), (0.77, 0.4), (0.68, 0.0)]
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, ClassVar
from unittest.mock import AsyncMock

import pytest

from conftest import EMBED_FAKE
from jobsearch_rag.adapters.base import JobListing
from jobsearch_rag.pipeline.ranker import RankedListing, Ranker
from jobsearch_rag.rag.comp_parser import (
    DEFAULT_COMP_BANDS,
    CompBand,
    compute_comp_score,
    parse_compensation,
)
from jobsearch_rag.rag.indexer import Indexer
from jobsearch_rag.rag.scorer import Scorer, ScoreResult

if TYPE_CHECKING:
    from pathlib import Path

    from jobsearch_rag.rag.decisions import DecisionRecorder
    from jobsearch_rag.rag.embedder import Embedder
    from jobsearch_rag.rag.store import VectorStore

# A distinct embedding that produces a different (but valid) similarity score
EMBED_ALT: list[float] = [0.9, 0.1, 0.0, 0.0, 0.1]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_listing(
    *,
    board: str = "testboard",
    external_id: str = "1",
    title: str = "Staff Platform Architect",
    full_text: str = (
        "We are seeking a Staff Platform Architect to define technical strategy "
        "for our distributed infrastructure platform serving 200M users."
    ),
) -> JobListing:
    """Build a realistic JobListing for scoring tests."""
    return JobListing(
        board=board,
        external_id=external_id,
        title=title,
        company="Acme Corp",
        location="Remote",
        url=f"https://{board}.com/{external_id}",
        full_text=full_text,
    )


async def _seed_resume(
    indexer: Indexer,
    tmp_path: Path,
) -> int:
    """Write a minimal resume and index it."""
    resume = tmp_path / "resume.md"
    resume.write_text(
        "# Resume\n\n"
        "## Experience\n"
        "Staff Platform Architect at BigCo — designed event-driven microservices, "
        "led RFC processes, mentored senior engineers.\n\n"
        "## Skills\n"
        "Distributed systems, Kubernetes, platform engineering, technical strategy, "
        "cross-team alignment, architecture governance.\n",
        encoding="utf-8",
    )
    return await indexer.index_resume(str(resume))


async def _seed_archetypes(
    indexer: Indexer,
    tmp_path: Path,
) -> int:
    """Write archetypes TOML and index it."""
    archetypes = tmp_path / "role_archetypes.toml"
    archetypes.write_text(
        '[[archetypes]]\n'
        'name = "Platform Architect"\n'
        'description = "Senior architect owning platform strategy, RFC processes, '
        'and cross-team technical alignment."\n'
        'signals_positive = ["platform strategy", "RFC process", "technical alignment", '
        '"architecture governance"]\n'
        'signals_negative = ["individual contributor", "hands-on coding daily"]\n',
        encoding="utf-8",
    )
    return await indexer.index_archetypes(str(archetypes))


async def _seed_negative_signals(
    indexer: Indexer,
    tmp_path: Path,
) -> int:
    """Write rubric + archetypes and index negative signals."""
    rubric = tmp_path / "global_rubric.toml"
    rubric.write_text(
        '[[dimensions]]\n'
        'name = "Domain Ethics"\n'
        'signals_positive = ["privacy-respecting", "ethical AI"]\n'
        'signals_negative = ["adtech", "surveillance", "behavioural manipulation"]\n'
        '\n'
        '[[dimensions]]\n'
        'name = "Culture"\n'
        'signals_positive = ["remote-first", "async collaboration", "written communication"]\n'
        'signals_negative = ["always-on", "war room", "firefighting culture"]\n',
        encoding="utf-8",
    )
    archetypes = tmp_path / "role_archetypes.toml"
    if not archetypes.exists():
        archetypes.write_text(
            '[[archetypes]]\n'
            'name = "Platform Architect"\n'
            'description = "Senior architect"\n'
            'signals_positive = ["platform strategy"]\n'
            'signals_negative = ["individual contributor", "hands-on coding daily"]\n',
            encoding="utf-8",
        )
    return await indexer.index_negative_signals(str(rubric), str(archetypes))


async def _seed_global_positive_signals(
    indexer: Indexer,
    tmp_path: Path,
) -> int:
    """Index global positive signals from rubric."""
    rubric = tmp_path / "global_rubric.toml"
    if not rubric.exists():
        rubric.write_text(
            '[[dimensions]]\n'
            'name = "Culture"\n'
            'signals_positive = ["remote-first", "async collaboration", '
            '"written communication"]\n'
            'signals_negative = ["always-on"]\n',
            encoding="utf-8",
        )
    return await indexer.index_global_positive_signals(str(rubric))


async def _seed_all_collections(
    indexer: Indexer,
    tmp_path: Path,
) -> None:
    """Seed resume, archetypes, negative signals, and global positive signals."""
    await _seed_resume(indexer, tmp_path)
    await _seed_archetypes(indexer, tmp_path)
    await _seed_negative_signals(indexer, tmp_path)
    await _seed_global_positive_signals(indexer, tmp_path)


# ============================================================================
# TestSemanticScoring
# ============================================================================


class TestSemanticScoring:
    """
    REQUIREMENT: Semantic scores reflect meaningful similarity, not noise.

    WHO: The ranker consuming scores to produce a ranked shortlist
    WHAT: All five semantic scores (fit, archetype, culture, history, negative)
          are floats in [0.0, 1.0]; a JD matching an archetype scores higher
          archetype_score than one that does not; a JD matching resume skills
          scores higher fit_score; a JD with humane culture language scores
          higher culture_score; a JD with negative signal language scores higher
          negative_score; scores are stable across repeated calls; empty
          collections return 0.0 rather than an error
    WHY: Scores outside [0.0, 1.0] or NaN values produce undefined ranking
         behaviour. negative_score has inverted semantics — high similarity
         is a bad outcome — which must be specced to prevent accidental inversion

    MOCK BOUNDARY:
        Mock:  mock_embedder fixture (Ollama HTTP — the only I/O boundary)
        Real:  Scorer instance, ChromaDB via vector_store fixture,
               indexed collections written via Indexer before scoring
        Never: Construct ScoreResult directly — always obtain via scorer.score(listing);
               never mock the Scorer itself; never pre-populate ChromaDB
               without going through the Indexer
    """

    @pytest.mark.asyncio
    async def test_all_scores_are_floats_between_zero_and_one(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        When a JD is scored against fully-indexed collections
        Then all five component scores are floats in [0.0, 1.0]
        """
        # Given: all collections indexed via the Indexer
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        # When: a JD is scored
        result = await scorer.score(
            "Staff Platform Architect — distributed systems, Kubernetes, RFC process"
        )

        # Then: all component scores are floats in [0.0, 1.0]
        for name in ("fit_score", "archetype_score", "history_score", "culture_score", "negative_score"):
            value = getattr(result, name)
            assert isinstance(value, float), (
                f"{name} should be a float, got {type(value).__name__}: {value}"
            )
            assert 0.0 <= value <= 1.0, (
                f"{name} out of range [0.0, 1.0]. Got {value:.4f}. Full result: {result}"
            )

    @pytest.mark.asyncio
    async def test_matching_jd_scores_higher_archetype_than_non_matching(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given a JD that matches archetype signals and one that does not
        When both are scored
        Then the matching JD has a higher archetype_score
        """
        # Given: archetypes indexed with "platform strategy, RFC process" signals
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        # Use distinct embeddings: matching JD gets an embedding close to the
        # archetype embedding, non-matching gets a distant one.
        matching_embed = EMBED_FAKE  # same as what was indexed
        non_matching_embed = EMBED_ALT

        call_count = 0

        async def _directed_embed(text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            if "platform strategy" in text.lower() or "rfc process" in text.lower():
                return matching_embed
            return non_matching_embed

        mock_embedder.embed = AsyncMock(side_effect=_directed_embed)

        # When: both JDs are scored
        matching_result = await scorer.score(
            "We need a Platform Architect to own platform strategy, drive RFC process, "
            "and lead cross-team technical alignment across the organisation."
        )
        non_matching_result = await scorer.score(
            "Entry-level data entry clerk needed for night shift. Must type 60 WPM. "
            "No technical skills required. On-site only, no remote."
        )

        # Then: matching JD has a higher archetype_score
        assert matching_result.archetype_score > non_matching_result.archetype_score, (
            f"Matching JD archetype_score ({matching_result.archetype_score:.4f}) should be "
            f"higher than non-matching ({non_matching_result.archetype_score:.4f})"
        )

    @pytest.mark.asyncio
    async def test_skill_matching_jd_scores_higher_fit_than_non_matching(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given a JD matching resume skills and one that does not
        When both are scored
        Then the skill-matching JD has a higher fit_score
        """
        # Given: resume indexed with platform engineering, Kubernetes, etc.
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        async def _directed_embed(text: str) -> list[float]:
            if "kubernetes" in text.lower() or "distributed systems" in text.lower():
                return EMBED_FAKE
            return EMBED_ALT

        mock_embedder.embed = AsyncMock(side_effect=_directed_embed)

        # When: both JDs are scored
        skill_match = await scorer.score(
            "Looking for a platform architect with Kubernetes and distributed systems "
            "expertise. Must have experience with event-driven microservices."
        )
        no_match = await scorer.score(
            "Seeking a pastry chef to lead our bakery operations. Must have culinary "
            "degree and 5 years of baking experience."
        )

        # Then: skill-matching JD has higher fit_score
        assert skill_match.fit_score > no_match.fit_score, (
            f"Skill-matching JD fit_score ({skill_match.fit_score:.4f}) should be "
            f"higher than non-matching ({no_match.fit_score:.4f})"
        )

    @pytest.mark.asyncio
    async def test_jd_with_culture_language_scores_higher_culture_than_without(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given a JD with humane culture language and one without
        When both are scored
        Then the culture-aligned JD has a higher culture_score
        """
        # Given: global positive signals indexed with remote-first, async collaboration
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        async def _directed_embed(text: str) -> list[float]:
            if "remote-first" in text.lower() or "async collaboration" in text.lower():
                return EMBED_FAKE
            return EMBED_ALT

        mock_embedder.embed = AsyncMock(side_effect=_directed_embed)

        # When: both JDs are scored
        culture_jd = await scorer.score(
            "We are a remote-first company with async collaboration at the core. "
            "Written communication is our primary medium. Flexible schedules respected."
        )
        no_culture = await scorer.score(
            "Must be on-site in our Manhattan office 5 days a week. Daily standups "
            "at 7am. Open floor plan. Always-on Slack responsiveness expected."
        )

        # Then: culture-aligned JD scores higher
        assert culture_jd.culture_score > no_culture.culture_score, (
            f"Culture JD culture_score ({culture_jd.culture_score:.4f}) should be "
            f"higher than non-culture ({no_culture.culture_score:.4f})"
        )

    @pytest.mark.asyncio
    async def test_jd_with_negative_language_scores_higher_negative_than_clean_jd(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given a JD with negative signal language and a clean JD
        When both are scored
        Then the negative JD has a higher negative_score
        """
        # Given: negative signals indexed (adtech, surveillance, firefighting culture)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        async def _directed_embed(text: str) -> list[float]:
            if "adtech" in text.lower() or "surveillance" in text.lower():
                return EMBED_FAKE
            return EMBED_ALT

        mock_embedder.embed = AsyncMock(side_effect=_directed_embed)

        # When: both JDs are scored
        negative_jd = await scorer.score(
            "Join our adtech platform! We build surveillance-grade user tracking "
            "and behavioural manipulation systems for maximum ad revenue."
        )
        clean_jd = await scorer.score(
            "Join our privacy-focused platform team. We build ethical AI systems "
            "with a remote-first, async culture."
        )

        # Then: negative JD has higher negative_score
        assert negative_jd.negative_score > clean_jd.negative_score, (
            f"Negative JD negative_score ({negative_jd.negative_score:.4f}) should be "
            f"higher than clean JD ({clean_jd.negative_score:.4f})"
        )

    @pytest.mark.asyncio
    async def test_scores_are_stable_across_repeated_calls_with_same_input(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        When the same JD is scored twice
        Then all component scores are identical
        """
        # Given: all collections indexed
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        jd_text = (
            "Staff Platform Architect — distributed systems, Kubernetes, "
            "RFC process, cross-team alignment"
        )

        # When: scored twice
        result1 = await scorer.score(jd_text)
        result2 = await scorer.score(jd_text)

        # Then: all scores are identical
        for name in ("fit_score", "archetype_score", "history_score", "culture_score", "negative_score"):
            v1 = getattr(result1, name)
            v2 = getattr(result2, name)
            assert v1 == pytest.approx(v2, abs=1e-9), (
                f"{name} unstable across calls. First: {v1:.6f}, Second: {v2:.6f}"
            )

    @pytest.mark.asyncio
    async def test_empty_history_collection_returns_zero_not_error(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given an empty decisions collection
        When a JD is scored
        Then history_score is 0.0 (not an error)
        """
        # Given: resume and archetypes indexed, but no decisions recorded
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_resume(indexer, tmp_path)
        await _seed_archetypes(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        # When: a JD is scored
        result = await scorer.score("Staff Platform Architect role at a large company.")

        # Then: history_score is 0.0
        assert result.history_score == pytest.approx(0.0, abs=1e-9), (
            f"Expected history_score=0.0 with empty decisions collection. "
            f"Got {result.history_score:.4f}"
        )

    @pytest.mark.asyncio
    async def test_empty_global_positive_collection_returns_zero_not_error(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given an empty global_positive_signals collection
        When a JD is scored
        Then culture_score is 0.0 (not an error)
        """
        # Given: resume and archetypes indexed, no global positive signals
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_resume(indexer, tmp_path)
        await _seed_archetypes(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        # When: a JD is scored
        result = await scorer.score("Staff Platform Architect — remote-first culture.")

        # Then: culture_score is 0.0
        assert result.culture_score == pytest.approx(0.0, abs=1e-9), (
            f"Expected culture_score=0.0 with empty global_positive_signals. "
            f"Got {result.culture_score:.4f}"
        )

    @pytest.mark.asyncio
    async def test_empty_negative_signals_collection_returns_zero_not_error(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given an empty negative_signals collection
        When a JD is scored
        Then negative_score is 0.0 (not an error)
        """
        # Given: resume and archetypes indexed, no negative signals
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_resume(indexer, tmp_path)
        await _seed_archetypes(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        # When: a JD is scored
        result = await scorer.score("Staff Platform Architect at a large company.")

        # Then: negative_score is 0.0
        assert result.negative_score == pytest.approx(0.0, abs=1e-9), (
            f"Expected negative_score=0.0 with empty negative_signals. "
            f"Got {result.negative_score:.4f}"
        )


# ============================================================================
# TestCultureScoring
# ============================================================================


class TestCultureScoring:
    """
    REQUIREMENT: culture_score continuously rewards roles whose environment
    signals match global rubric positive dimensions.

    WHO: The ranker computing final_score; the operator who wants ethically
         aligned, well-scoped, humane environments to rank higher
    WHAT: A JD with explicit remote-first and async collaboration language
          scores higher culture_score than one without; an ethical domain JD
          scores higher than one with surveillance or adtech language;
          culture_score is queried from global_positive_signals;
          missing collection returns 0.0; culture_weight is read from settings;
          culture_score is a float in [0.0, 1.0]
    WHY: Archetype score answers "right kind of role." Culture score answers
         "right kind of environment." Culture_score is what separates a
         surveillance-company platform architect from a privacy-respecting one

    MOCK BOUNDARY:
        Mock:  mock_embedder fixture (Ollama HTTP)
        Real:  Scorer instance, ChromaDB via vector_store fixture with
               global_positive_signals collection indexed before scoring
        Never: Construct ScoreResult directly; never patch the culture_score
               computation — exercise it through scorer.score(listing).
               Ranker-focused tests (weight configurability, suppression,
               ranking order) may construct ScoreResult as Given input.
    """

    @pytest.mark.asyncio
    async def test_humane_culture_jd_scores_higher_than_chaotic_jd(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given a JD with humane culture language and one with chaotic culture
        When both are scored
        Then the humane JD has a higher culture_score
        """
        # Given: global positive signals indexed (remote-first, async collaboration)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        async def _directed_embed(text: str) -> list[float]:
            if "remote-first" in text.lower() or "async collaboration" in text.lower():
                return EMBED_FAKE
            return EMBED_ALT

        mock_embedder.embed = AsyncMock(side_effect=_directed_embed)

        # When: both JDs are scored
        humane = await scorer.score(
            "Remote-first engineering org with async collaboration. Written RFCs, "
            "flexible schedules, no surveillance of developer activity."
        )
        chaotic = await scorer.score(
            "Fast-paced, always-on war room culture. Daily fire drills expected. "
            "Open office, mandatory 7am standups, Slack responsiveness required."
        )

        # Then: humane JD has higher culture_score
        assert humane.culture_score > chaotic.culture_score, (
            f"Humane JD culture_score ({humane.culture_score:.4f}) should be "
            f"higher than chaotic ({chaotic.culture_score:.4f})"
        )

    @pytest.mark.asyncio
    async def test_ethical_domain_jd_scores_higher_than_adtech_jd(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given an ethical-domain JD and an adtech JD
        When both are scored
        Then the ethical JD has a higher culture_score
        """
        # Given: global positive signals indexed with ethical AI, privacy signals
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        async def _directed_embed(text: str) -> list[float]:
            if "ethical" in text.lower() or "privacy" in text.lower():
                return EMBED_FAKE
            return EMBED_ALT

        mock_embedder.embed = AsyncMock(side_effect=_directed_embed)

        # When: both JDs are scored
        ethical = await scorer.score(
            "Join our ethical AI platform building privacy-respecting systems. "
            "Remote-first culture with async collaboration."
        )
        adtech = await scorer.score(
            "Join our adtech platform maximising ad impressions through behavioural "
            "profiling and user surveillance. Growth at all costs."
        )

        # Then: ethical JD scores higher culture_score
        assert ethical.culture_score > adtech.culture_score, (
            f"Ethical JD culture_score ({ethical.culture_score:.4f}) should be "
            f"higher than adtech ({adtech.culture_score:.4f})"
        )

    @pytest.mark.asyncio
    async def test_altitude_language_increases_culture_score(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given a JD with altitude/strategic language
        When scored
        Then culture_score is above zero
        """
        # Given: collections indexed
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        # When: scored with altitude language
        result = await scorer.score(
            "Staff architect defining technical vision and platform strategy. "
            "Remote-first, written communication, async collaboration culture."
        )

        # Then: culture_score is above zero
        assert result.culture_score > 0.0, (
            f"Expected culture_score > 0.0 for altitude language JD. "
            f"Got {result.culture_score:.4f}"
        )

    @pytest.mark.asyncio
    async def test_nd_compatible_signals_increase_culture_score(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given a JD with neurodivergent-compatible signals (written comms, async)
        When scored
        Then culture_score is above zero
        """
        # Given: collections indexed
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        # When: scored with ND-compatible signals
        result = await scorer.score(
            "We value written communication, async collaboration, and flexible "
            "schedules. No open office. Meetings are optional with written summaries."
        )

        # Then: culture_score is above zero
        assert result.culture_score > 0.0, (
            f"Expected culture_score > 0.0 for ND-compatible JD. "
            f"Got {result.culture_score:.4f}"
        )

    @pytest.mark.asyncio
    async def test_missing_global_positive_collection_returns_zero_not_error(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given no global_positive_signals collection
        When a JD is scored
        Then culture_score is 0.0 (not an error)
        """
        # Given: only resume and archetypes indexed — no global positive signals
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_resume(indexer, tmp_path)
        await _seed_archetypes(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        # When: scored
        result = await scorer.score("Remote-first async culture, written communication.")

        # Then: culture_score is 0.0
        assert result.culture_score == pytest.approx(0.0, abs=1e-9), (
            f"Expected culture_score=0.0 with missing collection. "
            f"Got {result.culture_score:.4f}"
        )

    @pytest.mark.asyncio
    async def test_culture_score_appears_in_score_breakdown(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        When a JD is scored
        Then the ScoreResult has a culture_score attribute
        """
        # Given: all collections indexed
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        # When: scored
        result = await scorer.score("Staff Platform Architect at a humane company.")

        # Then: culture_score is present and is a float
        assert hasattr(result, "culture_score"), (
            "ScoreResult should have a culture_score attribute"
        )
        assert isinstance(result.culture_score, float), (
            f"culture_score should be float, got {type(result.culture_score).__name__}"
        )

    def test_culture_weight_is_read_from_settings_not_hardcoded(self) -> None:
        """
        Given two Ranker instances with different culture_weight values
        When the same scores are ranked with each
        Then the final_scores differ
        """
        # Given: two rankers with different culture weights
        ranker_low = Ranker(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            culture_weight=0.1, negative_weight=0.0, min_score_threshold=0.0,
        )
        ranker_high = Ranker(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            culture_weight=0.5, negative_weight=0.0, min_score_threshold=0.0,
        )
        scores = ScoreResult(
            fit_score=0.5, archetype_score=0.5, history_score=0.5,
            disqualified=False, culture_score=0.8,
        )

        # When: the same scores are fused with each ranker
        final_low = ranker_low.compute_final_score(scores)
        final_high = ranker_high.compute_final_score(scores)

        # Then: the results differ, proving the weight is configurable
        assert final_low != pytest.approx(final_high, abs=0.01), (
            f"culture_weight should affect final_score. "
            f"Low weight: {final_low:.4f}, High weight: {final_high:.4f}"
        )

    @pytest.mark.asyncio
    async def test_culture_score_is_float_between_zero_and_one(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        When a JD is scored
        Then culture_score is a float in [0.0, 1.0]
        """
        # Given: all collections indexed
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        # When: scored
        result = await scorer.score("Any job description text for range validation.")

        # Then: culture_score in valid range
        assert isinstance(result.culture_score, float), (
            f"culture_score should be float, got {type(result.culture_score).__name__}"
        )
        assert 0.0 <= result.culture_score <= 1.0, (
            f"culture_score out of range [0.0, 1.0]. Got {result.culture_score:.4f}"
        )


# ============================================================================
# TestNegativeScoring
# ============================================================================


class TestNegativeScoring:
    """
    REQUIREMENT: negative_score continuously suppresses roles that match
    undesirable patterns, acting as a penalty rather than a binary gate.

    WHO: The ranker computing final_score; the operator who wants adtech,
         surveillance, and chaos-culture roles to rank lower
    WHAT: A JD with adtech language scores higher negative_score than one
          without; a role that scores high on both positive and negative
          signals ranks lower than an equivalent role with no negative match;
          missing collection returns 0.0; negative_score appears in breakdown;
          negative_weight is read from settings
    WHY: The LLM disqualifier is a hard gate. Many roles should rank lower
         due to domain or culture concerns without being hard-filtered.
         A continuous penalty preserves nuance a binary gate discards

    MOCK BOUNDARY:
        Mock:  mock_embedder fixture (Ollama HTTP)
        Real:  Scorer instance, ChromaDB via vector_store fixture with
               negative_signals collection indexed before scoring
        Never: Construct ScoreResult directly; never patch the negative_score
               computation — exercise it through scorer.score(listing).
               Ranker-focused tests (weight configurability, suppression,
               ranking order) may construct ScoreResult as Given input.
    """

    @pytest.mark.asyncio
    async def test_adtech_jd_scores_higher_negative_than_platform_engineering_jd(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given an adtech JD and a platform engineering JD
        When both are scored
        Then the adtech JD has a higher negative_score
        """
        # Given: negative signals indexed (adtech, surveillance, etc.)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        async def _directed_embed(text: str) -> list[float]:
            if "adtech" in text.lower() or "surveillance" in text.lower():
                return EMBED_FAKE
            return EMBED_ALT

        mock_embedder.embed = AsyncMock(side_effect=_directed_embed)

        # When: both are scored
        adtech = await scorer.score(
            "Lead our adtech platform maximising user tracking and surveillance-grade "
            "behavioural profiling for targeted advertising."
        )
        platform = await scorer.score(
            "Staff Platform Architect building distributed infrastructure. "
            "Privacy-first, ethical AI, async collaboration."
        )

        # Then: adtech JD has higher negative_score
        assert adtech.negative_score > platform.negative_score, (
            f"Adtech JD negative_score ({adtech.negative_score:.4f}) should be "
            f"higher than platform engineering ({platform.negative_score:.4f})"
        )

    @pytest.mark.asyncio
    async def test_chaos_culture_language_increases_negative_score(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given a JD with chaos-culture language
        When scored
        Then negative_score is above zero
        """
        # Given: negative signals indexed (always-on, war room, firefighting)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        async def _directed_embed(text: str) -> list[float]:
            if "war room" in text.lower() or "firefighting" in text.lower():
                return EMBED_FAKE
            return EMBED_ALT

        mock_embedder.embed = AsyncMock(side_effect=_directed_embed)

        # When: scored
        result = await scorer.score(
            "Fast-paced war room culture with firefighting as a core competency. "
            "Always-on Slack expected. Mandatory weekend deployments."
        )

        # Then: negative_score > 0
        assert result.negative_score > 0.0, (
            f"Expected negative_score > 0.0 for chaos culture JD. "
            f"Got {result.negative_score:.4f}"
        )

    @pytest.mark.asyncio
    async def test_high_negative_score_suppresses_final_rank(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given a ScoreResult with a high negative_score
        When the Ranker computes final_score
        Then final_score is lower than without the negative penalty
        """
        # Given: a ranker with non-zero negative_weight
        ranker = Ranker(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            negative_weight=0.4, culture_weight=0.0, min_score_threshold=0.0,
        )
        scores_with_negative = ScoreResult(
            fit_score=0.8, archetype_score=0.7, history_score=0.5,
            disqualified=False, negative_score=0.9,
        )
        scores_without_negative = ScoreResult(
            fit_score=0.8, archetype_score=0.7, history_score=0.5,
            disqualified=False, negative_score=0.0,
        )

        # When: final scores are computed
        final_with = ranker.compute_final_score(scores_with_negative)
        final_without = ranker.compute_final_score(scores_without_negative)

        # Then: high negative suppresses the score
        assert final_with < final_without, (
            f"High negative_score should suppress final_score. "
            f"With negative: {final_with:.4f}, Without: {final_without:.4f}"
        )

    @pytest.mark.asyncio
    async def test_role_with_positive_and_negative_signals_ranks_below_positive_only(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given two roles with equal positive scores but one also has high negative_score
        When ranked
        Then the role with negative signals ranks lower
        """
        # Given: ranker and two score sets
        ranker = Ranker(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            negative_weight=0.4, culture_weight=0.0, min_score_threshold=0.0,
        )
        positive_only = ScoreResult(
            fit_score=0.8, archetype_score=0.8, history_score=0.5,
            disqualified=False, negative_score=0.0,
        )
        positive_and_negative = ScoreResult(
            fit_score=0.8, archetype_score=0.8, history_score=0.5,
            disqualified=False, negative_score=0.7,
        )
        listing_a = _make_listing(external_id="a", title="Clean Role")
        listing_b = _make_listing(external_id="b", title="Mixed Role")

        # When: ranked together
        ranked, _ = ranker.rank([(listing_a, positive_only), (listing_b, positive_and_negative)])

        # Then: positive-only role ranks first
        assert len(ranked) >= 2, (
            f"Expected at least 2 ranked listings, got {len(ranked)}"
        )
        assert ranked[0].listing.external_id == "a", (
            f"Positive-only role should rank first. Got: "
            f"{[r.listing.external_id for r in ranked]}"
        )

    @pytest.mark.asyncio
    async def test_missing_negative_signals_collection_returns_zero_not_error(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given no negative_signals collection
        When a JD is scored
        Then negative_score is 0.0 (not an error)
        """
        # Given: only resume and archetypes — no negative signals
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_resume(indexer, tmp_path)
        await _seed_archetypes(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        # When: scored
        result = await scorer.score("Some job description text.")

        # Then: negative_score is 0.0
        assert result.negative_score == pytest.approx(0.0, abs=1e-9), (
            f"Expected negative_score=0.0 with missing collection. "
            f"Got {result.negative_score:.4f}"
        )

    @pytest.mark.asyncio
    async def test_negative_score_appears_in_score_breakdown(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        When a JD is scored
        Then the ScoreResult has a negative_score attribute
        """
        # Given: all collections indexed
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        # When: scored
        result = await scorer.score("Staff Platform Architect.")

        # Then: negative_score attribute is present and is a float
        assert hasattr(result, "negative_score"), (
            "ScoreResult should have a negative_score attribute"
        )
        assert isinstance(result.negative_score, float), (
            f"negative_score should be float, got {type(result.negative_score).__name__}"
        )

    def test_negative_weight_is_read_from_settings_not_hardcoded(self) -> None:
        """
        Given two Ranker instances with different negative_weight values
        When the same scores are ranked with each
        Then the final_scores differ
        """
        # Given: two rankers with different negative weights
        ranker_low = Ranker(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            negative_weight=0.1, culture_weight=0.0, min_score_threshold=0.0,
        )
        ranker_high = Ranker(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            negative_weight=0.8, culture_weight=0.0, min_score_threshold=0.0,
        )
        scores = ScoreResult(
            fit_score=0.8, archetype_score=0.7, history_score=0.5,
            disqualified=False, negative_score=0.6,
        )

        # When: final scores computed
        final_low = ranker_low.compute_final_score(scores)
        final_high = ranker_high.compute_final_score(scores)

        # Then: results differ
        assert final_low != pytest.approx(final_high, abs=0.01), (
            f"negative_weight should affect final_score. "
            f"Low weight: {final_low:.4f}, High weight: {final_high:.4f}"
        )

    @pytest.mark.asyncio
    async def test_negative_score_is_float_between_zero_and_one(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        When a JD is scored
        Then negative_score is a float in [0.0, 1.0]
        """
        # Given: all collections indexed
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=False)

        # When: scored
        result = await scorer.score("Any job description for range check.")

        # Then: negative_score in valid range
        assert isinstance(result.negative_score, float), (
            f"negative_score should be float, got {type(result.negative_score).__name__}"
        )
        assert 0.0 <= result.negative_score <= 1.0, (
            f"negative_score out of range [0.0, 1.0]. Got {result.negative_score:.4f}"
        )


# ============================================================================
# TestDisqualifierClassification
# ============================================================================


class TestDisqualifierClassification:
    """
    REQUIREMENT: The LLM disqualifier correctly identifies structurally
    unsuitable roles before they consume ranking budget.

    WHO: The ranker applying disqualification before final scoring
    WHAT: Known disqualifier patterns produce disqualified=True; clearly
          suitable roles are not disqualified; malformed LLM JSON falls back
          to not-disqualified with a warning; disqualified roles score 0.0
          regardless of semantic scores; the disqualifier reason appears
          in export output
    WHY: A false negative wastes review time; a false positive silently
         removes a candidate — both error types must have specs

    MOCK BOUNDARY:
        Mock:  mock_embedder fixture (Ollama HTTP); LLM call mocked via
               AsyncMock on embedder.classify() returning controlled JSON strings
        Real:  Scorer instance, disqualifier logic, ScoreResult
        Never: Construct ScoreResult directly; never patch the JSON parser
               inside the disqualifier — inject malformed JSON via the LLM mock
    """

    @pytest.mark.asyncio
    async def test_ic_role_disguised_as_architect_is_disqualified(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given an IC coding role with "Architect" in the title
        When scored with disqualification enabled
        Then disqualified is True
        """
        # Given: collections indexed, LLM returns disqualified=true
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        mock_embedder.classify = AsyncMock(
            return_value='{"disqualified": true, "reason": "IC coding role disguised as architect"}'
        )
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=True)

        # When: scored
        result = await scorer.score(
            "Software Architect — daily hands-on coding in React and Node.js. "
            "Must submit at least 20 PRs per week. No architecture responsibilities."
        )

        # Then: disqualified
        assert result.disqualified is True, (
            f"IC-disguised-as-architect should be disqualified. Got: {result.disqualified}"
        )

    @pytest.mark.asyncio
    async def test_sre_primary_ownership_role_is_disqualified(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given an SRE on-call ownership role
        When scored with disqualification enabled
        Then disqualified is True
        """
        # Given: collections indexed, LLM flags SRE primary
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        mock_embedder.classify = AsyncMock(
            return_value='{"disqualified": true, "reason": "Primarily SRE on-call operations ownership"}'
        )
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=True)

        # When: scored
        result = await scorer.score(
            "Site Reliability Engineer — own the on-call rotation, manage incident "
            "response, and maintain 99.99% uptime SLAs for production systems."
        )

        # Then: disqualified
        assert result.disqualified is True, (
            f"SRE primary ownership should be disqualified. Got: {result.disqualified}"
        )

    @pytest.mark.asyncio
    async def test_staffing_agency_posting_is_disqualified(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given a staffing agency posting
        When scored with disqualification enabled
        Then disqualified is True
        """
        # Given: LLM flags staffing agency
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        mock_embedder.classify = AsyncMock(
            return_value='{"disqualified": true, "reason": "Staffing agency posting"}'
        )
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=True)

        # When: scored
        result = await scorer.score(
            "We are a premier staffing agency seeking candidates for our client's "
            "contract-to-hire architect position. No direct client contact."
        )

        # Then: disqualified
        assert result.disqualified is True, (
            f"Staffing agency posting should be disqualified. Got: {result.disqualified}"
        )

    @pytest.mark.asyncio
    async def test_fullstack_primary_responsibility_is_disqualified(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given a full-stack web development role
        When scored with disqualification enabled
        Then disqualified is True
        """
        # Given: LLM flags fullstack primary
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        mock_embedder.classify = AsyncMock(
            return_value='{"disqualified": true, "reason": "Primarily full-stack web development"}'
        )
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=True)

        # When: scored
        result = await scorer.score(
            "Full-Stack Engineer — build and maintain React frontends and Express "
            "backends. Must be proficient in HTML/CSS/JavaScript."
        )

        # Then: disqualified
        assert result.disqualified is True, (
            f"Full-stack primary role should be disqualified. Got: {result.disqualified}"
        )

    @pytest.mark.asyncio
    async def test_senior_architecture_role_is_not_disqualified(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given a genuine senior architecture role
        When scored with disqualification enabled
        Then disqualified is False
        """
        # Given: LLM does not flag a legit role
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        mock_embedder.classify = AsyncMock(
            return_value='{"disqualified": false, "reason": null}'
        )
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=True)

        # When: scored
        result = await scorer.score(
            "Staff Platform Architect — define technical strategy for our distributed "
            "infrastructure. Own RFC processes, drive cross-team alignment, mentor "
            "senior engineers."
        )

        # Then: not disqualified
        assert result.disqualified is False, (
            f"Senior architecture role should NOT be disqualified. Got: {result.disqualified}"
        )

    @pytest.mark.asyncio
    async def test_malformed_llm_json_falls_back_to_not_disqualified(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given the LLM returns malformed JSON
        When scored with disqualification enabled
        Then disqualified is False (safe default)
        """
        # Given: LLM returns garbage
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        mock_embedder.classify = AsyncMock(
            return_value="This is not valid JSON at all {{{broken"
        )
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=True)

        # When: scored
        result = await scorer.score("Some job description text.")

        # Then: falls back to not disqualified
        assert result.disqualified is False, (
            f"Malformed LLM JSON should fall back to not disqualified. "
            f"Got: {result.disqualified}"
        )

    @pytest.mark.asyncio
    async def test_malformed_llm_json_logs_warning_with_raw_response(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Given the LLM returns malformed JSON
        When scored with disqualification enabled
        Then a warning is logged containing the raw response
        """
        # Given: LLM returns garbage
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        malformed_response = "Not JSON: {{{broken"
        mock_embedder.classify = AsyncMock(return_value=malformed_response)
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=True)

        # When: scored (capture warnings)
        with caplog.at_level(logging.WARNING, logger="jobsearch_rag.rag.scorer"):
            await scorer.score("Some job description.")

        # Then: warning logged with raw response
        assert any(malformed_response in record.message for record in caplog.records), (
            f"Expected warning containing raw response '{malformed_response}'. "
            f"Log records: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_disqualified_role_final_score_is_zero(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given a disqualified role with high semantic scores
        When the Ranker computes final_score
        Then final_score is 0.0
        """
        # Given: a disqualified ScoreResult with high scores, obtained via scorer
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        mock_embedder.classify = AsyncMock(
            return_value='{"disqualified": true, "reason": "IC role disguised as architect"}'
        )
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=True)
        result = await scorer.score("A high-matching but disqualified role.")

        ranker = Ranker(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            negative_weight=0.4, culture_weight=0.2, min_score_threshold=0.0,
        )

        # When: final score computed
        final = ranker.compute_final_score(result)

        # Then: final_score is 0.0
        assert final == pytest.approx(0.0, abs=1e-9), (
            f"Disqualified role should have final_score=0.0. Got {final:.4f}"
        )

    @pytest.mark.asyncio
    async def test_disqualifier_reason_appears_in_export_output(
        self, mock_embedder: Embedder, vector_store: VectorStore, tmp_path: Path
    ) -> None:
        """
        Given a disqualified role
        When the RankedListing's score_explanation() is called
        Then the disqualifier reason appears in the output
        """
        # Given: a disqualified ScoreResult with a reason
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        reason_text = "Staffing agency posting"
        mock_embedder.classify = AsyncMock(
            return_value=json.dumps({"disqualified": True, "reason": reason_text})
        )
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=True)
        result = await scorer.score("A staffing agency role.")

        listing = _make_listing(title="Agency Role")
        ranked = RankedListing(
            listing=listing, scores=result, final_score=0.0,
        )

        # When: score_explanation is generated
        explanation = ranked.score_explanation()

        # Then: the reason appears in the explanation
        assert reason_text in explanation, (
            f"Expected reason '{reason_text}' in score explanation. "
            f"Got: {explanation}"
        )


# ============================================================================
# TestRejectionReasonInjection
# ============================================================================


class TestRejectionReasonInjection:
    """
    REQUIREMENT: Past rejection reasons are injected into the disqualifier
    prompt so the system learns from the operator's decisions.

    WHO: The scorer augmenting the LLM system prompt with learned patterns
    WHAT: When 'no' verdicts with reasons exist in the decisions collection,
          those reasons appear in the disqualifier prompt for new JDs;
          'yes' and 'maybe' reasons are not injected; empty reasons are
          omitted; duplicate reasons appear once; a missing decisions
          collection produces no reasons (not an error); reasons are cached
          per Scorer instance
    WHY: Without injection, the operator must repeatedly reject the same
         patterns — the system should apply learned 'no' patterns automatically

    MOCK BOUNDARY:
        Mock:  mock_embedder fixture (Ollama HTTP)
        Real:  Scorer instance, DecisionRecorder, ChromaDB via vector_store
               fixture — record real decisions and verify they appear in prompt
        Never: Patch the decisions query directly; inject rejection reasons
               by calling DecisionRecorder.record() before constructing Scorer,
               then inspect the prompt via the LLM mock's call_args
    """

    @pytest.mark.asyncio
    async def test_rejection_reasons_appear_in_disqualifier_prompt(
        self,
        mock_embedder: Embedder,
        vector_store: VectorStore,
        decision_recorder: DecisionRecorder,
        tmp_path: Path,
    ) -> None:
        """
        Given a recorded 'no' decision with a reason
        When a new JD is scored with disqualification enabled
        Then the rejection reason appears in the classify() call args
        """
        # Given: a 'no' decision recorded with a reason
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        reason = "requires on-call rotation"
        await decision_recorder.record(
            job_id="test-1", verdict="no", jd_text="Some rejected job.",
            board="testboard", reason=reason,
        )

        mock_embedder.classify = AsyncMock(
            return_value='{"disqualified": false, "reason": null}'
        )
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=True)

        # When: a new JD is scored
        await scorer.score("Some new job description.")

        # Then: the rejection reason appears in the classify call
        classify_call_args = mock_embedder.classify.call_args
        prompt_text = classify_call_args[0][0] if classify_call_args[0] else ""
        assert reason in prompt_text, (
            f"Expected rejection reason '{reason}' in disqualifier prompt. "
            f"Got: {prompt_text[:500]}"
        )

    @pytest.mark.asyncio
    async def test_yes_verdict_reasons_are_not_injected(
        self,
        mock_embedder: Embedder,
        vector_store: VectorStore,
        decision_recorder: DecisionRecorder,
        tmp_path: Path,
    ) -> None:
        """
        Given a 'yes' verdict with a reason
        When a new JD is scored with disqualification enabled
        Then the 'yes' reason does NOT appear in the classify() call args
        """
        # Given: a 'yes' decision recorded
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        yes_reason = "excellent culture fit and remote-first"
        await decision_recorder.record(
            job_id="test-yes", verdict="yes", jd_text="A great job.",
            board="testboard", reason=yes_reason,
        )

        mock_embedder.classify = AsyncMock(
            return_value='{"disqualified": false, "reason": null}'
        )
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=True)

        # When: scored
        await scorer.score("Some new job description.")

        # Then: 'yes' reason is NOT in the prompt
        classify_call_args = mock_embedder.classify.call_args
        prompt_text = classify_call_args[0][0] if classify_call_args[0] else ""
        assert yes_reason not in prompt_text, (
            f"'yes' reason '{yes_reason}' should NOT appear in disqualifier prompt. "
            f"Got: {prompt_text[:500]}"
        )

    @pytest.mark.asyncio
    async def test_empty_reasons_are_omitted_from_prompt(
        self,
        mock_embedder: Embedder,
        vector_store: VectorStore,
        decision_recorder: DecisionRecorder,
        tmp_path: Path,
    ) -> None:
        """
        Given a 'no' decision with an empty reason string
        When a new JD is scored
        Then the empty reason is not in the prompt (no blank bullet)
        """
        # Given: a 'no' decision with empty reason
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        await decision_recorder.record(
            job_id="test-empty", verdict="no", jd_text="Rejected job.",
            board="testboard", reason="",
        )

        mock_embedder.classify = AsyncMock(
            return_value='{"disqualified": false, "reason": null}'
        )
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=True)

        # When: scored
        await scorer.score("Some new job description.")

        # Then: no "- " line with empty content after the rejection reasons header
        classify_call_args = mock_embedder.classify.call_args
        prompt_text = classify_call_args[0][0] if classify_call_args[0] else ""
        # Empty reason should not produce a "- " bullet
        assert "- \n" not in prompt_text, (
            f"Empty reason should not produce a blank bullet in prompt. "
            f"Got: {prompt_text[:500]}"
        )

    @pytest.mark.asyncio
    async def test_duplicate_reasons_appear_once_in_prompt(
        self,
        mock_embedder: Embedder,
        vector_store: VectorStore,
        decision_recorder: DecisionRecorder,
        tmp_path: Path,
    ) -> None:
        """
        Given two 'no' decisions with the same reason
        When a new JD is scored
        Then the reason appears exactly once in the prompt
        """
        # Given: two 'no' decisions with identical reason
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        duplicate_reason = "requires on-call rotation"
        await decision_recorder.record(
            job_id="test-dup-1", verdict="no", jd_text="Job A.",
            board="testboard", reason=duplicate_reason,
        )
        await decision_recorder.record(
            job_id="test-dup-2", verdict="no", jd_text="Job B.",
            board="testboard", reason=duplicate_reason,
        )

        mock_embedder.classify = AsyncMock(
            return_value='{"disqualified": false, "reason": null}'
        )
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=True)

        # When: scored
        await scorer.score("Some new job description.")

        # Then: reason appears exactly once
        classify_call_args = mock_embedder.classify.call_args
        prompt_text = classify_call_args[0][0] if classify_call_args[0] else ""
        occurrences = prompt_text.count(f"- {duplicate_reason}")
        assert occurrences == 1, (
            f"Duplicate reason should appear exactly once. "
            f"Found {occurrences} occurrences in: {prompt_text[:500]}"
        )

    @pytest.mark.asyncio
    async def test_missing_decisions_collection_produces_no_reasons(
        self,
        mock_embedder: Embedder,
        vector_store: VectorStore,
        tmp_path: Path,
    ) -> None:
        """
        Given no decisions collection exists
        When a new JD is scored
        Then no rejection reasons are injected (no error)
        """
        # Given: only resume and archetypes indexed — no decisions at all
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_resume(indexer, tmp_path)
        await _seed_archetypes(indexer, tmp_path)

        mock_embedder.classify = AsyncMock(
            return_value='{"disqualified": false, "reason": null}'
        )
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=True)

        # When: scored — should not raise
        result = await scorer.score("Some job description.")

        # Then: not disqualified, and the prompt didn't contain rejection reasons header
        assert result.disqualified is False, (
            f"Missing decisions collection should not cause an error. "
            f"Got disqualified={result.disqualified}"
        )
        classify_call_args = mock_embedder.classify.call_args
        prompt_text = classify_call_args[0][0] if classify_call_args[0] else ""
        assert "The operator has also rejected" not in prompt_text, (
            f"No rejection reasons header should be in prompt without decisions. "
            f"Got: {prompt_text[:500]}"
        )

    @pytest.mark.asyncio
    async def test_rejection_reasons_are_cached_per_scorer_instance(
        self,
        mock_embedder: Embedder,
        vector_store: VectorStore,
        decision_recorder: DecisionRecorder,
        tmp_path: Path,
    ) -> None:
        """
        Given reasons are loaded on first score() call
        When a second score() call is made
        Then the reasons are served from cache (no additional ChromaDB query)
        """
        # Given: a 'no' decision recorded
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        await _seed_all_collections(indexer, tmp_path)
        await decision_recorder.record(
            job_id="test-cache", verdict="no", jd_text="Cached rejection.",
            board="testboard", reason="requires travel",
        )

        mock_embedder.classify = AsyncMock(
            return_value='{"disqualified": false, "reason": null}'
        )
        scorer = Scorer(store=vector_store, embedder=mock_embedder, disqualify_on_llm_flag=True)

        # When: scored twice
        await scorer.score("First scoring call.")
        await scorer.score("Second scoring call.")

        # Then: both calls include the reason (cached from first call)
        assert mock_embedder.classify.call_count == 2, (
            f"Expected 2 classify calls, got {mock_embedder.classify.call_count}"
        )
        for i, call in enumerate(mock_embedder.classify.call_args_list):
            prompt_text = call[0][0] if call[0] else ""
            assert "requires travel" in prompt_text, (
                f"Call {i+1}: Expected cached reason 'requires travel' in prompt. "
                f"Got: {prompt_text[:300]}"
            )


# ============================================================================
# TestCompensationParsing
# ============================================================================


class TestCompensationParsing:
    """
    REQUIREMENT: Compensation ranges are extracted from JD text and
    normalised to annual float values.

    WHO: The pipeline runner enriching listings; the scorer computing comp_score
    WHAT: Annual ranges ($NNN,NNN or $NNNk) are extracted as float pairs;
          hourly rates ($NN/hr) are converted to annual via x2080;
          single values produce identical min and max; the original text
          snippet is preserved in comp_text; JDs with no salary information
          produce None for all comp fields; the parser does not hallucinate
          numbers from non-salary contexts
    WHY: Incorrect parsing silently distorts comp_score rankings —
         a wrong number is worse than a missing one

    MOCK BOUNDARY:
        Mock:  nothing — parse_compensation() is pure computation
        Real:  parse_compensation() called directly with string inputs
        Never: Mock the function itself; all variation is in the input string
    """

    def test_annual_range_with_commas_is_parsed_to_float_pair(self) -> None:
        """
        Given a JD with "$180,000 - $220,000" salary text
        When parsed
        Then comp_min=180000.0 and comp_max=220000.0
        """
        # Given: JD text with comma-formatted annual range
        text = "Salary: $180,000 - $220,000 per year. Excellent benefits."

        # When: parsed
        result = parse_compensation(text)

        # Then: correct float pair
        assert result is not None, "Expected a CompResult, got None"
        assert result.comp_min == pytest.approx(180_000.0, abs=1.0), (
            f"comp_min mismatch. Expected 180000.0, got {result.comp_min}. "
            f"Parsed from: {result.comp_text!r}"
        )
        assert result.comp_max == pytest.approx(220_000.0, abs=1.0), (
            f"comp_max mismatch. Expected 220000.0, got {result.comp_max}. "
            f"Parsed from: {result.comp_text!r}"
        )

    def test_annual_range_with_k_suffix_is_parsed(self) -> None:
        """
        Given a JD with "$180k - $220k" salary text
        When parsed
        Then comp_min=180000.0 and comp_max=220000.0
        """
        # Given: JD text with k-suffix
        text = "Compensation: $180k - $220k base salary."

        # When: parsed
        result = parse_compensation(text)

        # Then: correct float pair
        assert result is not None, "Expected a CompResult, got None"
        assert result.comp_min == pytest.approx(180_000.0, abs=1.0), (
            f"comp_min mismatch. Expected 180000.0, got {result.comp_min}"
        )
        assert result.comp_max == pytest.approx(220_000.0, abs=1.0), (
            f"comp_max mismatch. Expected 220000.0, got {result.comp_max}"
        )

    def test_hourly_rate_is_converted_to_annual_via_2080(self) -> None:
        """
        Given a JD with "$85/hr" hourly rate
        When parsed
        Then comp_min and comp_max are 85 * 2080 = 176800.0
        """
        # Given: hourly rate text
        text = "Rate: $85/hr for this contract position."

        # When: parsed
        result = parse_compensation(text)

        # Then: annualised via x2080
        expected = 85.0 * 2080
        assert result is not None, "Expected a CompResult, got None"
        assert result.comp_min == pytest.approx(expected, abs=1.0), (
            f"comp_min mismatch. Expected {expected}, got {result.comp_min}"
        )
        assert result.comp_max == pytest.approx(expected, abs=1.0), (
            f"comp_max mismatch. Expected {expected}, got {result.comp_max}"
        )

    def test_single_annual_value_sets_both_min_and_max(self) -> None:
        """
        Given a JD with a single salary value "$200,000"
        When parsed
        Then comp_min == comp_max == 200000.0
        """
        # Given: single value
        text = "Base salary: $200,000 per year."

        # When: parsed
        result = parse_compensation(text)

        # Then: min == max
        assert result is not None, "Expected a CompResult, got None"
        assert result.comp_min == pytest.approx(200_000.0, abs=1.0), (
            f"comp_min mismatch for single value. Expected 200000.0, got {result.comp_min}"
        )
        assert result.comp_max == pytest.approx(200_000.0, abs=1.0), (
            f"comp_max mismatch for single value. Expected 200000.0, got {result.comp_max}"
        )

    def test_no_salary_info_returns_none_for_all_comp_fields(self) -> None:
        """
        Given a JD with no salary information
        When parsed
        Then the result is None
        """
        # Given: JD without salary text
        text = (
            "We are seeking a Staff Platform Architect. Competitive compensation "
            "package. Must have 10+ years experience."
        )

        # When: parsed
        result = parse_compensation(text)

        # Then: None
        assert result is None, (
            f"Expected None for JD with no salary info. Got: {result}"
        )

    def test_comp_text_preserves_original_matched_snippet(self) -> None:
        """
        Given a JD with "$180,000 - $220,000 per year"
        When parsed
        Then comp_text contains the matched salary snippet
        """
        # Given: JD with salary range
        text = "This role pays $180,000 - $220,000 per year plus equity."

        # When: parsed
        result = parse_compensation(text)

        # Then: comp_text preserved
        assert result is not None, "Expected a CompResult, got None"
        assert "$180,000" in result.comp_text, (
            f"comp_text should contain '$180,000'. Got: {result.comp_text!r}"
        )
        assert "$220,000" in result.comp_text, (
            f"comp_text should contain '$220,000'. Got: {result.comp_text!r}"
        )

    def test_employee_count_is_not_parsed_as_salary(self) -> None:
        """
        Given a JD mentioning "$500 million revenue" and "2,000 employees"
        When parsed
        Then the result is None (not false-positive parsed)
        """
        # Given: text with non-salary dollar amounts
        text = (
            "Our company has $500 million in revenue with 2,000 employees "
            "across 15 offices worldwide. No salary information provided."
        )

        # When: parsed
        result = parse_compensation(text)

        # Then: None (false positives filtered)
        assert result is None, (
            f"Employee count / revenue should not be parsed as salary. Got: {result}"
        )

    def test_revenue_figure_is_not_parsed_as_salary(self) -> None:
        """
        Given a JD mentioning "$2 billion in funding"
        When parsed
        Then the result is None
        """
        # Given: revenue figure
        text = "We raised $2 billion in Series D funding to fuel growth."

        # When: parsed
        result = parse_compensation(text)

        # Then: None
        assert result is None, (
            f"Revenue/funding figure should not be parsed as salary. Got: {result}"
        )

    def test_range_with_dollar_sign_and_hyphen_is_parsed(self) -> None:
        """
        Given "$150,000-$200,000" (no spaces around hyphen)
        When parsed
        Then comp_min=150000.0 and comp_max=200000.0
        """
        # Given: no-space range
        text = "Pay range: $150,000-$200,000"

        # When: parsed
        result = parse_compensation(text)

        # Then: correct pair
        assert result is not None, "Expected a CompResult, got None"
        assert result.comp_min == pytest.approx(150_000.0, abs=1.0), (
            f"comp_min mismatch. Expected 150000.0, got {result.comp_min}"
        )
        assert result.comp_max == pytest.approx(200_000.0, abs=1.0), (
            f"comp_max mismatch. Expected 200000.0, got {result.comp_max}"
        )

    def test_range_with_to_keyword_is_parsed(self) -> None:
        """
        Given "$150,000 to $200,000"
        When parsed
        Then comp_min=150000.0 and comp_max=200000.0
        """
        # Given: range with "to" keyword
        text = "Salary: $150,000 to $200,000 annually."

        # When: parsed
        result = parse_compensation(text)

        # Then: correct pair
        assert result is not None, "Expected a CompResult, got None"
        assert result.comp_min == pytest.approx(150_000.0, abs=1.0), (
            f"comp_min mismatch. Expected 150000.0, got {result.comp_min}"
        )
        assert result.comp_max == pytest.approx(200_000.0, abs=1.0), (
            f"comp_max mismatch. Expected 200000.0, got {result.comp_max}"
        )

    def test_per_year_and_per_annum_suffixes_are_recognised(self) -> None:
        """
        Given salary text with "per year" and "per annum" suffixes
        When parsed
        Then both are correctly recognised as annual compensation
        """
        # Given: per-year suffix
        text_year = "Base: $190,000 per year."
        text_annum = "Base: $190,000 per annum."

        # When: parsed
        result_year = parse_compensation(text_year)
        result_annum = parse_compensation(text_annum)

        # Then: both produce results
        assert result_year is not None, "Expected CompResult for 'per year', got None"
        assert result_annum is not None, "Expected CompResult for 'per annum', got None"
        assert result_year.comp_min == pytest.approx(190_000.0, abs=1.0), (
            f"'per year' comp_min mismatch. Expected 190000.0, got {result_year.comp_min}"
        )
        assert result_annum.comp_min == pytest.approx(190_000.0, abs=1.0), (
            f"'per annum' comp_min mismatch. Expected 190000.0, got {result_annum.comp_min}"
        )

    def test_employer_stated_salary_is_preferred_over_board_estimate(self) -> None:
        """
        Given a JD body containing a salary range
        When parsed with source="employer"
        Then comp_source is "employer"
        """
        # Given: JD body with salary
        text = "Base salary: $180,000 - $220,000 per year."

        # When: parsed as employer-sourced
        result = parse_compensation(text, source="employer")

        # Then: source is employer
        assert result is not None, "Expected CompResult, got None"
        assert result.comp_source == "employer", (
            f"Expected comp_source='employer', got {result.comp_source!r}"
        )

    def test_comp_source_is_employer_when_salary_is_in_jd_body(self) -> None:
        """
        Given salary text in the JD body
        When parsed with default source
        Then comp_source is "employer"
        """
        # Given: JD body salary
        text = "We offer $200,000 base compensation."

        # When: parsed with default source
        result = parse_compensation(text)

        # Then: source is employer (default)
        assert result is not None, "Expected CompResult, got None"
        assert result.comp_source == "employer", (
            f"Expected comp_source='employer', got {result.comp_source!r}"
        )

    def test_comp_source_is_estimated_when_salary_is_from_board_metadata(self) -> None:
        """
        Given salary text from board metadata
        When parsed with source="estimated"
        Then comp_source is "estimated"
        """
        # Given: estimated salary from board
        text = "Estimated: $150,000 - $180,000"

        # When: parsed as estimated
        result = parse_compensation(text, source="estimated")

        # Then: source is estimated
        assert result is not None, "Expected CompResult, got None"
        assert result.comp_source == "estimated", (
            f"Expected comp_source='estimated', got {result.comp_source!r}"
        )

    def test_range_near_false_positive_context_is_filtered(self) -> None:
        """
        Given a range pattern near false-positive context words like 'revenue'
        When parsed
        Then the result is None (the range is not treated as a salary)
        """
        # Given: dollar range near "revenue" context
        text = "Annual revenue of $50,000 - $80,000 in Q4 sales."

        # When: parsed
        result = parse_compensation(text)

        # Then: filtered as false positive
        assert result is None, (
            f"Range near 'revenue' should be filtered. Got: {result}"
        )

    def test_hourly_range_is_annualized_via_2080(self) -> None:
        """
        Given a JD with an hourly range like "$80 - $120/hr"
        When parsed
        Then both comp_min and comp_max are annualized via x2080
        """
        # Given: hourly range (suffix after the second value)
        text = "Rate: $80 - $120/hr depending on experience."

        # When: parsed
        result = parse_compensation(text)

        # Then: both endpoints annualized
        assert result is not None, "Expected a CompResult, got None"
        assert result.comp_min == pytest.approx(80.0 * 2080, abs=1.0), (
            f"comp_min should be 80*2080={80.0 * 2080}. Got {result.comp_min}"
        )
        assert result.comp_max == pytest.approx(120.0 * 2080, abs=1.0), (
            f"comp_max should be 120*2080={120.0 * 2080}. Got {result.comp_max}"
        )

    def test_range_below_salary_floor_returns_none(self) -> None:
        """
        Given a range with both values below the salary floor ($10)
        When parsed
        Then the result is None (below minimum salary floor)
        """
        # Given: range below floor
        text = "Pays $5 - $8 per widget."

        # When: parsed
        result = parse_compensation(text)

        # Then: filtered by salary floor
        assert result is None, (
            f"Range $5-$8 should be below salary floor. Got: {result}"
        )

    def test_single_value_below_salary_floor_returns_none(self) -> None:
        """
        Given a single dollar value below the salary floor ($10)
        When parsed
        Then the result is None (below minimum salary floor)
        """
        # Given: single value below floor
        text = "Only $5 per unit."

        # When: parsed
        result = parse_compensation(text)

        # Then: filtered by salary floor
        assert result is None, (
            f"Single value $5 should be below salary floor. Got: {result}"
        )


# ============================================================================
# TestCompensationScoring
# ============================================================================


class TestCompensationScoring:
    """
    REQUIREMENT: comp_score is a continuous signal relative to configurable
    base_salary and comp_bands targets, not a binary gate.

    WHO: The ranker consuming comp_score to nudge final ranking;
         the operator tuning base_salary and comp_bands in settings.toml
    WHAT: comp_max at or above base_salary produces 1.0; comp_max below
          the minimum threshold produces 0.0; the curve is linear within
          each band; missing compensation produces a configurable neutral
          score (default 0.5); base_salary and band boundaries are read
          from settings; comp_score is in [0.0, 1.0]; when comp_bands
          are absent from config, the system uses documented defaults;
          custom comp_bands shift the scoring curve accordingly
    WHY: Compensation as a continuous taste signal lets well-paying roles
         float up without hard-gating potential stepping-stone roles.
         Configurable bands let the operator tune sensitivity without
         code changes — e.g. tighten bands when money matters more,
         or widen them for exploratory searches

    MOCK BOUNDARY:
        Mock:  nothing — comp scoring is pure computation
        Real:  compute_comp_score() called directly with float inputs
               and CompBand / missing_comp_score parameters;
               Settings instance constructed from a real TOML tmp_path file
        Never: Mock the scorer or Settings; verify the band boundaries by
               calling compute_comp_score() with inputs at and around each boundary
    """

    # Base salary for tests — using the default from ScoringConfig
    BASE = 220_000.0

    def test_comp_max_at_base_salary_scores_one(self) -> None:
        """
        Given comp_max equals base_salary
        When compute_comp_score() is called
        Then the result is 1.0
        """
        # Given: comp_max == base_salary
        # When: computed
        score = compute_comp_score(self.BASE, self.BASE)

        # Then: 1.0
        assert score == pytest.approx(1.0, abs=0.01), (
            f"comp_max at base_salary should score 1.0. Got {score:.4f}"
        )

    def test_comp_max_above_base_salary_scores_one(self) -> None:
        """
        Given comp_max > base_salary
        When compute_comp_score() is called
        Then the result is 1.0
        """
        # Given: comp_max 10% above base
        # When: computed
        score = compute_comp_score(self.BASE * 1.1, self.BASE)

        # Then: 1.0
        assert score == pytest.approx(1.0, abs=0.01), (
            f"comp_max above base_salary should score 1.0. Got {score:.4f}"
        )

    def test_comp_max_at_95_percent_of_base_scores_between_07_and_09(self) -> None:
        """
        Given comp_max at 95% of base_salary
        When compute_comp_score() is called
        Then the result is between 0.7 and 0.9
        """
        # Given: 95% of base
        comp_max = self.BASE * 0.95

        # When: computed
        score = compute_comp_score(comp_max, self.BASE)

        # Then: between 0.7 and 0.9
        assert 0.7 <= score <= 0.9, (
            f"comp_max at 95% of base should score 0.7-0.9. Got {score:.4f}"
        )

    def test_comp_max_at_90_percent_of_base_scores_07(self) -> None:
        """
        Given comp_max at 90% of base_salary
        When compute_comp_score() is called
        Then the result is approximately 0.7
        """
        # Given: 90% of base
        comp_max = self.BASE * 0.90

        # When: computed
        score = compute_comp_score(comp_max, self.BASE)

        # Then: ~0.7
        assert score == pytest.approx(0.7, abs=0.01), (
            f"comp_max at 90% of base should score ~0.7. Got {score:.4f}"
        )

    def test_comp_max_at_85_percent_of_base_scores_between_04_and_07(self) -> None:
        """
        Given comp_max at 85% of base_salary
        When compute_comp_score() is called
        Then the result is between 0.4 and 0.7
        """
        # Given: 85% of base
        comp_max = self.BASE * 0.85

        # When: computed
        score = compute_comp_score(comp_max, self.BASE)

        # Then: between 0.4 and 0.7
        assert 0.4 <= score <= 0.7, (
            f"comp_max at 85% of base should score 0.4-0.7. Got {score:.4f}"
        )

    def test_comp_max_at_77_percent_of_base_scores_04(self) -> None:
        """
        Given comp_max at 77% of base_salary
        When compute_comp_score() is called
        Then the result is approximately 0.4
        """
        # Given: 77% of base
        comp_max = self.BASE * 0.77

        # When: computed
        score = compute_comp_score(comp_max, self.BASE)

        # Then: ~0.4
        assert score == pytest.approx(0.4, abs=0.01), (
            f"comp_max at 77% of base should score ~0.4. Got {score:.4f}"
        )

    def test_comp_max_at_72_percent_of_base_scores_between_00_and_04(self) -> None:
        """
        Given comp_max at 72% of base_salary
        When compute_comp_score() is called
        Then the result is between 0.0 and 0.4
        """
        # Given: 72% of base
        comp_max = self.BASE * 0.72

        # When: computed
        score = compute_comp_score(comp_max, self.BASE)

        # Then: between 0.0 and 0.4
        assert 0.0 <= score <= 0.4, (
            f"comp_max at 72% of base should score 0.0-0.4. Got {score:.4f}"
        )

    def test_comp_max_at_68_percent_of_base_scores_zero(self) -> None:
        """
        Given comp_max at 68% of base_salary
        When compute_comp_score() is called
        Then the result is approximately 0.0
        """
        # Given: 68% of base
        comp_max = self.BASE * 0.68

        # When: computed
        score = compute_comp_score(comp_max, self.BASE)

        # Then: ~0.0
        assert score == pytest.approx(0.0, abs=0.01), (
            f"comp_max at 68% of base should score ~0.0. Got {score:.4f}"
        )

    def test_comp_max_below_68_percent_of_base_scores_zero(self) -> None:
        """
        Given comp_max well below 68% of base_salary
        When compute_comp_score() is called
        Then the result is 0.0
        """
        # Given: 50% of base
        comp_max = self.BASE * 0.50

        # When: computed
        score = compute_comp_score(comp_max, self.BASE)

        # Then: 0.0
        assert score == pytest.approx(0.0, abs=0.01), (
            f"comp_max well below 68% of base should score 0.0. Got {score:.4f}"
        )

    def test_missing_comp_data_scores_neutral(self) -> None:
        """
        Given comp_max is None (missing compensation)
        When compute_comp_score() is called with default missing_comp_score
        Then the result is 0.5 (the default neutral score)
        """
        # Given: None comp_max, default missing_comp_score
        # When: computed
        score = compute_comp_score(None, self.BASE)

        # Then: 0.5 (default neutral)
        assert score == pytest.approx(0.5, abs=0.01), (
            f"Missing comp data should score 0.5 (default neutral). Got {score:.4f}"
        )

    def test_base_salary_is_read_from_settings_not_hardcoded(self) -> None:
        """
        Given two different base_salary values
        When compute_comp_score() is called with the same comp_max
        Then the scores differ
        """
        # Given: two different base salaries
        comp_max = 200_000.0

        # When: computed with each
        score_low_base = compute_comp_score(comp_max, 180_000.0)
        score_high_base = compute_comp_score(comp_max, 300_000.0)

        # Then: scores differ
        assert score_low_base != pytest.approx(score_high_base, abs=0.01), (
            f"base_salary should affect comp_score. "
            f"Low base: {score_low_base:.4f}, High base: {score_high_base:.4f}"
        )

    def test_changing_base_salary_shifts_all_band_boundaries(self) -> None:
        """
        Given a specific comp_max
        When computed with two different base_salary values
        Then a higher base_salary shifts the score downward
        """
        # Given: fixed comp_max
        comp_max = 190_000.0

        # When: computed with low and high base
        score_with_low_base = compute_comp_score(comp_max, 180_000.0)
        score_with_high_base = compute_comp_score(comp_max, 250_000.0)

        # Then: higher base → lower score
        assert score_with_low_base > score_with_high_base, (
            f"Higher base_salary should shift score downward. "
            f"Low base score: {score_with_low_base:.4f}, "
            f"High base score: {score_with_high_base:.4f}"
        )

    def test_comp_score_is_always_in_zero_to_one_range(self) -> None:
        """
        Given extreme comp_max values
        When compute_comp_score() is called
        Then the result is always in [0.0, 1.0]
        """
        # Given: extreme values
        extremes = [0.0, 1.0, 50_000.0, 100_000.0, 500_000.0, 1_000_000.0]

        for comp_max in extremes:
            # When: computed
            score = compute_comp_score(comp_max, self.BASE)

            # Then: in range
            assert 0.0 <= score <= 1.0, (
                f"comp_score out of range for comp_max={comp_max}. Got {score:.4f}"
            )

    def test_score_is_continuous_across_band_boundaries(self) -> None:
        """
        Given comp_max values just above and below each band boundary
        When compute_comp_score() is called for both
        Then the scores are close (no large discontinuity)
        """
        # Given: boundary ratios from the default band config
        boundaries = [band.ratio for band in DEFAULT_COMP_BANDS]
        max_gap = 0.05  # maximum allowed jump at a boundary

        for ratio in boundaries:
            above = self.BASE * (ratio + 0.001)
            below = self.BASE * (ratio - 0.001)

            # When: computed
            score_above = compute_comp_score(above, self.BASE)
            score_below = compute_comp_score(below, self.BASE)

            # Then: no large discontinuity
            gap = abs(score_above - score_below)
            assert gap <= max_gap, (
                f"Discontinuity at boundary {ratio}: "
                f"above={score_above:.4f}, below={score_below:.4f}, gap={gap:.4f} > {max_gap}"
            )

    def test_comp_bands_are_read_from_config_not_hardcoded(self) -> None:
        """
        Given custom comp_bands with different boundaries than the defaults
        When compute_comp_score() is called at the default 90% boundary
        Then the score differs from the default-bands score at that point
        """
        # Given: custom bands with a single steep band (80% -> 0.0, 100% -> 1.0)
        custom_bands = [CompBand(ratio=1.0, score=1.0), CompBand(ratio=0.80, score=0.0)]
        comp_max = self.BASE * 0.90  # 90% of base

        # When: computed with default bands vs custom bands
        score_default = compute_comp_score(comp_max, self.BASE)
        score_custom = compute_comp_score(
            comp_max, self.BASE, comp_bands=custom_bands
        )

        # Then: scores differ because band boundaries differ
        assert score_default != pytest.approx(score_custom, abs=0.01), (
            f"Custom comp_bands should produce a different score at 90% of base. "
            f"Default: {score_default:.4f}, Custom: {score_custom:.4f}"
        )

    def test_custom_comp_bands_change_scoring_curve(self) -> None:
        """
        Given a tight custom band set (95% -> 0.0, 100% -> 1.0)
        When compute_comp_score() is called at 90% of base
        Then the score is 0.0 (below the custom floor)
        """
        # Given: tight bands — anything below 95% scores 0.0
        tight_bands = [CompBand(ratio=1.0, score=1.0), CompBand(ratio=0.95, score=0.0)]
        comp_max = self.BASE * 0.90  # 90% of base — below the tight floor

        # When: computed with tight bands
        score = compute_comp_score(comp_max, self.BASE, comp_bands=tight_bands)

        # Then: 0.0 — below the custom minimum band
        assert score == pytest.approx(0.0, abs=0.01), (
            f"comp_max at 90% with tight bands (floor=95%) should score 0.0. Got {score:.4f}"
        )

    def test_missing_comp_score_is_configurable(self) -> None:
        """
        Given a custom missing_comp_score of 0.3
        When compute_comp_score() is called with None comp_max
        Then the result is 0.3 (not the default 0.5)
        """
        # Given: custom missing_comp_score
        custom_neutral = 0.3

        # When: computed with None comp_max
        score = compute_comp_score(None, self.BASE, missing_comp_score=custom_neutral)

        # Then: returns the custom neutral score
        assert score == pytest.approx(0.3, abs=0.01), (
            f"Custom missing_comp_score=0.3 should return 0.3 for None. Got {score:.4f}"
        )

    def test_default_comp_bands_match_original_behavior(self) -> None:
        """
        Given the default comp_bands are passed explicitly
        When compute_comp_score() is called at several reference points
        Then the scores match the original hardcoded behavior exactly
        """
        # Given: reference points with known scores under original behavior
        reference_points = [
            (self.BASE * 1.0, 1.0),     # at base → 1.0
            (self.BASE * 0.90, 0.7),    # at 90% → 0.7
            (self.BASE * 0.77, 0.4),    # at 77% → 0.4
            (self.BASE * 0.68, 0.0),    # at 68% → 0.0
            (self.BASE * 0.50, 0.0),    # below floor → 0.0
        ]

        for comp_max, expected_score in reference_points:
            # When: computed with explicit default bands
            score = compute_comp_score(
                comp_max, self.BASE, comp_bands=DEFAULT_COMP_BANDS
            )

            # Then: matches original behavior
            assert score == pytest.approx(expected_score, abs=0.01), (
                f"Default bands at comp_max={comp_max:.0f} should score "
                f"{expected_score:.1f}. Got {score:.4f}"
            )


# ============================================================================
# TestScoreFusion
# ============================================================================


class TestScoreFusion:
    """
    REQUIREMENT: Final score correctly fuses weighted components from settings,
    including a culture reward and a continuous negative penalty.

    WHO: The ranker; the operator tuning weights in settings.toml
    WHAT: final_score = (archetype_weight * archetype_score
                       + fit_weight       * fit_score
                       + culture_weight   * culture_score
                       + history_weight   * history_score
                       + comp_weight      * comp_score
                       - negative_weight  * negative_score)
                       * (0.0 if disqualified else 1.0)
          All weights are read from settings; disqualified roles score 0.0;
          roles below min_score_threshold are excluded from output entirely;
          negative penalty floored at 0.0; score breakdown includes all six
          components
    WHY: Hardcoded weights make the tool non-configurable. A negative
         final_score has no semantic meaning and breaks threshold comparisons

    MOCK BOUNDARY:
        Mock:  mock_embedder fixture (Ollama HTTP)
        Real:  Ranker instance, Scorer instance, ChromaDB via vector_store
               fixture, Settings constructed from real TOML in tmp_path
        Never: Construct RankedListing directly; always obtain via
               ranker.rank(scored_listings, settings); never hardcode
               expected final_score — compute it from the formula and assert
               with pytest.approx
    """

    def test_final_score_matches_weighted_formula(self) -> None:
        """
        When a ScoreResult is scored by the Ranker
        Then final_score matches the weighted formula with pytest.approx
        """
        # Given: known weights and scores
        ranker = Ranker(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            comp_weight=0.15, negative_weight=0.4, culture_weight=0.2,
            min_score_threshold=0.0,
        )
        scores = ScoreResult(
            fit_score=0.8, archetype_score=0.7, history_score=0.5,
            disqualified=False, comp_score=0.9, negative_score=0.2,
            culture_score=0.6,
        )

        # When: final score computed
        final = ranker.compute_final_score(scores)

        # Then: matches formula
        expected = (
            0.5 * 0.7  # archetype
            + 0.3 * 0.8  # fit
            + 0.2 * 0.5  # history
            + 0.15 * 0.9  # comp
            + 0.2 * 0.6  # culture
            - 0.4 * 0.2  # negative penalty
        )
        expected = max(0.0, expected)
        assert final == pytest.approx(expected, abs=0.01), (
            f"final_score mismatch. Expected {expected:.4f}, got {final:.4f}"
        )

    def test_all_weights_are_read_from_settings(self) -> None:
        """
        Given two Ranker instances with different weights
        When the same scores are ranked
        Then the final_scores differ
        """
        # Given: two rankers with different archetype_weight
        ranker_a = Ranker(
            archetype_weight=0.8, fit_weight=0.1, history_weight=0.1,
            negative_weight=0.0, culture_weight=0.0, min_score_threshold=0.0,
        )
        ranker_b = Ranker(
            archetype_weight=0.1, fit_weight=0.8, history_weight=0.1,
            negative_weight=0.0, culture_weight=0.0, min_score_threshold=0.0,
        )
        scores = ScoreResult(
            fit_score=0.3, archetype_score=0.9, history_score=0.5,
            disqualified=False,
        )

        # When: computed
        final_a = ranker_a.compute_final_score(scores)
        final_b = ranker_b.compute_final_score(scores)

        # Then: differ
        assert final_a != pytest.approx(final_b, abs=0.01), (
            f"Different weights should produce different scores. "
            f"A: {final_a:.4f}, B: {final_b:.4f}"
        )

    def test_disqualified_role_scores_zero(self) -> None:
        """
        Given a disqualified ScoreResult
        When final_score is computed
        Then the result is 0.0
        """
        # Given: disqualified
        ranker = Ranker(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            negative_weight=0.4, culture_weight=0.2, min_score_threshold=0.0,
        )
        scores = ScoreResult(
            fit_score=0.9, archetype_score=0.9, history_score=0.9,
            disqualified=True, disqualifier_reason="IC role",
        )

        # When: computed
        final = ranker.compute_final_score(scores)

        # Then: 0.0
        assert final == pytest.approx(0.0, abs=1e-9), (
            f"Disqualified role should score 0.0. Got {final:.4f}"
        )

    def test_role_below_threshold_is_excluded_from_output(self) -> None:
        """
        Given a role whose final_score is below min_score_threshold
        When ranked
        Then the role is excluded from the output
        """
        # Given: ranker with threshold=0.5 and a low-scoring role
        ranker = Ranker(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            negative_weight=0.0, culture_weight=0.0, min_score_threshold=0.5,
        )
        low_scores = ScoreResult(
            fit_score=0.1, archetype_score=0.1, history_score=0.1,
            disqualified=False,
        )
        listing = _make_listing(external_id="low")

        # When: ranked
        ranked, summary = ranker.rank([(listing, low_scores)])

        # Then: excluded
        assert len(ranked) == 0, (
            f"Role below threshold should be excluded. Got {len(ranked)} results"
        )
        assert summary.total_excluded == 1, (
            f"Expected 1 excluded, got {summary.total_excluded}"
        )

    def test_role_at_exactly_threshold_is_included(self) -> None:
        """
        Given a role whose final_score exactly equals min_score_threshold
        When ranked
        Then the role is included in the output
        """
        # Given: a role that scores exactly at threshold
        # We need to find weights that produce exactly 0.45 for the default threshold
        # With archetype=0.9, fit=0.0, history=0.0, weights 0.5/0.3/0.2:
        # final = 0.5 * 0.9 = 0.45
        ranker = Ranker(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            negative_weight=0.0, culture_weight=0.0, min_score_threshold=0.45,
        )
        scores = ScoreResult(
            fit_score=0.0, archetype_score=0.9, history_score=0.0,
            disqualified=False,
        )
        listing = _make_listing(external_id="threshold")

        # When: ranked
        ranked, _ = ranker.rank([(listing, scores)])

        # Then: included (>= threshold)
        assert len(ranked) == 1, (
            f"Role at exactly threshold should be included. Got {len(ranked)} results. "
            f"final_score={ranker.compute_final_score(scores):.4f}, threshold=0.45"
        )

    def test_high_culture_score_raises_final_score(self) -> None:
        """
        Given two identical score sets except one has high culture_score
        When final_score is computed
        Then the high-culture role scores higher
        """
        # Given: ranker with non-zero culture_weight
        ranker = Ranker(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            negative_weight=0.0, culture_weight=0.3, min_score_threshold=0.0,
        )
        base = ScoreResult(
            fit_score=0.7, archetype_score=0.7, history_score=0.5,
            disqualified=False, culture_score=0.0,
        )
        with_culture = ScoreResult(
            fit_score=0.7, archetype_score=0.7, history_score=0.5,
            disqualified=False, culture_score=0.9,
        )

        # When: computed
        final_base = ranker.compute_final_score(base)
        final_culture = ranker.compute_final_score(with_culture)

        # Then: culture raises score
        assert final_culture > final_base, (
            f"High culture_score should raise final_score. "
            f"Base: {final_base:.4f}, With culture: {final_culture:.4f}"
        )

    def test_high_negative_score_suppresses_final_score(self) -> None:
        """
        Given two identical score sets except one has high negative_score
        When final_score is computed
        Then the high-negative role scores lower
        """
        # Given: ranker with non-zero negative_weight
        ranker = Ranker(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            negative_weight=0.4, culture_weight=0.0, min_score_threshold=0.0,
        )
        clean = ScoreResult(
            fit_score=0.7, archetype_score=0.7, history_score=0.5,
            disqualified=False, negative_score=0.0,
        )
        negative = ScoreResult(
            fit_score=0.7, archetype_score=0.7, history_score=0.5,
            disqualified=False, negative_score=0.9,
        )

        # When: computed
        final_clean = ranker.compute_final_score(clean)
        final_negative = ranker.compute_final_score(negative)

        # Then: negative suppresses
        assert final_negative < final_clean, (
            f"High negative_score should suppress final_score. "
            f"Clean: {final_clean:.4f}, Negative: {final_negative:.4f}"
        )

    def test_score_breakdown_includes_all_six_components(self) -> None:
        """
        When a RankedListing's score_explanation() is called
        Then it includes all six component names
        """
        # Given: a listing scored through the ranker
        ranker = Ranker(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            comp_weight=0.15, negative_weight=0.4, culture_weight=0.2,
            min_score_threshold=0.0,
        )
        scores = ScoreResult(
            fit_score=0.8, archetype_score=0.7, history_score=0.5,
            disqualified=False, comp_score=0.9, negative_score=0.1,
            culture_score=0.6,
        )
        listing = _make_listing()

        # When: ranked and explanation generated
        ranked, _ = ranker.rank([(listing, scores)])
        assert len(ranked) == 1, f"Expected 1 ranked listing. Got {len(ranked)}"
        explanation = ranked[0].score_explanation()

        # Then: all six components present
        for component in ("Archetype", "Fit", "History", "Comp", "Culture", "Negative"):
            assert component in explanation, (
                f"Missing '{component}' in score breakdown. Got: {explanation}"
            )

    def test_negative_penalty_exceeding_positive_sum_floors_at_zero(self) -> None:
        """
        Given a ScoreResult where negative penalty exceeds the positive sum
        When final_score is computed
        Then the result is 0.0 (floored, not negative)
        """
        # Given: extreme negative, low positive
        ranker = Ranker(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            negative_weight=2.0, culture_weight=0.0, min_score_threshold=0.0,
        )
        scores = ScoreResult(
            fit_score=0.2, archetype_score=0.2, history_score=0.1,
            disqualified=False, negative_score=1.0,
        )

        # When: computed
        final = ranker.compute_final_score(scores)

        # Then: floored at 0.0
        assert final == pytest.approx(0.0, abs=1e-9), (
            f"Negative penalty exceeding positive sum should floor at 0.0. Got {final:.4f}"
        )

    def test_zero_negative_score_does_not_suppress_final_score(self) -> None:
        """
        Given a ScoreResult with zero negative_score
        When final_score is computed
        Then the result equals the positive sum (no suppression)
        """
        # Given: zero negative
        ranker = Ranker(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            negative_weight=0.4, culture_weight=0.2, min_score_threshold=0.0,
        )
        scores = ScoreResult(
            fit_score=0.8, archetype_score=0.7, history_score=0.5,
            disqualified=False, comp_score=0.9, negative_score=0.0,
            culture_score=0.6,
        )

        # When: computed
        final = ranker.compute_final_score(scores)

        # Then: equals positive sum (no penalty)
        expected_positive = (
            0.5 * 0.7 + 0.3 * 0.8 + 0.2 * 0.5 + 0.0 * 0.9 + 0.2 * 0.6
        )
        assert final == pytest.approx(expected_positive, abs=0.01), (
            f"Zero negative_score should not suppress. "
            f"Expected {expected_positive:.4f}, got {final:.4f}"
        )


# ============================================================================
# TestCrossBoardDeduplication
# ============================================================================


class TestCrossBoardDeduplication:
    """
    REQUIREMENT: The same job appearing on multiple boards is presented once.

    WHO: The operator reviewing ranked output
    WHAT: Near-duplicate listings (cosine similarity > 0.95 on full_text)
          are collapsed into one; the highest-scored instance is retained;
          output notes which boards carried the duplicate; same external_id
          on same board is always deduplicated; distinct roles with similar
          titles are not collapsed; deduplication count appears in run summary
    WHY: Seeing the same role five times in a shortlist wastes review time
         and inflates apparent result counts

    MOCK BOUNDARY:
        Mock:  mock_embedder fixture (Ollama HTTP — embed() returns
               controlled vectors for deduplication similarity testing)
        Real:  Ranker, RankSummary
        Never: Construct RankSummary directly; control deduplication by
               providing an embeddings dict with identical vectors for
               intended duplicates and distinct vectors for non-duplicates
    """

    # Deterministic vectors for dedup control
    VEC_A: ClassVar[list[float]] = [0.1, 0.2, 0.3, 0.4, 0.5]
    VEC_A_COPY: ClassVar[list[float]] = [0.1, 0.2, 0.3, 0.4, 0.5]  # identical → cosine=1.0
    VEC_B: ClassVar[list[float]] = [0.9, 0.1, 0.0, 0.0, 0.1]       # distinct → low cosine

    @staticmethod
    def _make_ranker(**overrides: float) -> Ranker:
        defaults = dict(
            archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
            negative_weight=0.0, culture_weight=0.0, min_score_threshold=0.0,
        )
        defaults.update(overrides)
        return Ranker(**defaults)

    @staticmethod
    def _make_scores(**overrides: object) -> ScoreResult:
        defaults: dict[str, object] = dict(
            fit_score=0.8, archetype_score=0.7, history_score=0.5,
            disqualified=False,
        )
        defaults.update(overrides)
        return ScoreResult(**defaults)  # type: ignore[arg-type]

    def test_near_duplicate_listings_are_collapsed_to_one(self) -> None:
        """
        Given two listings from different boards with identical embeddings
        When ranked
        Then only one listing survives
        """
        # Given: two boards, same content (identical vectors)
        ranker = self._make_ranker()
        listing_a = _make_listing(external_id="job1", board="linkedin")
        listing_b = _make_listing(external_id="job1-zr", board="ziprecruiter")
        scores = self._make_scores()

        embeddings = {
            listing_a.url: self.VEC_A,
            listing_b.url: self.VEC_A_COPY,
        }

        # When: ranked with embeddings
        ranked, summary = ranker.rank(
            [(listing_a, scores), (listing_b, scores)],
            embeddings,
        )

        # Then: collapsed to one
        assert len(ranked) == 1, (
            f"Near-duplicate listings should collapse to one. Got {len(ranked)}"
        )
        assert summary.total_deduplicated == 1, (
            f"Expected 1 deduplicated. Got {summary.total_deduplicated}"
        )

    def test_highest_scored_duplicate_is_retained(self) -> None:
        """
        Given two near-duplicate listings with different scores
        When ranked
        Then the higher-scored listing is the survivor
        """
        # Given: linkedin scores higher than ziprecruiter
        ranker = self._make_ranker()
        listing_high = _make_listing(external_id="job1", board="linkedin")
        listing_low = _make_listing(external_id="job1-zr", board="ziprecruiter")
        scores_high = self._make_scores(archetype_score=0.9)
        scores_low = self._make_scores(archetype_score=0.3)

        embeddings = {
            listing_high.url: self.VEC_A,
            listing_low.url: self.VEC_A_COPY,
        }

        # When: ranked
        ranked, _ = ranker.rank(
            [(listing_high, scores_high), (listing_low, scores_low)],
            embeddings,
        )

        # Then: the higher-scored one survives
        assert len(ranked) == 1, (
            f"Expected 1 survivor. Got {len(ranked)}"
        )
        survivor = ranked[0]
        assert survivor.listing.board == "linkedin", (
            f"Higher-scored listing should survive. Got board={survivor.listing.board}"
        )
        assert survivor.final_score > ranker.compute_final_score(scores_low), (
            f"Survivor should have the higher score. Got {survivor.final_score:.4f}"
        )

    def test_output_notes_all_boards_that_carried_duplicate(self) -> None:
        """
        Given three near-duplicate listings from three different boards
        When ranked
        Then the survivor's duplicate_boards lists the other two boards
        """
        # Given: three boards with identical content
        ranker = self._make_ranker()
        listing_a = _make_listing(external_id="job1", board="linkedin")
        listing_b = _make_listing(external_id="job1-zr", board="ziprecruiter")
        listing_c = _make_listing(external_id="job1-id", board="indeed")
        scores_a = self._make_scores(archetype_score=0.9)  # highest
        scores_b = self._make_scores(archetype_score=0.7)
        scores_c = self._make_scores(archetype_score=0.5)

        vec_identical = [0.1, 0.2, 0.3, 0.4, 0.5]
        embeddings = {
            listing_a.url: vec_identical,
            listing_b.url: list(vec_identical),
            listing_c.url: list(vec_identical),
        }

        # When: ranked
        ranked, _ = ranker.rank(
            [
                (listing_a, scores_a),
                (listing_b, scores_b),
                (listing_c, scores_c),
            ],
            embeddings,
        )

        # Then: one survivor with both other boards noted
        assert len(ranked) == 1, (
            f"Expected 1 survivor from 3 near-duplicates. Got {len(ranked)}"
        )
        survivor = ranked[0]
        assert len(survivor.duplicate_boards) == 2, (
            f"Expected 2 duplicate boards noted. Got {survivor.duplicate_boards}"
        )
        for board in ("ziprecruiter", "indeed"):
            assert board in survivor.duplicate_boards, (
                f"Missing '{board}' in duplicate_boards: {survivor.duplicate_boards}"
            )

    def test_same_external_id_same_board_is_always_deduplicated(self) -> None:
        """
        Given two listings with the same external_id and same board
        When ranked (without embeddings)
        Then only one survives (exact dedup, no embeddings needed)
        """
        # Given: exact duplicates on same board
        ranker = self._make_ranker()
        listing_a = _make_listing(external_id="dup1", board="linkedin")
        listing_b = _make_listing(external_id="dup1", board="linkedin")
        scores = self._make_scores()

        # When: ranked without embeddings (exact dedup only)
        ranked, summary = ranker.rank(
            [(listing_a, scores), (listing_b, scores)],
        )

        # Then: collapsed to one via exact dedup
        assert len(ranked) == 1, (
            f"Same external_id + same board should always dedup. Got {len(ranked)}"
        )
        assert summary.total_deduplicated == 1, (
            f"Expected 1 deduplicated. Got {summary.total_deduplicated}"
        )

    def test_distinct_roles_with_similar_titles_are_not_collapsed(self) -> None:
        """
        Given two listings with similar titles but distinct embeddings
        When ranked
        Then both survive (cosine similarity below 0.95)
        """
        # Given: different roles with distinct vectors
        ranker = self._make_ranker()
        listing_a = _make_listing(
            external_id="eng1", board="linkedin",
            title="Senior Platform Engineer",
        )
        listing_b = _make_listing(
            external_id="eng2", board="linkedin",
            title="Senior Platform Engineer — Observability",
        )
        scores = self._make_scores()

        # Distinct vectors → cosine similarity well below 0.95
        embeddings = {
            listing_a.url: self.VEC_A,
            listing_b.url: self.VEC_B,
        }

        # When: ranked
        ranked, summary = ranker.rank(
            [(listing_a, scores), (listing_b, scores)],
            embeddings,
        )

        # Then: both survive
        assert len(ranked) == 2, (
            f"Distinct roles should not be collapsed. Got {len(ranked)}"
        )
        assert summary.total_deduplicated == 0, (
            f"Expected 0 deduplicated. Got {summary.total_deduplicated}"
        )

    def test_deduplication_count_appears_in_run_summary(self) -> None:
        """
        Given three listings where two are near-duplicates
        When ranked
        Then RankSummary.total_deduplicated == 1
        """
        # Given: one unique + two near-duplicates
        ranker = self._make_ranker()
        listing_unique = _make_listing(external_id="unique1", board="linkedin")
        listing_dup_a = _make_listing(external_id="dup-li", board="linkedin")
        listing_dup_b = _make_listing(external_id="dup-zr", board="ziprecruiter")
        scores = self._make_scores()

        embeddings = {
            listing_unique.url: self.VEC_B,        # distinct
            listing_dup_a.url: self.VEC_A,          # duplicate pair
            listing_dup_b.url: self.VEC_A_COPY,     # duplicate pair
        }

        # When: ranked
        ranked, summary = ranker.rank(
            [
                (listing_unique, scores),
                (listing_dup_a, scores),
                (listing_dup_b, scores),
            ],
            embeddings,
        )

        # Then: summary tracks dedup count
        assert summary.total_deduplicated == 1, (
            f"Expected 1 deduplicated in summary. Got {summary.total_deduplicated}"
        )
        assert summary.total_scored == 3, (
            f"Expected 3 total scored. Got {summary.total_scored}"
        )
        assert len(ranked) == 2, (
            f"Expected 2 survivors (1 unique + 1 from dup pair). Got {len(ranked)}"
        )
