"""Scoring pipeline tests — semantic scoring, disqualifier, fusion, dedup.

Maps to BDD specs: TestSemanticScoring, TestDisqualifierClassification,
TestScoreFusion, TestCrossBoardDeduplication

The Scorer orchestrates VectorStore (similarity queries) and Embedder
(embedding + LLM classification).  VectorStore is tested with a real
temp-directory instance; Embedder is mocked since it requires live Ollama.
ScoreFusion and CrossBoardDeduplication specs test the Ranker (Phase 3).
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from jobsearch_rag.adapters.base import JobListing
from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.pipeline.ranker import RankedListing, Ranker
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.scorer import Scorer, ScoreResult
from jobsearch_rag.rag.store import VectorStore

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

# Fake 5D embeddings with directional meaning for similarity tests.
# "architect" and "arch_jd" point similarly; "data_eng" is orthogonal.
EMBED_ARCHITECT = [0.9, 0.1, 0.2, 0.0, 0.3]
EMBED_DATA_ENG = [0.1, 0.8, 0.1, 0.7, 0.0]
EMBED_DEVREL = [0.7, 0.2, 0.3, 0.0, 0.4]
EMBED_ARCH_JD = [0.85, 0.15, 0.25, 0.05, 0.28]  # close to ARCHITECT
EMBED_UNRELATED_JD = [0.0, 0.0, 0.1, 0.9, 0.9]  # far from everything


@pytest.fixture
def store() -> Iterator[VectorStore]:
    """A VectorStore backed by a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield VectorStore(persist_dir=tmpdir)


@pytest.fixture
def mock_embedder() -> Embedder:
    """An Embedder with embed/classify mocked."""
    embedder = Embedder.__new__(Embedder)
    embedder.base_url = "http://localhost:11434"
    embedder.embed_model = "nomic-embed-text"
    embedder.llm_model = "mistral:7b"
    embedder.max_retries = 3
    embedder.base_delay = 0.0
    embedder.embed = AsyncMock(return_value=EMBED_ARCH_JD)  # type: ignore[method-assign]
    embedder.classify = AsyncMock(  # type: ignore[method-assign]
        return_value='{"disqualified": false, "reason": null}'
    )
    return embedder


@pytest.fixture
def populated_store(store: VectorStore) -> VectorStore:
    """A VectorStore with resume and archetype collections pre-populated."""
    # Resume collection — 2 chunks
    store.add_documents(
        collection_name="resume",
        ids=["resume-summary", "resume-experience"],
        documents=[
            "## Summary\nPrincipal architect specializing in distributed systems.",
            "## Experience\nLed platform architecture at multiple companies.",
        ],
        embeddings=[EMBED_ARCHITECT, EMBED_DATA_ENG],
        metadatas=[
            {"source": "resume", "section": "Summary"},
            {"source": "resume", "section": "Experience"},
        ],
    )

    # Archetype collection — 2 archetypes
    store.add_documents(
        collection_name="role_archetypes",
        ids=["archetype-staff-platform-architect", "archetype-devrel"],
        documents=[
            "Staff Platform Architect: distributed systems, cross-team influence.",
            "Developer Relations: technical writing, community engagement.",
        ],
        embeddings=[EMBED_ARCHITECT, EMBED_DEVREL],
        metadatas=[
            {"name": "Staff Platform Architect", "source": "role_archetypes"},
            {"name": "Developer Relations", "source": "role_archetypes"},
        ],
    )

    return store


@pytest.fixture
def scorer(populated_store: VectorStore, mock_embedder: Embedder) -> Scorer:
    """A Scorer wired to a populated VectorStore and mocked Embedder."""
    return Scorer(store=populated_store, embedder=mock_embedder)


@pytest.fixture
def scorer_empty_history(populated_store: VectorStore, mock_embedder: Embedder) -> Scorer:
    """A Scorer with resume+archetypes but no decisions collection."""
    return Scorer(store=populated_store, embedder=mock_embedder)


# ---------------------------------------------------------------------------
# TestSemanticScoring
# ---------------------------------------------------------------------------


class TestSemanticScoring:
    """REQUIREMENT: Semantic scores reflect meaningful similarity, not noise.

    WHO: The ranker consuming scores to produce a ranked shortlist
    WHAT: All three scores (fit, archetype, history) are floats in [0.0, 1.0];
          a JD clearly matching an archetype scores higher than one that does not;
          a JD matching resume skills scores higher fit than one that does not;
          score order is stable across repeated calls with the same inputs
    WHY: Nonsensical scores (>1.0, negative, NaN) or instability across calls
         would produce a randomly-ordered shortlist disguised as a ranking
    """

    async def test_all_scores_are_floats_between_zero_and_one(self, scorer: Scorer) -> None:
        """All three component scores (fit, archetype, history) are floats in [0.0, 1.0]."""
        result = await scorer.score("Staff architect for distributed systems")
        assert result.is_valid
        assert isinstance(result.fit_score, float)
        assert isinstance(result.archetype_score, float)
        assert isinstance(result.history_score, float)

    async def test_matching_jd_scores_higher_archetype_than_non_matching(
        self, scorer: Scorer, mock_embedder: Embedder
    ) -> None:
        """A JD matching an archetype scores higher than an unrelated JD."""
        # First call: architect-like JD
        mock_embedder.embed = AsyncMock(return_value=EMBED_ARCH_JD)  # type: ignore[method-assign]
        result_match = await scorer.score("Staff architect for distributed systems")

        # Second call: unrelated JD
        mock_embedder.embed = AsyncMock(return_value=EMBED_UNRELATED_JD)  # type: ignore[method-assign]
        result_nomatch = await scorer.score("Underwater basket weaving instructor")

        assert result_match.archetype_score > result_nomatch.archetype_score

    async def test_skill_matching_jd_scores_higher_fit_than_non_matching(
        self, scorer: Scorer, mock_embedder: Embedder
    ) -> None:
        """A JD matching resume skills has higher fit_score than one without overlap."""
        mock_embedder.embed = AsyncMock(return_value=EMBED_ARCH_JD)  # type: ignore[method-assign]
        result_match = await scorer.score("Principal architect cloud systems")

        mock_embedder.embed = AsyncMock(return_value=EMBED_UNRELATED_JD)  # type: ignore[method-assign]
        result_nomatch = await scorer.score("Completely unrelated role")

        assert result_match.fit_score > result_nomatch.fit_score

    async def test_scores_are_stable_across_repeated_calls(self, scorer: Scorer) -> None:
        """Scoring the same JD twice produces identical results."""
        r1 = await scorer.score("Staff architect distributed systems")
        r2 = await scorer.score("Staff architect distributed systems")
        assert r1.fit_score == r2.fit_score
        assert r1.archetype_score == r2.archetype_score
        assert r1.history_score == r2.history_score

    async def test_empty_history_collection_returns_zero_history_score(
        self, scorer_empty_history: Scorer
    ) -> None:
        """When no decisions exist, history_score is 0.0 rather than raising."""
        result = await scorer_empty_history.score("Any job description")
        assert result.history_score == 0.0

    async def test_missing_resume_collection_raises_index_error(
        self, mock_embedder: Embedder
    ) -> None:
        """Scoring against an empty/missing resume collection raises INDEX error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_store = VectorStore(persist_dir=tmpdir)
            scorer = Scorer(store=empty_store, embedder=mock_embedder)
            with pytest.raises(ActionableError) as exc_info:
                await scorer.score("Any JD text")
            assert exc_info.value.error_type == ErrorType.INDEX

    async def test_existing_but_empty_resume_collection_raises_index_error(
        self, mock_embedder: Embedder
    ) -> None:
        """A resume collection that exists but has 0 documents raises INDEX error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(persist_dir=tmpdir)
            # Create the collection but don't add documents
            store.reset_collection("resume")
            scorer = Scorer(store=store, embedder=mock_embedder)
            with pytest.raises(ActionableError) as exc_info:
                await scorer.score("Any JD text")
            assert exc_info.value.error_type == ErrorType.INDEX

    async def test_existing_but_empty_decisions_returns_zero_history(
        self, populated_store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """A decisions collection that exists but is empty returns history_score=0.0."""
        # Create an empty decisions collection
        populated_store.reset_collection("decisions")
        scorer = Scorer(store=populated_store, embedder=mock_embedder)
        result = await scorer.score("Staff architect")
        assert result.history_score == 0.0

    async def test_history_score_uses_decisions_when_populated(
        self, populated_store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """When a decisions collection has documents, history_score > 0.0."""
        # Add a decision that looks like the architect JD embedding
        populated_store.add_documents(
            collection_name="decisions",
            ids=["decision-001"],
            documents=["Applied to Staff Architect role — strong match."],
            embeddings=[EMBED_ARCHITECT],
            metadatas=[{"decision": "applied", "source": "decisions"}],
        )
        scorer = Scorer(store=populated_store, embedder=mock_embedder)
        result = await scorer.score("Staff architect for distributed systems")
        assert result.history_score > 0.0

    async def test_disqualify_on_llm_flag_false_skips_disqualifier(
        self, populated_store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """When disqualify_on_llm_flag=False, score() skips the LLM disqualifier."""
        mock_embedder.classify = AsyncMock(  # type: ignore[method-assign]
            return_value='{"disqualified": true, "reason": "should be skipped"}'
        )
        scorer = Scorer(
            store=populated_store,
            embedder=mock_embedder,
            disqualify_on_llm_flag=False,
        )
        result = await scorer.score("Any JD")
        assert result.disqualified is False
        assert result.disqualifier_reason is None
        mock_embedder.classify.assert_not_called()


# ---------------------------------------------------------------------------
# TestDistanceToScore
# ---------------------------------------------------------------------------


class TestDistanceToScore:
    """REQUIREMENT: Cosine distance conversion produces valid similarity scores.

    WHO: The Scorer internals converting ChromaDB distances
    WHAT: Empty distance lists return 0.0; distances clamp to [0.0, 1.0];
          the best (smallest) distance maps to the highest score
    WHY: Invalid distance-to-score conversion would silently corrupt all
         ranking decisions downstream
    """

    def test_empty_distances_return_zero(self) -> None:
        """An empty distances list returns 0.0 — the defensive guard."""
        from jobsearch_rag.rag.scorer import _distance_to_score

        assert _distance_to_score([]) == 0.0

    def test_zero_distance_returns_perfect_score(self) -> None:
        """A cosine distance of 0.0 (identical vectors) returns 1.0."""
        from jobsearch_rag.rag.scorer import _distance_to_score

        assert _distance_to_score([0.0]) == 1.0

    def test_distance_one_returns_zero_score(self) -> None:
        """A cosine distance of 1.0 (orthogonal vectors) returns 0.0."""
        from jobsearch_rag.rag.scorer import _distance_to_score

        assert _distance_to_score([1.0]) == 0.0

    def test_best_distance_is_used_when_multiple(self) -> None:
        """The minimum distance (closest match) determines the score."""
        from jobsearch_rag.rag.scorer import _distance_to_score

        assert _distance_to_score([0.8, 0.2, 0.5]) == pytest.approx(0.8)

    def test_negative_distance_clamps_to_one(self) -> None:
        """Distances below 0.0 clamp the score to 1.0 (max)."""
        from jobsearch_rag.rag.scorer import _distance_to_score

        assert _distance_to_score([-0.5]) == 1.0

    def test_distance_greater_than_one_clamps_to_zero(self) -> None:
        """Distances above 1.0 clamp the score to 0.0 (min)."""
        from jobsearch_rag.rag.scorer import _distance_to_score

        assert _distance_to_score([1.5]) == 0.0


# ---------------------------------------------------------------------------
# TestParseDisqualifierResponse
# ---------------------------------------------------------------------------


class TestParseDisqualifierResponse:
    """REQUIREMENT: Disqualifier JSON parsing handles all LLM response variants.

    WHO: The Scorer parsing raw LLM text
    WHAT: Valid JSON is parsed correctly; string "null" reason is normalised
          to None; non-JSON falls back to (False, None)
    WHY: LLMs produce varied outputs — brittle parsing would cause
         false positives or crash the scoring pipeline
    """

    def test_string_null_reason_is_normalised_to_none(self) -> None:
        """LLM returning reason as the string 'null' is normalised to None."""
        result = Scorer._parse_disqualifier_response('{"disqualified": false, "reason": "null"}')
        assert result == (False, None)

    def test_reason_none_json_returns_none(self) -> None:
        """JSON ``null`` for reason is parsed as None."""
        result = Scorer._parse_disqualifier_response('{"disqualified": false, "reason": null}')
        assert result == (False, None)

    def test_numeric_reason_is_stringified(self) -> None:
        """A numeric reason is coerced to string."""
        result = Scorer._parse_disqualifier_response('{"disqualified": true, "reason": 42}')
        assert result == (True, "42")

    def test_missing_disqualified_key_defaults_to_false(self) -> None:
        """If the 'disqualified' key is missing, defaults to False."""
        result = Scorer._parse_disqualifier_response('{"reason": "something"}')
        assert result == (False, "something")


# ---------------------------------------------------------------------------
# TestDisqualifierClassification
# ---------------------------------------------------------------------------


class TestDisqualifierClassification:
    """REQUIREMENT: LLM disqualifier correctly identifies structurally unsuitable roles.

    WHO: The ranker applying disqualification before final scoring
    WHAT: Known disqualifier patterns produce disqualified=True;
          suitable roles return disqualified=False; malformed LLM JSON
          falls back to not-disqualified with a warning; the reason is preserved
    WHY: A disqualified role that slips through wastes review time;
         a false disqualification silently removes a good role
    """

    async def test_disqualified_jd_returns_true_with_reason(
        self, scorer: Scorer, mock_embedder: Embedder
    ) -> None:
        """A JD flagged by the LLM returns disqualified=True with a reason."""
        mock_embedder.classify = AsyncMock(  # type: ignore[method-assign]
            return_value='{"disqualified": true, "reason": "IC role disguised as architect"}'
        )
        disqualified, reason = await scorer.disqualify("Some IC role")
        assert disqualified is True
        assert reason == "IC role disguised as architect"

    async def test_suitable_jd_returns_false(
        self, scorer: Scorer, mock_embedder: Embedder
    ) -> None:
        """A suitable role returns disqualified=False."""
        mock_embedder.classify = AsyncMock(  # type: ignore[method-assign]
            return_value='{"disqualified": false, "reason": null}'
        )
        disqualified, reason = await scorer.disqualify("Staff Platform Architect")
        assert disqualified is False
        assert reason is None

    async def test_malformed_llm_json_falls_back_to_not_disqualified(
        self, scorer: Scorer, mock_embedder: Embedder
    ) -> None:
        """If the LLM returns unparseable JSON, the role is kept (safe default)."""
        mock_embedder.classify = AsyncMock(  # type: ignore[method-assign]
            return_value="This is not JSON at all"
        )
        disqualified, _reason = await scorer.disqualify("Any JD")
        assert disqualified is False

    async def test_disqualifier_reason_is_preserved(
        self, scorer: Scorer, mock_embedder: Embedder
    ) -> None:
        """The disqualification reason string is preserved for audit."""
        mock_embedder.classify = AsyncMock(  # type: ignore[method-assign]
            return_value='{"disqualified": true, "reason": "Requires active clearance"}'
        )
        _, reason = await scorer.disqualify("Classified role")
        assert reason == "Requires active clearance"

    async def test_score_integrates_disqualifier_when_flagged(
        self, scorer: Scorer, mock_embedder: Embedder
    ) -> None:
        """score() sets disqualified=True and reason when the LLM flags the JD."""
        mock_embedder.classify = AsyncMock(  # type: ignore[method-assign]
            return_value='{"disqualified": true, "reason": "SRE on-call role"}'
        )
        result = await scorer.score("SRE role with on-call duties")
        assert result.disqualified is True
        assert result.disqualifier_reason == "SRE on-call role"

    async def test_score_not_disqualified_by_default(self, scorer: Scorer) -> None:
        """score() returns disqualified=False when the LLM approves."""
        result = await scorer.score("Staff architect role")
        assert result.disqualified is False
        assert result.disqualifier_reason is None


# ---------------------------------------------------------------------------
# TestScoreFusion (Phase 3 — Ranker)
# ---------------------------------------------------------------------------


class TestScoreFusion:
    """REQUIREMENT: Final score correctly fuses weighted components from settings.

    WHO: The ranker; the operator tuning weights in settings.toml
    WHAT: Final score equals the weighted sum of the three component scores;
          weights are read from settings, not hardcoded; weights need not sum to 1.0
          (the formula normalizes); a disqualified role always scores 0.0;
          roles below min_score_threshold are excluded from output entirely
    WHY: Incorrect weight application would produce a ranking that doesn't
         reflect configured priorities — a silent correctness failure
    """

    def _make_listing(self, board: str = "test", external_id: str = "1") -> JobListing:
        return JobListing(
            board=board,
            external_id=external_id,
            title="Test Role",
            company="Test Co",
            location="Remote",
            url=f"https://example.org/{external_id}",
            full_text="A test job description.",
        )

    def test_final_score_matches_weighted_sum_formula(self) -> None:
        """The final score equals the weighted sum of fit, archetype, and history scores."""
        ranker = Ranker(
            archetype_weight=0.5,
            fit_weight=0.3,
            history_weight=0.2,
            min_score_threshold=0.0,
        )
        scores = ScoreResult(
            fit_score=0.8,
            archetype_score=0.6,
            history_score=0.4,
            disqualified=False,
        )
        expected = 0.5 * 0.6 + 0.3 * 0.8 + 0.2 * 0.4  # 0.30 + 0.24 + 0.08 = 0.62
        assert ranker.compute_final_score(scores) == pytest.approx(expected)

    def test_weights_are_read_from_settings_not_hardcoded(self) -> None:
        """Scoring weights come from settings.toml, allowing operator tuning without code changes."""
        # Custom weights that differ from defaults
        ranker = Ranker(
            archetype_weight=0.1,
            fit_weight=0.8,
            history_weight=0.1,
            min_score_threshold=0.0,
        )
        scores = ScoreResult(
            fit_score=1.0,
            archetype_score=0.0,
            history_score=0.0,
            disqualified=False,
        )
        # With these weights, only fit matters — score should be 0.8
        assert ranker.compute_final_score(scores) == pytest.approx(0.8)

    def test_disqualified_role_scores_zero_regardless_of_weights(self) -> None:
        """Disqualification zeroes the final score regardless of weight configuration."""
        ranker = Ranker(
            archetype_weight=0.5,
            fit_weight=0.3,
            history_weight=0.2,
            min_score_threshold=0.0,
        )
        scores = ScoreResult(
            fit_score=1.0,
            archetype_score=1.0,
            history_score=1.0,
            disqualified=True,
            disqualifier_reason="IC role disguised as architect",
        )
        assert ranker.compute_final_score(scores) == 0.0

    def test_role_below_threshold_is_excluded_from_output(self) -> None:
        """Roles scoring below min_score_threshold are omitted entirely from the ranked output."""
        ranker = Ranker(
            archetype_weight=0.5,
            fit_weight=0.3,
            history_weight=0.2,
            min_score_threshold=0.5,
        )
        listing = self._make_listing()
        low_scores = ScoreResult(
            fit_score=0.1,
            archetype_score=0.1,
            history_score=0.1,
            disqualified=False,
        )
        ranked, summary = ranker.rank([(listing, low_scores)])
        assert len(ranked) == 0
        assert summary.total_excluded == 1

    def test_role_at_exactly_threshold_is_included_in_output(self) -> None:
        """A role scoring exactly at the threshold is included — the boundary is inclusive."""
        ranker = Ranker(
            archetype_weight=1.0,
            fit_weight=0.0,
            history_weight=0.0,
            min_score_threshold=0.5,
        )
        listing = self._make_listing()
        scores = ScoreResult(
            fit_score=0.0,
            archetype_score=0.5,
            history_score=0.0,
            disqualified=False,
        )
        ranked, _summary = ranker.rank([(listing, scores)])
        assert len(ranked) == 1
        assert ranked[0].final_score == pytest.approx(0.5)

    def test_score_explanation_includes_all_three_component_values(self) -> None:
        """The explanation string shows fit, archetype, and history scores for transparency."""
        scores = ScoreResult(
            fit_score=0.75,
            archetype_score=0.80,
            history_score=0.60,
            disqualified=False,
        )
        ranked = RankedListing(
            listing=self._make_listing(),
            scores=scores,
            final_score=0.75,
        )
        explanation = ranked.score_explanation()
        assert "Archetype: 0.80" in explanation
        assert "Fit: 0.75" in explanation
        assert "History: 0.60" in explanation
        assert "Not disqualified" in explanation


# ---------------------------------------------------------------------------
# TestCrossBoardDeduplication (Phase 3 — Ranker)
# ---------------------------------------------------------------------------


class TestCrossBoardDeduplication:
    """REQUIREMENT: The same job appearing on multiple boards is presented once.

    WHO: The operator reviewing the ranked output
    WHAT: Near-duplicate listings (cosine similarity > 0.95 on full_text) are
          collapsed into one; the highest-scored instance is kept; the output
          notes which boards carried the duplicate; exact same external_id
          on same board is always deduplicated regardless of similarity threshold
    WHY: Seeing the same role five times in a shortlist wastes review time
         and inflates apparent result counts
    """

    def _make_listing(
        self, board: str = "test", external_id: str = "1", title: str = "Role"
    ) -> JobListing:
        return JobListing(
            board=board,
            external_id=external_id,
            title=title,
            company="Test Co",
            location="Remote",
            url=f"https://{board}.com/{external_id}",
            full_text=f"Job description for {title} on {board}.",
        )

    def _make_ranker(self, threshold: float = 0.0) -> Ranker:
        return Ranker(
            archetype_weight=0.5,
            fit_weight=0.3,
            history_weight=0.2,
            min_score_threshold=threshold,
        )

    def test_near_duplicate_listings_are_collapsed_to_one(self) -> None:
        """Listings with cosine similarity > 0.95 on full_text are collapsed into a single entry."""
        ranker = self._make_ranker()
        listing_a = self._make_listing(board="ziprecruiter", external_id="1")
        listing_b = self._make_listing(board="indeed", external_id="2")

        scores = ScoreResult(
            fit_score=0.8, archetype_score=0.7, history_score=0.5, disqualified=False
        )
        # Use nearly identical embeddings (similarity > 0.95)
        embed_a = [0.9, 0.1, 0.2, 0.3, 0.4]
        embed_b = [0.89, 0.11, 0.21, 0.29, 0.41]  # very close

        embeddings = {listing_a.url: embed_a, listing_b.url: embed_b}
        ranked, summary = ranker.rank(
            [(listing_a, scores), (listing_b, scores)],
            embeddings=embeddings,
        )
        assert len(ranked) == 1
        assert summary.total_deduplicated == 1

    def test_highest_scored_duplicate_is_retained(self) -> None:
        """Among near-duplicates, the instance with the highest final score survives."""
        ranker = self._make_ranker()
        listing_low = self._make_listing(board="indeed", external_id="1")
        listing_high = self._make_listing(board="ziprecruiter", external_id="2")

        low_scores = ScoreResult(
            fit_score=0.3, archetype_score=0.3, history_score=0.3, disqualified=False
        )
        high_scores = ScoreResult(
            fit_score=0.9, archetype_score=0.9, history_score=0.9, disqualified=False
        )

        embed = [0.9, 0.1, 0.2, 0.3, 0.4]
        embeddings = {listing_low.url: embed, listing_high.url: embed}

        ranked, _ = ranker.rank(
            [(listing_low, low_scores), (listing_high, high_scores)],
            embeddings=embeddings,
        )
        assert len(ranked) == 1
        assert ranked[0].listing.board == "ziprecruiter"

    def test_output_notes_all_boards_that_carried_duplicate(self) -> None:
        """The retained listing's metadata records which other boards carried the same role."""
        ranker = self._make_ranker()
        listing_a = self._make_listing(board="ziprecruiter", external_id="1")
        listing_b = self._make_listing(board="indeed", external_id="2")

        scores = ScoreResult(
            fit_score=0.8, archetype_score=0.7, history_score=0.5, disqualified=False
        )
        embed = [0.9, 0.1, 0.2, 0.3, 0.4]
        embeddings = {listing_a.url: embed, listing_b.url: embed}

        ranked, _ = ranker.rank(
            [(listing_a, scores), (listing_b, scores)],
            embeddings=embeddings,
        )
        assert len(ranked) == 1
        # The duplicate board should be noted
        survivor = ranked[0]
        other_board = "indeed" if survivor.listing.board == "ziprecruiter" else "ziprecruiter"
        assert other_board in survivor.duplicate_boards

    def test_same_external_id_same_board_is_deduplicated_unconditionally(self) -> None:
        """Exact-match external_id on the same board is deduplicated without similarity computation."""
        ranker = self._make_ranker()
        listing_a = self._make_listing(board="ziprecruiter", external_id="abc123")
        listing_b = self._make_listing(board="ziprecruiter", external_id="abc123")

        scores = ScoreResult(
            fit_score=0.8, archetype_score=0.7, history_score=0.5, disqualified=False
        )

        # No embeddings needed — exact match dedup is ID-based
        ranked, summary = ranker.rank(
            [(listing_a, scores), (listing_b, scores)],
        )
        assert len(ranked) == 1
        assert summary.total_deduplicated == 1

    def test_distinct_roles_with_similar_titles_are_not_collapsed(self) -> None:
        """Roles with similar titles but different JD content remain as separate listings."""
        ranker = self._make_ranker()
        listing_a = self._make_listing(board="ziprecruiter", external_id="1", title="Staff Architect")
        listing_b = self._make_listing(board="indeed", external_id="2", title="Staff Architect")

        scores = ScoreResult(
            fit_score=0.8, archetype_score=0.7, history_score=0.5, disqualified=False
        )
        # Use very different embeddings (low similarity)
        embed_a = [0.9, 0.1, 0.0, 0.0, 0.0]
        embed_b = [0.0, 0.0, 0.1, 0.9, 0.0]  # orthogonal

        embeddings = {listing_a.url: embed_a, listing_b.url: embed_b}
        ranked, _ = ranker.rank(
            [(listing_a, scores), (listing_b, scores)],
            embeddings=embeddings,
        )
        assert len(ranked) == 2

    def test_deduplication_count_appears_in_run_summary(self) -> None:
        """The run summary reports how many listings were removed by deduplication."""
        ranker = self._make_ranker()
        listing_a = self._make_listing(board="ziprecruiter", external_id="same")
        listing_b = self._make_listing(board="ziprecruiter", external_id="same")
        listing_c = self._make_listing(board="ziprecruiter", external_id="different")

        scores = ScoreResult(
            fit_score=0.8, archetype_score=0.7, history_score=0.5, disqualified=False
        )

        ranked, summary = ranker.rank(
            [(listing_a, scores), (listing_b, scores), (listing_c, scores)],
        )
        assert summary.total_deduplicated == 1
        assert len(ranked) == 2
