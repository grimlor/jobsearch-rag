"""
Best archetype match surfacing tests.

Spec classes:
    TestBestArchetypeMatch

The scoring pipeline already queries the role_archetypes ChromaDB collection
and keeps the best cosine score, but discards *which* archetype matched.
This module specifies that the best-matching archetype name surfaces through
ScoreResult, score explanation, CSV export, JD file metadata, and review display.
"""

# Public API surface (from src/jobsearch_rag/rag/scorer):
#   ScoreResult(fit_score, archetype_score, history_score, disqualified,
#               disqualifier_reason, comp_score, negative_score, culture_score)
#   Scorer(store: VectorStore, embedder: Embedder, disqualify_on_llm_flag: bool)
#   scorer.score(jd_text: str) -> ScoreResult
#
# Public API surface (from src/jobsearch_rag/pipeline/ranker):
#   RankedListing(listing, scores, final_score, duplicate_boards)
#   RankedListing.score_explanation() -> str
#
# Public API surface (from src/jobsearch_rag/export/csv_export):
#   CSVExporter().export(listings, output_path, summary=None)
#
# Public API surface (from src/jobsearch_rag/export/jd_files):
#   JDFileExporter().export(listings, output_dir, summary=None, cleanup_stale=True)
#
# Public API surface (from src/jobsearch_rag/pipeline/review):
#   ReviewSession(ranked_listings, recorder, jd_dir=...)
#   review.format_listing(ranked, rank=, total=) -> str

from __future__ import annotations

import csv
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from jobsearch_rag.adapters.base import JobListing
from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.export.csv_export import CSVExporter
from jobsearch_rag.export.jd_files import JDFileExporter
from jobsearch_rag.pipeline.ranker import RankedListing
from jobsearch_rag.pipeline.review import ReviewSession
from jobsearch_rag.rag.decisions import DecisionRecorder
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.scorer import Scorer, ScoreResult
from jobsearch_rag.rag.store import VectorStore
from tests.conftest import make_mock_ollama_client, make_test_ollama_config

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Fake 5D embeddings with directional meaning for archetype similarity tests.
# ARCHITECT and ARCH_JD are close; DATA_ENG is orthogonal.
# ---------------------------------------------------------------------------
EMBED_ARCHITECT = [0.9, 0.1, 0.2, 0.0, 0.3]
EMBED_DATA_ENG = [0.1, 0.8, 0.1, 0.7, 0.0]
EMBED_ARCH_JD = [0.85, 0.15, 0.25, 0.05, 0.28]  # close to ARCHITECT
EMBED_ARCH_JD_CLOSER = [0.89, 0.11, 0.21, 0.01, 0.30]  # even closer — guarantees higher score
EMBED_DATA_JD = [0.15, 0.75, 0.15, 0.65, 0.05]  # close to DATA_ENG
EMBED_UNRELATED_JD = [0.0, 0.0, 0.1, 0.9, 0.9]  # far from everything


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_embed_response(embedder: Embedder, vector: list[float]) -> None:
    """Change the embedding vector returned by the mock ollama client."""
    response = MagicMock()
    response.embeddings = [vector]
    response.prompt_eval_count = 42
    embedder._client.embed.return_value = response  # type: ignore[union-attr]


def _set_embed_side_effect(embedder: Embedder, vectors: list[list[float]]) -> None:
    """Set a sequence of different embedding vectors for successive embed calls."""
    embedder._client.embed.side_effect = [  # type: ignore[union-attr]
        MagicMock(embeddings=[v], prompt_eval_count=42) for v in vectors
    ]


def _make_listing(
    *,
    title: str = "Staff Architect",
    company: str = "Acme Corp",
    board: str = "ziprecruiter",
    external_id: str = "ext-001",
    url: str = "https://example.org/job/1",
    full_text: str = "Build distributed systems at scale.",
    comp_min: float | None = None,
    comp_max: float | None = None,
) -> JobListing:
    return JobListing(
        board=board,
        external_id=external_id,
        title=title,
        company=company,
        location="Remote",
        url=url,
        full_text=full_text,
        comp_min=comp_min,
        comp_max=comp_max,
    )


def _make_ranked(
    *,
    title: str = "Staff Architect",
    company: str = "Acme Corp",
    board: str = "ziprecruiter",
    external_id: str = "ext-001",
    url: str = "https://example.org/job/1",
    full_text: str = "Build distributed systems at scale.",
    final_score: float = 0.78,
    fit: float = 0.74,
    archetype: float = 0.81,
    history: float = 0.62,
    comp: float = 0.5,
    best_archetype: str | None = "AI Systems Engineer",
    disqualified: bool = False,
    disqualifier_reason: str | None = None,
    duplicate_boards: list[str] | None = None,
) -> RankedListing:
    listing = _make_listing(
        title=title,
        company=company,
        board=board,
        external_id=external_id,
        url=url,
        full_text=full_text,
    )
    scores = ScoreResult(
        fit_score=fit,
        archetype_score=archetype,
        history_score=history,
        comp_score=comp,
        disqualified=disqualified,
        disqualifier_reason=disqualifier_reason,
        best_archetype=best_archetype,
    )
    return RankedListing(
        listing=listing,
        scores=scores,
        final_score=final_score,
        duplicate_boards=duplicate_boards or [],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> Iterator[VectorStore]:
    """A VectorStore backed by a temporary directory."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        s = VectorStore(persist_dir=tmpdir)
        yield s
        s.close()


@pytest.fixture
def mock_embedder() -> Embedder:
    """Real Embedder with ollama client stubbed at the I/O boundary."""
    mock_client = make_mock_ollama_client(
        embed_vector=EMBED_ARCH_JD,
        classify_response='{"disqualified": false, "reason": null}',
    )
    embedder = Embedder(make_test_ollama_config(max_retries=1, base_delay=0.0))
    embedder._client = mock_client  # type: ignore[attr-defined]
    return embedder


@pytest.fixture
def populated_store(store: VectorStore) -> VectorStore:
    """A VectorStore with resume and two archetypes seeded."""
    # Resume collection
    store.add_documents(
        collection_name="resume",
        ids=["resume-summary"],
        documents=["Principal architect specializing in distributed systems."],
        embeddings=[EMBED_ARCHITECT],
        metadatas=[{"source": "resume", "section": "Summary"}],
    )
    # Archetype collection — two archetypes with distinct embeddings
    store.add_documents(
        collection_name="role_archetypes",
        ids=["archetype-ai-systems-engineer", "archetype-data-platform-lead"],
        documents=[
            "AI Systems Engineer: ML infrastructure, distributed training, model deployment.",
            "Data Platform Lead: data pipelines, warehousing, analytics infrastructure.",
        ],
        embeddings=[EMBED_ARCHITECT, EMBED_DATA_ENG],
        metadatas=[
            {"name": "AI Systems Engineer", "source": "role_archetypes"},
            {"name": "Data Platform Lead", "source": "role_archetypes"},
        ],
    )
    return store


@pytest.fixture
def scorer(populated_store: VectorStore, mock_embedder: Embedder) -> Scorer:
    """A Scorer wired to a populated VectorStore and mocked Embedder."""
    return Scorer(store=populated_store, embedder=mock_embedder)


# ---------------------------------------------------------------------------
# TestBestArchetypeMatch
# ---------------------------------------------------------------------------


class TestBestArchetypeMatch:
    """
    REQUIREMENT: The scoring pipeline surfaces the name of the best-matching
    archetype per listing, flowing through ScoreResult, score explanation,
    CSV export, JD file metadata, and review display.

    WHO: The operator reviewing scored listings — needs to know which target
         role type each JD most resembles, for tuning and tailoring.
    WHAT: (1) ScoreResult includes a best_archetype field containing the name
              metadata from the closest archetype document.
          (2) When multiple archetypes exist, the archetype with the smallest
              distance (highest similarity) is selected as best_archetype.
          (3) When multiple chunks produce different best archetypes, the
              archetype from the chunk with the highest archetype_score wins.
          (4) When a single archetype exists in the collection, that archetype
              is always selected.
          (5) score_explanation() includes the best archetype name when present:
              "Archetype: 0.81 (AI Systems Engineer) · Fit: 0.74 · ..."
          (6) score_explanation() omits the parenthetical when best_archetype
              is None.
          (7) CSV export includes a 'best_archetype' column after 'archetype_score'.
          (8) JD file metadata includes a '**Best Archetype:**' line after the
              archetype score line.
          (9) Review display shows the best archetype name alongside the archetype
              score in the component breakdown.
    WHY: Without knowing which archetype matched, the operator cannot tune
         archetypes, understand why a listing ranked where it did, or tailor
         resume/cover letters per archetype. The data is already returned by
         ChromaDB but currently discarded.

    MOCK BOUNDARY:
        Mock:  mock_embedder fixture (Ollama HTTP — the only I/O boundary)
        Real:  Scorer instance, VectorStore (ChromaDB in temp dir),
               RankedListing, CSVExporter, JDFileExporter, ReviewSession
        Never: Construct ScoreResult directly for scorer tests (scenarios 1-4)
               - always obtain via scorer.score(); never mock the Scorer itself
        Note:  Export/explanation/review tests (scenarios 5-9) construct
               ScoreResult directly - these test presentation, not scoring.
               Collections are seeded via store.add_documents with controlled
               embeddings, following the established test_scorer.py pattern.
    """

    # --- Scorer: best_archetype surfaces through scorer.score() ---

    async def test_score_result_includes_best_archetype_name(
        self, scorer: Scorer, mock_embedder: Embedder
    ) -> None:
        """
        Given a role_archetypes collection with 'AI Systems Engineer' and 'Data Platform Lead'
        When scorer.score() processes a JD matching 'AI Systems Engineer'
        Then the returned ScoreResult.best_archetype equals 'AI Systems Engineer'
        """
        # Given: a scorer with two archetypes; JD embedding close to AI Systems Engineer
        _set_embed_response(mock_embedder, EMBED_ARCH_JD)

        # When: a JD is scored
        result = await scorer.score(
            "Design and deploy ML inference pipelines on distributed GPU clusters"
        )

        # Then: the best archetype name is surfaced
        assert result.best_archetype == "AI Systems Engineer", (
            f"Expected best_archetype='AI Systems Engineer', got {result.best_archetype!r}"
        )

    async def test_best_archetype_is_closest_match_when_multiple_exist(
        self, scorer: Scorer, mock_embedder: Embedder
    ) -> None:
        """
        Given a role_archetypes collection with multiple archetypes
        When scorer.score() processes a JD closer to 'Data Platform Lead'
        Then best_archetype is 'Data Platform Lead' (smallest distance)
        """
        # Given: a JD embedding close to the Data Platform Lead archetype
        _set_embed_response(mock_embedder, EMBED_DATA_JD)

        # When: a JD is scored
        result = await scorer.score(
            "Build and maintain Snowflake data warehouse and Airflow ETL pipelines"
        )

        # Then: the closest archetype wins
        assert result.best_archetype == "Data Platform Lead", (
            f"Expected best_archetype='Data Platform Lead', got {result.best_archetype!r}"
        )

    async def test_multi_chunk_jd_selects_archetype_from_best_scoring_chunk(
        self, populated_store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """
        Given a JD that is long enough to be split into multiple chunks
        When scorer.score() processes it and different chunks match different archetypes
        Then best_archetype comes from the chunk with the highest archetype_score
        """
        # Given: a scorer with a very small max_embed_chars to force chunking,
        # and zero overlap so chunks don't multiply
        mock_embedder.max_embed_chars = 60
        scorer = Scorer(store=populated_store, embedder=mock_embedder, chunk_overlap=0)

        # Chunk 1 produces DATA_JD embedding (closer to Data Platform Lead, score ~0.996)
        # Chunk 2 produces ARCH_JD_CLOSER embedding (closer to AI Systems Engineer, score ~0.9998)
        _set_embed_side_effect(mock_embedder, [EMBED_DATA_JD, EMBED_ARCH_JD_CLOSER])

        # When: a 2-chunk JD is scored (61-120 chars at 60 char chunk_size)
        long_jd = "A" * 61 + " " + "B" * 57  # total = 119 chars → 2 chunks
        result = await scorer.score(long_jd)

        # Then: the best archetype comes from the higher-scoring chunk
        assert result.best_archetype == "AI Systems Engineer", (
            f"Expected best_archetype='AI Systems Engineer' from the higher-scoring chunk, "
            f"got {result.best_archetype!r}"
        )

    async def test_single_archetype_collection_always_selects_that_archetype(
        self, store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """
        Given a role_archetypes collection with exactly one archetype
        When scorer.score() is called
        Then best_archetype is the name of that single archetype
        """
        # Given: resume + single archetype seeded
        store.add_documents(
            collection_name="resume",
            ids=["resume-summary"],
            documents=["Principal architect specializing in distributed systems."],
            embeddings=[EMBED_ARCHITECT],
            metadatas=[{"source": "resume", "section": "Summary"}],
        )
        store.add_documents(
            collection_name="role_archetypes",
            ids=["archetype-solo"],
            documents=["Solo Archetype: the only role type available."],
            embeddings=[EMBED_ARCHITECT],
            metadatas=[{"name": "Solo Archetype", "source": "role_archetypes"}],
        )
        scorer = Scorer(store=store, embedder=mock_embedder)
        _set_embed_response(mock_embedder, EMBED_UNRELATED_JD)

        # When: any JD is scored
        result = await scorer.score("Completely unrelated role description")

        # Then: the only archetype is always selected
        assert result.best_archetype == "Solo Archetype", (
            f"Expected best_archetype='Solo Archetype' (the only one), "
            f"got {result.best_archetype!r}"
        )

    # --- Score explanation includes archetype name ---

    def test_score_explanation_includes_archetype_name_when_present(self) -> None:
        """
        Given a ScoreResult with best_archetype='AI Systems Engineer'
        When score_explanation() is called on a RankedListing
        Then the explanation contains 'Archetype: 0.81 (AI Systems Engineer)'
        """
        # Given: a RankedListing with a known best_archetype
        ranked = _make_ranked(archetype=0.81, best_archetype="AI Systems Engineer")

        # When: score explanation is generated
        explanation = ranked.score_explanation()

        # Then: the archetype name appears in parentheses after the score
        assert "Archetype: 0.81 (AI Systems Engineer)" in explanation, (
            f"Expected 'Archetype: 0.81 (AI Systems Engineer)' in explanation, "
            f"got: {explanation!r}"
        )

    def test_score_explanation_omits_parenthetical_when_archetype_is_none(self) -> None:
        """
        Given a ScoreResult with best_archetype=None
        When score_explanation() is called on a RankedListing
        Then the explanation contains 'Archetype: 0.81' with no parenthetical
        """
        # Given: a RankedListing without a best_archetype
        ranked = _make_ranked(archetype=0.81, best_archetype=None)

        # When: score explanation is generated
        explanation = ranked.score_explanation()

        # Then: only the score appears, no parenthetical
        assert "Archetype: 0.81" in explanation, (
            f"Expected 'Archetype: 0.81' in explanation, got: {explanation!r}"
        )
        assert "(None)" not in explanation, (
            f"Explanation should not contain '(None)', got: {explanation!r}"
        )
        assert "Archetype: 0.81 (" not in explanation, (
            f"Explanation should not have parenthetical after score, got: {explanation!r}"
        )

    # --- CSV export ---

    def test_csv_export_includes_best_archetype_column(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """
        Given a list of RankedListings with best_archetype values
        When CSVExporter.export() is called
        Then the output CSV has a 'best_archetype' column after 'archetype_score'
             and cell values match each listing's best_archetype
        """
        # Given: two ranked listings with different best archetypes
        listings = [
            _make_ranked(
                external_id="ext-001",
                best_archetype="AI Systems Engineer",
                final_score=0.85,
            ),
            _make_ranked(
                external_id="ext-002",
                best_archetype="Data Platform Lead",
                final_score=0.72,
            ),
        ]
        output_path = str(tmp_path / "results.csv")  # type: ignore[operator]

        # When: CSV is exported
        CSVExporter().export(listings, output_path)

        # Then: the CSV contains a best_archetype column with correct values
        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert "best_archetype" in (reader.fieldnames or []), (
            f"Expected 'best_archetype' column in CSV header, got columns: {reader.fieldnames}"
        )
        # Verify column order: best_archetype follows archetype_score
        headers = list(reader.fieldnames or [])
        arch_idx = headers.index("archetype_score")
        best_idx = headers.index("best_archetype")
        assert best_idx == arch_idx + 1, (
            f"Expected 'best_archetype' immediately after 'archetype_score', "
            f"but archetype_score is at index {arch_idx} and best_archetype at {best_idx}"
        )
        # Verify cell values
        assert rows[0]["best_archetype"] == "AI Systems Engineer", (
            f"Expected first row best_archetype='AI Systems Engineer', "
            f"got {rows[0]['best_archetype']!r}"
        )
        assert rows[1]["best_archetype"] == "Data Platform Lead", (
            f"Expected second row best_archetype='Data Platform Lead', "
            f"got {rows[1]['best_archetype']!r}"
        )

    # --- JD file metadata ---

    def test_jd_file_includes_best_archetype_line(self, tmp_path: pytest.TempPathFactory) -> None:
        """
        Given a RankedListing with best_archetype='Data Platform Lead'
        When JDFileExporter writes the JD file
        Then the Score section includes '- **Best Archetype:** Data Platform Lead'
             after the archetype score line
        """
        # Given: a ranked listing with a best archetype
        ranked = _make_ranked(
            best_archetype="Data Platform Lead",
            external_id="ext-jd-001",
        )
        output_dir = str(tmp_path / "jds")  # type: ignore[operator]

        # When: JD file is exported
        paths = JDFileExporter().export([ranked], output_dir)

        # Then: the file contains the best archetype line
        assert len(paths) == 1, f"Expected 1 JD file, got {len(paths)}"
        content = paths[0].read_text()
        assert "- **Best Archetype:** Data Platform Lead" in content, (
            f"Expected '- **Best Archetype:** Data Platform Lead' in JD file, "
            f"got content:\n{content}"
        )
        # Verify order: best archetype line appears after archetype score line
        lines = content.split("\n")
        arch_score_lines = [i for i, line in enumerate(lines) if "**Archetype Score:**" in line]
        best_arch_lines = [i for i, line in enumerate(lines) if "**Best Archetype:**" in line]
        assert arch_score_lines and best_arch_lines, (
            "Expected both '**Archetype Score:**' and '**Best Archetype:**' lines in JD file"
        )
        assert best_arch_lines[0] == arch_score_lines[0] + 1, (
            f"Expected '**Best Archetype:**' immediately after '**Archetype Score:**', "
            f"but archetype score at line {arch_score_lines[0]} and "
            f"best archetype at line {best_arch_lines[0]}"
        )

    # --- Review display ---

    def test_review_display_shows_best_archetype(
        self, vector_store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """
        Given a RankedListing with best_archetype='AI Systems Engineer'
        When format_listing() renders the terminal display
        Then the output includes 'Archetype: 0.81 (AI Systems Engineer)'
        """
        # Given: a review session with a ranked listing that has a best archetype
        ranked = _make_ranked(archetype=0.81, best_archetype="AI Systems Engineer")

        recorder = DecisionRecorder(
            store=vector_store,
            embedder=mock_embedder,
            decisions_dir=tempfile.mkdtemp(),
        )
        session = ReviewSession(ranked_listings=[ranked], recorder=recorder)

        # When: the listing is formatted for display
        output = session.format_listing(ranked, rank=1, total=1)

        # Then: the archetype name appears alongside the score
        assert "Archetype: 0.81 (AI Systems Engineer)" in output, (
            f"Expected 'Archetype: 0.81 (AI Systems Engineer)' in review display, got:\n{output}"
        )

    # --- JD file with None archetype ---

    def test_jd_file_omits_best_archetype_line_when_none(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        """
        Given a RankedListing with best_archetype=None
        When JDFileExporter writes the JD file
        Then the Score section does not include a '**Best Archetype:**' line
        """
        # Given: a ranked listing with no best archetype
        ranked = _make_ranked(
            best_archetype=None,
            external_id="ext-jd-none",
        )
        output_dir = str(tmp_path / "jds_none")  # type: ignore[operator]

        # When: JD file is exported
        paths = JDFileExporter().export([ranked], output_dir)

        # Then: no best archetype line appears
        assert len(paths) == 1, f"Expected 1 JD file, got {len(paths)}"
        content = paths[0].read_text()
        assert "**Best Archetype:**" not in content, (
            f"Expected no '**Best Archetype:**' line when best_archetype is None, "
            f"but found it in:\n{content}"
        )

    # --- Archetype without metadata falls back ---

    async def test_archetype_without_metadata_falls_back_to_none(
        self, store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """
        Given a role_archetypes document was added without metadata
        When scorer.score() processes a JD
        Then best_archetype is None (graceful fallback)
        """
        # Given: resume seeded, one archetype added WITHOUT metadata
        store.add_documents(
            collection_name="resume",
            ids=["resume-summary"],
            documents=["Principal architect specializing in distributed systems."],
            embeddings=[EMBED_ARCHITECT],
            metadatas=[{"source": "resume", "section": "Summary"}],
        )
        store.add_documents(
            collection_name="role_archetypes",
            ids=["archetype-no-meta"],
            documents=["Some archetype without metadata."],
            embeddings=[EMBED_ARCHITECT],
        )
        scorer = Scorer(store=store, embedder=mock_embedder)
        _set_embed_response(mock_embedder, EMBED_ARCH_JD)

        # When: a JD is scored
        result = await scorer.score("Design ML inference pipelines")

        # Then: best_archetype falls back to None
        assert result.best_archetype is None, (
            f"Expected best_archetype=None when metadata is absent, got {result.best_archetype!r}"
        )

    # --- Empty archetypes collection error path ---

    async def test_empty_archetypes_collection_raises_index_error(
        self, store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """
        Given a resume collection is populated but role_archetypes is empty
        When scorer.score() is called
        Then an ActionableError of type INDEX is raised
        """
        # Given: resume seeded but no archetypes
        store.add_documents(
            collection_name="resume",
            ids=["resume-summary"],
            documents=["Principal architect specializing in distributed systems."],
            embeddings=[EMBED_ARCHITECT],
            metadatas=[{"source": "resume", "section": "Summary"}],
        )
        store.reset_collection("role_archetypes")
        scorer = Scorer(store=store, embedder=mock_embedder)

        # When/Then: scoring raises an INDEX error about empty archetypes
        with pytest.raises(ActionableError) as exc_info:
            await scorer.score("Any job description")

        assert exc_info.value.error_type == ErrorType.INDEX, (
            f"Expected INDEX error type, got {exc_info.value.error_type}"
        )
        assert "role_archetypes" in str(exc_info.value), (
            f"Error should mention 'role_archetypes', got: {exc_info.value}"
        )
