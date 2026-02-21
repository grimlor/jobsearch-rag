"""Indexer tests — resume chunking, archetype ingestion, and negative signal indexing.

Maps to BDD specs: TestResumeIndexing, TestArchetypeIndexing,
TestArchetypeEmbeddingSynthesis, TestGlobalRubricLoading, TestNegativeSignalIndexing

The Indexer coordinates VectorStore and Embedder to ingest documents.
VectorStore is tested with a real temp-directory instance; Embedder
is mocked since it requires a live Ollama server.
"""

from __future__ import annotations

import tempfile
import textwrap
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.indexer import Indexer, build_archetype_embedding_text
from jobsearch_rag.rag.store import VectorStore

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1, 0.2, 0.3, 0.4, 0.5]

SAMPLE_RESUME = textwrap.dedent("""\
    # Jack Pines

    Contact info line

    ## Summary

    Principal-level Systems Architect specializing in distributed systems.
    Fifteen years of experience designing cloud architectures.

    ## Core Strengths

    - Distributed Systems & Data Platforms
    - Cloud & Infrastructure
    - AI-Integrated Tooling

    ## Experience

    ### Acme Corp — Staff Engineer

    _Remote | Jan 2023 - Present_

    - Led platform architecture for streaming data systems.
    - Mentored junior engineers on system design.

    ### Beta Inc — Senior Engineer

    _Remote | Jun 2020 - Dec 2022_

    - Built ingestion pipelines for real-time analytics.

    ## Earlier Roles

    - Previous Company — Engineer (2018-2020)
""")

SAMPLE_ARCHETYPES_TOML = textwrap.dedent("""\
    [[archetypes]]
    name = "Staff Platform Architect"
    description = \"\"\"
    Defines technical strategy for distributed systems.
    Cross-team influence. Cloud-native infrastructure.
    \"\"\"

    [[archetypes]]
    name = "Principal Data Engineer"
    description = \"\"\"
    Leads data pipeline architecture and platform governance.
    Deep experience with cloud data stacks.
    \"\"\"
""")

SAMPLE_ARCHETYPES_WITH_SIGNALS_TOML = textwrap.dedent("""\
    [[archetypes]]
    name = "Staff Platform Architect"
    description = \"\"\"
    Defines technical strategy for distributed systems.
    Cross-team influence. Cloud-native infrastructure.
    \"\"\"
    signals_positive = [
        "Cross-team architecture ownership",
        "Distributed systems design",
    ]
    signals_negative = [
        "Primarily hands-on feature development",
        "No cross-team scope mentioned",
    ]

    [[archetypes]]
    name = "Principal Data Engineer"
    description = \"\"\"
    Leads data pipeline architecture and platform governance.
    Deep experience with cloud data stacks.
    \"\"\"
    signals_positive = [
        "Data pipeline architecture ownership",
        "Cloud data stack expertise",
    ]
    signals_negative = [
        "Data analyst or BI developer role",
        "Junior data engineering with no architecture",
    ]
""")

SAMPLE_RUBRIC_TOML = textwrap.dedent("""\
    [[dimensions]]
    name = "Role Scope"
    signals_negative = [
        "IC coding role disguised with architect title",
        "No cross-team influence mentioned",
    ]

    [[dimensions]]
    name = "Industry Alignment"
    signals_negative = [
        "Adtech or surveillance platform",
        "Gambling technology",
    ]
""")


@pytest.fixture
def store() -> Iterator[VectorStore]:
    """A VectorStore backed by a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield VectorStore(persist_dir=tmpdir)


@pytest.fixture
def mock_embedder() -> Embedder:
    """An Embedder with embed() mocked to return a fake vector."""
    embedder = Embedder.__new__(Embedder)
    embedder.base_url = "http://localhost:11434"
    embedder.embed_model = "nomic-embed-text"
    embedder.llm_model = "mistral:7b"
    embedder.max_retries = 3
    embedder.base_delay = 0.0
    embedder.embed = AsyncMock(return_value=FAKE_EMBEDDING)  # type: ignore[method-assign]
    return embedder


@pytest.fixture
def indexer(store: VectorStore, mock_embedder: Embedder) -> Indexer:
    """An Indexer wired to a real VectorStore and a mocked Embedder."""
    return Indexer(store=store, embedder=mock_embedder)


@pytest.fixture
def resume_path(tmp_path: Path) -> Path:
    """Write sample resume to a temp file and return its path."""
    p = tmp_path / "resume.md"
    p.write_text(SAMPLE_RESUME)
    return p


@pytest.fixture
def archetypes_path(tmp_path: Path) -> Path:
    """Write sample archetypes to a temp file and return its path."""
    p = tmp_path / "role_archetypes.toml"
    p.write_text(SAMPLE_ARCHETYPES_TOML)
    return p


@pytest.fixture
def archetypes_with_signals_path(tmp_path: Path) -> Path:
    """Write sample archetypes with signals to a temp file and return its path."""
    p = tmp_path / "role_archetypes_signals.toml"
    p.write_text(SAMPLE_ARCHETYPES_WITH_SIGNALS_TOML)
    return p


@pytest.fixture
def rubric_path(tmp_path: Path) -> Path:
    """Write sample global rubric to a temp file and return its path."""
    p = tmp_path / "global_rubric.toml"
    p.write_text(SAMPLE_RUBRIC_TOML)
    return p


# ---------------------------------------------------------------------------
# TestResumeChunking
# ---------------------------------------------------------------------------


class TestResumeChunking:
    """REQUIREMENT: Resume is chunked by ## headings for semantic coherence.

    WHO: The indexer preparing resume content for embedding
    WHAT: The resume is split on ## section headings; each chunk carries
          its heading as context; the title line (# Name) is excluded
          from chunks; nested ### headings stay with their parent section
    WHY: Chunking at section boundaries preserves semantic coherence —
         embedding a mix of "Core Strengths" and "Experience" dilutes both
    """

    async def test_resume_is_chunked_by_section_heading(
        self, indexer: Indexer, resume_path: Path
    ) -> None:
        """The resume is split on ## headings so each chunk carries coherent, section-scoped context."""
        count = await indexer.index_resume(str(resume_path))
        # Sample resume has 4 sections: Summary, Core Strengths, Experience, Earlier Roles
        assert count == 4

    async def test_each_chunk_starts_with_section_heading(
        self, indexer: Indexer, store: VectorStore, resume_path: Path
    ) -> None:
        """Each chunk starts with its ## heading for semantic context."""
        await indexer.index_resume(str(resume_path))
        result = store.get_documents("resume", ids=["resume-summary"])
        assert result["documents"][0].startswith("## Summary")

    async def test_nested_headings_stay_with_parent_section(
        self, indexer: Indexer, store: VectorStore, resume_path: Path
    ) -> None:
        """### sub-headings are included in their parent ## section's chunk."""
        await indexer.index_resume(str(resume_path))
        result = store.get_documents("resume", ids=["resume-experience"])
        doc = result["documents"][0]
        assert "### Acme Corp" in doc
        assert "### Beta Inc" in doc

    async def test_chunk_ids_are_derived_from_heading(
        self, indexer: Indexer, store: VectorStore, resume_path: Path
    ) -> None:
        """Chunk IDs are slugified from the heading for stable, readable identifiers."""
        await indexer.index_resume(str(resume_path))
        # Should be able to retrieve by predictable IDs
        result = store.get_documents("resume", ids=["resume-summary", "resume-core-strengths"])
        assert len(result["documents"]) == 2

    async def test_chunks_contain_at_least_one_complete_sentence(
        self, indexer: Indexer, store: VectorStore, resume_path: Path
    ) -> None:
        """Chunks never split mid-sentence, preserving semantic coherence for embedding."""
        await indexer.index_resume(str(resume_path))
        result = store.get_documents("resume", ids=["resume-summary"])
        doc = result["documents"][0]
        # The summary has two sentences — both should be present, no truncation
        assert "distributed systems" in doc
        assert "cloud architectures" in doc

    async def test_title_line_is_excluded_from_chunks(
        self, indexer: Indexer, store: VectorStore, resume_path: Path
    ) -> None:
        """The # Name line at the top is not its own chunk."""
        count = await indexer.index_resume(str(resume_path))
        # 4 sections, not 5 (title excluded)
        assert count == 4

    async def test_resume_with_no_section_headings_returns_zero_chunks(
        self, indexer: Indexer, store: VectorStore, tmp_path: Path
    ) -> None:
        """A resume with no ## headings produces zero chunks — _chunk_resume returns []."""
        flat_resume = tmp_path / "flat.md"
        flat_resume.write_text("# Just a Title\n\nSome text without section headings.\n")
        count = await indexer.index_resume(str(flat_resume))
        assert count == 0


# ---------------------------------------------------------------------------
# TestResumeIndexing
# ---------------------------------------------------------------------------


class TestResumeIndexing:
    """REQUIREMENT: Resume is indexed into ChromaDB before scoring can proceed.

    WHO: The scorer computing fit_score; the operator running first-time setup
    WHAT: Each chunk is embedded and stored; re-indexing replaces previous
          content; the return value confirms chunk count; metadata records source
    WHY: An empty resume collection silently produces zero fit_scores for all
         roles — a harder bug to catch than an explicit missing-index error
    """

    async def test_embedder_is_called_for_each_chunk(
        self, indexer: Indexer, mock_embedder: Embedder, resume_path: Path
    ) -> None:
        """Every chunk is embedded via the Embedder."""
        count = await indexer.index_resume(str(resume_path))
        assert mock_embedder.embed.call_count == count  # type: ignore[attr-defined]

    async def test_reindex_replaces_previous_content(
        self, indexer: Indexer, store: VectorStore, resume_path: Path
    ) -> None:
        """Re-indexing clears previous content before inserting, preventing stale chunk accumulation."""
        await indexer.index_resume(str(resume_path))
        assert store.collection_count("resume") == 4
        # Index again — count should stay the same, not double
        await indexer.index_resume(str(resume_path))
        assert store.collection_count("resume") == 4

    async def test_index_returns_chunk_count(self, indexer: Indexer, resume_path: Path) -> None:
        """The index method returns the number of chunks created for operator feedback."""
        count = await indexer.index_resume(str(resume_path))
        assert isinstance(count, int)
        assert count > 0

    async def test_chunk_metadata_records_source(
        self, indexer: Indexer, store: VectorStore, resume_path: Path
    ) -> None:
        """Each chunk's metadata records the source file for traceability."""
        await indexer.index_resume(str(resume_path))
        result = store.get_documents("resume", ids=["resume-summary"])
        assert result["metadatas"][0]["source"] == "resume"

    async def test_missing_resume_file_tells_operator_to_create_it(self, indexer: Indexer) -> None:
        """A nonexistent resume file tells the operator to create it with actionable guidance."""
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_resume("/nonexistent/resume.md")
        err = exc_info.value
        assert err.error_type == ErrorType.CONFIG
        assert err.suggestion is not None
        assert err.troubleshooting is not None
        assert len(err.troubleshooting.steps) > 0


# ---------------------------------------------------------------------------
# TestArchetypeIndexing
# ---------------------------------------------------------------------------


class TestArchetypeIndexing:
    """REQUIREMENT: Role archetypes are loaded from TOML and embedded correctly.

    WHO: The scorer computing archetype_score
    WHAT: Each archetype in role_archetypes.toml produces one ChromaDB document;
          malformed TOML raises a parse error at index time; an empty file raises
          early; whitespace in descriptions is normalized before embedding
    WHY: Missing or malformed archetypes silently score all roles equally —
         the most insidious failure mode since ranking still appears to work
    """

    async def test_each_toml_archetype_produces_one_chroma_document(
        self, indexer: Indexer, store: VectorStore, archetypes_path: Path
    ) -> None:
        """Every archetype entry in role_archetypes.toml becomes exactly one ChromaDB document."""
        count = await indexer.index_archetypes(str(archetypes_path))
        assert count == 2
        assert store.collection_count("role_archetypes") == 2

    async def test_archetype_name_is_stored_as_document_metadata(
        self, indexer: Indexer, store: VectorStore, archetypes_path: Path
    ) -> None:
        """The archetype name is stored in document metadata for score explanation and debugging."""
        await indexer.index_archetypes(str(archetypes_path))
        result = store.get_documents("role_archetypes", ids=["archetype-staff-platform-architect"])
        assert result["metadatas"][0]["name"] == "Staff Platform Architect"

    async def test_archetype_description_is_the_document_text(
        self, indexer: Indexer, store: VectorStore, archetypes_path: Path
    ) -> None:
        """The archetype description (normalized) is stored as the document text."""
        await indexer.index_archetypes(str(archetypes_path))
        result = store.get_documents("role_archetypes", ids=["archetype-staff-platform-architect"])
        doc = result["documents"][0]
        assert "distributed systems" in doc
        assert "Cloud-native" in doc

    async def test_archetype_description_whitespace_is_normalized(
        self, indexer: Indexer, mock_embedder: Embedder, archetypes_path: Path
    ) -> None:
        """Extra whitespace in descriptions is normalized before embedding."""
        await indexer.index_archetypes(str(archetypes_path))
        # Check what was passed to embed — should have no leading/trailing whitespace
        # and no excessive internal whitespace
        for call in mock_embedder.embed.call_args_list:  # type: ignore[attr-defined]
            text = call.args[0] if call.args else call.kwargs.get("text", "")
            assert text == text.strip()
            assert "  " not in text  # no double spaces
            assert "\n\n" not in text  # no double newlines

    async def test_malformed_toml_identifies_syntax_error_and_file_path(
        self, indexer: Indexer, tmp_path: Path
    ) -> None:
        """Invalid TOML syntax names the error and file path so the operator can fix the syntax."""
        bad_toml = tmp_path / "bad.toml"
        bad_toml.write_text("[[archetypes]\nname = broken")  # missing closing bracket
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_archetypes(str(bad_toml))
        err = exc_info.value
        assert err.error_type == ErrorType.PARSE
        assert err.suggestion is not None
        assert err.troubleshooting is not None

    async def test_empty_archetypes_tells_operator_to_add_entries(
        self, indexer: Indexer, tmp_path: Path
    ) -> None:
        """An empty archetypes file tells the operator to add archetype entries before searching."""
        empty = tmp_path / "empty.toml"
        empty.write_text("")
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_archetypes(str(empty))
        err = exc_info.value
        assert err.error_type == ErrorType.VALIDATION
        assert err.suggestion is not None
        assert err.troubleshooting is not None

    async def test_missing_archetypes_file_tells_operator_to_create_it(
        self, indexer: Indexer
    ) -> None:
        """A nonexistent archetypes file tells the operator to create it with actionable guidance."""
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_archetypes("/nonexistent/archetypes.toml")
        err = exc_info.value
        assert err.error_type == ErrorType.CONFIG
        assert err.suggestion is not None
        assert err.troubleshooting is not None

    async def test_reindex_replaces_previous_archetypes(
        self, indexer: Indexer, store: VectorStore, archetypes_path: Path
    ) -> None:
        """Re-indexing archetypes replaces previous content."""
        await indexer.index_archetypes(str(archetypes_path))
        assert store.collection_count("role_archetypes") == 2
        await indexer.index_archetypes(str(archetypes_path))
        assert store.collection_count("role_archetypes") == 2


# ---------------------------------------------------------------------------
# TestArchetypeEmbeddingSynthesis
# ---------------------------------------------------------------------------


class TestArchetypeEmbeddingSynthesis:
    """REQUIREMENT: Archetype embeddings synthesize description + positive signals.

    WHO: The indexer preparing archetype documents for embedding
    WHAT: When an archetype has ``signals_positive``, the embedding text
          combines the normalized description with those signals so the
          resulting vector captures both the narrative and keyword anchors;
          when signals are absent, the description alone is used
    WHY: Pure prose descriptions embed well for general similarity but miss
         specific keyword anchors the scorer needs to distinguish roles —
         e.g. "cross-team architecture ownership" as a distinct signal
    """

    def test_synthesis_includes_description_and_signals(self) -> None:
        """Synthesized text includes both description and positive signals."""
        archetype: dict[str, object] = {
            "description": "  Distributed systems architect.  ",
            "signals_positive": ["Cross-team influence", "API governance"],
        }
        result = build_archetype_embedding_text(archetype)
        assert "Distributed systems architect." in result, (
            f"Description missing from synthesized text: {result!r}"
        )
        assert "Cross-team influence" in result, (
            f"Signal 'Cross-team influence' missing from synthesized text: {result!r}"
        )
        assert "API governance" in result, (
            f"Signal 'API governance' missing from synthesized text: {result!r}"
        )

    def test_synthesis_normalizes_description_whitespace(self) -> None:
        """Extra whitespace in the description is normalized before synthesis."""
        archetype: dict[str, object] = {
            "description": "  Lots    of   whitespace   here  ",
            "signals_positive": ["Signal A"],
        }
        result = build_archetype_embedding_text(archetype)
        assert "  " not in result.split("\n")[0], (
            f"Double spaces remain in description: {result!r}"
        )

    def test_synthesis_without_signals_returns_description_only(self) -> None:
        """An archetype with no signals_positive returns just the normalized description."""
        archetype: dict[str, object] = {"description": "  A simple role.  "}
        result = build_archetype_embedding_text(archetype)
        assert result == "A simple role.", f"Expected clean description only, got: {result!r}"

    def test_synthesis_with_empty_signals_returns_description_only(self) -> None:
        """An archetype with an empty signals_positive list returns just the description."""
        archetype: dict[str, object] = {"description": "A role.", "signals_positive": []}
        result = build_archetype_embedding_text(archetype)
        assert result == "A role.", f"Expected description only for empty signals, got: {result!r}"

    async def test_index_archetypes_uses_synthesized_text(
        self, indexer: Indexer, store: VectorStore, archetypes_with_signals_path: Path
    ) -> None:
        """index_archetypes() stores the synthesized text (description + signals) as the document."""
        await indexer.index_archetypes(str(archetypes_with_signals_path))
        result = store.get_documents("role_archetypes", ids=["archetype-staff-platform-architect"])
        doc = result["documents"][0]
        assert "Cross-team architecture ownership" in doc, (
            f"Positive signal missing from stored document: {doc!r}"
        )
        assert "distributed systems" in doc.lower(), (
            f"Description content missing from stored document: {doc!r}"
        )


# ---------------------------------------------------------------------------
# TestGlobalRubricLoading
# ---------------------------------------------------------------------------


class TestGlobalRubricLoading:
    """REQUIREMENT: Global rubric TOML is loaded and parsed for negative signal extraction.

    WHO: The indexer building the negative_signals collection
    WHAT: Each dimension's signals_negative entries are loaded from
          global_rubric.toml and used as source material for the
          negative signals collection; malformed TOML produces a
          parse error; a missing file produces a config error
    WHY: The global rubric defines universal evaluation criteria that
         apply to all listings regardless of archetype — without it,
         negative scoring is incomplete
    """

    async def test_rubric_signals_are_loaded_and_indexed(
        self,
        indexer: Indexer,
        store: VectorStore,
        rubric_path: Path,
        archetypes_with_signals_path: Path,
    ) -> None:
        """Negative signals from the global rubric are indexed into the negative_signals collection."""
        count = await indexer.index_negative_signals(
            str(rubric_path), str(archetypes_with_signals_path)
        )
        # 4 rubric signals + 4 archetype signals = 8 total
        assert count == 8, f"Expected 8 negative signals (4 rubric + 4 archetype), got {count}"
        assert store.collection_count("negative_signals") == 8

    async def test_missing_rubric_file_tells_operator_to_create_it(
        self, indexer: Indexer, archetypes_with_signals_path: Path
    ) -> None:
        """A nonexistent rubric file tells the operator to create it with actionable guidance."""
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_negative_signals(
                "/nonexistent/rubric.toml", str(archetypes_with_signals_path)
            )
        err = exc_info.value
        assert err.error_type == ErrorType.CONFIG, f"Expected CONFIG error, got {err.error_type}"
        assert err.suggestion is not None
        assert err.troubleshooting is not None

    async def test_malformed_rubric_identifies_syntax_error(
        self, indexer: Indexer, tmp_path: Path, archetypes_with_signals_path: Path
    ) -> None:
        """Invalid rubric TOML syntax produces a PARSE error."""
        bad_rubric = tmp_path / "bad_rubric.toml"
        bad_rubric.write_text("[[dimensions]\nname = broken")
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_negative_signals(
                str(bad_rubric), str(archetypes_with_signals_path)
            )
        err = exc_info.value
        assert err.error_type == ErrorType.PARSE, f"Expected PARSE error, got {err.error_type}"
        assert err.suggestion is not None


# ---------------------------------------------------------------------------
# TestNegativeSignalIndexing
# ---------------------------------------------------------------------------


class TestNegativeSignalIndexing:
    """REQUIREMENT: Negative signals from rubric and archetypes are indexed for penalty scoring.

    WHO: The scorer computing negative_score for each listing
    WHAT: Negative signals from both global_rubric.toml dimensions and
          per-archetype signals_negative are combined into a single
          negative_signals ChromaDB collection; each signal is individually
          embedded; metadata tracks the source; re-indexing is idempotent
    WHY: A JD about adtech should score high on negative signals regardless
         of which archetype it matches — the global rubric catches universal
         dealbreakers while archetype negatives catch role-specific mismatches
    """

    async def test_rubric_and_archetype_negatives_are_combined(
        self,
        indexer: Indexer,
        store: VectorStore,
        rubric_path: Path,
        archetypes_with_signals_path: Path,
    ) -> None:
        """Both rubric dimensions and archetype signals_negative contribute to the collection."""
        count = await indexer.index_negative_signals(
            str(rubric_path), str(archetypes_with_signals_path)
        )
        assert count > 0, "Expected at least one negative signal indexed"
        assert store.collection_count("negative_signals") == count

    async def test_each_signal_is_individually_embedded(
        self,
        indexer: Indexer,
        mock_embedder: Embedder,
        rubric_path: Path,
        archetypes_with_signals_path: Path,
    ) -> None:
        """Each negative signal text is embedded individually via the Embedder."""
        count = await indexer.index_negative_signals(
            str(rubric_path), str(archetypes_with_signals_path)
        )
        # Each signal should trigger one embed() call
        assert mock_embedder.embed.call_count == count, (  # type: ignore[attr-defined]
            f"Expected {count} embed calls, got {mock_embedder.embed.call_count}"  # type: ignore[attr-defined]
        )

    async def test_signal_metadata_records_source(
        self,
        indexer: Indexer,
        store: VectorStore,
        rubric_path: Path,
        archetypes_with_signals_path: Path,
    ) -> None:
        """Each signal's metadata records whether it came from a rubric dimension or archetype."""
        await indexer.index_negative_signals(str(rubric_path), str(archetypes_with_signals_path))
        # Get all documents via the underlying collection
        collection = store._get_existing_collection("negative_signals")
        result = collection.get(include=["metadatas"])
        metadatas = result["metadatas"]
        assert metadatas is not None, "Expected metadatas in collection result"
        for meta in metadatas:
            source = str(meta.get("source", ""))
            assert source.startswith(("rubric:", "archetype:")), (
                f"Signal source should start with 'rubric:' or 'archetype:', got: {source!r}"
            )

    async def test_reindex_replaces_previous_signals(
        self,
        indexer: Indexer,
        store: VectorStore,
        rubric_path: Path,
        archetypes_with_signals_path: Path,
    ) -> None:
        """Re-indexing negative signals replaces previous content, not appends."""
        count1 = await indexer.index_negative_signals(
            str(rubric_path), str(archetypes_with_signals_path)
        )
        count2 = await indexer.index_negative_signals(
            str(rubric_path), str(archetypes_with_signals_path)
        )
        assert count1 == count2, f"Signal count changed on re-index: {count1} → {count2}"
        assert store.collection_count("negative_signals") == count1

    async def test_empty_rubric_dimensions_indexes_only_archetype_negatives(
        self,
        indexer: Indexer,
        store: VectorStore,
        tmp_path: Path,
        archetypes_with_signals_path: Path,
    ) -> None:
        """An empty rubric (no dimensions) still indexes archetype negative signals."""
        empty_rubric = tmp_path / "empty_rubric.toml"
        empty_rubric.write_text("# No dimensions\n")
        count = await indexer.index_negative_signals(
            str(empty_rubric), str(archetypes_with_signals_path)
        )
        # 4 archetype negatives (2 per archetype x 2 archetypes)
        assert count == 4, f"Expected 4 archetype-only negative signals, got {count}"

    async def test_missing_archetypes_file_tells_operator(
        self, indexer: Indexer, rubric_path: Path
    ) -> None:
        """A nonexistent archetypes file produces a CONFIG error with actionable guidance."""
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_negative_signals(str(rubric_path), "/nonexistent/archetypes.toml")
        err = exc_info.value
        assert err.error_type == ErrorType.CONFIG, f"Expected CONFIG error, got {err.error_type}"

    async def test_malformed_archetypes_toml_produces_actionable_parse_error(
        self, indexer: Indexer, rubric_path: Path, tmp_path: Path
    ) -> None:
        """
        When the archetypes file contains invalid TOML syntax
        Then a PARSE error is raised naming the file and suggesting a fix
        """
        bad_toml = tmp_path / "bad_archetypes.toml"
        bad_toml.write_text('[[archetypes]]\nname = "broken\n')

        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_negative_signals(str(rubric_path), str(bad_toml))
        err = exc_info.value
        assert err.error_type == ErrorType.PARSE, (
            f"Expected PARSE error for malformed archetypes TOML, got {err.error_type}"
        )
        assert "role_archetypes" in err.error.lower() or "syntax" in err.error.lower(), (
            f"Error should mention the file or syntax issue: {err.error!r}"
        )

    async def test_zero_signals_from_both_sources_returns_empty_collection(
        self, indexer: Indexer, store: VectorStore, tmp_path: Path
    ) -> None:
        """
        When both rubric dimensions and archetypes have no signals_negative
        Then the method returns 0 and the collection is empty (no crash)
        """
        # Rubric with dimensions but no negative signals
        empty_rubric = tmp_path / "no_neg_rubric.toml"
        empty_rubric.write_text(
            '[[dimensions]]\nname = "Culture"\nsignals_positive = ["async-first"]\n'
        )
        # Archetypes with no negative signals
        empty_archetypes = tmp_path / "no_neg_archetypes.toml"
        empty_archetypes.write_text(
            '[[archetypes]]\nname = "Architect"\ndescription = "Designs systems."\n'
        )

        count = await indexer.index_negative_signals(str(empty_rubric), str(empty_archetypes))
        assert count == 0, f"Expected 0 negative signals when both sources are empty, got {count}"
        assert store.collection_count("negative_signals") == 0, (
            "Collection should be empty when no negative signals exist"
        )
