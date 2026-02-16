"""Indexer tests — resume chunking and archetype ingestion.

Maps to BDD specs: TestResumeIndexing, TestArchetypeIndexing

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
from jobsearch_rag.rag.indexer import Indexer
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
        result = store.get_documents(
            "resume", ids=["resume-summary", "resume-core-strengths"]
        )
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

    async def test_index_returns_chunk_count(
        self, indexer: Indexer, resume_path: Path
    ) -> None:
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

    async def test_missing_resume_file_raises_config_error(
        self, indexer: Indexer
    ) -> None:
        """A nonexistent resume file raises a CONFIG error with the path."""
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_resume("/nonexistent/resume.md")
        assert exc_info.value.error_type == ErrorType.CONFIG


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
        result = store.get_documents(
            "role_archetypes", ids=["archetype-staff-platform-architect"]
        )
        assert result["metadatas"][0]["name"] == "Staff Platform Architect"

    async def test_archetype_description_is_the_document_text(
        self, indexer: Indexer, store: VectorStore, archetypes_path: Path
    ) -> None:
        """The archetype description (normalized) is stored as the document text."""
        await indexer.index_archetypes(str(archetypes_path))
        result = store.get_documents(
            "role_archetypes", ids=["archetype-staff-platform-architect"]
        )
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

    async def test_malformed_toml_raises_parse_error(
        self, indexer: Indexer, tmp_path: Path
    ) -> None:
        """Invalid TOML syntax raises a PARSE error during indexing."""
        bad_toml = tmp_path / "bad.toml"
        bad_toml.write_text("[[archetypes]\nname = broken")  # missing closing bracket
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_archetypes(str(bad_toml))
        assert exc_info.value.error_type == ErrorType.PARSE

    async def test_empty_archetypes_file_raises_validation_error(
        self, indexer: Indexer, tmp_path: Path
    ) -> None:
        """An empty archetypes file raises a VALIDATION error."""
        empty = tmp_path / "empty.toml"
        empty.write_text("")
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_archetypes(str(empty))
        assert exc_info.value.error_type == ErrorType.VALIDATION

    async def test_missing_archetypes_file_raises_config_error(
        self, indexer: Indexer
    ) -> None:
        """A nonexistent archetypes file raises a CONFIG error."""
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_archetypes("/nonexistent/archetypes.toml")
        assert exc_info.value.error_type == ErrorType.CONFIG

    async def test_reindex_replaces_previous_archetypes(
        self, indexer: Indexer, store: VectorStore, archetypes_path: Path
    ) -> None:
        """Re-indexing archetypes replaces previous content."""
        await indexer.index_archetypes(str(archetypes_path))
        assert store.collection_count("role_archetypes") == 2
        await indexer.index_archetypes(str(archetypes_path))
        assert store.collection_count("role_archetypes") == 2
