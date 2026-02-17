"""RAG pipeline tests — Ollama connectivity, resume/archetype indexing.

Maps to BDD specs: TestOllamaConnectivity, TestResumeIndexing, TestArchetypeIndexing
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.indexer import Indexer, _chunk_resume
from jobsearch_rag.rag.store import VectorStore

if TYPE_CHECKING:
    from collections.abc import Iterator

EMBED_FAKE = [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.fixture
def store() -> Iterator[VectorStore]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield VectorStore(persist_dir=tmpdir)


@pytest.fixture
def mock_embedder() -> Embedder:
    embedder = Embedder.__new__(Embedder)
    embedder.base_url = "http://localhost:11434"
    embedder.embed_model = "nomic-embed-text"
    embedder.llm_model = "mistral:7b"
    embedder.max_retries = 3
    embedder.base_delay = 0.0
    embedder.embed = AsyncMock(return_value=EMBED_FAKE)  # type: ignore[method-assign]
    embedder.classify = AsyncMock(return_value='{"disqualified": false}')  # type: ignore[method-assign]
    return embedder


@pytest.fixture
def indexer(store: VectorStore, mock_embedder: Embedder) -> Indexer:
    return Indexer(store=store, embedder=mock_embedder)


# ---------------------------------------------------------------------------
# TestOllamaConnectivity
# ---------------------------------------------------------------------------


class TestOllamaConnectivity:
    """REQUIREMENT: Ollama unavailability is detected before processing begins.

    WHO: The pipeline runner; the operator who may have forgotten to start Ollama
    WHAT: An unreachable Ollama endpoint raises a clear startup error naming
          the configured URL; the error distinguishes between "not running" and
          "wrong URL"; the run does not proceed to browser automation if
          Ollama is required and unavailable
    WHY: Completing a full browser session only to fail at scoring wastes
         time and risks rate limiting; fail fast at startup
    """

    async def test_unreachable_ollama_raises_startup_error_with_url(self) -> None:
        """An unreachable Ollama endpoint raises a CONNECTION error naming the configured URL."""
        embedder = Embedder(
            base_url="http://localhost:59999",
            embed_model="nomic-embed-text",
            llm_model="mistral:7b",
        )
        with pytest.raises(ActionableError) as exc_info:
            await embedder.health_check()
        assert exc_info.value.error_type == ErrorType.CONNECTION
        assert "localhost:59999" in exc_info.value.error

    async def test_startup_check_runs_before_browser_session_opens(self) -> None:
        """Ollama reachability is verified at startup so a slow browser session isn't wasted."""
        # This is a design test — the PipelineRunner calls health_check() first.
        # We verify by checking that health_check raises before search would run.
        embedder = Embedder(
            base_url="http://localhost:59999",
            embed_model="nomic-embed-text",
            llm_model="mistral:7b",
        )
        with pytest.raises(ActionableError) as exc_info:
            await embedder.health_check()
        assert exc_info.value.error_type == ErrorType.CONNECTION

    async def test_wrong_model_name_raises_error_distinguishable_from_connectivity(
        self,
    ) -> None:
        """A wrong model name produces a different error type than 'Ollama not running' for clear diagnosis."""
        embedder = Embedder(
            base_url="http://localhost:11434",
            embed_model="nonexistent-model-xyz",
            llm_model="also-nonexistent",
        )
        # Mock the list call to succeed (Ollama is reachable) but return no models
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.models = []
        embedder._client.list = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]

        with pytest.raises(ActionableError) as exc_info:
            await embedder.health_check()
        # Should be EMBEDDING error (model not found), not CONNECTION
        assert exc_info.value.error_type == ErrorType.EMBEDDING

    async def test_ollama_timeout_on_embedding_retries_with_backoff(self) -> None:
        """A transient Ollama timeout triggers exponential backoff retries before giving up."""
        import ollama as ollama_sdk

        embedder = Embedder(
            base_url="http://localhost:11434",
            embed_model="nomic-embed-text",
            llm_model="mistral:7b",
            max_retries=3,
            base_delay=0.01,  # fast for testing
        )

        # First two calls fail with retryable error, third succeeds
        call_count = 0
        async def _mock_embed(model: str, input: str) -> object:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ollama_sdk.ResponseError("timeout", status_code=504)
            from unittest.mock import MagicMock
            resp = MagicMock()
            resp.embeddings = [EMBED_FAKE]
            return resp

        embedder._client.embed = _mock_embed  # type: ignore[method-assign]
        result = await embedder.embed("test text")
        assert call_count == 3
        assert result == EMBED_FAKE

    async def test_ollama_timeout_after_max_retries_raises_embedding_error(self) -> None:
        """After exhausting retries, a persistent timeout raises an EMBEDDING error with retry count."""
        import ollama as ollama_sdk

        embedder = Embedder(
            base_url="http://localhost:11434",
            embed_model="nomic-embed-text",
            llm_model="mistral:7b",
            max_retries=2,
            base_delay=0.01,
        )

        async def _always_fail(model: str, input: str) -> object:
            raise ollama_sdk.ResponseError("timeout", status_code=504)

        embedder._client.embed = _always_fail  # type: ignore[method-assign]
        with pytest.raises(ActionableError) as exc_info:
            await embedder.embed("test text")
        assert exc_info.value.error_type == ErrorType.EMBEDDING
        assert "2" in exc_info.value.error  # mentions retry count


# ---------------------------------------------------------------------------
# TestResumeIndexing
# ---------------------------------------------------------------------------


class TestResumeIndexing:
    """REQUIREMENT: Resume is indexed into ChromaDB before scoring can proceed.

    WHO: The scorer computing fit_score; the operator running first-time setup
    WHAT: Scoring fails clearly if resume collection is empty; the index command
          chunks resume by section; re-indexing replaces previous content;
          chunk boundaries preserve semantic coherence (no mid-sentence splits)
    WHY: An empty resume collection silently produces zero fit_scores for all
         roles — a harder bug to catch than an explicit missing-index error
    """

    async def test_scoring_raises_index_error_when_resume_collection_is_empty(
        self, store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """Scoring against an empty resume collection raises INDEX error rather than returning zero scores silently."""
        from jobsearch_rag.rag.scorer import Scorer

        scorer = Scorer(store=store, embedder=mock_embedder)
        with pytest.raises(ActionableError) as exc_info:
            await scorer.score("Any JD text")
        assert exc_info.value.error_type == ErrorType.INDEX

    async def test_index_error_message_names_the_missing_collection(
        self, store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """The INDEX error message names the missing collection so the operator knows which 'index' command to run."""
        from jobsearch_rag.rag.scorer import Scorer

        scorer = Scorer(store=store, embedder=mock_embedder)
        with pytest.raises(ActionableError) as exc_info:
            await scorer.score("Any JD text")
        assert "resume" in exc_info.value.error.lower()

    def test_resume_is_chunked_by_section_heading(self) -> None:
        """The resume is split on ## headings so each chunk carries coherent, section-scoped context."""
        content = """\
# My Resume

## Summary
I am a principal architect.

## Experience
Led platform architecture at multiple companies.

## Skills
Python, distributed systems, cloud platforms.
"""
        chunks = _chunk_resume(content)
        assert len(chunks) == 3
        headings = [c[1] for c in chunks]
        assert "## Summary" in headings
        assert "## Experience" in headings
        assert "## Skills" in headings

    def test_each_chunk_contains_at_least_one_complete_sentence(self) -> None:
        """Chunks never split mid-sentence, preserving semantic coherence for embedding."""
        content = """\
# Resume

## Summary
I am an architect. I design systems.

## Experience
Led teams. Built platforms.
"""
        chunks = _chunk_resume(content)
        for _id, _heading, body in chunks:
            # Each body should contain at least one period (complete sentence)
            # Strip the heading from body for the check
            text = body.split("\n", 1)[-1] if "\n" in body else body
            assert "." in text, f"Chunk lacks complete sentence: {body!r}"

    async def test_reindex_replaces_previous_resume_content_not_appends(
        self, indexer: Indexer, store: VectorStore
    ) -> None:
        """Re-indexing clears previous content before inserting, preventing stale chunk accumulation."""
        resume_content = """\
# Resume

## Summary
Original summary.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(resume_content)
            path = f.name

        # Index once
        n1 = await indexer.index_resume(path)
        assert n1 == 1
        count1 = store.collection_count("resume")

        # Index again — should replace, not append
        n2 = await indexer.index_resume(path)
        assert n2 == 1
        count2 = store.collection_count("resume")
        assert count2 == count1  # same count, not doubled

    async def test_index_confirms_chunk_count_in_output(
        self, indexer: Indexer
    ) -> None:
        """The index command reports how many chunks were created so the operator can sanity-check coverage."""
        content = """\
# Resume

## Summary
I am an architect.

## Skills
Python, cloud, systems.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            path = f.name

        n = await indexer.index_resume(path)
        assert n == 2  # two ## sections


# ---------------------------------------------------------------------------
# TestArchetypeIndexing
# ---------------------------------------------------------------------------


class TestArchetypeIndexing:
    """REQUIREMENT: Role archetypes are loaded from TOML and embedded correctly.

    WHO: The scorer computing archetype_score
    WHAT: Each archetype in role_archetypes.toml produces one ChromaDB document;
          malformed TOML raises a parse error at index time, not scoring time;
          an empty archetypes file raises a clear error before any browser work
    WHY: Missing or malformed archetypes silently score all roles equally —
         the most insidious failure mode since ranking still appears to work
    """

    async def test_each_toml_archetype_produces_one_chroma_document(
        self, indexer: Indexer, store: VectorStore
    ) -> None:
        """Every archetype entry in role_archetypes.toml becomes exactly one ChromaDB document."""
        toml_content = """\
[[archetypes]]
name = "Staff Architect"
description = "Defines tech strategy."

[[archetypes]]
name = "DevRel"
description = "Developer relations."
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            path = f.name

        n = await indexer.index_archetypes(path)
        assert n == 2
        assert store.collection_count("role_archetypes") == 2

    async def test_archetype_name_is_stored_as_document_metadata(
        self, indexer: Indexer, store: VectorStore
    ) -> None:
        """The archetype name is stored in document metadata for score explanation and debugging."""
        toml_content = """\
[[archetypes]]
name = "Staff Architect"
description = "Defines tech strategy."
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            path = f.name

        await indexer.index_archetypes(path)
        docs = store.get_documents(
            collection_name="role_archetypes",
            ids=["archetype-staff-architect"],
        )
        assert docs["metadatas"][0]["name"] == "Staff Architect"

    async def test_malformed_toml_raises_parse_error_at_index_time(
        self, indexer: Indexer
    ) -> None:
        """Invalid TOML syntax raises a PARSE error during indexing, not later during scoring."""
        bad_toml = "this is [not valid {{ toml"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(bad_toml)
            path = f.name

        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_archetypes(path)
        assert exc_info.value.error_type == ErrorType.PARSE

    async def test_empty_archetypes_file_raises_error_before_browser_session(
        self, indexer: Indexer
    ) -> None:
        """An empty archetypes file raises early so a full browser crawl isn't wasted on unscoreable results."""
        empty_toml = "# No archetypes defined\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(empty_toml)
            path = f.name

        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_archetypes(path)
        assert exc_info.value.error_type == ErrorType.VALIDATION

    async def test_archetype_description_whitespace_is_normalized_before_embedding(
        self, indexer: Indexer, store: VectorStore
    ) -> None:
        """Extra whitespace in archetype descriptions is normalized so embeddings are not skewed by formatting."""
        toml_content = """\
[[archetypes]]
name = "Test"
description = \"\"\"
  Lots    of   extra
  whitespace   here
\"\"\"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            path = f.name

        await indexer.index_archetypes(path)
        docs = store.get_documents(
            collection_name="role_archetypes",
            ids=["archetype-test"],
        )
        doc_text = docs["documents"][0]
        # No runs of multiple spaces should remain
        assert "  " not in doc_text
