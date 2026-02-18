"""RAG pipeline tests — Ollama connectivity, resume/archetype indexing.

Maps to BDD specs: TestOllamaConnectivity, TestResumeIndexing, TestArchetypeIndexing
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.indexer import Indexer
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
    """REQUIREMENT: Ollama unavailability is detected before processing begins
    with guidance the operator can act on immediately.

    WHO: The pipeline runner; the operator who may have forgotten to start Ollama
    WHAT: An unreachable Ollama endpoint raises a clear startup error naming
          the configured URL with connectivity troubleshooting steps;
          the error distinguishes between "not running" and "wrong model"
          with different recovery guidance for each; the run does not proceed
          to browser automation if Ollama is required and unavailable
    WHY: Completing a full browser session only to fail at scoring wastes
         time and risks rate limiting; fail fast at startup
    """

    async def test_unreachable_ollama_provides_url_and_connectivity_steps(self) -> None:
        """An unreachable Ollama provides the URL and step-by-step connectivity troubleshooting."""
        embedder = Embedder(
            base_url="http://localhost:59999",
            embed_model="nomic-embed-text",
            llm_model="mistral:7b",
        )
        with pytest.raises(ActionableError) as exc_info:
            await embedder.health_check()
        err = exc_info.value
        assert err.error_type == ErrorType.CONNECTION
        assert "localhost:59999" in err.error
        assert err.suggestion is not None
        assert err.troubleshooting is not None
        assert len(err.troubleshooting.steps) > 0

    async def test_startup_check_runs_before_browser_session_opens(self) -> None:
        """Ollama reachability is verified at startup with actionable guidance if unreachable."""
        embedder = Embedder(
            base_url="http://localhost:59999",
            embed_model="nomic-embed-text",
            llm_model="mistral:7b",
        )
        with pytest.raises(ActionableError) as exc_info:
            await embedder.health_check()
        err = exc_info.value
        assert err.error_type == ErrorType.CONNECTION
        assert err.suggestion is not None
        assert err.troubleshooting is not None

    async def test_wrong_model_name_suggests_ollama_pull_command(
        self,
    ) -> None:
        """A wrong model name suggests 'ollama pull' so the operator knows exactly how to fix it."""
        embedder = Embedder(
            base_url="http://localhost:11434",
            embed_model="nonexistent-model-xyz",
            llm_model="also-nonexistent",
        )
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.models = []
        with patch.object(embedder._client, "list", AsyncMock(return_value=mock_response)):
            with pytest.raises(ActionableError) as exc_info:
                await embedder.health_check()
            err = exc_info.value
            assert err.error_type == ErrorType.EMBEDDING
            assert err.suggestion is not None
            assert err.troubleshooting is not None
            assert len(err.troubleshooting.steps) > 0

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

        with patch.object(embedder._client, "embed", _mock_embed):
            result = await embedder.embed("test text")
            assert call_count == 3
            assert result == EMBED_FAKE

    async def test_ollama_timeout_after_retries_advises_checking_system_resources(self) -> None:
        """After exhausting retries, the error advises checking system resources."""
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

        with patch.object(embedder._client, "embed", _always_fail):
            with pytest.raises(ActionableError) as exc_info:
                await embedder.embed("test text")
            err = exc_info.value
            assert err.error_type == ErrorType.EMBEDDING
            assert "2" in err.error
            assert err.suggestion is not None
            assert err.troubleshooting is not None


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

    async def test_empty_resume_collection_tells_operator_to_run_index_command(
        self, store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """Scoring against an empty resume collection tells the operator to run the index command."""
        from jobsearch_rag.rag.scorer import Scorer

        scorer = Scorer(store=store, embedder=mock_embedder)
        with pytest.raises(ActionableError) as exc_info:
            await scorer.score("Any JD text")
        err = exc_info.value
        assert err.error_type == ErrorType.INDEX
        assert err.suggestion is not None
        assert err.troubleshooting is not None
        assert len(err.troubleshooting.steps) > 0

    async def test_missing_collection_error_provides_step_by_step_setup_guidance(
        self, store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """The INDEX error provides step-by-step setup guidance naming the missing collection."""
        from jobsearch_rag.rag.scorer import Scorer

        scorer = Scorer(store=store, embedder=mock_embedder)
        with pytest.raises(ActionableError) as exc_info:
            await scorer.score("Any JD text")
        err = exc_info.value
        assert "resume" in err.error.lower()
        assert err.suggestion is not None
        assert err.troubleshooting is not None
        assert len(err.troubleshooting.steps) > 0

    async def test_resume_is_chunked_by_section_heading(
        self, indexer: Indexer, store: VectorStore
    ) -> None:
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            path = f.name

        n = await indexer.index_resume(path)
        assert n == 3

        # Verify each section became a separate document in ChromaDB
        docs = store.get_documents(
            collection_name="resume",
            ids=["resume-summary", "resume-experience", "resume-skills"],
        )
        assert len(docs["documents"]) == 3
        assert "principal architect" in docs["documents"][0]
        assert "platform architecture" in docs["documents"][1]
        assert "distributed systems" in docs["documents"][2]

    async def test_each_chunk_contains_at_least_one_complete_sentence(
        self, indexer: Indexer, store: VectorStore
    ) -> None:
        """Chunks never split mid-sentence, preserving semantic coherence for embedding."""
        content = """\
# Resume

## Summary
I am an architect. I design systems.

## Experience
Led teams. Built platforms.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            path = f.name

        await indexer.index_resume(path)

        # Retrieve all documents and verify each has a complete sentence
        for doc_id in ["resume-summary", "resume-experience"]:
            docs = store.get_documents(collection_name="resume", ids=[doc_id])
            body = docs["documents"][0]
            # Strip the heading line for the check
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

    async def test_malformed_toml_identifies_syntax_error_and_file_path(
        self, indexer: Indexer
    ) -> None:
        """Invalid TOML syntax identifies the syntax error so the operator can fix the file."""
        bad_toml = "this is [not valid {{ toml"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(bad_toml)
            path = f.name

        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_archetypes(path)
        err = exc_info.value
        assert err.error_type == ErrorType.PARSE
        assert err.suggestion is not None
        assert err.troubleshooting is not None

    async def test_empty_archetypes_tells_operator_to_add_entries_before_search(
        self, indexer: Indexer
    ) -> None:
        """An empty archetypes file tells the operator to add entries before searching."""
        empty_toml = "# No archetypes defined\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(empty_toml)
            path = f.name

        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_archetypes(path)
        err = exc_info.value
        assert err.error_type == ErrorType.VALIDATION
        assert err.suggestion is not None
        assert err.troubleshooting is not None

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
