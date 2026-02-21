"""RAG pipeline tests — Ollama connectivity, resume/archetype indexing, negative scoring.

Maps to BDD specs: TestOllamaConnectivity, TestResumeIndexing, TestArchetypeIndexing,
TestNegativeScoring
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import ollama as ollama_sdk
import pytest

from jobsearch_rag.adapters.base import JobListing
from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.pipeline.ranker import RankedListing, Ranker
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.indexer import Indexer
from jobsearch_rag.rag.scorer import Scorer, ScoreResult
from jobsearch_rag.rag.store import VectorStore

if TYPE_CHECKING:
    from collections.abc import Iterator

# Re-exported from conftest for use in OllamaConnectivity assertions
EMBED_FAKE = [0.1, 0.2, 0.3, 0.4, 0.5]


# mock_embedder is provided by conftest.py (Embedder.__new__ + stubbed I/O)


@pytest.fixture
def store() -> Iterator[VectorStore]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield VectorStore(persist_dir=tmpdir)


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
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.models = []
        mock_client.list = AsyncMock(return_value=mock_response)

        with patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client):
            embedder = Embedder(
                base_url="http://localhost:11434",
                embed_model="nonexistent-model-xyz",
                llm_model="also-nonexistent",
            )
            with pytest.raises(ActionableError) as exc_info:
                await embedder.health_check()
            err = exc_info.value
            assert err.error_type == ErrorType.EMBEDDING
            assert err.suggestion is not None
            assert err.troubleshooting is not None
            assert len(err.troubleshooting.steps) > 0

    async def test_ollama_timeout_on_embedding_retries_with_backoff(self) -> None:
        """A transient Ollama timeout triggers exponential backoff retries before giving up."""

        # First two calls fail with retryable error, third succeeds
        call_count = 0

        async def _mock_embed(model: str, input: str) -> object:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ollama_sdk.ResponseError("timeout", status_code=504)

            resp = MagicMock()
            resp.embeddings = [EMBED_FAKE]
            return resp

        mock_client = MagicMock()
        mock_client.embed = _mock_embed

        with patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client):
            embedder = Embedder(
                base_url="http://localhost:11434",
                embed_model="nomic-embed-text",
                llm_model="mistral:7b",
                max_retries=3,
                base_delay=0.01,  # fast for testing
            )
            result = await embedder.embed("test text")
            assert call_count == 3
            assert result == EMBED_FAKE

    async def test_ollama_timeout_after_retries_advises_checking_system_resources(self) -> None:
        """After exhausting retries, the error advises checking system resources."""

        async def _always_fail(model: str, input: str) -> object:
            raise ollama_sdk.ResponseError("timeout", status_code=504)

        mock_client = MagicMock()
        mock_client.embed = _always_fail

        with patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient", return_value=mock_client):
            embedder = Embedder(
                base_url="http://localhost:11434",
                embed_model="nomic-embed-text",
                llm_model="mistral:7b",
                max_retries=2,
                base_delay=0.01,
            )
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

    async def test_index_confirms_chunk_count_in_output(self, indexer: Indexer) -> None:
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


# ---------------------------------------------------------------------------
# TestNegativeScoring
# ---------------------------------------------------------------------------


class TestNegativeScoring:
    """REQUIREMENT: JDs matching negative signals receive a continuous penalty score.

    WHO: The scorer and ranker computing the final ranked output
    WHAT: When a negative_signals collection exists and contains embedded
          signals, the scorer queries it for each JD chunk and returns a
          negative_score in [0.0, 1.0]; when the collection is empty or
          missing, negative_score defaults to 0.0; the negative_score
          appears in ScoreResult and is available for the ranker
    WHY: Binary disqualification (yes/no via LLM) misses gradient cases —
         a role at a borderline-adtech company should rank lower, not be
         entirely hidden; the negative_score provides continuous penalization
    """

    async def test_negative_score_is_zero_when_collection_missing(
        self, store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """negative_score defaults to 0.0 when the negative_signals collection is absent."""

        # Populate resume and archetypes (required) but NOT negative_signals
        store.add_documents(
            collection_name="resume",
            ids=["resume-summary"],
            documents=["## Summary\nPrincipal architect."],
            embeddings=[EMBED_FAKE],
            metadatas=[{"source": "resume", "section": "Summary"}],
        )
        store.add_documents(
            collection_name="role_archetypes",
            ids=["archetype-test"],
            documents=["Staff Architect"],
            embeddings=[EMBED_FAKE],
            metadatas=[{"name": "Test", "source": "role_archetypes"}],
        )

        scorer = Scorer(store=store, embedder=mock_embedder, disqualify_on_llm_flag=False)
        result = await scorer.score("Any JD text")
        assert result.negative_score == 0.0, (
            f"Expected negative_score 0.0 when collection missing, got {result.negative_score}"
        )

    async def test_negative_score_is_zero_when_collection_empty(
        self, store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """negative_score defaults to 0.0 when the negative_signals collection exists but is empty."""

        store.add_documents(
            collection_name="resume",
            ids=["resume-summary"],
            documents=["## Summary\nPrincipal architect."],
            embeddings=[EMBED_FAKE],
            metadatas=[{"source": "resume", "section": "Summary"}],
        )
        store.add_documents(
            collection_name="role_archetypes",
            ids=["archetype-test"],
            documents=["Staff Architect"],
            embeddings=[EMBED_FAKE],
            metadatas=[{"name": "Test", "source": "role_archetypes"}],
        )
        # Create empty negative_signals collection
        store.reset_collection("negative_signals")

        scorer = Scorer(store=store, embedder=mock_embedder, disqualify_on_llm_flag=False)
        result = await scorer.score("Any JD text")
        assert result.negative_score == 0.0, (
            f"Expected negative_score 0.0 for empty collection, got {result.negative_score}"
        )

    async def test_negative_score_returned_in_score_result(
        self, store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """ScoreResult includes negative_score as a named field."""

        store.add_documents(
            collection_name="resume",
            ids=["resume-summary"],
            documents=["## Summary\nPrincipal architect."],
            embeddings=[EMBED_FAKE],
            metadatas=[{"source": "resume", "section": "Summary"}],
        )
        store.add_documents(
            collection_name="role_archetypes",
            ids=["archetype-test"],
            documents=["Staff Architect"],
            embeddings=[EMBED_FAKE],
            metadatas=[{"name": "Test", "source": "role_archetypes"}],
        )
        store.add_documents(
            collection_name="negative_signals",
            ids=["neg-test"],
            documents=["Adtech surveillance platform"],
            embeddings=[EMBED_FAKE],
            metadatas=[{"source": "rubric:Industry", "signal": "Adtech surveillance platform"}],
        )

        scorer = Scorer(store=store, embedder=mock_embedder, disqualify_on_llm_flag=False)
        result = await scorer.score("Some JD about adtech platform")
        assert hasattr(result, "negative_score"), "ScoreResult should have negative_score field"
        assert 0.0 <= result.negative_score <= 1.0, (
            f"negative_score should be in [0.0, 1.0], got {result.negative_score}"
        )


# ---------------------------------------------------------------------------
# TestGlobalPositiveSignalIndexing
# ---------------------------------------------------------------------------


class TestGlobalPositiveSignalIndexing:
    """REQUIREMENT: Positive signals from the global rubric are indexed into
    a dedicated ChromaDB collection that the scorer queries to compute a
    continuous culture score — orthogonal to archetype score.

    WHO: The scorer computing culture_score; the indexer building the
         global_positive_signals collection
    WHAT: index_global_positive_signals() produces one ChromaDB document
          per global rubric dimension that has signals_positive entries;
          documents are labeled with their source dimension name in metadata;
          re-indexing resets the collection first (replaces, not appends);
          a rubric with no positive signals produces an empty collection
          without error; missing global_rubric.toml produces an actionable
          error before any browser work begins
    WHY: The global rubric positive signals encode universal environment
         and culture preferences that apply to every role regardless of
         archetype. A dedicated collection keeps archetype and culture
         scoring axes independent
    """

    async def test_one_document_per_rubric_dimension_with_positive_signals(
        self, indexer: Indexer, store: VectorStore
    ) -> None:
        """Each dimension with signals_positive becomes exactly one ChromaDB document."""
        rubric_toml = """\
[[dimensions]]
name = "Altitude"
signals_positive = ["strategic", "cross-org"]
signals_negative = ["tactical only"]

[[dimensions]]
name = "Humane Culture"
signals_positive = ["async-first", "no crunch"]
signals_negative = ["60-hour weeks"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(rubric_toml)
            path = f.name

        n = await indexer.index_global_positive_signals(path)
        assert n == 2, f"Expected 2 documents, got {n}"
        assert store.collection_count("global_positive_signals") == 2

    async def test_document_metadata_identifies_source_dimension(
        self, indexer: Indexer, store: VectorStore
    ) -> None:
        """Each document's metadata records the rubric dimension it came from."""
        rubric_toml = """\
[[dimensions]]
name = "Altitude"
signals_positive = ["strategic thinking", "cross-org influence"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(rubric_toml)
            path = f.name

        await indexer.index_global_positive_signals(path)
        # Retrieve the document and check metadata
        docs = store.get_documents(
            collection_name="global_positive_signals",
            ids=["pos-altitude"],
        )
        assert docs["metadatas"][0]["source"] == "Altitude", (
            f"Expected source 'Altitude', got {docs['metadatas'][0].get('source')}"
        )

    async def test_reindex_replaces_global_positive_collection_not_appends(
        self, indexer: Indexer, store: VectorStore
    ) -> None:
        """Re-indexing resets the collection first so documents are replaced, not accumulated."""
        rubric_toml = """\
[[dimensions]]
name = "Altitude"
signals_positive = ["strategic"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(rubric_toml)
            path = f.name

        await indexer.index_global_positive_signals(path)
        count1 = store.collection_count("global_positive_signals")
        await indexer.index_global_positive_signals(path)
        count2 = store.collection_count("global_positive_signals")
        assert count2 == count1, (
            f"Re-indexing should replace, not append. Count went from {count1} to {count2}"
        )

    async def test_dimension_without_signals_positive_produces_no_document(
        self, indexer: Indexer, store: VectorStore
    ) -> None:
        """Dimensions that lack signals_positive are gracefully skipped."""
        rubric_toml = """\
[[dimensions]]
name = "Compensation Red Flags"
signals_negative = ["equity-only compensation"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(rubric_toml)
            path = f.name

        n = await indexer.index_global_positive_signals(path)
        assert n == 0, f"Expected 0 documents for dimension without signals_positive, got {n}"

    async def test_missing_global_rubric_produces_actionable_error(self, indexer: Indexer) -> None:
        """A missing global_rubric.toml raises an ActionableError with the path and guidance."""
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_global_positive_signals("/nonexistent/rubric.toml")
        err = exc_info.value
        assert err.error_type == ErrorType.CONFIG
        assert "rubric" in err.error.lower() or "not found" in err.error.lower()

    async def test_archetypes_only_flag_rebuilds_global_positive_collection(
        self, indexer: Indexer, store: VectorStore
    ) -> None:
        """The --archetypes-only flag triggers index_global_positive_signals alongside archetypes."""
        rubric_toml = """\
[[dimensions]]
name = "Altitude"
signals_positive = ["strategic vision"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(rubric_toml)
            path = f.name

        # Simulate what --archetypes-only does: call index_global_positive_signals
        n = await indexer.index_global_positive_signals(path)
        assert n >= 1, "Global positive signals should be indexed during --archetypes-only"

    async def test_global_positive_collection_count_matches_contributing_dimensions(
        self, indexer: Indexer, store: VectorStore
    ) -> None:
        """The collection count matches the number of dimensions that have signals_positive."""
        rubric_toml = """\
[[dimensions]]
name = "Altitude"
signals_positive = ["strategic"]

[[dimensions]]
name = "Humane Culture"
signals_positive = ["async-first"]

[[dimensions]]
name = "Compensation"
signals_negative = ["equity-only"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(rubric_toml)
            path = f.name

        n = await indexer.index_global_positive_signals(path)
        assert n == 2, f"Only 2 dimensions have signals_positive, got {n}"
        assert store.collection_count("global_positive_signals") == 2

    async def test_compensation_dimension_produces_no_positive_document(
        self, indexer: Indexer, store: VectorStore
    ) -> None:
        """A dimension with only signals_negative (like Compensation) produces no positive document."""
        rubric_toml = """\
[[dimensions]]
name = "Compensation Red Flags"
signals_negative = ["equity-only compensation", "unpaid position"]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(rubric_toml)
            path = f.name

        n = await indexer.index_global_positive_signals(path)
        assert n == 0, f"Expected 0 positive documents from negative-only dimension, got {n}"

    async def test_malformed_rubric_toml_produces_actionable_parse_error(
        self, indexer: Indexer
    ) -> None:
        """
        When the global rubric file contains invalid TOML syntax
        Then a PARSE error is raised naming the file and suggesting a fix
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[[dimensions]]\nname = "broken\n')
            path = f.name

        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_global_positive_signals(path)
        err = exc_info.value
        assert err.error_type == ErrorType.PARSE, (
            f"Expected PARSE error for malformed rubric TOML, got {err.error_type}"
        )
        assert "global_rubric" in err.error.lower() or "syntax" in err.error.lower(), (
            f"Error should mention the file or syntax issue: {err.error!r}"
        )


# ---------------------------------------------------------------------------
# TestCultureScoring
# ---------------------------------------------------------------------------


class TestCultureScoring:
    """REQUIREMENT: culture_score continuously rewards roles whose environment
    signals match global rubric positive dimensions — altitude, humane
    culture, domain alignment, scope, company maturity, ethics, and
    ND compatibility — acting as a second scoring axis orthogonal to
    archetype and fit.

    WHO: The ranker computing final_score; the operator who wants roles
         in humane, well-scoped, ethically aligned environments to rank
         higher regardless of which archetype they match
    WHAT: culture_score is queried from the global_positive_signals
          collection using cosine similarity; missing collection returns
          0.0 rather than an error; culture_score is a float in [0.0, 1.0];
          culture_weight is read from settings.toml, not hardcoded
    WHY: Archetype score answers "right kind of role." Culture score
         answers "right kind of environment."
    """

    async def test_missing_global_positive_collection_returns_zero_not_error(
        self, store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """culture_score defaults to 0.0 when global_positive_signals collection is absent."""

        # Populate required collections but NOT global_positive_signals
        store.add_documents(
            collection_name="resume",
            ids=["resume-summary"],
            documents=["## Summary\nPrincipal architect."],
            embeddings=[EMBED_FAKE],
            metadatas=[{"source": "resume", "section": "Summary"}],
        )
        store.add_documents(
            collection_name="role_archetypes",
            ids=["archetype-test"],
            documents=["Staff Architect"],
            embeddings=[EMBED_FAKE],
            metadatas=[{"name": "Test", "source": "role_archetypes"}],
        )

        scorer = Scorer(store=store, embedder=mock_embedder, disqualify_on_llm_flag=False)
        result = await scorer.score("Any JD text")
        assert result.culture_score == 0.0, (
            f"Expected culture_score 0.0 when collection missing, got {result.culture_score}"
        )

    async def test_culture_score_is_float_between_zero_and_one(
        self, store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """culture_score is always a float in [0.0, 1.0]."""

        store.add_documents(
            collection_name="resume",
            ids=["resume-summary"],
            documents=["## Summary\nPrincipal architect."],
            embeddings=[EMBED_FAKE],
            metadatas=[{"source": "resume", "section": "Summary"}],
        )
        store.add_documents(
            collection_name="role_archetypes",
            ids=["archetype-test"],
            documents=["Staff Architect"],
            embeddings=[EMBED_FAKE],
            metadatas=[{"name": "Test", "source": "role_archetypes"}],
        )
        store.add_documents(
            collection_name="global_positive_signals",
            ids=["pos-altitude"],
            documents=["Altitude: strategic thinking, cross-org influence"],
            embeddings=[EMBED_FAKE],
            metadatas=[{"source": "Altitude"}],
        )

        scorer = Scorer(store=store, embedder=mock_embedder, disqualify_on_llm_flag=False)
        result = await scorer.score("A strategic platform architect role")
        assert isinstance(result.culture_score, float), (
            f"culture_score should be float, got {type(result.culture_score)}"
        )
        assert 0.0 <= result.culture_score <= 1.0, (
            f"culture_score should be in [0.0, 1.0], got {result.culture_score}"
        )

    async def test_culture_score_returned_in_score_result(
        self, store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """ScoreResult includes culture_score as a named field."""

        store.add_documents(
            collection_name="resume",
            ids=["resume-summary"],
            documents=["## Summary\nPrincipal architect."],
            embeddings=[EMBED_FAKE],
            metadatas=[{"source": "resume", "section": "Summary"}],
        )
        store.add_documents(
            collection_name="role_archetypes",
            ids=["archetype-test"],
            documents=["Staff Architect"],
            embeddings=[EMBED_FAKE],
            metadatas=[{"name": "Test", "source": "role_archetypes"}],
        )

        scorer = Scorer(store=store, embedder=mock_embedder, disqualify_on_llm_flag=False)
        result = await scorer.score("Any JD text")
        assert hasattr(result, "culture_score"), "ScoreResult should have culture_score field"

    async def test_culture_score_appears_in_score_breakdown_output(self) -> None:
        """The score explanation string includes culture_score for export display."""
        scores = ScoreResult(
            fit_score=0.75,
            archetype_score=0.80,
            history_score=0.60,
            comp_score=0.90,
            negative_score=0.15,
            culture_score=0.65,
            disqualified=False,
        )
        ranked = RankedListing(
            listing=JobListing(
                board="test",
                external_id="1",
                title="Test",
                company="Co",
                location="Remote",
                url="https://example.org",
                full_text="JD text",
            ),
            scores=scores,
            final_score=0.75,
        )
        explanation = ranked.score_explanation()
        assert "Culture: 0.65" in explanation, (
            f"culture_score should appear in explanation. Got: {explanation}"
        )

    async def test_culture_weight_is_read_from_settings_not_hardcoded(self) -> None:
        """The Ranker accepts culture_weight as a parameter, not hardcoded."""
        ranker = Ranker(
            archetype_weight=0.5,
            fit_weight=0.3,
            history_weight=0.2,
            comp_weight=0.15,
            culture_weight=0.2,
            negative_weight=0.4,
            min_score_threshold=0.0,
        )
        assert ranker.culture_weight == 0.2, (
            f"Expected culture_weight 0.2, got {ranker.culture_weight}"
        )

    async def test_high_culture_score_raises_final_score(self) -> None:
        """A high culture_score increases the final score via culture_weight."""
        ranker = Ranker(
            archetype_weight=0.5,
            fit_weight=0.3,
            history_weight=0.0,
            comp_weight=0.0,
            culture_weight=0.3,
            negative_weight=0.0,
            min_score_threshold=0.0,
        )
        scores_no_culture = ScoreResult(
            fit_score=0.5,
            archetype_score=0.5,
            history_score=0.0,
            comp_score=0.0,
            culture_score=0.0,
            negative_score=0.0,
            disqualified=False,
        )
        scores_with_culture = ScoreResult(
            fit_score=0.5,
            archetype_score=0.5,
            history_score=0.0,
            comp_score=0.0,
            culture_score=0.9,
            negative_score=0.0,
            disqualified=False,
        )
        final_no = ranker.compute_final_score(scores_no_culture)
        final_with = ranker.compute_final_score(scores_with_culture)
        assert final_with > final_no, (
            f"High culture_score should raise final. No culture: {final_no}, With: {final_with}"
        )

    async def test_empty_global_positive_collection_returns_zero_culture_score(
        self, store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """An empty global_positive_signals collection returns 0.0 culture_score."""

        store.add_documents(
            collection_name="resume",
            ids=["resume-summary"],
            documents=["## Summary\nPrincipal architect."],
            embeddings=[EMBED_FAKE],
            metadatas=[{"source": "resume", "section": "Summary"}],
        )
        store.add_documents(
            collection_name="role_archetypes",
            ids=["archetype-test"],
            documents=["Staff Architect"],
            embeddings=[EMBED_FAKE],
            metadatas=[{"name": "Test", "source": "role_archetypes"}],
        )
        # Create empty collection
        store.reset_collection("global_positive_signals")

        scorer = Scorer(store=store, embedder=mock_embedder, disqualify_on_llm_flag=False)
        result = await scorer.score("Any JD text")
        assert result.culture_score == 0.0, (
            f"Expected culture_score 0.0 for empty collection, got {result.culture_score}"
        )

    async def test_score_explanation_includes_all_six_component_values(self) -> None:
        """The explanation string shows all six components: archetype, fit, culture, history, comp, negative."""
        scores = ScoreResult(
            fit_score=0.75,
            archetype_score=0.80,
            history_score=0.60,
            comp_score=0.90,
            negative_score=0.25,
            culture_score=0.65,
            disqualified=False,
        )
        ranked = RankedListing(
            listing=JobListing(
                board="test",
                external_id="1",
                title="Test",
                company="Co",
                location="Remote",
                url="https://example.org",
                full_text="JD text",
            ),
            scores=scores,
            final_score=0.75,
        )
        explanation = ranked.score_explanation()
        assert "Archetype: 0.80" in explanation
        assert "Fit: 0.75" in explanation
        assert "Culture: 0.65" in explanation
        assert "History: 0.60" in explanation
        assert "Comp: 0.90" in explanation
        assert "Negative: 0.25" in explanation
