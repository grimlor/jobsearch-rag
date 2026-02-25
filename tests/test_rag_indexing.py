"""BDD specs for RAG indexing: resume chunking, archetype synthesis, and signal indexing.

Covers: TestResumeIndexing, TestArchetypeIndexing, TestArchetypeEmbeddingSynthesis,
        TestGlobalRubricLoading, TestGlobalPositiveSignalIndexing, TestNegativeSignalIndexing
Spec doc: BDD Specifications — rag-indexing.md
"""

# Public API surface (from src/jobsearch_rag/rag/indexer.py):
#   build_archetype_embedding_text(archetype: dict[str, object]) -> str  (module-level)
#   Indexer(store: VectorStore, mock_embedder: Embedder)
#   indexer.index_resume(resume_path: str) -> int                        (async)
#   indexer.index_archetypes(archetypes_path: str) -> int                (async)
#   indexer.index_negative_signals(rubric_path: str, archetypes_path: str) -> int  (async)
#   indexer.index_global_positive_signals(rubric_path: str) -> int       (async)
#
# From src/jobsearch_rag/rag/store.py:
#   VectorStore(persist_dir: str)
#   store.collection_count(name: str) -> int
#   store.get_or_create_collection(name: str) -> chromadb.Collection
#   store.get_documents(collection_name: str, *, ids: list[str]) -> dict
#   store.get_by_metadata(collection_name: str, *, where: dict, include: list | None) -> dict
#
# From src/jobsearch_rag/errors.py:
#   ActionableError.config(field_name, reason, suggestion)
#   ActionableError.parse(board, selector, raw_error, suggestion)
#   ActionableError.validation(field_name, reason, suggestion)

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jobsearch_rag.errors import ActionableError
from jobsearch_rag.rag.indexer import Indexer, build_archetype_embedding_text

if TYPE_CHECKING:
    from pathlib import Path

    from jobsearch_rag.rag.embedder import Embedder
    from jobsearch_rag.rag.store import VectorStore

# ---------------------------------------------------------------------------
# Helpers — TOML content generators
# ---------------------------------------------------------------------------

_THREE_SECTION_RESUME = """\
# John Doe

## Experience
Worked at Acme Corp building distributed systems for five years.
Led platform engineering initiatives across three product teams.

## Skills
Python, Kubernetes, Terraform, and distributed systems design.
Strong background in cloud-native architecture and observability.

## Education
B.S. Computer Science from State University, graduated magna cum laude.
"""

_TWO_SECTION_RESUME = """\
# Jane Smith

## Projects
Built an open-source CLI tool for infrastructure automation.

## Certifications
AWS Solutions Architect Professional certification obtained in 2023.
"""

_VALID_ARCHETYPES_TOML = """\
[[archetypes]]
name = "Platform Engineer"
description = "Builds infrastructure tooling and internal platforms."
signals_positive = ["Kubernetes", "Terraform", "CI/CD pipelines"]
signals_negative = ["Front-end only", "No infrastructure scope"]

[[archetypes]]
name = "Data Engineer"
description = "Designs data pipelines and warehouse architectures."
signals_positive = ["Spark", "Airflow", "Data modeling"]
signals_negative = ["Manual reporting", "Excel-only analytics"]
"""

_SINGLE_ARCHETYPE_NO_SIGNALS_TOML = """\
[[archetypes]]
name = "Generalist"
description = "A broad role covering many engineering disciplines."
"""

_VALID_RUBRIC_TOML = """\
[[dimensions]]
name = "Role Scope"
signals_positive = ["Cross-team influence", "Strategic leadership"]
signals_negative = ["No cross-team scope", "Single product team only"]

[[dimensions]]
name = "Technical Depth"
signals_positive = ["Deep infrastructure work", "Platform engineering"]
signals_negative = ["Front-end heavy", "CMS-centric stack"]

[[dimensions]]
name = "Compensation Red Flags"
signals_negative = ["Equity-only compensation", "Unpaid position"]
"""

_RUBRIC_NO_POSITIVE_TOML = """\
[[dimensions]]
name = "Compensation Red Flags"
signals_negative = ["Equity-only compensation", "Unpaid position"]
"""

_RUBRIC_WITH_NUMERIC_FIELDS_TOML = """\
[[dimensions]]
name = "Role Scope"
description = "Measures the breadth of architectural influence."
minimum_target = 0.6
weight_fit = 0.3
signals_positive = ["Cross-team influence", "Strategic leadership"]
signals_negative = ["No cross-team scope"]
"""


# ---------------------------------------------------------------------------
# TestResumeIndexing
# ---------------------------------------------------------------------------


class TestResumeIndexing:
    """
    REQUIREMENT: The resume is indexed into ChromaDB as section chunks before
    scoring can produce meaningful fit scores.

    WHO: The scorer computing fit_score; the operator running first-time setup
    WHAT: Indexer.index_resume() chunks the resume file by section heading;
          each chunk contains at least one complete sentence; re-indexing
          replaces previous content rather than appending; the chunk count
          is returned; an empty or missing resume file produces an actionable
          error naming the path and the setup command
    WHY: An empty resume collection silently produces zero fit_scores for all
         roles — a harder bug to diagnose than an explicit missing-index error

    MOCK BOUNDARY:
        Mock:  mock_embedder fixture (Ollama HTTP — embed() is AsyncMock)
        Real:  Indexer instance, ChromaDB via vector_store fixture,
               resume file written to tmp_path
        Never: Pre-populate ChromaDB directly — always call index_resume()
               and assert on its return value and collection state
    """

    async def test_resume_is_chunked_by_section_heading(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a resume file with three section headings
        When index_resume() is called
        Then the resume collection contains three documents
        """
        # Given: a resume with 3 ## headings written to tmp_path
        resume_path = tmp_path / "resume.md"
        resume_path.write_text(_THREE_SECTION_RESUME)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_resume() is called
        count = await indexer.index_resume(str(resume_path))

        # Then: the resume collection contains exactly 3 documents
        collection_count = vector_store.collection_count("resume")
        assert collection_count == 3, (
            f"Expected 3 resume chunks (Experience, Skills, Education), "
            f"got {collection_count}"
        )
        assert count == 3, (
            f"index_resume() should return 3, got {count}"
        )

    async def test_each_chunk_contains_at_least_one_complete_sentence(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a resume file with multi-sentence sections
        When index_resume() is called
        Then no chunk ends mid-sentence
        """
        # Given: a resume with multi-sentence sections
        resume_path = tmp_path / "resume.md"
        resume_path.write_text(_THREE_SECTION_RESUME)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_resume() is called
        await indexer.index_resume(str(resume_path))

        # Then: every chunk ends with sentence-ending punctuation
        result = vector_store.get_by_metadata(
            "resume", where={"source": "resume"}, include=["documents"]
        )
        documents = result["documents"]
        assert documents, "Expected at least one document in the resume collection"
        for i, doc in enumerate(documents):
            stripped = doc.strip()
            assert stripped[-1] in ".!?", (
                f"Chunk {i} appears to end mid-sentence: "
                f"...{stripped[-60:]!r}"
            )

    async def test_reindex_replaces_previous_resume_content(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given index_resume() has been called once producing N chunks
        When index_resume() is called again with a different resume
        Then the collection contains the new document count, not N + new
        """
        # Given: first resume indexed (3 sections)
        resume_v1 = tmp_path / "resume_v1.md"
        resume_v1.write_text(_THREE_SECTION_RESUME)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        first_count = await indexer.index_resume(str(resume_v1))
        assert first_count == 3, f"First index should produce 3 chunks, got {first_count}"

        # When: second resume indexed (2 sections)
        resume_v2 = tmp_path / "resume_v2.md"
        resume_v2.write_text(_TWO_SECTION_RESUME)
        second_count = await indexer.index_resume(str(resume_v2))

        # Then: collection has 2 documents, not 3 + 2 = 5
        collection_count = vector_store.collection_count("resume")
        assert collection_count == 2, (
            f"Re-indexing should replace, not append. "
            f"Expected 2 chunks, got {collection_count} "
            f"(first index produced {first_count})"
        )
        assert second_count == 2, (
            f"index_resume() should return new count 2, got {second_count}"
        )

    async def test_index_returns_chunk_count(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        When index_resume() completes successfully
        Then it returns an integer >= 1 confirming chunks were stored
        """
        # Given: a valid resume file
        resume_path = tmp_path / "resume.md"
        resume_path.write_text(_THREE_SECTION_RESUME)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_resume() is called
        count = await indexer.index_resume(str(resume_path))

        # Then: the return value is an integer >= 1
        assert isinstance(count, int), (
            f"Expected int return type, got {type(count).__name__}"
        )
        assert count >= 1, (
            f"Expected chunk count >= 1 for a non-empty resume, got {count}"
        )

    async def test_missing_resume_file_produces_actionable_error(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given the resume file path does not exist
        When index_resume() is called
        Then an ActionableError is raised whose message names the missing path
        """
        # Given: a path that does not exist
        missing_path = str(tmp_path / "nonexistent" / "resume.md")
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_resume() is called with a missing path
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_resume(missing_path)

        # Then: the error message names the missing path
        error_msg = str(exc_info.value)
        assert "nonexistent" in error_msg or "resume.md" in error_msg, (
            f"Error should name the missing path. Got: {error_msg}"
        )


# ---------------------------------------------------------------------------
# TestArchetypeIndexing
# ---------------------------------------------------------------------------


class TestArchetypeIndexing:
    """
    REQUIREMENT: Role archetypes are loaded from TOML and indexed as
    synthesized embedding documents, not raw description strings.

    WHO: The scorer computing archetype_score
    WHAT: Each archetype in role_archetypes.toml produces one ChromaDB document
          in the role_archetypes collection; the document text is synthesized
          from description + signals_positive (signals_negative is indexed
          separately into the negative_signals collection); the archetype
          name is stored as metadata; malformed TOML produces an actionable error
          at index time; an empty archetypes file produces an actionable error;
          re-indexing resets the collection first
    WHY: Missing or malformed archetypes silently score all roles equally.
         Embedding only the description discards the signal arrays that provide
         the discrimination power the archetype system exists to deliver

    MOCK BOUNDARY:
        Mock:  mock_embedder fixture (Ollama HTTP)
        Real:  Indexer instance, ChromaDB via vector_store fixture,
               TOML files written to tmp_path
        Never: Pre-populate ChromaDB directly; never patch build_archetype_embedding_text —
               test through index_archetype_roles() so the synthesis and storage
               path is exercised together
    """

    async def test_each_toml_archetype_produces_one_collection_document(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a TOML file with two archetypes
        When index_archetypes() is called
        Then the role_archetypes collection contains two documents
        """
        # Given: a TOML file with 2 archetypes
        toml_path = tmp_path / "role_archetypes.toml"
        toml_path.write_text(_VALID_ARCHETYPES_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_archetypes() is called
        count = await indexer.index_archetypes(str(toml_path))

        # Then: collection has exactly 2 documents
        collection_count = vector_store.collection_count("role_archetypes")
        assert collection_count == 2, (
            f"Expected 2 archetype documents (Platform Engineer, Data Engineer), "
            f"got {collection_count}"
        )
        assert count == 2, (
            f"index_archetypes() should return 2, got {count}"
        )

    async def test_archetype_name_is_stored_as_document_metadata(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a TOML file with a named archetype
        When index_archetypes() is called
        Then the document metadata contains the archetype name
        """
        # Given: a TOML with a "Platform Engineer" archetype
        toml_path = tmp_path / "role_archetypes.toml"
        toml_path.write_text(_VALID_ARCHETYPES_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_archetypes() is called
        await indexer.index_archetypes(str(toml_path))

        # Then: metadata for each document includes "name"
        result = vector_store.get_by_metadata(
            "role_archetypes",
            where={"source": "role_archetypes"},
            include=["metadatas"],
        )
        metadatas = result["metadatas"]
        assert metadatas is not None
        names = [m["name"] for m in metadatas]
        assert "Platform Engineer" in names, (
            f"Expected 'Platform Engineer' in metadata names, got {names}"
        )
        assert "Data Engineer" in names, (
            f"Expected 'Data Engineer' in metadata names, got {names}"
        )

    async def test_synthesized_document_includes_description_text(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given an archetype with a description
        When index_archetypes() is called
        Then the stored document text contains the description content
        """
        # Given: a TOML with archetypes that have descriptions
        toml_path = tmp_path / "role_archetypes.toml"
        toml_path.write_text(_VALID_ARCHETYPES_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_archetypes() is called
        await indexer.index_archetypes(str(toml_path))

        # Then: description text appears in at least one stored document
        result = vector_store.get_by_metadata(
            "role_archetypes",
            where={"source": "role_archetypes"},
            include=["documents"],
        )
        assert result["documents"] is not None
        all_text = " ".join(result["documents"])
        assert "infrastructure tooling" in all_text, (
            f"Expected description text 'infrastructure tooling' in indexed documents. "
            f"Got: {all_text[:200]!r}"
        )

    async def test_synthesized_document_includes_positive_signals(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given an archetype with signals_positive entries
        When index_archetypes() is called
        Then the stored document text includes the positive signal phrases
        """
        # Given: a TOML with archetypes that have positive signals
        toml_path = tmp_path / "role_archetypes.toml"
        toml_path.write_text(_VALID_ARCHETYPES_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_archetypes() is called
        await indexer.index_archetypes(str(toml_path))

        # Then: positive signals appear in the stored document text
        result = vector_store.get_by_metadata(
            "role_archetypes",
            where={"source": "role_archetypes"},
            include=["documents"],
        )
        assert result["documents"] is not None
        all_text = " ".join(result["documents"])
        assert "Kubernetes" in all_text, (
            f"Expected positive signal 'Kubernetes' in synthesized document. "
            f"Got: {all_text[:300]!r}"
        )
        assert "Terraform" in all_text, (
            f"Expected positive signal 'Terraform' in synthesized document. "
            f"Got: {all_text[:300]!r}"
        )

    async def test_synthesized_document_omits_negative_signals(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given an archetype with signals_negative entries
        When index_archetypes() is called
        Then the stored document text does NOT include negative signal phrases
             (they are indexed separately into the negative_signals collection)
        """
        # Given: a TOML with archetypes that have negative signals
        toml_path = tmp_path / "role_archetypes.toml"
        toml_path.write_text(_VALID_ARCHETYPES_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_archetypes() is called
        await indexer.index_archetypes(str(toml_path))

        # Then: negative signals are NOT in the archetype embedding documents
        result = vector_store.get_by_metadata(
            "role_archetypes",
            where={"source": "role_archetypes"},
            include=["documents"],
        )
        assert result["documents"] is not None
        all_text = " ".join(result["documents"])
        assert "Front-end only" not in all_text, (
            f"Negative signal 'Front-end only' should not appear in archetype "
            f"embedding (indexed separately). Got: {all_text[:300]!r}"
        )

    async def test_archetype_without_signals_embeds_description_only(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given an archetype with no signals_positive or signals_negative
        When index_archetypes() is called
        Then the document text is the description alone
        """
        # Given: an archetype with only a description
        toml_path = tmp_path / "role_archetypes.toml"
        toml_path.write_text(_SINGLE_ARCHETYPE_NO_SIGNALS_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_archetypes() is called
        await indexer.index_archetypes(str(toml_path))

        # Then: the stored document contains only the description text
        result = vector_store.get_by_metadata(
            "role_archetypes",
            where={"source": "role_archetypes"},
            include=["documents"],
        )
        assert result["documents"] is not None
        assert len(result["documents"]) == 1, (
            f"Expected 1 document, got {len(result['documents'])}"
        )
        doc_text = result["documents"][0]
        assert "broad role" in doc_text, (
            f"Expected description 'broad role' in document. Got: {doc_text!r}"
        )
        # Should not contain signal prefixes since there are none
        assert "Key signals:" not in doc_text, (
            f"Document should not contain signal section when no signals exist. "
            f"Got: {doc_text!r}"
        )

    async def test_malformed_toml_names_syntax_error_and_file_path(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a TOML file with invalid syntax
        When index_archetypes() is called
        Then an ActionableError names the syntax error and file path
        """
        # Given: malformed TOML content
        toml_path = tmp_path / "bad_archetypes.toml"
        toml_path.write_text("[[archetypes]\nname = broken")  # missing closing bracket
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_archetypes() is called
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_archetypes(str(toml_path))

        # Then: the error mentions the syntax issue
        error_msg = str(exc_info.value)
        assert "bad_archetypes.toml" in error_msg or "TOML" in error_msg, (
            f"Error should reference the file or TOML syntax. Got: {error_msg}"
        )

    async def test_empty_archetypes_file_produces_actionable_error(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a TOML file with no [[archetypes]] entries
        When index_archetypes() is called
        Then an ActionableError is raised indicating no archetypes found
        """
        # Given: a valid TOML file with no archetypes array
        toml_path = tmp_path / "empty_archetypes.toml"
        toml_path.write_text("# Empty config\n")
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_archetypes() is called
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_archetypes(str(toml_path))

        # Then: the error names the missing archetypes
        error_msg = str(exc_info.value)
        assert "archetype" in error_msg.lower(), (
            f"Error should mention archetypes. Got: {error_msg}"
        )

    async def test_reindex_replaces_previous_archetype_content_not_appends(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given index_archetypes() has been called once
        When index_archetypes() is called again with different content
        Then the collection contains only the new archetypes, not old + new
        """
        # Given: first indexing with 2 archetypes
        toml_v1 = tmp_path / "archetypes_v1.toml"
        toml_v1.write_text(_VALID_ARCHETYPES_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        first_count = await indexer.index_archetypes(str(toml_v1))
        assert first_count == 2, f"First index should produce 2, got {first_count}"

        # When: re-index with 1 archetype
        toml_v2 = tmp_path / "archetypes_v2.toml"
        toml_v2.write_text(_SINGLE_ARCHETYPE_NO_SIGNALS_TOML)
        second_count = await indexer.index_archetypes(str(toml_v2))

        # Then: collection has 1 document, not 2 + 1 = 3
        collection_count = vector_store.collection_count("role_archetypes")
        assert collection_count == 1, (
            f"Re-indexing should replace, not append. "
            f"Expected 1 document, got {collection_count}"
        )
        assert second_count == 1, (
            f"index_archetypes() should return 1, got {second_count}"
        )


# ---------------------------------------------------------------------------
# TestArchetypeEmbeddingSynthesis
# ---------------------------------------------------------------------------


class TestArchetypeEmbeddingSynthesis:
    """
    REQUIREMENT: Archetype embedding text is correctly synthesized from
    TOML fields at index time, keeping config clean and embeddings rich.

    WHO: The indexer building archetype documents for ChromaDB
    WHAT: build_archetype_embedding_text() concatenates description and
          positive signals with a clear prefix; signals_negative is not
          included (it is indexed separately into the negative_signals
          collection); missing signal arrays are omitted gracefully;
          scoring guidance fields (thresholds, weights) are never included;
          whitespace in description is normalized before synthesis
    WHY: Embedding metadata fields alongside semantic content adds noise
         that degrades cosine similarity — this function is the filter
         that ensures only signal-bearing text reaches the embedding model

    MOCK BOUNDARY:
        Mock:  nothing — build_archetype_embedding_text() is pure computation
        Real:  build_archetype_embedding_text() called directly with dict inputs
        Never: Mock the function itself or patch any of its internals;
               all test variation is in the input dict passed to the function
    """

    def test_synthesis_includes_description_text(self) -> None:
        """
        When build_archetype_embedding_text() is called with a description
        Then the result contains the description text
        """
        # Given: an archetype dict with a description
        archetype: dict[str, object] = {"description": "Builds infrastructure tooling and platforms."}

        # When: build_archetype_embedding_text is called
        result = build_archetype_embedding_text(archetype)

        # Then: the description text appears in the result
        assert "Builds infrastructure tooling and platforms." in result, (
            f"Expected description in synthesis result. Got: {result!r}"
        )

    def test_synthesis_includes_positive_signals_with_prefix(self) -> None:
        """
        When build_archetype_embedding_text() is called with signals_positive
        Then each signal appears with a prefix marker
        """
        # Given: an archetype with positive signals
        archetype: dict[str, object] = {
            "description": "Platform engineer role.",
            "signals_positive": ["Kubernetes", "Terraform"],
        }

        # When: build_archetype_embedding_text is called
        result = build_archetype_embedding_text(archetype)

        # Then: each signal appears with the "- " prefix
        assert "- Kubernetes" in result, (
            f"Expected '- Kubernetes' in synthesis. Got: {result!r}"
        )
        assert "- Terraform" in result, (
            f"Expected '- Terraform' in synthesis. Got: {result!r}"
        )

    def test_synthesis_omits_negative_signals_from_embedding(self) -> None:
        """
        When build_archetype_embedding_text() is called with signals_negative
        Then negative signals do NOT appear in the result
             (they are indexed separately into the negative_signals collection)
        """
        # Given: an archetype with negative signals
        archetype: dict[str, object] = {
            "description": "Platform engineer role.",
            "signals_negative": ["Front-end only", "No infrastructure scope"],
        }

        # When: build_archetype_embedding_text is called
        result = build_archetype_embedding_text(archetype)

        # Then: negative signals are not in the synthesized text
        assert "Front-end only" not in result, (
            f"Negative signal should not appear in archetype embedding. Got: {result!r}"
        )
        assert "No infrastructure scope" not in result, (
            f"Negative signal should not appear in archetype embedding. Got: {result!r}"
        )

    def test_synthesis_omits_positive_section_when_absent(self) -> None:
        """
        Given an archetype dict with no signals_positive key
        When build_archetype_embedding_text() is called
        Then the result contains only the description without signal prefix
        """
        # Given: no signals_positive key
        archetype: dict[str, object] = {"description": "A generalist engineering role."}

        # When: synthesis is called
        result = build_archetype_embedding_text(archetype)

        # Then: no signal section prefix appears
        assert "Key signals:" not in result, (
            f"Expected no signal prefix when signals_positive is absent. Got: {result!r}"
        )
        assert "A generalist engineering role." in result, (
            f"Expected description text in result. Got: {result!r}"
        )

    def test_synthesis_omits_negative_section_when_absent(self) -> None:
        """
        Given an archetype dict with no signals_negative key
        When build_archetype_embedding_text() is called
        Then the result does not contain a negative signal section
        """
        # Given: signals_positive present but no signals_negative
        archetype: dict[str, object] = {
            "description": "Platform engineer role.",
            "signals_positive": ["Kubernetes"],
        }

        # When: synthesis is called
        result = build_archetype_embedding_text(archetype)

        # Then: no negative signal section appears
        assert "negative" not in result.lower(), (
            f"Expected no negative signal section. Got: {result!r}"
        )

    def test_synthesis_normalizes_description_whitespace(self) -> None:
        """
        Given a description with excessive whitespace
        When build_archetype_embedding_text() is called
        Then whitespace in the description is normalized to single spaces
        """
        # Given: a description with extra whitespace
        archetype: dict[str, object] = {"description": "  Builds   infrastructure\n\ttooling  and   platforms.  "}

        # When: synthesis is called
        result = build_archetype_embedding_text(archetype)

        # Then: whitespace is normalized
        assert "Builds infrastructure tooling and platforms." in result, (
            f"Expected normalized whitespace. Got: {result!r}"
        )

    def test_synthesis_does_not_include_scoring_guidance_fields(self) -> None:
        """
        Given an archetype dict with scoring fields (preference_weight, thresholds)
        When build_archetype_embedding_text() is called
        Then those fields do not appear in the synthesis result
        """
        # Given: an archetype with scoring metadata fields
        archetype: dict[str, object] = {
            "description": "Platform engineer role.",
            "preference_weight": 1.0,
            "minimum_target": 0.6,
            "signals_positive": ["Kubernetes"],
        }

        # When: synthesis is called
        result = build_archetype_embedding_text(archetype)

        # Then: numeric scoring fields are not in the text
        assert "1.0" not in result, (
            f"preference_weight value should not appear in synthesis. Got: {result!r}"
        )
        assert "0.6" not in result, (
            f"minimum_target value should not appear in synthesis. Got: {result!r}"
        )
        assert "preference_weight" not in result, (
            f"preference_weight key should not appear in synthesis. Got: {result!r}"
        )

    def test_synthesis_result_is_non_empty_string(self) -> None:
        """
        When build_archetype_embedding_text() is called with valid input
        Then the result is a non-empty string
        """
        # Given: a minimal valid archetype
        archetype: dict[str, object] = {"description": "A valid role description."}

        # When: synthesis is called
        result = build_archetype_embedding_text(archetype)

        # Then: the result is a non-empty string
        assert isinstance(result, str), (
            f"Expected str return type, got {type(result).__name__}"
        )
        assert len(result) > 0, "Synthesis result should not be empty"


# ---------------------------------------------------------------------------
# TestGlobalRubricLoading
# ---------------------------------------------------------------------------


class TestGlobalRubricLoading:
    """
    REQUIREMENT: The global rubric is loaded from TOML and its signal arrays
    are available for embedding synthesis while non-signal fields are excluded.

    WHO: The indexer building the global_positive_signals and negative_signals collections
    WHAT: global_rubric.toml is parsed into dimensions with signals_positive
          and signals_negative lists; description, minimum_target, and weight_*
          fields are present in the parsed dict but excluded from embedding
          synthesis; malformed or missing TOML produces actionable errors;
          dimensions without a signal array are skipped gracefully
    WHY: Embedding description prose and numeric config fields alongside
         signal phrases dilutes both collections' discrimination power

    MOCK BOUNDARY:
        Mock:  mock_embedder fixture (Ollama HTTP)
        Real:  Indexer instance, TOML files written to tmp_path
        Never: Mock the TOML parser; write real TOML files to tmp_path
               so parse errors are triggered by actual malformed content
    """

    async def test_global_rubric_loads_all_dimensions(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a rubric TOML with three dimensions
        When index_global_positive_signals() is called
        Then the number of indexed dimensions matches dimensions with signals_positive
        """
        # Given: a rubric with 3 dimensions (2 have signals_positive)
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_VALID_RUBRIC_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_global_positive_signals() is called
        count = await indexer.index_global_positive_signals(str(rubric_path))

        # Then: 2 dimensions with signals_positive are indexed
        assert count == 2, (
            f"Expected 2 dimensions with signals_positive "
            f"(Role Scope, Technical Depth — Compensation Red Flags has none), "
            f"got {count}"
        )

    async def test_each_dimension_exposes_signals_positive_list(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a rubric TOML with dimensions containing signals_positive
        When index_global_positive_signals() is called
        Then each indexed document contains the positive signal text
        """
        # Given: a rubric with dimensions that have signals_positive
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_VALID_RUBRIC_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_global_positive_signals() is called
        await indexer.index_global_positive_signals(str(rubric_path))

        # Then: positive signals appear in stored documents
        result = vector_store.get_by_metadata(
            "global_positive_signals",
            where={"source": "Role Scope"},
            include=["documents"],
        )
        assert result["documents"] is not None
        all_text = " ".join(result["documents"])
        assert "Cross-team influence" in all_text, (
            f"Expected positive signal 'Cross-team influence' in indexed documents. "
            f"Got: {all_text[:300]!r}"
        )

    async def test_each_dimension_exposes_signals_negative_list(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a rubric TOML with dimensions containing signals_negative
        When index_negative_signals() is called
        Then each signal produces a document in the negative_signals collection
        """
        # Given: a rubric with dimensions that have signals_negative + archetypes file
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_VALID_RUBRIC_TOML)
        arch_path = tmp_path / "role_archetypes.toml"
        arch_path.write_text(_SINGLE_ARCHETYPE_NO_SIGNALS_TOML)  # no negative signals
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_negative_signals() is called
        count = await indexer.index_negative_signals(str(rubric_path), str(arch_path))

        # Then: rubric has 2+2+2 = 6 negative signals (Role Scope: 2, Technical Depth: 2, Comp: 2)
        assert count == 6, (
            f"Expected 6 negative signals from rubric dimensions, got {count}"
        )

    async def test_description_field_is_accessible_but_not_embedded(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a rubric dimension with a description field
        When index_global_positive_signals() is called
        Then the description text does not appear in the stored embedding document
        """
        # Given: a rubric with description fields on dimensions
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_RUBRIC_WITH_NUMERIC_FIELDS_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_global_positive_signals() is called
        await indexer.index_global_positive_signals(str(rubric_path))

        # Then: the description prose is not in the positive signal document
        result = vector_store.get_by_metadata(
            "global_positive_signals",
            where={"source": "Role Scope"},
            include=["documents"],
        )
        assert result["documents"] is not None
        all_text = " ".join(result["documents"])
        assert "breadth of architectural influence" not in all_text, (
            f"Description prose should not be embedded in positive signals collection. "
            f"Got: {all_text[:300]!r}"
        )

    async def test_numeric_fields_are_accessible_but_not_embedded(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a rubric dimension with minimum_target and weight fields
        When index_global_positive_signals() is called
        Then those numeric values do not appear in the stored document text
        """
        # Given: a rubric with numeric fields
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_RUBRIC_WITH_NUMERIC_FIELDS_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_global_positive_signals() is called
        await indexer.index_global_positive_signals(str(rubric_path))

        # Then: numeric values are not in the document text
        result = vector_store.get_by_metadata(
            "global_positive_signals",
            where={"source": "Role Scope"},
            include=["documents"],
        )
        assert result["documents"] is not None
        all_text = " ".join(result["documents"])
        assert "0.6" not in all_text, (
            f"minimum_target value should not be in embedding text. Got: {all_text[:300]!r}"
        )
        assert "0.3" not in all_text, (
            f"weight_fit value should not be in embedding text. Got: {all_text[:300]!r}"
        )

    async def test_missing_global_rubric_file_produces_actionable_error(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given the global_rubric.toml file does not exist
        When index_global_positive_signals() is called
        Then an ActionableError is raised naming the missing file
        """
        # Given: a path that does not exist
        missing_path = str(tmp_path / "nonexistent_rubric.toml")
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_global_positive_signals() is called
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_global_positive_signals(missing_path)

        # Then: the error names the missing file
        error_msg = str(exc_info.value)
        assert "nonexistent_rubric.toml" in error_msg or "rubric" in error_msg.lower(), (
            f"Error should reference the missing rubric file. Got: {error_msg}"
        )

    async def test_malformed_toml_names_syntax_error_and_file_path(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a global_rubric.toml with invalid syntax
        When index_global_positive_signals() is called
        Then an ActionableError names the syntax error
        """
        # Given: malformed TOML
        rubric_path = tmp_path / "bad_rubric.toml"
        rubric_path.write_text("[[dimensions]\nname = broken")
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_global_positive_signals() is called
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_global_positive_signals(str(rubric_path))

        # Then: error references TOML syntax
        error_msg = str(exc_info.value)
        assert "TOML" in error_msg or "syntax" in error_msg.lower(), (
            f"Error should reference TOML syntax issue. Got: {error_msg}"
        )

    async def test_dimension_without_signals_positive_is_skipped_gracefully(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a rubric with a dimension that has no signals_positive
        When index_global_positive_signals() is called
        Then that dimension produces no document and no error
        """
        # Given: a rubric where only "Compensation Red Flags" has no signals_positive
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_RUBRIC_NO_POSITIVE_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_global_positive_signals() is called
        count = await indexer.index_global_positive_signals(str(rubric_path))

        # Then: no documents indexed (the only dimension lacks signals_positive)
        assert count == 0, (
            f"Expected 0 documents for rubric with no signals_positive, got {count}"
        )

    async def test_dimension_without_signals_negative_is_skipped_gracefully(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a rubric dimension with no signals_negative
        When index_negative_signals() is called
        Then that dimension contributes no negative signal documents
        """
        # Given: a rubric with one dimension that has only signals_positive, no signals_negative
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_toml = """\
[[dimensions]]
name = "Positive Only"
signals_positive = ["Good signal"]
"""
        rubric_path.write_text(rubric_toml)
        arch_path = tmp_path / "role_archetypes.toml"
        arch_path.write_text(_SINGLE_ARCHETYPE_NO_SIGNALS_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_negative_signals() is called
        count = await indexer.index_negative_signals(str(rubric_path), str(arch_path))

        # Then: no negative signal documents (dimension has no signals_negative, archetype has none)
        assert count == 0, (
            f"Expected 0 negative signals for dimension without signals_negative, got {count}"
        )


# ---------------------------------------------------------------------------
# TestGlobalPositiveSignalIndexing
# ---------------------------------------------------------------------------


class TestGlobalPositiveSignalIndexing:
    """
    REQUIREMENT: Positive signals from the global rubric are indexed into
    a dedicated collection that the scorer queries to compute culture_score.

    WHO: The scorer computing culture_score; the indexer building the
         global_positive_signals collection
    WHAT: index_global_positive_signals() produces one document per rubric
          dimension that has signals_positive entries; documents carry the
          source dimension name in metadata; re-indexing resets the collection;
          a rubric with no positive signals produces an empty collection without
          error; missing global_rubric.toml produces an actionable error
    WHY: Folding culture signals into archetype embeddings would make all
         archetypes score similarly against culture language. A dedicated
         collection keeps the two scoring axes independent

    MOCK BOUNDARY:
        Mock:  mock_embedder fixture (Ollama HTTP)
        Real:  Indexer instance, ChromaDB via vector_store fixture,
               TOML files written to tmp_path
        Never: Pre-populate ChromaDB directly; verify collection state
               only through ChromaDB query after indexing completes
    """

    async def test_one_document_per_dimension_with_positive_signals(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a rubric with two dimensions having signals_positive
        When index_global_positive_signals() is called
        Then the collection contains exactly two documents
        """
        # Given: a rubric with 2 dimensions having signals_positive
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_VALID_RUBRIC_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_global_positive_signals() is called
        count = await indexer.index_global_positive_signals(str(rubric_path))

        # Then: exactly 2 documents
        collection_count = vector_store.collection_count("global_positive_signals")
        assert collection_count == 2, (
            f"Expected 2 pos-signal documents (Role Scope + Technical Depth), "
            f"got {collection_count}"
        )
        assert count == 2, (
            f"index_global_positive_signals() should return 2, got {count}"
        )

    async def test_document_metadata_identifies_source_dimension(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a rubric dimension named "Role Scope"
        When index_global_positive_signals() is called
        Then the document metadata 'source' contains "Role Scope"
        """
        # Given: a rubric
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_VALID_RUBRIC_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_global_positive_signals() is called
        await indexer.index_global_positive_signals(str(rubric_path))

        # Then: metadata identifies source dimensions
        role_scope = vector_store.get_by_metadata(
            "global_positive_signals", where={"source": "Role Scope"}
        )
        assert role_scope["ids"], (
            "Expected documents for 'Role Scope' dimension"
        )
        tech_depth = vector_store.get_by_metadata(
            "global_positive_signals", where={"source": "Technical Depth"}
        )
        assert tech_depth["ids"], (
            "Expected documents for 'Technical Depth' dimension"
        )

    async def test_reindex_replaces_collection_not_appends(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given index_global_positive_signals() has been called once
        When it is called again
        Then the collection contains only the latest documents
        """
        # Given: first indexing
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_VALID_RUBRIC_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        first_count = await indexer.index_global_positive_signals(str(rubric_path))

        # When: re-index with a rubric that has no positive signals
        rubric_v2 = tmp_path / "rubric_v2.toml"
        rubric_v2.write_text(_RUBRIC_NO_POSITIVE_TOML)
        second_count = await indexer.index_global_positive_signals(str(rubric_v2))

        # Then: collection is now empty (not first_count + 0)
        collection_count = vector_store.collection_count("global_positive_signals")
        assert collection_count == 0, (
            f"Re-indexing should replace. Expected 0, got {collection_count} "
            f"(first index had {first_count})"
        )
        assert second_count == 0, (
            f"Expected 0 from second indexing, got {second_count}"
        )

    async def test_dimension_without_positive_signals_produces_no_document(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a rubric with only dimensions lacking signals_positive
        When index_global_positive_signals() is called
        Then the collection is empty
        """
        # Given: a rubric with only negative signals
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_RUBRIC_NO_POSITIVE_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index is called
        count = await indexer.index_global_positive_signals(str(rubric_path))

        # Then: no documents, no error
        assert count == 0, (
            f"Expected 0 for rubric without positive signals, got {count}"
        )

    async def test_missing_global_rubric_produces_actionable_error(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given the global_rubric.toml does not exist
        When index_global_positive_signals() is called
        Then an ActionableError is raised
        """
        # Given: missing path
        missing_path = str(tmp_path / "missing_rubric.toml")
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: called with missing file
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_global_positive_signals(missing_path)

        # Then: error names the file
        error_msg = str(exc_info.value)
        assert "missing_rubric" in error_msg or "rubric" in error_msg.lower(), (
            f"Error should reference the missing file. Got: {error_msg}"
        )

    async def test_malformed_rubric_toml_produces_actionable_parse_error(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a global_rubric.toml with invalid syntax
        When index_global_positive_signals() is called
        Then an ActionableError is raised
        """
        # Given: malformed TOML
        rubric_path = tmp_path / "bad_rubric.toml"
        rubric_path.write_text("[[dimensions]\nbroken")
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: called
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_global_positive_signals(str(rubric_path))

        # Then: error references syntax
        error_msg = str(exc_info.value)
        assert "TOML" in error_msg or "syntax" in error_msg.lower(), (
            f"Error should reference TOML syntax. Got: {error_msg}"
        )

    async def test_archetypes_only_flag_rebuilds_global_positive_collection(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given the global_positive_signals collection already contains data
        When index_global_positive_signals() is called again (as --archetypes-only does)
        Then the collection is rebuilt from the rubric file
        """
        # Given: existing data in the collection
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_VALID_RUBRIC_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        first_count = await indexer.index_global_positive_signals(str(rubric_path))
        assert first_count == 2, f"Setup: expected 2 documents, got {first_count}"

        # When: called again (simulating --archetypes-only re-index path)
        second_count = await indexer.index_global_positive_signals(str(rubric_path))

        # Then: collection still has exactly 2 (replaced, not doubled)
        collection_count = vector_store.collection_count("global_positive_signals")
        assert collection_count == 2, (
            f"Re-index should replace. Expected 2, got {collection_count}"
        )
        assert second_count == 2, (
            f"Expected 2 from re-index, got {second_count}"
        )

    async def test_collection_document_count_matches_contributing_dimensions(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a rubric with N dimensions having signals_positive
        When index_global_positive_signals() is called
        Then the collection document count equals N
        """
        # Given: rubric with 2 contributing dimensions
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_VALID_RUBRIC_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index is called
        count = await indexer.index_global_positive_signals(str(rubric_path))

        # Then: count and collection match
        collection_count = vector_store.collection_count("global_positive_signals")
        assert count == collection_count, (
            f"Return value ({count}) should match collection count ({collection_count})"
        )

    async def test_compensation_dimension_produces_no_positive_document(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a rubric with a "Compensation Red Flags" dimension lacking signals_positive
        When index_global_positive_signals() is called
        Then that dimension produces no document
        """
        # Given: only the compensation dimension (no signals_positive)
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_RUBRIC_NO_POSITIVE_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index is called
        count = await indexer.index_global_positive_signals(str(rubric_path))

        # Then: no documents
        assert count == 0, (
            f"Compensation dimension without signals_positive should produce 0 docs, got {count}"
        )


# ---------------------------------------------------------------------------
# TestNegativeSignalIndexing
# ---------------------------------------------------------------------------


class TestNegativeSignalIndexing:
    """
    REQUIREMENT: Negative signals from the global rubric and archetype
    definitions are indexed into a dedicated collection for continuous penalty scoring.

    WHO: The scorer computing negative_score; the indexer building the
         negative_signals collection
    WHAT: index_negative_signals() produces one document per global rubric
          dimension with signals_negative, plus one document per archetype
          with signals_negative; documents carry the source name in metadata;
          re-indexing resets the collection; missing rubric produces an
          actionable error; --archetypes-only rebuilds this collection
    WHY: Without a dedicated negative collection, scoring is purely additive.
         A continuous penalty provides suppression where a binary gate
         would be too blunt

    MOCK BOUNDARY:
        Mock:  mock_embedder fixture (Ollama HTTP)
        Real:  Indexer instance, ChromaDB via vector_store fixture,
               TOML files written to tmp_path
        Never: Pre-populate ChromaDB directly; write real TOML to tmp_path
               to verify both rubric and archetype source documents appear
    """

    async def test_one_document_per_rubric_dimension_with_negative_signals(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a rubric with three dimensions having signals_negative
        When index_negative_signals() is called
        Then the collection contains one document per negative signal string
        """
        # Given: rubric with 3 dimensions having signals_negative (2+2+2=6 signals)
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_VALID_RUBRIC_TOML)
        arch_path = tmp_path / "role_archetypes.toml"
        arch_path.write_text(_SINGLE_ARCHETYPE_NO_SIGNALS_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_negative_signals() is called
        count = await indexer.index_negative_signals(str(rubric_path), str(arch_path))

        # Then: 6 negative signal documents from rubric (archetype has no negative signals)
        assert count == 6, (
            f"Expected 6 negative signal docs (Role Scope: 2, Technical Depth: 2, "
            f"Compensation: 2), got {count}"
        )

    async def test_one_document_per_archetype_with_negative_signals(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given archetypes with signals_negative entries
        When index_negative_signals() is called
        Then one document per archetype negative signal is indexed
        """
        # Given: rubric with no negative signals, archetypes with negative signals
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_no_neg = """\
[[dimensions]]
name = "Positive Only"
signals_positive = ["Good signal"]
"""
        rubric_path.write_text(rubric_no_neg)
        arch_path = tmp_path / "role_archetypes.toml"
        arch_path.write_text(_VALID_ARCHETYPES_TOML)  # 2 archetypes, each with 2 neg signals
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_negative_signals() is called
        count = await indexer.index_negative_signals(str(rubric_path), str(arch_path))

        # Then: 4 negative signals from archetypes (2 per archetype x 2 archetypes)
        assert count == 4, (
            f"Expected 4 negative signal docs from archetypes "
            f"(Platform Engineer: 2, Data Engineer: 2), got {count}"
        )

    async def test_document_metadata_identifies_source(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given rubric and archetype negative signals
        When index_negative_signals() is called
        Then document metadata identifies whether source is rubric or archetype
        """
        # Given: rubric and archetypes with negative signals
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_VALID_RUBRIC_TOML)
        arch_path = tmp_path / "role_archetypes.toml"
        arch_path.write_text(_VALID_ARCHETYPES_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index_negative_signals() is called
        await indexer.index_negative_signals(str(rubric_path), str(arch_path))

        # Then: metadata identifies sources
        result = vector_store.get_by_metadata(
            "negative_signals",
            where={"source": {"$ne": ""}},
            include=["metadatas"],
        )
        assert result["metadatas"] is not None
        sources = [str(m["source"]) for m in result["metadatas"]]
        rubric_sources = [s for s in sources if s.startswith("rubric:")]
        archetype_sources = [s for s in sources if s.startswith("archetype:")]
        assert len(rubric_sources) > 0, (
            f"Expected rubric-sourced documents. Sources: {sources}"
        )
        assert len(archetype_sources) > 0, (
            f"Expected archetype-sourced documents. Sources: {sources}"
        )
        assert "rubric:Role Scope" in sources, (
            f"Expected 'rubric:Role Scope' in sources, got {sources}"
        )
        assert "archetype:Platform Engineer" in sources, (
            f"Expected 'archetype:Platform Engineer' in sources, got {sources}"
        )

    async def test_reindex_replaces_collection_not_appends(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given index_negative_signals() has been called once
        When it is called again
        Then the collection has only the new documents
        """
        # Given: first indexing with rubric + archetypes
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_VALID_RUBRIC_TOML)
        arch_path = tmp_path / "role_archetypes.toml"
        arch_path.write_text(_VALID_ARCHETYPES_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        first_count = await indexer.index_negative_signals(str(rubric_path), str(arch_path))
        assert first_count > 0, f"Setup: expected > 0 signals, got {first_count}"

        # When: re-index with no negative signals
        rubric_v2 = tmp_path / "rubric_v2.toml"
        rubric_v2.write_text("[[dimensions]]\nname = 'Empty'\n")
        arch_v2 = tmp_path / "arch_v2.toml"
        arch_v2.write_text(_SINGLE_ARCHETYPE_NO_SIGNALS_TOML)
        second_count = await indexer.index_negative_signals(str(rubric_v2), str(arch_v2))

        # Then: collection is empty (not first + 0)
        collection_count = vector_store.collection_count("negative_signals")
        assert collection_count == 0, (
            f"Re-indexing should replace. Expected 0, got {collection_count} "
            f"(first index had {first_count})"
        )
        assert second_count == 0, (
            f"Expected 0 from second indexing, got {second_count}"
        )

    async def test_dimension_without_negative_signals_produces_no_document(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a rubric dimension with no signals_negative
        When index_negative_signals() is called
        Then that dimension contributes no negative documents
        """
        # Given: rubric dimension without negative signals, archetype without negative signals
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text("[[dimensions]]\nname = 'Positive Only'\nsignals_positive = ['Good']\n")
        arch_path = tmp_path / "role_archetypes.toml"
        arch_path.write_text(_SINGLE_ARCHETYPE_NO_SIGNALS_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index is called
        count = await indexer.index_negative_signals(str(rubric_path), str(arch_path))

        # Then: no negative signal documents
        assert count == 0, (
            f"Expected 0 for sources without signals_negative, got {count}"
        )

    async def test_missing_global_rubric_produces_actionable_error(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given the global_rubric.toml does not exist
        When index_negative_signals() is called
        Then an ActionableError is raised
        """
        # Given: missing rubric, valid archetypes
        missing_rubric = str(tmp_path / "missing_rubric.toml")
        arch_path = tmp_path / "role_archetypes.toml"
        arch_path.write_text(_SINGLE_ARCHETYPE_NO_SIGNALS_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: called with missing rubric
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_negative_signals(missing_rubric, str(arch_path))

        # Then: error references the missing file
        error_msg = str(exc_info.value)
        assert "missing_rubric" in error_msg or "rubric" in error_msg.lower(), (
            f"Error should reference missing rubric file. Got: {error_msg}"
        )

    async def test_malformed_archetypes_toml_produces_actionable_parse_error(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given the role_archetypes.toml has invalid syntax
        When index_negative_signals() is called
        Then an ActionableError is raised
        """
        # Given: valid rubric, malformed archetypes
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_VALID_RUBRIC_TOML)
        bad_arch = tmp_path / "bad_archetypes.toml"
        bad_arch.write_text("[[archetypes]\nbroken")
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: called with malformed archetypes
        with pytest.raises(ActionableError) as exc_info:
            await indexer.index_negative_signals(str(rubric_path), str(bad_arch))

        # Then: error references TOML syntax
        error_msg = str(exc_info.value)
        assert "TOML" in error_msg or "syntax" in error_msg.lower(), (
            f"Error should reference TOML syntax. Got: {error_msg}"
        )

    async def test_archetypes_only_flag_rebuilds_negative_signals_collection(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given the negative_signals collection already contains data
        When index_negative_signals() is called again (as --archetypes-only does)
        Then the collection is rebuilt
        """
        # Given: existing data
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_VALID_RUBRIC_TOML)
        arch_path = tmp_path / "role_archetypes.toml"
        arch_path.write_text(_VALID_ARCHETYPES_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)
        first_count = await indexer.index_negative_signals(str(rubric_path), str(arch_path))
        assert first_count > 0, f"Setup: expected > 0, got {first_count}"

        # When: re-index (simulating --archetypes-only)
        second_count = await indexer.index_negative_signals(str(rubric_path), str(arch_path))

        # Then: same count (replaced, not doubled)
        collection_count = vector_store.collection_count("negative_signals")
        assert collection_count == first_count, (
            f"Re-index should replace. Expected {first_count}, "
            f"got {collection_count}"
        )
        assert second_count == first_count, (
            f"Expected {first_count} from re-index, got {second_count}"
        )

    async def test_collection_count_matches_contributing_sources(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given rubric and archetype sources with known signal counts
        When index_negative_signals() is called
        Then the collection count equals total rubric + archetype negative signals
        """
        # Given: rubric has 6 negative signals, archetypes have 4
        rubric_path = tmp_path / "global_rubric.toml"
        rubric_path.write_text(_VALID_RUBRIC_TOML)
        arch_path = tmp_path / "role_archetypes.toml"
        arch_path.write_text(_VALID_ARCHETYPES_TOML)
        indexer = Indexer(store=vector_store, embedder=mock_embedder)

        # When: index is called
        count = await indexer.index_negative_signals(str(rubric_path), str(arch_path))

        # Then: total 10 (6 from rubric + 4 from archetypes)
        expected = 10  # Role Scope:2 + Technical Depth:2 + Comp:2 + PlatformEng:2 + DataEng:2
        collection_count = vector_store.collection_count("negative_signals")
        assert count == expected, (
            f"Expected {expected} total negative signals "
            f"(6 rubric + 4 archetype), got {count}"
        )
        assert count == collection_count, (
            f"Return value ({count}) should match collection count ({collection_count})"
        )
