# Public API surface (from src/jobsearch_rag/rag/decisions.py):
#   DecisionRecorder(*, store: VectorStore, embedder: Embedder, decisions_dir: str | Path)
#   recorder.record(*, job_id: str, verdict: str, jd_text: str, board: str,
#                   title: str = "", company: str = "", reason: str = "") -> None    (async)
#   recorder.get_decision(job_id: str) -> dict[str, str] | None
#   recorder.history_count() -> int
#
# From src/jobsearch_rag/rag/store.py:
#   VectorStore(persist_dir: str)
#   store.add_documents(collection_name, *, ids, documents, embeddings, metadatas=None)
#   store.get_documents(collection_name, *, ids) -> dict
#   store.get_by_metadata(collection_name, *, where, include=None) -> dict
#   store.get_or_create_collection(name) -> chromadb.Collection
#   store.collection_count(name) -> int
#
# From src/jobsearch_rag/errors.py:
#   ActionableError (dataclass exception)
#   ActionableError.index(collection, suggestion=None)
#   ActionableError.validation(field_name, reason, suggestion=None)
#   ErrorType.INDEX, ErrorType.VALIDATION, ErrorType.DECISION
"""BDD specs for decision history: metadata queries and decision recording.

Covers: TestVectorStoreMetadataQuery, TestDecisionRecording
Spec doc: BDD Specifications — decision-history.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.rag.decisions import DecisionRecorder
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.store import VectorStore

# Canonical fake embedding — matches conftest.EMBED_FAKE dimensionality.
EMBED_FAKE: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5]


# ---------------------------------------------------------------------------
# TestVectorStoreMetadataQuery
# ---------------------------------------------------------------------------


class TestVectorStoreMetadataQuery:
    """
    REQUIREMENT: Documents can be queried by metadata filters, not just
    by embedding similarity.

    WHO: The scorer fetching rejection reasons from the decisions collection
    WHAT: get_by_metadata() returns documents matching a metadata filter;
          non-matching filters return an empty result;
          querying a nonexistent collection produces an error with guidance
    WHY: Rejection reason injection requires fetching decisions by verdict
         type — a metadata filter access pattern, not similarity search

    MOCK BOUNDARY:
        Mock:  nothing — VectorStore wraps ChromaDB with no additional I/O
        Real:  VectorStore instance, ChromaDB via vector_store fixture (tmp_path)
        Never: Mock ChromaDB; populate collections via VectorStore.add()
               before querying so the storage and retrieval path is exercised
    """

    def test_get_by_metadata_returns_matching_documents(
        self, vector_store: VectorStore
    ) -> None:
        """
        Given a collection with documents carrying different verdict metadata
        When get_by_metadata is called with a filter matching one verdict
        Then only documents with that verdict are returned
        """
        # Given: a decisions collection with yes and no verdicts
        vector_store.add_documents(
            "decisions",
            ids=["decision-job-1", "decision-job-2", "decision-job-3"],
            documents=[
                "Staff architect role at Acme Corp",
                "Junior frontend developer position",
                "Senior platform engineer at Widgets Inc",
            ],
            embeddings=[EMBED_FAKE, EMBED_FAKE, EMBED_FAKE],
            metadatas=[
                {"verdict": "yes", "job_id": "job-1"},
                {"verdict": "no", "job_id": "job-2"},
                {"verdict": "yes", "job_id": "job-3"},
            ],
        )

        # When: querying for yes verdicts only
        results = vector_store.get_by_metadata(
            "decisions",
            where={"verdict": "yes"},
            include=["metadatas"],
        )

        # Then: only the two yes-verdict documents are returned
        returned_ids = results.get("ids", [])
        assert len(returned_ids) == 2, (
            f"Expected 2 documents with verdict='yes', got {len(returned_ids)}: "
            f"{returned_ids}"
        )
        for meta in results["metadatas"]:
            assert meta["verdict"] == "yes", (
                f"Expected all returned documents to have verdict='yes', "
                f"got verdict='{meta['verdict']}' in {meta}"
            )

    def test_get_by_metadata_returns_empty_when_no_match(
        self, vector_store: VectorStore
    ) -> None:
        """
        Given a collection with only 'yes' verdict documents
        When get_by_metadata is called filtering for 'no' verdicts
        Then an empty result is returned
        """
        # Given: a decisions collection with only yes verdicts
        vector_store.add_documents(
            "decisions",
            ids=["decision-job-10"],
            documents=["Platform engineering lead at Cloud Corp"],
            embeddings=[EMBED_FAKE],
            metadatas=[{"verdict": "yes", "job_id": "job-10"}],
        )

        # When: querying for no verdicts
        results = vector_store.get_by_metadata(
            "decisions",
            where={"verdict": "no"},
            include=["metadatas"],
        )

        # Then: no documents are returned
        returned_ids = results.get("ids", [])
        assert len(returned_ids) == 0, (
            f"Expected 0 documents with verdict='no', got {len(returned_ids)}: "
            f"{returned_ids}"
        )

    def test_get_by_metadata_nonexistent_collection_produces_actionable_error(
        self, vector_store: VectorStore
    ) -> None:
        """
        When get_by_metadata is called on a collection that does not exist
        Then an ActionableError of type INDEX is raised with guidance
        """
        # Given: no 'nonexistent_collection' has been created

        # When / Then: querying a nonexistent collection raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            vector_store.get_by_metadata(
                "nonexistent_collection",
                where={"verdict": "yes"},
            )

        assert exc_info.value.error_type == ErrorType.INDEX, (
            f"Expected error_type=INDEX, got {exc_info.value.error_type}. "
            f"Full error: {exc_info.value}"
        )
        assert "nonexistent_collection" in str(exc_info.value), (
            f"Error should name the missing collection 'nonexistent_collection'. "
            f"Got: {exc_info.value}"
        )


# ---------------------------------------------------------------------------
# TestDecisionRecording
# ---------------------------------------------------------------------------


class TestDecisionRecording:
    """
    REQUIREMENT: Operator decisions are recorded and progressively build
    the history signal that improves future scoring.

    WHO: The scorer computing history_score on future runs
    WHAT: A yes/no/maybe verdict is stored with the JD embedding and job_id;
          an optional reason enriches the embedding vector and is persisted
          in both ChromaDB metadata and the JSONL audit log; empty reasons
          do not alter the embedding; only 'yes' decisions contribute to
          history_score; an invalid verdict string produces an error naming
          both the job_id and the invalid string; duplicate decisions on the
          same job_id overwrite rather than append; empty JD text produces
          a validation error; history_count() returns the total document
          count in the decisions collection (all verdicts, not just yes);
          missing collections degrade gracefully
    WHY: 'no' decisions contributing to scoring would suppress roles similar
         to rejected ones — but rejections have too many confounding reasons
         that have nothing to do with role quality

    MOCK BOUNDARY:
        Mock:  mock_embedder fixture (Ollama HTTP — embed() is AsyncMock)
        Real:  DecisionRecorder instance, ChromaDB via vector_store fixture,
               JSONL audit log written to tmp_path via output directory guard
        Never: Insert directly into ChromaDB collections — always call
               recorder.record() and verify state through recorder.get_recent()
               or recorder.get_decision(); never patch _store
    """

    @pytest.mark.asyncio
    async def test_yes_verdict_is_stored_in_history_collection(
        self, decision_recorder: DecisionRecorder, vector_store: VectorStore
    ) -> None:
        """
        When a 'yes' verdict is recorded for a job listing
        Then the decision is stored in the decisions collection with scoring_signal='true'
        """
        # Given: a decision recorder ready to record

        # When: recording a yes verdict
        await decision_recorder.record(
            job_id="zr-100",
            verdict="yes",
            jd_text="Staff Platform Architect building cloud infrastructure at scale.",
            board="ziprecruiter",
            title="Staff Platform Architect",
            company="Acme Corp",
        )

        # Then: the decision is retrievable and marked for scoring
        decision = decision_recorder.get_decision("zr-100")
        assert decision is not None, (
            "Expected decision for 'zr-100' to be stored, got None"
        )
        assert decision["verdict"] == "yes", (
            f"Expected verdict='yes', got '{decision['verdict']}'"
        )
        assert decision["scoring_signal"] == "true", (
            f"Expected scoring_signal='true' for yes verdict, "
            f"got '{decision['scoring_signal']}'"
        )

    @pytest.mark.asyncio
    async def test_reason_is_stored_in_metadata(
        self, decision_recorder: DecisionRecorder
    ) -> None:
        """
        Given a non-empty reason is provided with the verdict
        When the decision is recorded
        Then the reason is persisted in the ChromaDB metadata
        """
        # Given: a non-empty reason
        reason_text = "Strong platform focus and distributed systems experience"

        # When: recording with a reason
        await decision_recorder.record(
            job_id="zr-200",
            verdict="yes",
            jd_text="Senior infrastructure engineer designing cloud-native platforms.",
            board="ziprecruiter",
            title="Senior Infrastructure Engineer",
            company="Cloud Widgets",
            reason=reason_text,
        )

        # Then: the reason appears in the stored metadata
        decision = decision_recorder.get_decision("zr-200")
        assert decision is not None, (
            "Expected decision for 'zr-200' to be stored, got None"
        )
        assert decision["reason"] == reason_text, (
            f"Expected reason='{reason_text}', got '{decision['reason']}'"
        )

    @pytest.mark.asyncio
    async def test_empty_reason_stored_without_modifying_embedding(
        self, decision_recorder: DecisionRecorder, mock_embedder: Embedder
    ) -> None:
        """
        Given an empty reason is provided
        When the decision is recorded
        Then embed() is called with the raw JD text only (no appended reason)
        """
        # Given: empty reason
        jd_text = "Backend engineer working on payment microservices."

        # When: recording with empty reason
        await decision_recorder.record(
            job_id="zr-300",
            verdict="yes",
            jd_text=jd_text,
            board="ziprecruiter",
            title="Backend Engineer",
            company="PayCo",
            reason="",
        )

        # Then: embed() was called with the JD text only, not enriched
        mock_embedder.embed.assert_called_with(jd_text)  # type: ignore[union-attr]
        call_arg = mock_embedder.embed.call_args[0][0]  # type: ignore[union-attr]
        assert "Operator reasoning" not in call_arg, (
            f"Expected embed text to NOT contain 'Operator reasoning' for empty reason, "
            f"but got: {call_arg!r}"
        )

    @pytest.mark.asyncio
    async def test_reason_enriches_embedding_when_non_empty(
        self, decision_recorder: DecisionRecorder, mock_embedder: Embedder
    ) -> None:
        """
        Given a non-empty reason is provided
        When the decision is recorded
        Then embed() is called with JD text enriched by the reason
        """
        # Given: a non-empty reason
        jd_text = "Principal engineer leading API platform team."
        reason = "Excellent scope and leadership opportunity"

        # When: recording with a reason
        await decision_recorder.record(
            job_id="zr-400",
            verdict="yes",
            jd_text=jd_text,
            board="ziprecruiter",
            title="Principal Engineer",
            company="BigCo",
            reason=reason,
        )

        # Then: embed() was called with enriched text
        expected_embed_text = f"{jd_text}\n\nOperator reasoning: {reason}"
        mock_embedder.embed.assert_called_with(expected_embed_text)  # type: ignore[union-attr]
        call_arg = mock_embedder.embed.call_args[0][0]  # type: ignore[union-attr]
        assert "Operator reasoning:" in call_arg, (
            f"Expected embed text to contain 'Operator reasoning:', "
            f"but got: {call_arg!r}"
        )

    @pytest.mark.asyncio
    async def test_no_verdict_is_stored_but_excluded_from_history_score(
        self, decision_recorder: DecisionRecorder
    ) -> None:
        """
        When a 'no' verdict is recorded
        Then it is stored with scoring_signal='false'
        """
        # Given: a decision recorder ready to record

        # When: recording a no verdict
        await decision_recorder.record(
            job_id="zr-500",
            verdict="no",
            jd_text="Junior data entry clerk processing invoices.",
            board="ziprecruiter",
            title="Junior Data Entry Clerk",
            company="BoreCorp",
        )

        # Then: stored but excluded from scoring
        decision = decision_recorder.get_decision("zr-500")
        assert decision is not None, (
            "Expected decision for 'zr-500' to be stored, got None"
        )
        assert decision["verdict"] == "no", (
            f"Expected verdict='no', got '{decision['verdict']}'"
        )
        assert decision["scoring_signal"] == "false", (
            f"Expected scoring_signal='false' for no verdict, "
            f"got '{decision['scoring_signal']}'"
        )

    @pytest.mark.asyncio
    async def test_maybe_verdict_is_stored_but_excluded_from_history_score(
        self, decision_recorder: DecisionRecorder
    ) -> None:
        """
        When a 'maybe' verdict is recorded
        Then it is stored with scoring_signal='false'
        """
        # Given: a decision recorder ready to record

        # When: recording a maybe verdict
        await decision_recorder.record(
            job_id="zr-600",
            verdict="maybe",
            jd_text="Mid-level full-stack developer at a growing startup.",
            board="ziprecruiter",
            title="Full-Stack Developer",
            company="StartupInc",
        )

        # Then: stored but excluded from scoring
        decision = decision_recorder.get_decision("zr-600")
        assert decision is not None, (
            "Expected decision for 'zr-600' to be stored, got None"
        )
        assert decision["verdict"] == "maybe", (
            f"Expected verdict='maybe', got '{decision['verdict']}'"
        )
        assert decision["scoring_signal"] == "false", (
            f"Expected scoring_signal='false' for maybe verdict, "
            f"got '{decision['scoring_signal']}'"
        )

    @pytest.mark.asyncio
    async def test_invalid_verdict_error_names_the_job_id_and_invalid_string(
        self, decision_recorder: DecisionRecorder
    ) -> None:
        """
        Given an invalid verdict string is provided
        When the decision is recorded
        Then an ActionableError of type DECISION is raised naming both the job_id and the invalid string
        """
        # Given: an invalid verdict

        # When / Then: recording with invalid verdict raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            await decision_recorder.record(
                job_id="zr-999",
                verdict="invalid_verdict",
                jd_text="Some job description text for testing.",
                board="ziprecruiter",
                title="Test Job",
                company="TestCo",
            )

        assert exc_info.value.error_type == ErrorType.DECISION, (
            f"Expected error_type=DECISION, got {exc_info.value.error_type}. "
            f"Full error: {exc_info.value}"
        )
        assert "zr-999" in str(exc_info.value), (
            f"Error should name the job_id 'zr-999'. Got: {exc_info.value}"
        )
        assert "invalid_verdict" in str(exc_info.value), (
            f"Error should name the invalid verdict 'invalid_verdict'. "
            f"Got: {exc_info.value}"
        )

    @pytest.mark.asyncio
    async def test_history_collection_count_increases_after_each_decision(
        self, decision_recorder: DecisionRecorder
    ) -> None:
        """
        Given no decisions have been recorded
        When multiple yes decisions are recorded sequentially
        Then history_count() increases after each recording
        """
        # Given: no decisions yet
        initial_count = decision_recorder.history_count()
        assert initial_count == 0, (
            f"Expected initial history_count=0, got {initial_count}"
        )

        # When: recording first yes decision
        await decision_recorder.record(
            job_id="zr-701",
            verdict="yes",
            jd_text="Staff reliability engineer designing fault-tolerant systems.",
            board="ziprecruiter",
            title="Staff SRE",
            company="HighAvail Corp",
        )

        # Then: count is now 1
        count_after_first = decision_recorder.history_count()
        assert count_after_first == 1, (
            f"Expected history_count=1 after first yes, got {count_after_first}"
        )

        # When: recording second yes decision
        await decision_recorder.record(
            job_id="zr-702",
            verdict="yes",
            jd_text="Principal engineer leading database internals team.",
            board="ziprecruiter",
            title="Principal Engineer — Databases",
            company="DataWorks",
        )

        # Then: count is now 2
        count_after_second = decision_recorder.history_count()
        assert count_after_second == 2, (
            f"Expected history_count=2 after second yes, got {count_after_second}"
        )

    @pytest.mark.asyncio
    async def test_duplicate_decision_on_same_job_id_overwrites_not_appends(
        self, decision_recorder: DecisionRecorder
    ) -> None:
        """
        Given a yes decision has already been recorded for a job_id
        When a no decision is recorded for the same job_id
        Then the stored verdict is updated to 'no' and the count does not increase
        """
        # Given: record an initial yes verdict
        await decision_recorder.record(
            job_id="zr-800",
            verdict="yes",
            jd_text="Senior platform engineer at a cloud-native company.",
            board="ziprecruiter",
            title="Senior Platform Engineer",
            company="CloudNative Inc",
        )
        count_after_first = decision_recorder.history_count()
        assert count_after_first == 1, (
            f"Expected history_count=1 after first record, got {count_after_first}"
        )

        # When: overwrite with a no verdict on the same job_id
        await decision_recorder.record(
            job_id="zr-800",
            verdict="no",
            jd_text="Senior platform engineer at a cloud-native company.",
            board="ziprecruiter",
            title="Senior Platform Engineer",
            company="CloudNative Inc",
        )

        # Then: verdict is updated, count stays the same (upsert, not insert)
        decision = decision_recorder.get_decision("zr-800")
        assert decision is not None, (
            "Expected decision for 'zr-800' to still exist after overwrite"
        )
        assert decision["verdict"] == "no", (
            f"Expected verdict='no' after overwrite, got '{decision['verdict']}'"
        )
        count_after_overwrite = decision_recorder.history_count()
        assert count_after_overwrite == count_after_first, (
            f"Expected history_count to stay at {count_after_first} after overwrite, "
            f"got {count_after_overwrite}"
        )

    @pytest.mark.asyncio
    async def test_empty_jd_text_produces_validation_error(
        self, decision_recorder: DecisionRecorder
    ) -> None:
        """
        Given empty JD text is provided
        When the decision is recorded
        Then an ActionableError of type VALIDATION is raised
        """
        # Given: empty jd_text

        # When / Then: recording with empty jd_text raises validation error
        with pytest.raises(ActionableError) as exc_info:
            await decision_recorder.record(
                job_id="zr-900",
                verdict="yes",
                jd_text="",
                board="ziprecruiter",
                title="Test Job",
                company="TestCo",
            )

        assert exc_info.value.error_type == ErrorType.VALIDATION, (
            f"Expected error_type=VALIDATION, got {exc_info.value.error_type}. "
            f"Full error: {exc_info.value}"
        )
        assert "jd_text" in str(exc_info.value), (
            f"Error should reference the 'jd_text' field. Got: {exc_info.value}"
        )

    @pytest.mark.asyncio
    async def test_reason_is_written_to_jsonl_audit_log(
        self, decision_recorder: DecisionRecorder, tmp_path: Path
    ) -> None:
        """
        When a decision with a reason is recorded
        Then the reason appears in the daily JSONL audit log file
        """
        # Given: a decision with a reason
        reason_text = "Great tech stack alignment and team culture"

        # When: recording the decision
        await decision_recorder.record(
            job_id="zr-1000",
            verdict="yes",
            jd_text="Lead engineer designing event-driven microservices.",
            board="ziprecruiter",
            title="Lead Engineer",
            company="EventCo",
            reason=reason_text,
        )

        # Then: the JSONL audit log contains the reason
        decisions_dir = tmp_path / "decisions"
        jsonl_files = list(decisions_dir.glob("*.jsonl"))
        assert len(jsonl_files) >= 1, (
            f"Expected at least one JSONL file in {decisions_dir}, "
            f"found {len(jsonl_files)}: {jsonl_files}"
        )
        log_content = jsonl_files[0].read_text(encoding="utf-8")
        records = [json.loads(line) for line in log_content.strip().split("\n")]
        matching = [r for r in records if r.get("job_id") == "zr-1000"]
        assert len(matching) == 1, (
            f"Expected exactly 1 JSONL record for 'zr-1000', "
            f"got {len(matching)}: {matching}"
        )
        assert matching[0]["reason"] == reason_text, (
            f"Expected reason='{reason_text}' in JSONL record, "
            f"got '{matching[0].get('reason')}'"
        )

    def test_get_decision_returns_none_when_collection_missing(
        self, tmp_path: Path, mock_embedder: Embedder
    ) -> None:
        """
        Given the decisions collection has never been created
        When get_decision is called
        Then None is returned (graceful degradation)
        """
        # Given: a fresh VectorStore with no decisions collection
        fresh_store = VectorStore(persist_dir=str(tmp_path / "chroma_empty"))
        recorder = DecisionRecorder(
            store=fresh_store,
            embedder=mock_embedder,
            decisions_dir=tmp_path / "decisions_empty",
        )

        # When: looking up a decision in a nonexistent collection
        result = recorder.get_decision("nonexistent-job")

        # Then: None is returned, not an error
        assert result is None, (
            f"Expected None when decisions collection is missing, got {result}"
        )

    def test_get_decision_returns_none_when_no_result_found(
        self, decision_recorder: DecisionRecorder
    ) -> None:
        """
        Given the decisions collection exists but contains no matching job_id
        When get_decision is called with an unknown job_id
        Then None is returned
        """
        # Given: an empty decisions collection (created by the fixture)

        # When: looking up a job_id that was never recorded
        result = decision_recorder.get_decision("never-recorded-job")

        # Then: None is returned
        assert result is None, (
            f"Expected None for unrecorded job_id 'never-recorded-job', got {result}"
        )

    def test_history_count_returns_zero_when_collection_missing(
        self, tmp_path: Path, mock_embedder: Embedder
    ) -> None:
        """
        Given the decisions collection has never been created
        When history_count is called
        Then 0 is returned (graceful degradation)
        """
        # Given: a fresh VectorStore with no decisions collection
        fresh_store = VectorStore(persist_dir=str(tmp_path / "chroma_no_decisions"))
        recorder = DecisionRecorder(
            store=fresh_store,
            embedder=mock_embedder,
            decisions_dir=tmp_path / "decisions_no_coll",
        )

        # When: getting the history count
        count = recorder.history_count()

        # Then: 0 is returned, not an error
        assert count == 0, (
            f"Expected history_count=0 when decisions collection is missing, "
            f"got {count}"
        )
