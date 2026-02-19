"""Decision history tests.

Maps to BDD spec: TestDecisionRecording
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.rag.decisions import DecisionRecorder
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.store import VectorStore

if TYPE_CHECKING:
    from collections.abc import Iterator

EMBED_TEST = [0.5, 0.5, 0.5, 0.5, 0.5]


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
    embedder.embed = AsyncMock(return_value=EMBED_TEST)  # type: ignore[method-assign]
    return embedder


@pytest.fixture
def recorder(store: VectorStore, mock_embedder: Embedder) -> Iterator[DecisionRecorder]:
    with tempfile.TemporaryDirectory() as decisions_dir:
        # Ensure decisions collection exists
        store.get_or_create_collection("decisions")
        yield DecisionRecorder(
            store=store, embedder=mock_embedder, decisions_dir=decisions_dir
        )


class TestDecisionRecording:
    """REQUIREMENT: User decisions are recorded and build the history signal over time.

    WHO: The scorer computing history_score on future runs
    WHAT: A verdict (yes/no/maybe) is stored with the JD embedding and job_id;
          recording a decision for an unknown job_id raises a clear error;
          the history collection grows with each decision;
          only 'yes' decisions contribute to history_score (maybes and nos do not)
    WHY: If 'no' decisions contributed to scoring, roles similar to rejected ones
         would score lower — but the signal we want is 'what did I like',
         not 'what did I reject' (rejections have too many confounding reasons)
    """

    async def test_yes_verdict_is_stored_in_history_collection(
        self, recorder: DecisionRecorder
    ) -> None:
        """A 'yes' verdict is persisted and becomes part of the history signal for future scoring."""
        await recorder.record(
            job_id="zr-123",
            verdict="yes",
            jd_text="Staff Platform Architect role at great company.",
            board="ziprecruiter",
            title="Staff Architect",
            company="Acme",
        )
        decision = recorder.get_decision("zr-123")
        assert decision is not None
        assert decision["verdict"] == "yes"
        assert decision["scoring_signal"] == "true"

    async def test_reason_is_stored_in_chromadb_metadata(
        self, recorder: DecisionRecorder
    ) -> None:
        """An optional reason is persisted alongside the verdict so the operator's reasoning is preserved."""
        await recorder.record(
            job_id="zr-reason",
            verdict="no",
            jd_text="Interesting role but fully on-site.",
            board="ziprecruiter",
            title="SRE",
            company="OnSite Corp",
            reason="Requires on-site 5 days/week, no remote option",
        )
        decision = recorder.get_decision("zr-reason")
        assert decision is not None
        assert decision["reason"] == "Requires on-site 5 days/week, no remote option"

    async def test_empty_reason_stored_when_not_provided(
        self, recorder: DecisionRecorder
    ) -> None:
        """When no reason is given, an empty string is stored — the field is always present."""
        await recorder.record(
            job_id="zr-noreason",
            verdict="yes",
            jd_text="Great remote role.",
            board="ziprecruiter",
            title="Staff Architect",
            company="Remote Co",
        )
        decision = recorder.get_decision("zr-noreason")
        assert decision is not None
        assert decision["reason"] == ""

    async def test_reason_enriches_embedding_vector(
        self, recorder: DecisionRecorder, mock_embedder: Embedder
    ) -> None:
        """GIVEN a verdict with a reason
        WHEN the decision is recorded
        THEN the embedder receives JD text + reason so the vector captures operator intent.
        """
        jd = "Staff Platform Architect role at Acme Corp."
        reason = "Fully remote with architecture leadership"
        await recorder.record(
            job_id="zr-enrich",
            verdict="yes",
            jd_text=jd,
            board="ziprecruiter",
            reason=reason,
        )
        mock_embedder.embed.assert_called_once_with(
            f"{jd}\n\nOperator reasoning: {reason}"
        )

    async def test_empty_reason_does_not_enrich_embedding(
        self, recorder: DecisionRecorder, mock_embedder: Embedder
    ) -> None:
        """GIVEN a verdict with no reason
        WHEN the decision is recorded
        THEN the embedder receives only the bare JD text — no enrichment suffix.
        """
        jd = "Staff Platform Architect role at Acme Corp."
        await recorder.record(
            job_id="zr-bare",
            verdict="yes",
            jd_text=jd,
            board="ziprecruiter",
        )
        mock_embedder.embed.assert_called_once_with(jd)

    async def test_no_verdict_is_stored_but_excluded_from_scoring_signal(
        self, recorder: DecisionRecorder
    ) -> None:
        """A 'no' verdict is recorded for audit but does not influence history_score — rejections have confounding reasons."""
        await recorder.record(
            job_id="zr-456",
            verdict="no",
            jd_text="SRE on-call role, not a fit.",
            board="ziprecruiter",
            title="SRE",
            company="Other Co",
        )
        decision = recorder.get_decision("zr-456")
        assert decision is not None
        assert decision["verdict"] == "no"
        assert decision["scoring_signal"] == "false"

    async def test_maybe_verdict_is_stored_but_excluded_from_scoring_signal(
        self, recorder: DecisionRecorder
    ) -> None:
        """A 'maybe' verdict is recorded but excluded from scoring — only clear 'yes' signals are useful."""
        await recorder.record(
            job_id="zr-789",
            verdict="maybe",
            jd_text="Interesting but unclear role.",
            board="ziprecruiter",
            title="Ambiguous Role",
            company="Maybe Co",
        )
        decision = recorder.get_decision("zr-789")
        assert decision is not None
        assert decision["verdict"] == "maybe"
        assert decision["scoring_signal"] == "false"

    async def test_unknown_job_id_names_the_id_and_suggests_checking_latest_output(
        self, recorder: DecisionRecorder
    ) -> None:
        """An invalid verdict names the ID and suggests checking the latest output."""
        with pytest.raises(ActionableError) as exc_info:
            await recorder.record(
                job_id="zr-000",
                verdict="invalid_verdict",
                jd_text="Some text.",
                board="test",
            )
        err = exc_info.value
        assert err.error_type == ErrorType.DECISION
        assert err.suggestion is not None
        assert err.troubleshooting is not None

    async def test_history_collection_count_increases_after_each_decision(
        self, recorder: DecisionRecorder
    ) -> None:
        """Each new decision adds one document to the history collection, growing the signal over time."""
        initial = recorder.history_count()
        await recorder.record(
            job_id="job-a",
            verdict="yes",
            jd_text="First role description.",
            board="test",
        )
        assert recorder.history_count() == initial + 1

        await recorder.record(
            job_id="job-b",
            verdict="no",
            jd_text="Second role description.",
            board="test",
        )
        assert recorder.history_count() == initial + 2

    async def test_duplicate_decision_on_same_job_id_overwrites_not_appends(
        self, recorder: DecisionRecorder
    ) -> None:
        """Changing a verdict on the same job replaces the previous decision rather than duplicating it."""
        await recorder.record(
            job_id="zr-dup",
            verdict="maybe",
            jd_text="A role I was unsure about.",
            board="ziprecruiter",
        )
        count_after_first = recorder.history_count()

        # Change verdict — should overwrite, not append
        await recorder.record(
            job_id="zr-dup",
            verdict="yes",
            jd_text="A role I was unsure about.",
            board="ziprecruiter",
        )
        assert recorder.history_count() == count_after_first  # no increase

        decision = recorder.get_decision("zr-dup")
        assert decision is not None
        assert decision["verdict"] == "yes"

    async def test_empty_jd_text_tells_operator_to_provide_content(
        self, recorder: DecisionRecorder
    ) -> None:
        """Empty JD text produces a VALIDATION error telling the operator to provide content."""
        with pytest.raises(ActionableError) as exc_info:
            await recorder.record(
                job_id="zr-empty",
                verdict="yes",
                jd_text="   ",
                board="ziprecruiter",
            )
        err = exc_info.value
        assert err.error_type == ErrorType.VALIDATION
        assert err.suggestion is not None
        assert err.troubleshooting is not None

    async def test_reason_is_written_to_jsonl_audit_log(
        self, store: VectorStore, mock_embedder: Embedder
    ) -> None:
        """The reason field appears in the daily JSONL audit file alongside the verdict."""
        import json as json_mod

        with tempfile.TemporaryDirectory() as decisions_dir:
            store.get_or_create_collection("decisions")
            rec = DecisionRecorder(
                store=store, embedder=mock_embedder, decisions_dir=decisions_dir
            )
            await rec.record(
                job_id="zr-audit",
                verdict="no",
                jd_text="On-site only role.",
                board="ziprecruiter",
                title="SRE",
                company="Corp",
                reason="No remote option",
            )
            # Read the JSONL file
            jsonl_files = list(Path(decisions_dir).glob("*.jsonl"))
            assert len(jsonl_files) == 1
            records = [
                json_mod.loads(line)
                for line in jsonl_files[0].read_text().strip().splitlines()
            ]
            assert len(records) == 1
            assert records[0]["reason"] == "No remote option"

    def test_get_decision_returns_none_when_collection_missing(
        self, mock_embedder: Embedder
    ) -> None:
        """GIVEN a store with no decisions collection
        WHEN get_decision() is called
        THEN None is returned instead of raising.
        """
        from unittest.mock import MagicMock

        mock_store = MagicMock()
        mock_store.get_documents.side_effect = ActionableError(
            error="Collection not found",
            error_type=ErrorType.INDEX,
            service="chromadb",
        )
        recorder = DecisionRecorder(store=mock_store, embedder=mock_embedder)
        assert recorder.get_decision("nonexistent-job") is None

    def test_get_decision_returns_none_when_no_results_found(
        self, mock_embedder: Embedder
    ) -> None:
        """GIVEN a decisions collection that exists but has no matching document
        WHEN get_decision() is called
        THEN None is returned.
        """
        from unittest.mock import MagicMock

        mock_store = MagicMock()
        mock_store.get_documents.return_value = {"ids": [], "metadatas": []}
        recorder = DecisionRecorder(store=mock_store, embedder=mock_embedder)
        assert recorder.get_decision("unknown-id") is None

    def test_history_count_returns_zero_when_collection_missing(
        self, mock_embedder: Embedder
    ) -> None:
        """GIVEN a store where the decisions collection does not exist
        WHEN history_count() is called
        THEN 0 is returned instead of raising.
        """
        from unittest.mock import MagicMock

        mock_store = MagicMock()
        mock_store.collection_count.side_effect = ActionableError(
            error="Collection not found",
            error_type=ErrorType.INDEX,
            service="chromadb",
        )
        recorder = DecisionRecorder(store=mock_store, embedder=mock_embedder)
        assert recorder.history_count() == 0
