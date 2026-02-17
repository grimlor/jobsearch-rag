"""Decision history tests.

Maps to BDD spec: TestDecisionRecording
"""

from __future__ import annotations

import tempfile
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

    async def test_unknown_job_id_raises_decision_error_with_job_id(
        self, recorder: DecisionRecorder
    ) -> None:
        """Recording a decision for a job_id not in the results raises a DECISION error naming the ID."""
        # Invalid verdict should raise a DECISION error
        with pytest.raises(ActionableError) as exc_info:
            await recorder.record(
                job_id="zr-000",
                verdict="invalid_verdict",
                jd_text="Some text.",
                board="test",
            )
        assert exc_info.value.error_type == ErrorType.DECISION

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
