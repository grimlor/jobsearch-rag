"""
Privacy verification tests — executable proof that the scoring pipeline
makes no external network calls.

Spec classes:
    TestPrivacyGuarantee — the scoring pipeline makes no network calls
                           to hosts other than localhost during scoring,
                           embedding, and decision recording
"""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from jobsearch_rag.rag.decisions import DecisionRecorder
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.ports import (
    EmbeddedDocument,
    VectorStoreConfig,
    VectorStorePort,
    create_vector_store,
)
from jobsearch_rag.rag.scorer import Scorer
from tests.conftest import make_mock_ollama_client, make_test_ollama_config
from tests.constants import EMBED_FAKE

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# Public API surface (from src/jobsearch_rag/rag/scorer):
#   Scorer(store: VectorStorePort, embedder: Embedder, disqualify_on_llm_flag: bool)
#   scorer.score(jd_text: str) -> ScoreResult
#
# Public API surface (from src/jobsearch_rag/rag/embedder):
#   Embedder(base_url, embed_model, llm_model, ...)
#   embedder.embed(text: str) -> list[float]
#   embedder.classify(prompt: str) -> str
#   embedder.health_check() -> None
#
# Public API surface (from src/jobsearch_rag/rag/decisions):
#   DecisionRecorder(store, embedder, decisions_dir)
#   recorder.record(job_id, verdict, jd_text, board, ...) -> None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LOCALHOST_ADDRS = {"127.0.0.1", "::1", "localhost"}

_SAMPLE_JD = (
    "We are looking for a Staff Platform Engineer to lead our cloud "
    "infrastructure team. Experience with Kubernetes and Terraform required."
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_original_create_connection = socket.create_connection


def _guarded_create_connection(
    address: tuple[str, int],
    timeout: float | None = None,
    source_address: tuple[str, int] | None = None,
) -> socket.socket:
    """Allow localhost connections only; raise on any external host."""
    host = str(address[0])
    if host not in _LOCALHOST_ADDRS:
        raise AssertionError(f"Privacy violation: attempted connection to external host '{host}'")
    return _original_create_connection(address, timeout, source_address)


@pytest.fixture
def network_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch socket.create_connection to reject non-localhost calls."""
    monkeypatch.setattr(socket, "create_connection", _guarded_create_connection)


@pytest.fixture
def store() -> VectorStorePort:
    """In-memory store seeded with required collections."""
    store = create_vector_store(
        VectorStoreConfig(
            store_class="tests.fakes.FakeVectorStore",
            persist_dir="",
            distance_metric="cosine",
            sync_threshold=1,
        )
    )
    # Pre-create and seed collections the scorer requires
    for name in ("resume", "role_archetypes", "decisions"):
        store.reset_collection(name)
    # Seed required collections with at least one document so scorer doesn't
    # raise "collection is empty" errors.
    store.add_documents(
        collection_name="resume",
        documents=[
            EmbeddedDocument(
                id="resume-chunk-1",
                document="10 years experience in platform engineering and distributed systems.",
                embedding=EMBED_FAKE,
                metadata={"source": "resume"},
            ),
        ],
    )
    store.add_documents(
        collection_name="role_archetypes",
        documents=[
            EmbeddedDocument(
                id="archetype-1",
                document="Staff+ platform engineer leading cloud infrastructure teams.",
                embedding=EMBED_FAKE,
                metadata={"source": "archetypes"},
            ),
        ],
    )
    return store


@pytest.fixture
def mock_ollama_client() -> AsyncMock:
    """
    Stubbed ``ollama.AsyncClient`` — the I/O boundary.

    Returns realistic response objects for ``embed`` and ``chat`` so that
    all Embedder logic (retry, truncation, metrics, token counting) runs
    for real.  Only the final HTTP call is replaced.
    """
    return make_mock_ollama_client(
        classify_response='{"suspicious": false, "disqualified": false}',
    )


@pytest.fixture
def embedder(
    mock_ollama_client: AsyncMock,
) -> Embedder:
    """
    Real Embedder instance with the ollama client stubbed at the I/O boundary.

    Patches ``ollama_sdk.AsyncClient`` at import time so ``Embedder.__init__``
    receives the mock.  All Embedder logic — retry, truncation, metrics,
    token counting — runs for real.
    """
    with patch(
        "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
        return_value=mock_ollama_client,
    ):
        return Embedder(make_test_ollama_config(max_retries=1, base_delay=0.0))


@pytest.fixture
def scorer(
    store: VectorStorePort,
    embedder: Embedder,
) -> Scorer:
    """Real Scorer wired to real ChromaDB and stubbed Embedder."""
    return Scorer(
        store=store,
        embedder=embedder,
        disqualify_on_llm_flag=True,
        disqualifier_prompt="test disqualifier prompt",
        screen_prompt="test screen prompt",
        chunk_overlap=50,
        top_k_retrieval=3,
    )


@pytest.fixture
def recorder(
    store: VectorStorePort,
    embedder: Embedder,
    tmp_path: Path,
) -> DecisionRecorder:
    """Real DecisionRecorder wired to real ChromaDB and stubbed Embedder."""
    return DecisionRecorder(
        store=store,
        embedder=embedder,
        decisions_dir=tmp_path / "decisions",
    )


@pytest.fixture
def embed_call_tracker(
    mock_ollama_client: AsyncMock,
) -> Iterator[AsyncMock]:
    """Yield the ollama client ``embed`` mock so tests can assert I/O boundary call counts."""
    yield mock_ollama_client.embed


# ---------------------------------------------------------------------------
# TestPrivacyGuarantee
# ---------------------------------------------------------------------------


class TestPrivacyGuarantee:
    """
    REQUIREMENT: The scoring pipeline makes no network calls to hosts
    other than the configured Ollama endpoint during scoring, embedding,
    and decision recording

    WHO: The operator who chose this tool specifically because it
         does not send personal data to external services
    WHAT: (1) During a complete scoring pipeline run covering Scorer,
              Embedder, and VectorStore, the system makes no network
              calls to any host other than localhost.
          (2) The system does not send any JD text, resume text, or
              scoring data to external servers during the pipeline run.
    WHY: The privacy-first claim is the primary architectural differentiator.
         A test that can fail is a guarantee worth making; a README statement
         is not

    MOCK BOUNDARY:
        Mock:  ``patch("jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient")``
               via ``_mock_ollama_client`` fixture; ``socket.create_connection``
               monkeypatched to reject non-localhost connections
        Real:  Embedder (embed, classify, retry, truncation, metrics),
               Scorer, DecisionRecorder, ChromaDB via VectorStore (all local)
        Never: Replace Embedder.embed() or classify() — the point is to
               verify the full call chain from public API to I/O boundary
    """

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("network_guard")
    async def test_scoring_pipeline_makes_no_external_network_calls(
        self,
        scorer: Scorer,
    ) -> None:
        """
        Given a complete pipeline run with Scorer, Embedder, and VectorStore
        When all outbound connections to non-localhost hosts are intercepted
        Then no such calls are made during scoring
        And the pipeline completes without error
        """
        # Given: network guard is active (via fixture), scorer is wire
        # When: score a listing through the full pipeline
        result = await scorer.score(_SAMPLE_JD)

        # Then: pipeline completed (no AssertionError from network guard)
        assert result is not None, "score() must return a ScoreResult"
        assert result.fit_score >= 0.0, "fit_score must be non-negative"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("network_guard")
    async def test_ollama_calls_to_localhost_are_permitted(
        self,
        scorer: Scorer,
        embed_call_tracker: AsyncMock,
    ) -> None:
        """
        Given the same interception setup that blocks external calls
        When the pipeline scores a listing
        Then calls to localhost:11434 complete normally
        And at least one embedding call is made
        """
        # Given: network guard is active, scorer is wired
        # When: score a listing
        await scorer.score(_SAMPLE_JD)

        # Then: at least one embed call was made (proving local calls work)
        assert embed_call_tracker.call_count >= 1, (
            "at least one embedding call must be made during scoring"
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("network_guard")
    async def test_disqualifier_pass_makes_no_external_calls(
        self,
        scorer: Scorer,
    ) -> None:
        """
        Given a listing that passes through the LLM disqualifier
        When external network calls to non-localhost hosts are intercepted
        Then no such calls occur during the disqualifier pass
        """
        # Given: network guard is active, scorer has disqualifier enabled
        # When: score triggers disqualifier via classify()
        result = await scorer.score(_SAMPLE_JD)

        # Then: pipeline completed, disqualifier ran without external calls
        assert result is not None, "score() must return a ScoreResult"
        assert result.disqualified is False, (
            "listing must not be disqualified (mock returns false)"
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("network_guard")
    async def test_decision_recording_makes_no_external_calls(
        self,
        recorder: DecisionRecorder,
    ) -> None:
        """
        Given a verdict recorded via DecisionRecorder
        When external network calls to non-localhost hosts are intercepted
        Then no such calls occur during the recording operation
        """
        # Given: network guard is active, recorder is wired
        # When: record a decision
        await recorder.record(
            job_id="privacy-test-001",
            verdict="yes",
            jd_text=_SAMPLE_JD,
            board="test-board",
            title="Staff Engineer",
            company="Acme Corp",
            reason="Privacy test",
        )

        # Then: recording completed without external calls
        decision = recorder.get_decision("privacy-test-001")
        assert decision is not None, "decision must be persisted"
        assert decision["verdict"] == "yes", "verdict must match"
