"""Embedder tests — Ollama wrapper with retry and error handling.

Maps to BDD specs from TestOllamaConnectivity plus Embedder-specific
behaviour: embedding, classification, health check, retry logic.

All tests mock ollama.AsyncClient so no live Ollama is required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.rag.embedder import Embedder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_EMBEDDING = [0.1, 0.2, 0.3, 0.4, 0.5]
BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral:7b"


def _mock_embed_response(embeddings: list[list[float]]) -> MagicMock:
    """Create a mock EmbedResponse with the given embeddings."""
    resp = MagicMock()
    resp.embeddings = embeddings
    return resp


def _mock_chat_response(content: str) -> MagicMock:
    """Create a mock ChatResponse with the given message content."""
    resp = MagicMock()
    resp.message = MagicMock()
    resp.message.content = content
    return resp


def _mock_list_response(model_names: list[str]) -> MagicMock:
    """Create a mock ListResponse with the given model names."""
    models = []
    for name in model_names:
        m = MagicMock()
        m.model = name
        models.append(m)
    resp = MagicMock()
    resp.models = models
    return resp


@pytest.fixture
def embedder() -> Embedder:
    """An Embedder configured for testing."""
    return Embedder(base_url=BASE_URL, embed_model=EMBED_MODEL, llm_model=LLM_MODEL)


# ---------------------------------------------------------------------------
# TestEmbedding
# ---------------------------------------------------------------------------


class TestEmbedding:
    """REQUIREMENT: Text is embedded into a vector via Ollama.

    WHO: The indexer converting resume chunks and archetypes to vectors
    WHAT: A text string is sent to Ollama's embed endpoint and a list of
          floats is returned; the configured model name is used
    WHY: Wrong model or mangled vectors would silently corrupt all similarity
         scores — the entire scoring pipeline depends on correct embeddings
    """

    async def test_embed_returns_float_vector(self, embedder: Embedder) -> None:
        """Embedding a text string returns a list of floats."""
        with patch.object(embedder, "_client") as mock_client:
            mock_client.embed = AsyncMock(
                return_value=_mock_embed_response([FAKE_EMBEDDING])
            )
            result = await embedder.embed("Staff Platform Architect")
            assert result == FAKE_EMBEDDING
            assert all(isinstance(v, float) for v in result)

    async def test_embed_uses_configured_model(self, embedder: Embedder) -> None:
        """The embed call passes the configured embed_model to Ollama."""
        with patch.object(embedder, "_client") as mock_client:
            mock_client.embed = AsyncMock(
                return_value=_mock_embed_response([FAKE_EMBEDDING])
            )
            await embedder.embed("some text")
            mock_client.embed.assert_called_once_with(
                model=EMBED_MODEL, input="some text"
            )

    async def test_embed_strips_whitespace_before_sending(
        self, embedder: Embedder
    ) -> None:
        """Leading/trailing whitespace is stripped so embeddings are consistent."""
        with patch.object(embedder, "_client") as mock_client:
            mock_client.embed = AsyncMock(
                return_value=_mock_embed_response([FAKE_EMBEDDING])
            )
            await embedder.embed("  padded text  ")
            mock_client.embed.assert_called_once_with(
                model=EMBED_MODEL, input="padded text"
            )

    async def test_embed_empty_string_raises_validation_error(
        self, embedder: Embedder
    ) -> None:
        """Embedding an empty string raises a VALIDATION error — no point calling Ollama."""
        with pytest.raises(ActionableError) as exc_info:
            await embedder.embed("")
        assert exc_info.value.error_type == ErrorType.VALIDATION

    async def test_embed_whitespace_only_raises_validation_error(
        self, embedder: Embedder
    ) -> None:
        """Embedding whitespace-only text raises a VALIDATION error after stripping."""
        with pytest.raises(ActionableError) as exc_info:
            await embedder.embed("   \n\t  ")
        assert exc_info.value.error_type == ErrorType.VALIDATION


# ---------------------------------------------------------------------------
# TestClassification
# ---------------------------------------------------------------------------


class TestClassification:
    """REQUIREMENT: LLM classification prompts are sent via Ollama chat.

    WHO: The scorer's disqualifier checking for hard-no signals
    WHAT: A prompt is sent to the LLM and the raw response text is returned;
          the system message establishes the classification role
    WHY: The disqualifier must receive unmodified LLM output to make
         accept/reject decisions — any transformation could flip the result
    """

    async def test_classify_returns_llm_response_text(
        self, embedder: Embedder
    ) -> None:
        """Classification returns the raw LLM response content."""
        with patch.object(embedder, "_client") as mock_client:
            mock_client.chat = AsyncMock(
                return_value=_mock_chat_response("DISQUALIFIED: requires clearance")
            )
            result = await embedder.classify("Is this job suitable?")
            assert result == "DISQUALIFIED: requires clearance"

    async def test_classify_uses_configured_llm_model(
        self, embedder: Embedder
    ) -> None:
        """The classify call passes the configured llm_model to Ollama."""
        with patch.object(embedder, "_client") as mock_client:
            mock_client.chat = AsyncMock(
                return_value=_mock_chat_response("OK")
            )
            await embedder.classify("some prompt")
            call_kwargs = mock_client.chat.call_args
            assert call_kwargs.kwargs["model"] == LLM_MODEL

    async def test_classify_sends_user_message(
        self, embedder: Embedder
    ) -> None:
        """The prompt is sent as a user message in the chat conversation."""
        with patch.object(embedder, "_client") as mock_client:
            mock_client.chat = AsyncMock(
                return_value=_mock_chat_response("OK")
            )
            await embedder.classify("evaluate this listing")
            call_kwargs = mock_client.chat.call_args
            messages = call_kwargs.kwargs["messages"]
            user_msgs = [m for m in messages if m["role"] == "user"]
            assert len(user_msgs) == 1
            assert user_msgs[0]["content"] == "evaluate this listing"


# ---------------------------------------------------------------------------
# TestHealthCheck
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """REQUIREMENT: Ollama unavailability is detected before processing begins.

    WHO: The pipeline runner; the operator who may have forgotten to start Ollama
    WHAT: The health check verifies Ollama is reachable and both configured
          models (embed + LLM) are available; distinct errors for "not running"
          vs "wrong model"
    WHY: Completing a full browser session only to fail at scoring wastes
         time and risks rate limiting; fail fast at startup
    """

    async def test_health_check_passes_when_both_models_available(
        self, embedder: Embedder
    ) -> None:
        """Health check succeeds when both embed and LLM models are listed."""
        with patch.object(embedder, "_client") as mock_client:
            mock_client.list = AsyncMock(
                return_value=_mock_list_response([EMBED_MODEL, LLM_MODEL])
            )
            await embedder.health_check()  # Should not raise

    async def test_unreachable_ollama_raises_connection_error(
        self, embedder: Embedder
    ) -> None:
        """An unreachable Ollama endpoint raises a CONNECTION error naming the URL."""
        with patch.object(embedder, "_client") as mock_client:
            mock_client.list = AsyncMock(
                side_effect=ConnectionError("could not connect")
            )
            with pytest.raises(ActionableError) as exc_info:
                await embedder.health_check()
            assert exc_info.value.error_type == ErrorType.CONNECTION
            assert BASE_URL in exc_info.value.error

    async def test_missing_embed_model_raises_embedding_error(
        self, embedder: Embedder
    ) -> None:
        """A missing embed model raises an EMBEDDING error naming the model."""
        with patch.object(embedder, "_client") as mock_client:
            mock_client.list = AsyncMock(
                return_value=_mock_list_response([LLM_MODEL])  # embed model missing
            )
            with pytest.raises(ActionableError) as exc_info:
                await embedder.health_check()
            assert exc_info.value.error_type == ErrorType.EMBEDDING
            assert EMBED_MODEL in exc_info.value.error

    async def test_missing_llm_model_raises_embedding_error(
        self, embedder: Embedder
    ) -> None:
        """A missing LLM model raises an EMBEDDING error naming the model."""
        with patch.object(embedder, "_client") as mock_client:
            mock_client.list = AsyncMock(
                return_value=_mock_list_response([EMBED_MODEL])  # LLM model missing
            )
            with pytest.raises(ActionableError) as exc_info:
                await embedder.health_check()
            assert exc_info.value.error_type == ErrorType.EMBEDDING
            assert LLM_MODEL in exc_info.value.error


# ---------------------------------------------------------------------------
# TestRetryLogic
# ---------------------------------------------------------------------------


class TestRetryLogic:
    """REQUIREMENT: Transient Ollama failures are retried with backoff.

    WHO: The embedding/classification caller during scoring
    WHAT: Transient timeouts and server errors trigger exponential backoff
          retries; after max retries, a clear EMBEDDING error is raised
          with the retry count
    WHY: Ollama can be slow under load — a single timeout shouldn't abort
         a scoring run that's already invested browser-scraping time
    """

    async def test_transient_error_is_retried(self, embedder: Embedder) -> None:
        """A transient error on the first call is retried and succeeds on the second."""
        from ollama import ResponseError

        with patch.object(embedder, "_client") as mock_client:
            mock_client.embed = AsyncMock(
                side_effect=[
                    ResponseError("server busy", status_code=503),
                    _mock_embed_response([FAKE_EMBEDDING]),
                ]
            )
            result = await embedder.embed("retry me")
            assert result == FAKE_EMBEDDING
            assert mock_client.embed.call_count == 2

    async def test_max_retries_exhausted_raises_embedding_error(
        self, embedder: Embedder
    ) -> None:
        """After exhausting retries, a persistent error raises an EMBEDDING error."""
        from ollama import ResponseError

        with patch.object(embedder, "_client") as mock_client:
            mock_client.embed = AsyncMock(
                side_effect=ResponseError("server busy", status_code=503)
            )
            with pytest.raises(ActionableError) as exc_info:
                await embedder.embed("fail forever")
            assert exc_info.value.error_type == ErrorType.EMBEDDING

    async def test_non_retryable_error_fails_immediately(
        self, embedder: Embedder
    ) -> None:
        """A non-retryable error (e.g., model not found) fails without retrying."""
        from ollama import ResponseError

        with patch.object(embedder, "_client") as mock_client:
            mock_client.embed = AsyncMock(
                side_effect=ResponseError("model 'fake' not found", status_code=404)
            )
            with pytest.raises(ActionableError) as exc_info:
                await embedder.embed("bad model")
            assert exc_info.value.error_type == ErrorType.EMBEDDING
            assert mock_client.embed.call_count == 1

    async def test_classify_retries_on_transient_error(
        self, embedder: Embedder
    ) -> None:
        """Classification also retries on transient errors."""
        from ollama import ResponseError

        with patch.object(embedder, "_client") as mock_client:
            mock_client.chat = AsyncMock(
                side_effect=[
                    ResponseError("server busy", status_code=503),
                    _mock_chat_response("OK"),
                ]
            )
            result = await embedder.classify("retry me")
            assert result == "OK"
            assert mock_client.chat.call_count == 2
