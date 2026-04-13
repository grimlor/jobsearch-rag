"""
BDD specs for test fakes that satisfy port protocols.

Covers: FakeEmbedder (D2).

Public API surface (from tests/fakes):
    FakeEmbedder(
        embed_vector: list[float] = [0.0, ...],
        classify_response: str = "{}",
        embed_side_effect: Callable | None = None,
        classify_side_effect: Callable | None = None,
    )
    await fake.embed(text: str) -> list[float]
    await fake.classify(prompt: str) -> str
    fake.embed_calls: list[str]      # recorded arguments
    fake.embed_call_count: int
    fake.classify_call_count: int

Public API surface (from src/jobsearch_rag/ports):
    EmbeddingPort  — Protocol (embed, classify)
    HealthCheckable — Protocol (health_check)
    MetricsProvider — Protocol (metrics property)
"""

from __future__ import annotations

from typing import cast

import pytest

from jobsearch_rag.ports import EmbeddingPort, HealthCheckable, MetricsProvider
from tests.fakes import FakeEmbedder


class TestFakeEmbedder:
    """
    REQUIREMENT: FakeEmbedder is a test double that satisfies EmbeddingPort
    (but NOT HealthCheckable or MetricsProvider) with configurable,
    deterministic behavior and no Ollama dependency.

    WHO: All unit tests that need an embedder — replaces the mock_embedder
         fixture backed by patched ollama_sdk.AsyncClient.
    WHAT: (1) FakeEmbedder satisfies EmbeddingPort (isinstance check).
          (2) FakeEmbedder does NOT satisfy HealthCheckable.
          (3) FakeEmbedder does NOT satisfy MetricsProvider.
          (4) embed() returns a configurable fixed vector.
          (5) classify() returns a configurable fixed response string.
          (6) embed() can be configured with a side_effect callable for
              per-call behavior (e.g., error injection, varying vectors).
          (7) classify() can be configured with a side_effect callable.
          (8) embed() records call arguments for inspection.
          (9) embed() tracks call count.
          (10) classify() tracks call count.
    WHY: FakeEmbedder eliminates all ollama_sdk.AsyncClient patching and
         ~15 type: ignore[union-attr] suppressions from _client access.
         Tests express intent ("embedder returns this vector") instead of
         SDK internals ("mock_client.embed.return_value.embeddings = ...").
         Not implementing HealthCheckable/MetricsProvider keeps the fake
         minimal and validates the isinstance guard pattern in PipelineRunner.

    MOCK BOUNDARY:
        Mock:  Nothing — FakeEmbedder IS the test double
        Real:  FakeEmbedder, EmbeddingPort protocol check
        Never: Mock FakeEmbedder internals
    """

    def test_fake_embedder_satisfies_embedding_port(self) -> None:
        """
        Given a FakeEmbedder instance
        When isinstance(fake, EmbeddingPort) is checked
        Then it returns True
        """
        # Given: a FakeEmbedder instance

        fake = cast("object", FakeEmbedder())

        # When: isinstance check against EmbeddingPort
        result = isinstance(fake, EmbeddingPort)

        # Then: FakeEmbedder satisfies EmbeddingPort
        assert result is True, (
            f"FakeEmbedder should satisfy EmbeddingPort. isinstance returned {result}"
        )

    def test_fake_embedder_does_not_satisfy_health_checkable(self) -> None:
        """
        Given a FakeEmbedder instance
        When isinstance(fake, HealthCheckable) is checked
        Then it returns False
        """
        # Given: a FakeEmbedder instance

        fake = cast("object", FakeEmbedder())

        # When: isinstance check against HealthCheckable
        result = isinstance(fake, HealthCheckable)

        # Then: FakeEmbedder does NOT satisfy HealthCheckable
        assert result is False, (
            f"FakeEmbedder should NOT satisfy HealthCheckable. isinstance returned {result}"
        )

    def test_fake_embedder_does_not_satisfy_metrics_provider(self) -> None:
        """
        Given a FakeEmbedder instance
        When isinstance(fake, MetricsProvider) is checked
        Then it returns False
        """
        # Given: a FakeEmbedder instance

        fake = cast("object", FakeEmbedder())

        # When: isinstance check against MetricsProvider
        result = isinstance(fake, MetricsProvider)

        # Then: FakeEmbedder does NOT satisfy MetricsProvider
        assert result is False, (
            f"FakeEmbedder should NOT satisfy MetricsProvider. isinstance returned {result}"
        )

    @pytest.mark.anyio()
    async def test_embed_returns_configured_vector(self) -> None:
        """
        Given a FakeEmbedder configured with embed_vector=[0.1, 0.2, 0.3]
        When embed("any text") is awaited
        Then it returns [0.1, 0.2, 0.3]
        """
        # Given: a FakeEmbedder with a specific embed_vector
        fake = FakeEmbedder(embed_vector=[0.1, 0.2, 0.3])

        # When: embed is awaited
        result = await fake.embed("any text")

        # Then: the configured vector is returned
        assert result == [0.1, 0.2, 0.3], f"Expected [0.1, 0.2, 0.3], got {result}"

    @pytest.mark.anyio()
    async def test_classify_returns_configured_response(self) -> None:
        """
        Given a FakeEmbedder configured with classify_response='{"disqualified": false}'
        When classify("any prompt") is awaited
        Then it returns '{"disqualified": false}'
        """
        # Given: a FakeEmbedder with a specific classify_response
        fake = FakeEmbedder(classify_response='{"disqualified": false}')

        # When: classify is awaited
        result = await fake.classify("any prompt")

        # Then: the configured response is returned
        assert result == '{"disqualified": false}', (
            f"Expected '{{\"disqualified\": false}}', got {result!r}"
        )

    @pytest.mark.anyio()
    async def test_embed_side_effect_overrides_fixed_vector(self) -> None:
        """
        Given a FakeEmbedder with a side_effect callable on embed
        When embed() is awaited
        Then the side_effect return value is used instead of the fixed vector
        """
        # Given: a FakeEmbedder with embed_side_effect returning a custom vector
        custom_vector = [9.0, 8.0, 7.0]
        fake = FakeEmbedder(
            embed_vector=[0.0, 0.0, 0.0],
            embed_side_effect=lambda text: custom_vector,
        )

        # When: embed is awaited
        result = await fake.embed("ignored input")

        # Then: side_effect return value overrides the fixed vector
        assert result == custom_vector, (
            f"Expected side_effect vector {custom_vector}, got {result}"
        )

    @pytest.mark.anyio()
    async def test_classify_side_effect_overrides_fixed_response(self) -> None:
        """
        Given a FakeEmbedder with a side_effect callable on classify
        When classify() is awaited
        Then the side_effect return value is used instead of the fixed response
        """
        # Given: a FakeEmbedder with classify_side_effect returning a custom response
        custom_response = "side_effect_response"
        fake = FakeEmbedder(
            classify_response="default_response",
            classify_side_effect=lambda prompt: custom_response,
        )

        # When: classify is awaited
        result = await fake.classify("ignored prompt")

        # Then: side_effect return value overrides the fixed response
        assert result == custom_response, (
            f"Expected side_effect response {custom_response!r}, got {result!r}"
        )

    @pytest.mark.anyio()
    async def test_embed_records_call_arguments(self) -> None:
        """
        Given a FakeEmbedder
        When embed("specific text") is awaited
        Then the call arguments are recorded and inspectable
        """
        # Given: a FakeEmbedder
        fake = FakeEmbedder()

        # When: embed is called with specific text
        await fake.embed("specific text")

        # Then: the call argument is recorded
        assert fake.embed_calls == ["specific text"], (
            f"Expected embed_calls=['specific text'], got {fake.embed_calls}"
        )

    @pytest.mark.anyio()
    async def test_embed_tracks_call_count(self) -> None:
        """
        Given a FakeEmbedder
        When embed() is called 3 times
        Then the call count is 3
        """
        # Given: a FakeEmbedder
        fake = FakeEmbedder()

        # When: embed is called 3 times
        await fake.embed("one")
        await fake.embed("two")
        await fake.embed("three")

        # Then: call count is 3
        assert fake.embed_call_count == 3, (
            f"Expected embed_call_count=3, got {fake.embed_call_count}"
        )

    @pytest.mark.anyio()
    async def test_classify_tracks_call_count(self) -> None:
        """
        Given a FakeEmbedder
        When classify() is called 2 times
        Then the call count is 2
        """
        # Given: a FakeEmbedder
        fake = FakeEmbedder()

        # When: classify is called 2 times
        await fake.classify("prompt one")
        await fake.classify("prompt two")

        # Then: call count is 2
        assert fake.classify_call_count == 2, (
            f"Expected classify_call_count=2, got {fake.classify_call_count}"
        )
