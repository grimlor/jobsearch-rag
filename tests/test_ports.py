"""
BDD specifications for hexagonal port protocols and result dataclasses.

Verifies that EmbeddingPort, HealthCheckable, MetricsProvider, and
VectorStorePort protocols are correctly defined, that concrete
implementations satisfy them structurally, and that QueryResult/GetResult
dataclasses provide typed access to vector store responses.

Implements D1 of Feature -- hexagonal-port-interfaces.
"""

# Public API surface (from src/jobsearch_rag/ports):
#   EmbeddingPort  -- Protocol: embed(text: str) -> list[float],
#                              classify(prompt: str) -> str
#   HealthCheckable -- Protocol: health_check() -> None
#   MetricsProvider -- Protocol: metrics -> InferenceMetrics (property)
#   VectorStorePort -- Protocol: add_documents, query, get_documents,
#                     get_by_metadata, get_all_documents, delete_by_id,
#                     collection_count, reset_collection, close
#   QueryResult -- dataclass: ids, documents, metadatas, distances
#   GetResult   -- dataclass: ids, documents, metadatas
#
# Public API surface (from src/jobsearch_rag/rag/embedder):
#   InferenceMetrics -- dataclass: embed_calls, embed_tokens_total, etc.
#   Embedder(config: OllamaConfig) -- satisfies EmbeddingPort,
#            HealthCheckable, MetricsProvider
#
# Public API surface (from src/jobsearch_rag/rag/store):
#   VectorStore(persist_dir: str, distance_metric: str)
#     -- satisfies VectorStorePort

from __future__ import annotations

import inspect
from typing import Any, cast, get_type_hints
from unittest.mock import patch

# D1 imports -- will be created in src/jobsearch_rag/ports.py
from jobsearch_rag.ports import (
    EmbeddingPort,
    GetResult,
    HealthCheckable,
    MetricsProvider,
    QueryResult,
    VectorStorePort,
)
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.store import VectorStore
from tests.conftest import make_test_ollama_config

# ---------------------------------------------------------------------------
# Helper: minimal embed-only stub for protocol rejection tests
# ---------------------------------------------------------------------------
# Used in D1 to test that HealthCheckable and MetricsProvider correctly
# reject instances that only satisfy EmbeddingPort.  Replaced by the
# full FakeEmbedder in D2.


class _EmbedOnlyStub:
    """Minimal stub satisfying only EmbeddingPort -- no health_check, no metrics."""

    async def embed(self, text: str) -> list[float]:
        return [0.0]

    async def classify(self, prompt: str) -> str:
        return ""


# ============================================================================
# TestEmbeddingPortProtocol
# ============================================================================


class TestEmbeddingPortProtocol:
    """
    REQUIREMENT: EmbeddingPort defines a Protocol that captures the core
    embedding and LLM classification operations the domain calls on Embedder.
    Observability concerns (health_check, metrics) live in separate protocols.

    WHO: Domain classes (Scorer, DecisionRecorder, Indexer) -- they
         depend on this protocol instead of the concrete Embedder class.
    WHAT: (1) EmbeddingPort is a typing.Protocol with @runtime_checkable.
          (2) It declares async method embed(text: str) -> list[float].
          (3) It declares async method classify(prompt: str) -> str.
          (4) It does NOT declare health_check (belongs to HealthCheckable).
          (5) It does NOT declare metrics (belongs to MetricsProvider).
          (6) The concrete Embedder class satisfies EmbeddingPort structurally
              (isinstance check passes without inheritance).
    WHY: Without a protocol, domain classes import the concrete Embedder,
         coupling them to Ollama. Tests must mock at SDK internals.
         With the protocol, any implementation satisfying the contract
         can be substituted -- including test fakes. Keeping the port
         narrow (embed + classify only) means fakes don't need health
         or metrics stubs.

    MOCK BOUNDARY:
        Mock:  Nothing -- these are pure protocol/type contract tests
        Real:  EmbeddingPort definition, Embedder class, isinstance checks
        Never: Mock the protocol itself
    """

    def test_embedding_port_is_runtime_checkable_protocol(self) -> None:
        """
        Given the EmbeddingPort protocol
        When inspected at runtime
        Then it is a typing.Protocol with @runtime_checkable
        """
        # Given: the EmbeddingPort protocol class

        # When: checking protocol attributes
        is_protocol = getattr(EmbeddingPort, "_is_protocol", False)
        is_runtime = getattr(EmbeddingPort, "_is_runtime_protocol", False)

        # Then: it is a Protocol and runtime_checkable
        assert is_protocol, (
            f"EmbeddingPort should be a typing.Protocol. Got _is_protocol={is_protocol}"
        )
        assert is_runtime, "EmbeddingPort should be @runtime_checkable"

    def test_embedding_port_declares_embed_method(self) -> None:
        """
        Given the EmbeddingPort protocol
        When its members are inspected
        Then it declares an async method embed(text: str) -> list[float]
        """
        # Given: the EmbeddingPort protocol

        # When: inspecting the embed member
        assert hasattr(EmbeddingPort, "embed"), "EmbeddingPort should declare an 'embed' method"
        sig = inspect.signature(EmbeddingPort.embed)
        params = list(sig.parameters.keys())
        hints = get_type_hints(EmbeddingPort.embed)

        # Then: embed takes (self, text: str) -> list[float]
        assert "text" in params, f"embed() should have a 'text' parameter. Got params: {params}"
        assert hints.get("text") is str, (
            f"embed(text) should be typed as str. Got: {hints.get('text')}"
        )
        assert hints.get("return") == list[float], (
            f"embed() return type should be list[float]. Got: {hints.get('return')}"
        )
        assert inspect.iscoroutinefunction(EmbeddingPort.embed), (
            "embed() should be an async method"
        )

    def test_embedding_port_declares_classify_method(self) -> None:
        """
        Given the EmbeddingPort protocol
        When its members are inspected
        Then it declares an async method classify(prompt: str) -> str
        """
        # Given: the EmbeddingPort protocol

        # When: inspecting the classify member
        assert hasattr(EmbeddingPort, "classify"), (
            "EmbeddingPort should declare a 'classify' method"
        )
        sig = inspect.signature(EmbeddingPort.classify)
        params = list(sig.parameters.keys())
        hints = get_type_hints(EmbeddingPort.classify)

        # Then: classify takes (self, prompt: str) -> str
        assert "prompt" in params, (
            f"classify() should have a 'prompt' parameter. Got params: {params}"
        )
        assert hints.get("prompt") is str, (
            f"classify(prompt) should be typed as str. Got: {hints.get('prompt')}"
        )
        assert hints.get("return") is str, (
            f"classify() return type should be str. Got: {hints.get('return')}"
        )
        assert inspect.iscoroutinefunction(EmbeddingPort.classify), (
            "classify() should be an async method"
        )

    def test_embedding_port_does_not_declare_health_check(self) -> None:
        """
        Given the EmbeddingPort protocol
        When its members are inspected
        Then health_check is not a member (it belongs to HealthCheckable)
        """
        # Given: the EmbeddingPort protocol

        # When: checking for health_check on the protocol's own members
        own_members = {name for name in dir(EmbeddingPort) if not name.startswith("_")}

        # Then: health_check is not among EmbeddingPort's declared members
        assert "health_check" not in own_members, (
            f"EmbeddingPort should NOT declare health_check "
            f"(it belongs to HealthCheckable). Found members: {own_members}"
        )

    def test_embedding_port_does_not_declare_metrics(self) -> None:
        """
        Given the EmbeddingPort protocol
        When its members are inspected
        Then metrics is not a member (it belongs to MetricsProvider)
        """
        # Given: the EmbeddingPort protocol

        # When: checking for metrics on the protocol's own members
        own_members = {name for name in dir(EmbeddingPort) if not name.startswith("_")}

        # Then: metrics is not among EmbeddingPort's declared members
        assert "metrics" not in own_members, (
            f"EmbeddingPort should NOT declare metrics "
            f"(it belongs to MetricsProvider). Found members: {own_members}"
        )

    def test_embedder_satisfies_embedding_port(self) -> None:
        """
        Given a real Embedder instance (with mocked ollama client at I/O boundary)
        When isinstance(embedder, EmbeddingPort) is checked
        Then it returns True (structural conformance, no inheritance required)
        """
        # Given: a real Embedder with the ollama client mocked at I/O boundary
        with patch(
            "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
        ):
            embedder = Embedder(make_test_ollama_config(max_retries=1, base_delay=0.0))

        # When: checking isinstance (cast to object to test runtime conformance)
        instance = cast("object", embedder)
        result = isinstance(instance, EmbeddingPort)

        # Then: structural conformance -- no inheritance required
        assert result, (
            f"Embedder should satisfy EmbeddingPort structurally. "
            f"isinstance returned {result}. Embedder bases: {type(embedder).__mro__}"
        )


# ============================================================================
# TestHealthCheckableProtocol
# ============================================================================


class TestHealthCheckableProtocol:
    """
    REQUIREMENT: HealthCheckable is a supplementary protocol for
    implementations that support pre-flight connectivity verification.

    WHO: PipelineRunner.run() -- uses isinstance guard to optionally
         run health_check before pipeline execution.
    WHAT: (1) HealthCheckable is a typing.Protocol with @runtime_checkable.
          (2) It declares async method health_check() -> None.
          (3) Embedder satisfies HealthCheckable structurally.
          (4) An embed-only implementation does NOT satisfy HealthCheckable
              (by design -- fakes don't need observability).
    WHY: health_check is an observability concern, not an embedding
         operation. Splitting it into a separate protocol keeps
         EmbeddingPort narrow and fakes minimal.

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (for Embedder construction only)
        Real:  HealthCheckable definition, Embedder, _EmbedOnlyStub, isinstance
        Never: Mock the protocol itself
    """

    def test_health_checkable_is_runtime_checkable_protocol(self) -> None:
        """
        Given the HealthCheckable protocol
        When inspected at runtime
        Then it is a typing.Protocol with @runtime_checkable
        """
        # Given: the HealthCheckable protocol class

        # When: checking protocol attributes
        is_protocol = getattr(HealthCheckable, "_is_protocol", False)
        is_runtime = getattr(HealthCheckable, "_is_runtime_protocol", False)

        # Then: it is a Protocol and runtime_checkable
        assert is_protocol, (
            f"HealthCheckable should be a typing.Protocol. Got _is_protocol={is_protocol}"
        )
        assert is_runtime, "HealthCheckable should be @runtime_checkable"

    def test_health_checkable_declares_health_check(self) -> None:
        """
        Given the HealthCheckable protocol
        When its members are inspected
        Then it declares an async method health_check() -> None
        """
        # Given: the HealthCheckable protocol

        # When: inspecting the health_check member
        assert hasattr(HealthCheckable, "health_check"), (
            "HealthCheckable should declare a 'health_check' method"
        )
        hints = get_type_hints(HealthCheckable.health_check)

        # Then: health_check() -> None and is async
        assert hints.get("return") is type(None), (
            f"health_check() return type should be None. Got: {hints.get('return')}"
        )
        assert inspect.iscoroutinefunction(HealthCheckable.health_check), (
            "health_check() should be an async method"
        )

    def test_embedder_satisfies_health_checkable(self) -> None:
        """
        Given a real Embedder instance (with mocked ollama client at I/O boundary)
        When isinstance(embedder, HealthCheckable) is checked
        Then it returns True
        """
        # Given: a real Embedder with I/O boundary mocked
        with patch(
            "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
        ):
            embedder = Embedder(make_test_ollama_config(max_retries=1, base_delay=0.0))

        # When: checking isinstance (cast to object to test runtime conformance)
        instance = cast("object", embedder)
        result = isinstance(instance, HealthCheckable)

        # Then: Embedder has health_check → satisfies HealthCheckable
        assert result, f"Embedder should satisfy HealthCheckable. isinstance returned {result}"

    def test_embed_only_does_not_satisfy_health_checkable(self) -> None:
        """
        Given an embed-only stub (satisfies EmbeddingPort but not HealthCheckable)
        When isinstance(stub, HealthCheckable) is checked
        Then it returns False
        """
        # Given: an _EmbedOnlyStub instance (no health_check method)
        stub = _EmbedOnlyStub()

        # When: checking isinstance
        result = isinstance(stub, HealthCheckable)

        # Then: the stub lacks health_check → does not satisfy HealthCheckable
        assert not result, (
            f"_EmbedOnlyStub should NOT satisfy HealthCheckable. isinstance returned {result}"
        )


# ============================================================================
# TestMetricsProviderProtocol
# ============================================================================


class TestMetricsProviderProtocol:
    """
    REQUIREMENT: MetricsProvider is a supplementary protocol for
    implementations that expose inference metrics.

    WHO: PipelineRunner.run() -- uses isinstance guard to optionally
         collect session metrics after pipeline execution.
    WHAT: (1) MetricsProvider is a typing.Protocol with @runtime_checkable.
          (2) It declares a read-only property metrics -> InferenceMetrics.
          (3) Embedder satisfies MetricsProvider structurally.
          (4) An embed-only implementation does NOT satisfy MetricsProvider
              (by design).
    WHY: Metrics are an observability concern. Splitting them into a
         separate protocol keeps EmbeddingPort narrow and fakes minimal.

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (for Embedder construction only)
        Real:  MetricsProvider definition, Embedder, _EmbedOnlyStub, isinstance
        Never: Mock the protocol itself
    """

    def test_metrics_provider_is_runtime_checkable_protocol(self) -> None:
        """
        Given the MetricsProvider protocol
        When inspected at runtime
        Then it is a typing.Protocol with @runtime_checkable
        """
        # Given: the MetricsProvider protocol class

        # When: checking protocol attributes
        is_protocol = getattr(MetricsProvider, "_is_protocol", False)
        is_runtime = getattr(MetricsProvider, "_is_runtime_protocol", False)

        # Then: it is a Protocol and runtime_checkable
        assert is_protocol, (
            f"MetricsProvider should be a typing.Protocol. Got _is_protocol={is_protocol}"
        )
        assert is_runtime, "MetricsProvider should be @runtime_checkable"

    def test_metrics_provider_declares_metrics_property(self) -> None:
        """
        Given the MetricsProvider protocol
        When its members are inspected
        Then it declares a read-only property metrics -> InferenceMetrics
        """
        # Given: the MetricsProvider protocol

        # When: inspecting the metrics member
        assert hasattr(MetricsProvider, "metrics"), (
            "MetricsProvider should declare a 'metrics' member"
        )
        prop = vars(MetricsProvider).get("metrics")
        assert isinstance(prop, property), (
            f"metrics should be a property descriptor. Got: {type(prop)}"
        )
        fget = prop.fget
        assert fget is not None, "metrics property should have a getter"
        annotations = fget.__annotations__

        # Then: metrics return is typed as InferenceMetrics
        return_annotation = annotations.get("return", "")
        assert "InferenceMetrics" in str(return_annotation), (
            f"metrics property return type should be InferenceMetrics. "
            f"Got annotation: {return_annotation}"
        )

    def test_embedder_satisfies_metrics_provider(self) -> None:
        """
        Given a real Embedder instance (with mocked ollama client at I/O boundary)
        When isinstance(embedder, MetricsProvider) is checked
        Then it returns True
        """
        # Given: a real Embedder with I/O boundary mocked
        with patch(
            "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
        ):
            embedder = Embedder(make_test_ollama_config(max_retries=1, base_delay=0.0))

        # When: checking isinstance (cast to object to test runtime conformance)
        instance = cast("object", embedder)
        result = isinstance(instance, MetricsProvider)

        # Then: Embedder has metrics property → satisfies MetricsProvider
        assert result, f"Embedder should satisfy MetricsProvider. isinstance returned {result}"

    def test_embed_only_does_not_satisfy_metrics_provider(self) -> None:
        """
        Given an embed-only stub (satisfies EmbeddingPort but not MetricsProvider)
        When isinstance(stub, MetricsProvider) is checked
        Then it returns False
        """
        # Given: an _EmbedOnlyStub instance (no metrics property)
        stub = _EmbedOnlyStub()

        # When: checking isinstance
        result = isinstance(stub, MetricsProvider)

        # Then: the stub lacks metrics → does not satisfy MetricsProvider
        assert not result, (
            f"_EmbedOnlyStub should NOT satisfy MetricsProvider. isinstance returned {result}"
        )


# ============================================================================
# TestVectorStorePortProtocol
# ============================================================================


class TestVectorStorePortProtocol:
    """
    REQUIREMENT: VectorStorePort defines a Protocol that captures all
    vector storage operations the domain calls on VectorStore.

    WHO: Domain classes (Scorer, DecisionRecorder, Indexer, EvalRunner)
         -- they depend on this protocol instead of the concrete VectorStore.
    WHAT: (1) VectorStorePort is a typing.Protocol with @runtime_checkable.
          (2) It declares add_documents(collection_name, *, ids, documents,
              embeddings, metadatas=None) -> None.
          (3) It declares query(collection_name, *, query_embedding,
              n_results=5) -> QueryResult.
          (4) It declares get_documents(collection_name, *, ids) -> GetResult.
          (5) It declares get_by_metadata(collection_name, *, where,
              include=None) -> GetResult.
          (6) It declares get_all_documents(collection_name, *,
              include=None) -> GetResult.
          (7) It declares delete_by_id(collection_name, *, ids) -> None.
          (8) It declares collection_count(name: str) -> int.
          (9) It declares reset_collection(name: str) -> None.
          (10) It declares close() -> None.
          (11) The concrete VectorStore class satisfies VectorStorePort
               structurally (isinstance check passes without inheritance).
    WHY: Without a protocol, domain classes import VectorStore which imports
         chromadb. Tests use real ChromaDB temp dirs or mock at SDK internals.
         With the protocol, InMemoryVectorStore can replace ChromaDB in unit
         tests -- no WAL isolation issues, no SDK mocking.

    MOCK BOUNDARY:
        Mock:  Nothing -- pure protocol/type contract tests
        Real:  VectorStorePort definition, VectorStore class, isinstance checks
        Never: Mock the protocol itself
    """

    def test_vector_store_port_is_runtime_checkable_protocol(self) -> None:
        """
        Given the VectorStorePort protocol
        When inspected at runtime
        Then it is a typing.Protocol with @runtime_checkable
        """
        # Given: the VectorStorePort protocol class

        # When: checking protocol attributes
        is_protocol = getattr(VectorStorePort, "_is_protocol", False)
        is_runtime = getattr(VectorStorePort, "_is_runtime_protocol", False)

        # Then: it is a Protocol and runtime_checkable
        assert is_protocol, (
            f"VectorStorePort should be a typing.Protocol. Got _is_protocol={is_protocol}"
        )
        assert is_runtime, "VectorStorePort should be @runtime_checkable"

    def test_vector_store_port_declares_add_documents(self) -> None:
        """
        Given the VectorStorePort protocol
        When its members are inspected
        Then it declares add_documents(collection_name, *, ids, documents,
             embeddings, metadatas=None) -> None
        """
        # Given: the VectorStorePort protocol

        # When: inspecting add_documents
        assert hasattr(VectorStorePort, "add_documents"), (
            "VectorStorePort should declare 'add_documents'"
        )
        sig = inspect.signature(VectorStorePort.add_documents)
        params = sig.parameters
        hints = get_type_hints(VectorStorePort.add_documents)

        # Then: correct signature with keyword-only args after collection_name
        assert "collection_name" in params, (
            f"add_documents should have 'collection_name'. Got: {list(params)}"
        )
        assert "ids" in params, f"add_documents should have 'ids'. Got: {list(params)}"
        assert "documents" in params, f"add_documents should have 'documents'. Got: {list(params)}"
        assert "embeddings" in params, (
            f"add_documents should have 'embeddings'. Got: {list(params)}"
        )
        assert "metadatas" in params, f"add_documents should have 'metadatas'. Got: {list(params)}"
        assert params["metadatas"].default is None, (
            f"metadatas default should be None. Got: {params['metadatas'].default}"
        )
        assert hints.get("return") is type(None), (
            f"add_documents return type should be None. Got: {hints.get('return')}"
        )

    def test_vector_store_port_declares_query(self) -> None:
        """
        Given the VectorStorePort protocol
        When its members are inspected
        Then it declares query(collection_name, *, query_embedding,
             n_results=5) -> QueryResult
        """
        # Given: the VectorStorePort protocol

        # When: inspecting query
        assert hasattr(VectorStorePort, "query"), "VectorStorePort should declare 'query'"
        sig = inspect.signature(VectorStorePort.query)
        params = sig.parameters
        hints = get_type_hints(VectorStorePort.query)

        # Then: correct signature
        assert "collection_name" in params, (
            f"query should have 'collection_name'. Got: {list(params)}"
        )
        assert "query_embedding" in params, (
            f"query should have 'query_embedding'. Got: {list(params)}"
        )
        assert "n_results" in params, f"query should have 'n_results'. Got: {list(params)}"
        assert params["n_results"].default == 5, (
            f"n_results default should be 5. Got: {params['n_results'].default}"
        )
        assert hints.get("return") is QueryResult, (
            f"query return type should be QueryResult. Got: {hints.get('return')}"
        )

    def test_vector_store_port_declares_get_documents(self) -> None:
        """
        Given the VectorStorePort protocol
        When its members are inspected
        Then it declares get_documents(collection_name, *, ids) -> GetResult
        """
        # Given: the VectorStorePort protocol

        # When: inspecting get_documents
        assert hasattr(VectorStorePort, "get_documents"), (
            "VectorStorePort should declare 'get_documents'"
        )
        sig = inspect.signature(VectorStorePort.get_documents)
        params = sig.parameters
        hints = get_type_hints(VectorStorePort.get_documents)

        # Then: correct signature
        assert "collection_name" in params, (
            f"get_documents should have 'collection_name'. Got: {list(params)}"
        )
        assert "ids" in params, f"get_documents should have 'ids'. Got: {list(params)}"
        assert hints.get("return") is GetResult, (
            f"get_documents return type should be GetResult. Got: {hints.get('return')}"
        )

    def test_vector_store_port_declares_get_by_metadata(self) -> None:
        """
        Given the VectorStorePort protocol
        When its members are inspected
        Then it declares get_by_metadata(collection_name, *, where,
             include=None) -> GetResult
        """
        # Given: the VectorStorePort protocol

        # When: inspecting get_by_metadata
        assert hasattr(VectorStorePort, "get_by_metadata"), (
            "VectorStorePort should declare 'get_by_metadata'"
        )
        sig = inspect.signature(VectorStorePort.get_by_metadata)
        params = sig.parameters
        hints = get_type_hints(VectorStorePort.get_by_metadata)

        # Then: correct signature
        assert "collection_name" in params, (
            f"get_by_metadata should have 'collection_name'. Got: {list(params)}"
        )
        assert "where" in params, f"get_by_metadata should have 'where'. Got: {list(params)}"
        assert "include" in params, f"get_by_metadata should have 'include'. Got: {list(params)}"
        assert params["include"].default is None, (
            f"include default should be None. Got: {params['include'].default}"
        )
        assert hints.get("return") is GetResult, (
            f"get_by_metadata return type should be GetResult. Got: {hints.get('return')}"
        )

    def test_vector_store_port_declares_get_all_documents(self) -> None:
        """
        Given the VectorStorePort protocol
        When its members are inspected
        Then it declares get_all_documents(collection_name, *,
             include=None) -> GetResult
        """
        # Given: the VectorStorePort protocol

        # When: inspecting get_all_documents
        assert hasattr(VectorStorePort, "get_all_documents"), (
            "VectorStorePort should declare 'get_all_documents'"
        )
        sig = inspect.signature(VectorStorePort.get_all_documents)
        params = sig.parameters
        hints = get_type_hints(VectorStorePort.get_all_documents)

        # Then: correct signature
        assert "collection_name" in params, (
            f"get_all_documents should have 'collection_name'. Got: {list(params)}"
        )
        assert "include" in params, f"get_all_documents should have 'include'. Got: {list(params)}"
        assert params["include"].default is None, (
            f"include default should be None. Got: {params['include'].default}"
        )
        assert hints.get("return") is GetResult, (
            f"get_all_documents return type should be GetResult. Got: {hints.get('return')}"
        )

    def test_vector_store_port_declares_delete_by_id(self) -> None:
        """
        Given the VectorStorePort protocol
        When its members are inspected
        Then it declares delete_by_id(collection_name, *, ids) -> None
        """
        # Given: the VectorStorePort protocol

        # When: inspecting delete_by_id
        assert hasattr(VectorStorePort, "delete_by_id"), (
            "VectorStorePort should declare 'delete_by_id'"
        )
        sig = inspect.signature(VectorStorePort.delete_by_id)
        params = sig.parameters
        hints = get_type_hints(VectorStorePort.delete_by_id)

        # Then: correct signature
        assert "collection_name" in params, (
            f"delete_by_id should have 'collection_name'. Got: {list(params)}"
        )
        assert "ids" in params, f"delete_by_id should have 'ids'. Got: {list(params)}"
        assert hints.get("return") is type(None), (
            f"delete_by_id return type should be None. Got: {hints.get('return')}"
        )

    def test_vector_store_port_declares_collection_count(self) -> None:
        """
        Given the VectorStorePort protocol
        When its members are inspected
        Then it declares collection_count(name: str) -> int
        """
        # Given: the VectorStorePort protocol

        # When: inspecting collection_count
        assert hasattr(VectorStorePort, "collection_count"), (
            "VectorStorePort should declare 'collection_count'"
        )
        sig = inspect.signature(VectorStorePort.collection_count)
        params = sig.parameters
        hints = get_type_hints(VectorStorePort.collection_count)

        # Then: correct signature
        assert "name" in params, f"collection_count should have 'name'. Got: {list(params)}"
        assert hints.get("name") is str, (
            f"collection_count(name) should be str. Got: {hints.get('name')}"
        )
        assert hints.get("return") is int, (
            f"collection_count return type should be int. Got: {hints.get('return')}"
        )

    def test_vector_store_port_declares_reset_collection(self) -> None:
        """
        Given the VectorStorePort protocol
        When its members are inspected
        Then it declares reset_collection(name: str) -> None
        """
        # Given: the VectorStorePort protocol

        # When: inspecting reset_collection
        assert hasattr(VectorStorePort, "reset_collection"), (
            "VectorStorePort should declare 'reset_collection'"
        )
        sig = inspect.signature(VectorStorePort.reset_collection)
        params = sig.parameters
        hints = get_type_hints(VectorStorePort.reset_collection)

        # Then: correct signature
        assert "name" in params, f"reset_collection should have 'name'. Got: {list(params)}"
        assert hints.get("name") is str, (
            f"reset_collection(name) should be str. Got: {hints.get('name')}"
        )
        assert hints.get("return") is type(None), (
            f"reset_collection return type should be None. Got: {hints.get('return')}"
        )

    def test_vector_store_port_declares_close(self) -> None:
        """
        Given the VectorStorePort protocol
        When its members are inspected
        Then it declares close() -> None
        """
        # Given: the VectorStorePort protocol

        # When: inspecting close
        assert hasattr(VectorStorePort, "close"), "VectorStorePort should declare 'close'"
        hints = get_type_hints(VectorStorePort.close)

        # Then: close() -> None
        assert hints.get("return") is type(None), (
            f"close() return type should be None. Got: {hints.get('return')}"
        )

    def test_vector_store_satisfies_vector_store_port(self, tmp_path: Any) -> None:
        """
        Given a real VectorStore instance (ChromaDB temp dir)
        When isinstance(store, VectorStorePort) is checked
        Then it returns True (structural conformance, no inheritance required)
        """
        # Given: a real VectorStore backed by a temp directory
        store = VectorStore(
            persist_dir=str(tmp_path / "chroma"),
            distance_metric="cosine",
        )

        # When: checking isinstance
        try:
            result = isinstance(cast("object", store), VectorStorePort)
        finally:
            store.close()

        # Then: structural conformance -- no inheritance required
        assert result, (
            f"VectorStore should satisfy VectorStorePort structurally. "
            f"isinstance returned {result}"
        )


# ============================================================================
# TestQueryResultDataclass
# ============================================================================


class TestQueryResultDataclass:
    """
    REQUIREMENT: QueryResult is a typed dataclass that replaces
    dict[str, Any] for VectorStorePort.query() return values.

    WHO: Scorer._query_collection, Scorer._query_archetypes -- they consume
         query results and need typed attribute access instead of string-key
         dict access.
    WHAT: (1) QueryResult has field ids: list[list[str]].
          (2) QueryResult has field documents: list[list[str]].
          (3) QueryResult has field metadatas: list[list[dict[str, Any]]].
          (4) QueryResult has field distances: list[list[float]].
          (5) All fields default to nested empty lists when constructed
              without arguments (empty result).
    WHY: dict[str, Any] returns leak ChromaDB's response shape through
         the port boundary. Callers use untyped string keys that pyright
         cannot verify. Typed dataclasses make the contract explicit and
         catch access errors at type-check time.

    MOCK BOUNDARY:
        Mock:  Nothing -- pure dataclass construction tests
        Real:  QueryResult dataclass
        Never: N/A
    """

    def test_query_result_has_ids_field(self) -> None:
        """
        Given a QueryResult constructed with ids=[["doc-1"]]
        When result.ids is accessed
        Then it returns [["doc-1"]]
        """
        # Given: a QueryResult with specific ids
        result = QueryResult(ids=[["doc-1"]])

        # When: accessing the ids field
        ids = result.ids

        # Then: the value matches what was provided
        assert ids == [["doc-1"]], f"QueryResult.ids should be [['doc-1']]. Got: {ids}"

    def test_query_result_has_documents_field(self) -> None:
        """
        Given a QueryResult constructed with documents=[["text"]]
        When result.documents is accessed
        Then it returns [["text"]]
        """
        # Given: a QueryResult with specific documents
        result = QueryResult(documents=[["text"]])

        # When: accessing the documents field
        docs = result.documents

        # Then: the value matches what was provided
        assert docs == [["text"]], f"QueryResult.documents should be [['text']]. Got: {docs}"

    def test_query_result_has_metadatas_field(self) -> None:
        """
        Given a QueryResult constructed with metadatas=[[{"key": "val"}]]
        When result.metadatas is accessed
        Then it returns [[{"key": "val"}]]
        """
        # Given: a QueryResult with specific metadatas
        result = QueryResult(metadatas=[[{"key": "val"}]])

        # When: accessing the metadatas field
        metas = result.metadatas

        # Then: the value matches what was provided
        assert metas == [[{"key": "val"}]], (
            f"QueryResult.metadatas should be [[{{'key': 'val'}}]]. Got: {metas}"
        )

    def test_query_result_has_distances_field(self) -> None:
        """
        Given a QueryResult constructed with distances=[[0.1, 0.3]]
        When result.distances is accessed
        Then it returns [[0.1, 0.3]]
        """
        # Given: a QueryResult with specific distances
        result = QueryResult(distances=[[0.1, 0.3]])

        # When: accessing the distances field
        dists = result.distances

        # Then: the value matches what was provided
        assert dists == [[0.1, 0.3]], f"QueryResult.distances should be [[0.1, 0.3]]. Got: {dists}"

    def test_query_result_defaults_to_empty(self) -> None:
        """
        Given a QueryResult constructed with no arguments
        When all fields are accessed
        Then ids, documents, metadatas, and distances are all [[]]
        """
        # Given: a default-constructed QueryResult
        result = QueryResult()

        # When/Then: all fields default to nested empty lists
        assert result.ids == [[]], f"Default QueryResult.ids should be [[]]. Got: {result.ids}"
        assert result.documents == [[]], (
            f"Default QueryResult.documents should be [[]]. Got: {result.documents}"
        )
        assert result.metadatas == [[]], (
            f"Default QueryResult.metadatas should be [[]]. Got: {result.metadatas}"
        )
        assert result.distances == [[]], (
            f"Default QueryResult.distances should be [[]]. Got: {result.distances}"
        )


# ============================================================================
# TestGetResultDataclass
# ============================================================================


class TestGetResultDataclass:
    """
    REQUIREMENT: GetResult is a typed dataclass that replaces
    dict[str, Any] for get_documents() and get_by_metadata() returns.

    WHO: DecisionRecorder.get_decision, Scorer._get_rejection_reasons,
         EvalRunner._load_decisions -- they consume get results.
    WHAT: (1) GetResult has field ids: list[str].
          (2) GetResult has field documents: list[str | None].
          (3) GetResult has field metadatas: list[dict[str, Any]].
          (4) All fields default to empty lists when constructed without
              arguments (empty result).
    WHY: Same typing rationale as QueryResult. Flat shape (not nested)
         matches ChromaDB's get() response shape, distinct from query()
         batch shape.

    MOCK BOUNDARY:
        Mock:  Nothing -- pure dataclass construction tests
        Real:  GetResult dataclass
        Never: N/A
    """

    def test_get_result_has_ids_field(self) -> None:
        """
        Given a GetResult constructed with ids=["doc-1", "doc-2"]
        When result.ids is accessed
        Then it returns ["doc-1", "doc-2"]
        """
        # Given: a GetResult with specific ids
        result = GetResult(ids=["doc-1", "doc-2"])

        # When: accessing the ids field
        ids = result.ids

        # Then: the value matches what was provided
        assert ids == ["doc-1", "doc-2"], f"GetResult.ids should be ['doc-1', 'doc-2']. Got: {ids}"

    def test_get_result_has_documents_field(self) -> None:
        """
        Given a GetResult constructed with documents=["text", None]
        When result.documents is accessed
        Then it returns ["text", None]
        """
        # Given: a GetResult with mixed documents (some None)
        result = GetResult(documents=["text", None])

        # When: accessing the documents field
        docs = result.documents

        # Then: the value matches including None entries
        assert docs == ["text", None], f"GetResult.documents should be ['text', None]. Got: {docs}"

    def test_get_result_has_metadatas_field(self) -> None:
        """
        Given a GetResult constructed with metadatas=[{"k": "v"}]
        When result.metadatas is accessed
        Then it returns [{"k": "v"}]
        """
        # Given: a GetResult with specific metadatas
        result = GetResult(metadatas=[{"k": "v"}])

        # When: accessing the metadatas field
        metas = result.metadatas

        # Then: the value matches what was provided
        assert metas == [{"k": "v"}], f"GetResult.metadatas should be [{{'k': 'v'}}]. Got: {metas}"

    def test_get_result_defaults_to_empty(self) -> None:
        """
        Given a GetResult constructed with no arguments
        When all fields are accessed
        Then ids, documents, and metadatas are all []
        """
        # Given: a default-constructed GetResult
        result = GetResult()

        # When/Then: all fields default to empty lists
        assert result.ids == [], f"Default GetResult.ids should be []. Got: {result.ids}"
        assert result.documents == [], (
            f"Default GetResult.documents should be []. Got: {result.documents}"
        )
        assert result.metadatas == [], (
            f"Default GetResult.metadatas should be []. Got: {result.metadatas}"
        )
