"""Global test configuration — shared fixtures and safety guards.

This conftest provides:

1. **Output redirect** — per-test ``autouse`` fixture that redirects all
   application output paths (output/jds/, output/results.md,
   output/results.csv, data/logs/, data/decisions/) to subdirectories of
   ``tmp_path``.  Tests actively write to isolated tmp directories.

2. **Shared I/O-boundary fixtures** — ``mock_embedder`` (Embedder with
   stubbed Ollama methods), ``vector_store`` (real ChromaDB backed by
   ``tmp_path``), and ``decision_recorder`` (real recorder wired to
   the above).  Individual test files may shadow these with local
   fixtures that use different return values.

3. **Runner-layer factories** — ``make_settings``, ``make_listing``,
   ``make_runner_with_mocks``, and ``mock_board_io`` provide the
   foundational infrastructure for pipeline orchestration tests.
   All runner construction uses real VectorStore and DecisionRecorder;
   only Ollama network I/O (Embedder, Scorer) and browser I/O (adapter,
   session) are mocked.  The ``mock_embedder`` fixture is reused — the factory
   does not create its own local mock.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import jobsearch_rag.logging as _logging_mod
import jobsearch_rag.rag.decisions as _decisions_mod
from jobsearch_rag.adapters.base import JobListing
from jobsearch_rag.config import (
    BoardConfig,
    ChromaConfig,
    OllamaConfig,
    OutputConfig,
    ScoringConfig,
    Settings,
)
from jobsearch_rag.pipeline.runner import PipelineRunner
from jobsearch_rag.rag.decisions import DecisionRecorder
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.scorer import ScoreResult
from jobsearch_rag.rag.store import VectorStore

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Canonical fake embedding used across test files.  Individual tests that
# need a different vector can define their own constant.
EMBED_FAKE: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5]


# ---------------------------------------------------------------------------
# Shared I/O-boundary fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedder() -> Embedder:
    """Embedder with stubbed I/O methods — no Ollama connection needed.

    Uses ``Embedder.__new__`` to create a real instance without calling
    ``__init__`` (which would create an ``ollama.AsyncClient``).  All
    async methods are replaced with ``AsyncMock`` stubs that return
    deterministic values.

    Use for any test class whose MOCK BOUNDARY includes:
        Mock: mock_embedder fixture (Ollama HTTP)
    """
    mock_embedder = Embedder.__new__(Embedder)
    mock_embedder.base_url = "http://localhost:11434"
    mock_embedder.embed_model = "nomic-embed-text"
    mock_embedder.llm_model = "mistral:7b"
    mock_embedder.max_retries = 3
    mock_embedder.base_delay = 0.0
    mock_embedder.embed = AsyncMock(return_value=EMBED_FAKE)  # type: ignore[method-assign]
    mock_embedder.classify = AsyncMock(  # type: ignore[method-assign]
        return_value='{"disqualified": false}',
    )
    mock_embedder.health_check = AsyncMock()  # type: ignore[method-assign]
    return mock_embedder


@pytest.fixture
def vector_store(tmp_path: Path) -> VectorStore:
    """Real ChromaDB VectorStore backed by a per-test temp directory."""
    return VectorStore(persist_dir=str(tmp_path / "chroma"))


@pytest.fixture
def decision_recorder(
    vector_store: VectorStore,
    mock_embedder: Embedder,
    tmp_path: Path,
) -> DecisionRecorder:
    """Real DecisionRecorder with real ChromaDB and stubbed Embedder.

    The ``decisions`` collection is pre-created so ``get_decision``
    works even before the first ``record()`` call.
    """
    vector_store.get_or_create_collection("decisions")
    return DecisionRecorder(
        store=vector_store,
        embedder=mock_embedder,
        decisions_dir=tmp_path / "decisions",
    )


# ---------------------------------------------------------------------------
# Runner-layer factories
# ---------------------------------------------------------------------------


@pytest.fixture
def make_settings(tmp_path: Path) -> Callable[..., Settings]:
    """Factory fixture — returns a callable that produces a Settings instance.

    The ChromaDB persist directory is always rooted under ``tmp_path`` so
    each test gets an isolated store.  The output directory is similarly
    isolated under ``tmp_path``.

    Usage::

        def test_something(make_settings):
            settings = make_settings()
            settings = make_settings(enabled_boards=["ziprecruiter"])
            settings = make_settings(
                enabled_boards=["board_a"],
                overnight_boards=["linkedin"],
            )
    """

    def _factory(
        enabled_boards: list[str] | None = None,
        overnight_boards: list[str] | None = None,
    ) -> Settings:
        boards = enabled_boards or ["testboard"]
        board_configs: dict[str, BoardConfig] = {}
        for name in boards:
            board_configs[name] = BoardConfig(
                name=name,
                searches=[f"https://{name}.com/search"],
                max_pages=1,
                headless=True,
            )
        for name in overnight_boards or []:
            if name not in board_configs:
                board_configs[name] = BoardConfig(
                    name=name,
                    searches=[f"https://{name}.com/search"],
                    max_pages=1,
                    headless=False,
                )
        return Settings(
            enabled_boards=boards,
            overnight_boards=overnight_boards or [],
            boards=board_configs,
            scoring=ScoringConfig(),
            ollama=OllamaConfig(),
            output=OutputConfig(output_dir=str(tmp_path / "output")),
            chroma=ChromaConfig(persist_dir=str(tmp_path / "chroma")),
        )

    return _factory


@pytest.fixture
def make_listing() -> Callable[..., JobListing]:
    """Factory fixture — returns a callable that produces a JobListing instance.

    Defaults produce a fully populated listing with non-empty ``full_text``
    so it passes the extraction-quality gate without needing ``extract_detail``.

    Usage::

        def test_something(make_listing):
            listing = make_listing()
            listing = make_listing(board="ziprecruiter", external_id="zr-42")
            listing = make_listing(full_text="")  # simulate extraction failure
    """

    def _factory(
        board: str = "testboard",
        external_id: str = "1",
        title: str = "Staff Architect",
        full_text: str = "A detailed job description for a staff architect role.",
    ) -> JobListing:
        return JobListing(
            board=board,
            external_id=external_id,
            title=title,
            company="Acme Corp",
            location="Remote",
            url=f"https://{board}.com/{external_id}",
            full_text=full_text,
        )

    return _factory


@pytest.fixture
def make_runner_with_mocks(tmp_path: Path, mock_embedder: Embedder) -> Callable[..., tuple[PipelineRunner, Embedder, MagicMock]]:
    """Factory fixture — returns a callable that produces a wired PipelineRunner.

    VectorStore and DecisionRecorder use **real** instances backed by
    ``tmp_path``.  The ``mock_embedder`` fixture is reused for Ollama I/O;
    Scorer is mocked.

    When ``populate_store=True`` (default), minimal documents are seeded into
    the ``resume``, ``role_archetypes``, and ``global_positive_signals``
    collections via the public ``VectorStore.add_documents`` API before the
    runner is constructed, so auto-indexing is skipped.  Pass
    ``populate_store=False`` for tests that verify the auto-index behaviour.

    Returns ``(runner, mock_embedder, mock_scorer)``.

    Usage::

        async def test_something(make_runner_with_mocks, make_settings):
            settings = make_settings()
            runner, mock_embedder, mock_scorer = make_runner_with_mocks(settings)

        async def test_auto_index(make_runner_with_mocks, make_settings):
            settings = make_settings()
            runner, _, _ = make_runner_with_mocks(settings, populate_store=False)
    """

    def _populate(store: VectorStore) -> None:
        for name in ("resume", "role_archetypes", "global_positive_signals"):
            store.add_documents(
                name,
                ids=[f"{name}-seed"],
                documents=[f"Seed document for {name}"],
                embeddings=[EMBED_FAKE],
            )

    def _factory(
        settings: Settings,
        *,
        populate_store: bool = True,
    ) -> tuple[PipelineRunner, Embedder, MagicMock]:
        # Build a real VectorStore from the settings path — seed it via
        # the public API *before* the runner is constructed.
        store = VectorStore(persist_dir=settings.chroma.persist_dir)
        if populate_store:
            _populate(store)

        mock_scorer = MagicMock()
        mock_scorer.score = AsyncMock(
            return_value=ScoreResult(
                fit_score=0.8,
                archetype_score=0.7,
                history_score=0.5,
                disqualified=False,
            )
        )

        with (
            patch("jobsearch_rag.pipeline.runner.Embedder", return_value=mock_embedder),
            patch("jobsearch_rag.pipeline.runner.Scorer", return_value=mock_scorer),
            patch("jobsearch_rag.pipeline.runner.VectorStore", return_value=store),
        ):
            runner = PipelineRunner(settings)

        return runner, mock_embedder, mock_scorer

    return _factory


@pytest.fixture
def mock_board_io() -> Callable[..., tuple[MagicMock, MagicMock, MagicMock]]:
    """Factory fixture — returns a callable that produces mocked browser I/O boundaries.

    Creates a mock adapter, session, and registry representing the three
    browser I/O boundaries for a board search.  The adapter returns
    ``search_results`` from ``search()`` (default: empty list).

    The registry mock wraps the adapter as a computation placeholder only —
    for tests that need registry behaviour, construct a real AdapterRegistry
    and register the mock adapter directly.

    Returns ``(mock_adapter, mock_session, mock_registry)``.

    Usage::

        async def test_something(mock_board_io, make_listing):
            listing = make_listing()
            mock_adapter, mock_session, mock_registry = mock_board_io()
            mock_adapter, mock_session, mock_registry = mock_board_io(
                search_results=[listing]
            )
    """

    def _factory(
        search_results: list[JobListing] | None = None,
    ) -> tuple[MagicMock, MagicMock, MagicMock]:
        mock_adapter = MagicMock()
        mock_adapter.board_name = "testboard"
        mock_adapter.rate_limit_seconds = (0.0, 0.0)
        mock_adapter.authenticate = AsyncMock()
        mock_adapter.search = AsyncMock(
            return_value=search_results if search_results is not None else []
        )
        mock_adapter.extract_detail = AsyncMock()

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.new_page = AsyncMock(return_value=MagicMock())
        mock_session.save_storage_state = AsyncMock()

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_adapter

        return mock_adapter, mock_session, mock_registry

    return _factory


# ---------------------------------------------------------------------------
# Output redirect — per-test isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def redirect_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect all application output paths to ``tmp_path`` subdirectories.

    Redirected paths:
        output/          → tmp_path/output/
        output/jds/      → tmp_path/output/jds/
        data/logs/       → tmp_path/logs/
        data/decisions/  → tmp_path/decisions/

    Applied automatically to every test — never bypass it.
    """
    out_dir = tmp_path / "output"
    jds_dir = out_dir / "jds"
    log_dir = tmp_path / "logs"
    decisions_dir = tmp_path / "decisions"

    for d in (out_dir, jds_dir, log_dir, decisions_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Redirect OutputConfig default so any Settings built without
    # make_settings still lands in tmp_path.
    monkeypatch.setattr(OutputConfig, "output_dir", str(out_dir))

    # Redirect file-logging default directory.
    monkeypatch.setattr(_logging_mod, "DEFAULT_LOG_DIR", str(log_dir))

    # Redirect decision recorder default directory.
    monkeypatch.setattr(_decisions_mod, "_DECISIONS_DIR", decisions_dir)
