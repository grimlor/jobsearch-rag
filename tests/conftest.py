"""Global test configuration — shared fixtures and safety guards.

This conftest provides:

1. **Output guard** — makes the real ``output/`` directory read-only so
   tests that forget to use ``tmp_path`` get an immediate ``PermissionError``.

2. **Shared I/O-boundary fixtures** — ``mock_embedder`` (Embedder with
   stubbed Ollama methods), ``vector_store`` (real ChromaDB backed by
   ``tmp_path``), and ``decision_recorder`` (real recorder wired to the
   above).  Individual test files may shadow these with local fixtures
   that use different return values.

3. **Runner-layer factories** — ``make_settings``, ``make_listing``,
   ``make_runner_with_mocks``, and ``mock_board_io`` provide the
   foundational infrastructure for pipeline orchestration tests.
   All runner construction uses real VectorStore and DecisionRecorder;
   only Ollama network I/O (Embedder, Scorer) and browser I/O (adapter,
   session) are mocked.
"""

from __future__ import annotations

import contextlib
import stat
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    from collections.abc import Iterator

_PROJECT_OUTPUT = Path(__file__).resolve().parent.parent / "output"

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
    """
    embedder = Embedder.__new__(Embedder)
    embedder.base_url = "http://localhost:11434"
    embedder.embed_model = "nomic-embed-text"
    embedder.llm_model = "mistral:7b"
    embedder.max_retries = 3
    embedder.base_delay = 0.0
    embedder.embed = AsyncMock(return_value=EMBED_FAKE)  # type: ignore[method-assign]
    embedder.classify = AsyncMock(  # type: ignore[method-assign]
        return_value='{"disqualified": false}',
    )
    embedder.health_check = AsyncMock()  # type: ignore[method-assign]
    return embedder


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
def make_settings(tmp_path: Path):
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
def make_listing():
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
def make_runner_with_mocks(tmp_path: Path):
    """Factory fixture — returns a callable that produces a wired PipelineRunner.

    VectorStore and DecisionRecorder use **real** instances backed by
    ``tmp_path``.  Embedder and Scorer are mocked because they make network
    I/O to Ollama.

    When ``populate_store=True`` (default), minimal documents are seeded into
    the ``resume``, ``role_archetypes``, and ``global_positive_signals``
    collections so that auto-indexing is skipped.  Pass ``populate_store=False``
    for tests that verify the auto-index behaviour.

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
    ) -> tuple[PipelineRunner, MagicMock, MagicMock]:
        mock_embedder = MagicMock()
        mock_embedder.health_check = AsyncMock()
        mock_embedder.embed = AsyncMock(return_value=EMBED_FAKE)

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
        ):
            runner = PipelineRunner(settings)

        if populate_store:
            _populate(runner._store)

        return runner, mock_embedder, mock_scorer

    return _factory


@pytest.fixture
def mock_board_io():
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
# Output safety guard
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def _guard_real_output_dir() -> Iterator[None]:
    """Make the real output/ directory read-only during tests.

    Restores original permissions after the session, even on failure.
    If the directory does not exist the guard is silently skipped —
    CI environments may not have it.
    """
    if not _PROJECT_OUTPUT.is_dir():
        yield
        return

    # Save original permissions for output/ and key subdirs
    dirs_to_guard = [_PROJECT_OUTPUT]
    for child in _PROJECT_OUTPUT.iterdir():
        if child.is_dir():
            dirs_to_guard.append(child)

    original_modes: dict[Path, int] = {}
    for d in dirs_to_guard:
        original_modes[d] = d.stat().st_mode
        # Remove write permission (owner, group, other)
        d.chmod(original_modes[d] & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    try:
        yield
    finally:
        # Restore original permissions
        for d, mode in original_modes.items():
            with contextlib.suppress(OSError):
                d.chmod(mode)
