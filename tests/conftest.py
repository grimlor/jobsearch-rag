"""Global test configuration — shared fixtures and safety guards.

This conftest provides:

1. **Output guard** — makes the real ``output/`` directory read-only so
   tests that forget to use ``tmp_path`` get an immediate ``PermissionError``.

2. **Shared I/O-boundary fixtures** — ``mock_embedder`` (Embedder with
   stubbed Ollama methods), ``vector_store`` (real ChromaDB backed by
   ``tmp_path``), and ``decision_recorder`` (real recorder wired to the
   above).  Individual test files may shadow these with local fixtures
   that use different return values.
"""

from __future__ import annotations

import contextlib
import stat
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from jobsearch_rag.rag.decisions import DecisionRecorder
from jobsearch_rag.rag.embedder import Embedder
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
