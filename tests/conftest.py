"""
Global test configuration — shared fixtures and safety guards.

This conftest provides:

1. **Output guard** — makes the real ``output/`` directory read-only so
   tests that forget to use ``tmp_path`` get an immediate ``PermissionError``.

2. **Shared I/O-boundary fixtures** — ``mock_embedder`` (real Embedder with
   ollama client stubbed at the I/O boundary), ``mock_ollama_client`` (the
   raw mock), ``vector_store`` (real ChromaDB backed by ``tmp_path``), and
   ``decision_recorder`` (real recorder wired to the above).  Individual
   test files may shadow these with local fixtures that use different
   return values.

3. **Windows ChromaDB cleanup guard** — ``TemporaryDirectory`` is patched
   to use ``ignore_cleanup_errors=True`` on Windows, preventing
   ``PermissionError`` when ChromaDB file handles are still open during
   temp directory cleanup.
"""

from __future__ import annotations

import contextlib
import stat
import sys
import tempfile as _tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from jobsearch_rag.config import (
    AdaptersConfig,
    BoardConfig,
    ChromaConfig,
    CompBand,
    OllamaConfig,
    OutputConfig,
    ScoringConfig,
    SecurityConfig,
    Settings,
)
from jobsearch_rag.rag.decisions import DecisionRecorder
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.store import VectorStore
from tests.constants import EMBED_FAKE as EMBED_FAKE  # re-export for fixtures below

if TYPE_CHECKING:
    from collections.abc import Iterator

_PROJECT_OUTPUT = Path(__file__).resolve().parent.parent / "output"


# ---------------------------------------------------------------------------
# Windows: ChromaDB file-handle cleanup guard
# ---------------------------------------------------------------------------
# ChromaDB's PersistentClient holds open file handles on SQLite and HNSW
# segment files.  On Windows, TemporaryDirectory.__exit__ cannot delete
# these files while they're still open, raising PermissionError.  The file
# handles are released when the PersistentClient is garbage-collected (after
# the test method returns), so the error is benign — the OS reclaims the
# temp files.  Patching TemporaryDirectory globally is simpler and safer
# than modifying 100+ inline call sites across the test suite.
if sys.platform == "win32":
    _OriginalTemporaryDirectory = _tempfile.TemporaryDirectory

    class _WinSafeTemporaryDirectory(_OriginalTemporaryDirectory):  # type: ignore[type-arg]
        """TemporaryDirectory that tolerates cleanup errors on Windows."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs.setdefault("ignore_cleanup_errors", True)
            super().__init__(*args, **kwargs)

    _tempfile.TemporaryDirectory = _WinSafeTemporaryDirectory  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test config factories — explicit values, no code-side defaults
# ---------------------------------------------------------------------------

_TEST_COMP_BANDS: list[CompBand] = [
    CompBand(ratio=1.0, score=1.0),
    CompBand(ratio=0.90, score=0.7),
    CompBand(ratio=0.77, score=0.4),
    CompBand(ratio=0.68, score=0.0),
]


def make_test_scoring_config(**overrides: Any) -> ScoringConfig:
    """Build a ScoringConfig with all required fields specified."""
    defaults: dict[str, Any] = {
        "archetype_weight": 0.5,
        "fit_weight": 0.3,
        "history_weight": 0.2,
        "comp_weight": 0.15,
        "negative_weight": 0.4,
        "culture_weight": 0.2,
        "base_salary": 220_000,
        "disqualify_on_llm_flag": False,
        "min_score_threshold": 0.45,
        "comp_bands": list(_TEST_COMP_BANDS),
        "missing_comp_score": 0.5,
        "chunk_overlap": 2000,
        "dedup_similarity_threshold": 0.95,
        "top_k_retrieval": 3,
        "salary_floor": 10.0,
        "salary_ceiling": 1_000_000.0,
        "hours_per_year": 2080,
    }
    defaults.update(overrides)
    return ScoringConfig(**defaults)


def make_test_ollama_config(**overrides: Any) -> OllamaConfig:
    """Build an OllamaConfig with all required fields specified."""
    defaults: dict[str, Any] = {
        "base_url": "http://localhost:11434",
        "llm_model": "mistral:7b",
        "embed_model": "nomic-embed-text",
        "slow_llm_threshold_ms": 30_000,
        "classify_system_prompt": "You are a job listing classifier.",
        "max_retries": 3,
        "base_delay": 1.0,
        "max_embed_chars": 8_000,
        "head_ratio": 0.6,
        "retryable_status_codes": [408, 429, 500, 502, 503, 504],
    }
    defaults.update(overrides)
    return OllamaConfig(**defaults)


def make_test_output_config(**overrides: Any) -> OutputConfig:
    """Build an OutputConfig with all required fields specified."""
    defaults: dict[str, Any] = {
        "default_format": "markdown",
        "output_dir": "./output",
        "open_top_n": 5,
        "jd_dir": "output/jds",
        "decisions_dir": "data/decisions",
        "log_dir": "data/logs",
        "eval_history_path": "data/eval_history.jsonl",
        "max_slug_length": 80,
    }
    defaults.update(overrides)
    return OutputConfig(**defaults)


def make_test_security_config(**overrides: Any) -> SecurityConfig:
    """Build a SecurityConfig with all required fields specified."""
    defaults: dict[str, Any] = {
        "screen_prompt": "Review the following job description text.",
    }
    defaults.update(overrides)
    return SecurityConfig(**defaults)


def make_test_settings(
    tmpdir: str,
    enabled_boards: list[str] | None = None,
    overnight_boards: list[str] | None = None,
    *,
    resume_path: str | None = None,
    archetypes_path: str | None = None,
    global_rubric_path: str | None = None,
    scoring_overrides: dict[str, Any] | None = None,
    ollama_overrides: dict[str, Any] | None = None,
) -> Settings:
    """Build a Settings with all required fields for test use."""
    boards = enabled_boards or ["testboard"]
    board_configs: dict[str, BoardConfig] = {}
    for name in boards:
        board_configs[name] = BoardConfig(
            name=name,
            searches=[f"https://{name}.com/search"],
            max_pages=1,
            headless=True,
            rate_limit_range=(1.5, 3.5),
        )
    for name in overnight_boards or []:
        if name not in board_configs:
            board_configs[name] = BoardConfig(
                name=name,
                searches=[f"https://{name}.com/search"],
                max_pages=1,
                headless=False,
                rate_limit_range=(1.5, 3.5),
            )
    tmpdir_path = Path(tmpdir)
    return Settings(
        enabled_boards=boards,
        overnight_boards=overnight_boards or [],
        boards=board_configs,
        scoring=make_test_scoring_config(**(scoring_overrides or {})),
        ollama=make_test_ollama_config(**(ollama_overrides or {})),
        output=make_test_output_config(
            output_dir=str(tmpdir_path / "output"),
            jd_dir=str(tmpdir_path / "output" / "jds"),
            decisions_dir=str(tmpdir_path / "decisions"),
            log_dir=str(tmpdir_path / "logs"),
        ),
        chroma=ChromaConfig(
            persist_dir=str(tmpdir_path / "chroma"),
            distance_metric="cosine",
        ),
        security=make_test_security_config(),
        resume_path=resume_path or "data/resume.md",
        archetypes_path=archetypes_path or "config/role_archetypes.toml",
        global_rubric_path=global_rubric_path or "config/global_rubric.toml",
        session_storage_dir=str(tmpdir_path),
        adapters=AdaptersConfig(
            browser_paths={},
            cdp_timeout=15.0,
            max_full_text_chars=250_000,
            viewport_width=1440,
            viewport_height=900,
        ),
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_mock_ollama_client(
    embed_vector: list[float] | None = None,
    classify_response: str = '{"disqualified": false}',
) -> AsyncMock:
    """
    Build a stubbed ``ollama.AsyncClient`` — the I/O boundary.

    Returns realistic response objects for ``embed`` and ``chat`` so that
    all Embedder logic (retry, truncation, metrics, token counting) runs
    for real.  Only the final HTTP call is replaced.

    The ``embed_vector`` defaults to ``EMBED_FAKE`` if not provided.
    """
    if embed_vector is None:
        embed_vector = list(EMBED_FAKE)  # copy to avoid mutation

    client = AsyncMock()

    # embed() → response with embeddings list and token count
    embed_response = MagicMock()
    embed_response.embeddings = [embed_vector]
    embed_response.prompt_eval_count = 42
    client.embed = AsyncMock(return_value=embed_response)

    # chat() → response with message.content
    chat_message = MagicMock()
    chat_message.content = classify_response
    chat_response = MagicMock()
    chat_response.message = chat_message
    chat_response.prompt_eval_count = 100
    chat_response.eval_count = 20
    client.chat = AsyncMock(return_value=chat_response)

    # list() → response with available models (for health_check)
    model_embed = MagicMock()
    model_embed.model = "nomic-embed-text"
    model_llm = MagicMock()
    model_llm.model = "mistral:7b"
    client.list = AsyncMock(return_value=MagicMock(models=[model_embed, model_llm]))

    return client


# ---------------------------------------------------------------------------
# Shared I/O-boundary fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ollama_client() -> AsyncMock:  # pyright: ignore[reportUnusedFunction]
    """Stubbed ollama.AsyncClient — the I/O boundary."""
    return make_mock_ollama_client()


@pytest.fixture
def mock_embedder(mock_ollama_client: AsyncMock) -> Embedder:
    """
    Real Embedder with ollama client stubbed at the I/O boundary.

    All Embedder logic — retry, truncation, metrics, token counting —
    runs for real.  Only the ollama HTTP call is replaced.
    """
    embedder = Embedder(make_test_ollama_config(max_retries=1, base_delay=0.0))
    embedder._client = mock_ollama_client  # type: ignore[attr-defined]
    return embedder


@pytest.fixture
def vector_store(tmp_path: Path) -> Iterator[VectorStore]:
    """Real ChromaDB VectorStore backed by a per-test temp directory."""
    store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    yield store
    store.close()


@pytest.fixture
def decision_recorder(
    vector_store: VectorStore,
    mock_embedder: Embedder,
    tmp_path: Path,
) -> DecisionRecorder:
    """
    Real DecisionRecorder with real ChromaDB and stubbed Embedder.

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
def _guard_real_output_dir() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]  # autouse fixture
    """
    Make the real output/ directory read-only during tests.

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
