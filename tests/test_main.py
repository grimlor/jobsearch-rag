"""
CLI entry point dispatch and error display tests.

Spec classes:
    TestMainDispatch — main() routes each subcommand to its handler
    TestMainErrorDisplay — main() formats errors for the operator
    TestMainModuleEntryPoint — ``python -m jobsearch_rag`` invokes main()
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING, Any, Self
from unittest.mock import MagicMock, patch

import pytest

from jobsearch_rag.__main__ import HANDLERS, main
from jobsearch_rag.errors import ActionableError, ErrorType

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from jobsearch_rag.rag.ports import (
        DocumentRecord,
        EmbeddedDocument,
        MetadataFilter,
        QueryResults,
    )

# ---------------------------------------------------------------------------
# Sentinel store — raises on first real use, proving the composition root
# created it and passed it to the handler.
# ---------------------------------------------------------------------------

_SENTINEL_MSG = "SentinelStore: composition root delivered the store to the handler"


class _SentinelStore:
    """
    VectorStorePort implementation that raises immediately on any domain method.

    Proves the store was created by the composition root (via factory/config)
    and delivered to the handler. The handler calls a method → raises →
    main() catches the exception → test asserts on stderr.
    """

    def __init__(self, **_kwargs: Any) -> None:
        """Accept and ignore factory kwargs (persist_dir, etc.)."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass

    def close(self) -> None:
        pass

    # Every domain method raises the sentinel
    def collection_count(self, name: str) -> int:
        raise RuntimeError(_SENTINEL_MSG)

    def reset_collection(self, name: str) -> None:
        raise RuntimeError(_SENTINEL_MSG)

    def add_documents(self, collection_name: str, *, documents: list[EmbeddedDocument]) -> None:
        raise RuntimeError(_SENTINEL_MSG)

    def get_documents(self, collection_name: str, *, ids: list[str]) -> list[DocumentRecord]:
        raise RuntimeError(_SENTINEL_MSG)

    def get_all_documents(self, collection_name: str) -> list[DocumentRecord]:
        raise RuntimeError(_SENTINEL_MSG)

    def get_by_metadata(
        self, collection_name: str, *, where: MetadataFilter
    ) -> list[DocumentRecord]:
        raise RuntimeError(_SENTINEL_MSG)

    def delete_by_id(self, collection_name: str, *, ids: list[str]) -> None:
        raise RuntimeError(_SENTINEL_MSG)

    def query(
        self,
        collection_name: str,
        *,
        query_embedding: list[float],
        n_results: int,
    ) -> QueryResults:
        raise RuntimeError(_SENTINEL_MSG)


# ---------------------------------------------------------------------------
# TestMainDispatch
# ---------------------------------------------------------------------------


class TestMainDispatch:
    """
    REQUIREMENT: main() dispatches each CLI subcommand to the correct handler

    WHO: The operator invoking ``python -m jobsearch_rag <command>``
    WHAT: (1) Store-independent commands (boards, login, export) are dispatched
              directly without creating a store.
          (2) Store-dependent commands (index, search, decide, decisions,
              review, rescore, eval, reset) receive a VectorStorePort
              created by the composition root.
    WHY: The shim is the only coupling between argparse and handler
         functions — incorrect wiring silently runs the wrong command.
         ``main()`` is the single composition root: it creates the store
         once and passes it to every handler that needs it.

    MOCK BOUNDARY:
        Mock: sys.argv (process state), filesystem (config file — temp dir)
        Real: main(), build_parser(), load_settings(), create_vector_store(),
              handler functions (run until first store method call)
        Never: Patch our own functions (load_settings, create_vector_store,
               handlers); never assert on mock call_args
    """

    @pytest.mark.parametrize(
        ("command", "extra_argv"),
        [
            ("boards", []),
            ("export", []),
            ("login", ["--board", "ziprecruiter"]),
        ],
        ids=["boards", "export", "login"],
    )
    def test_store_independent_command_dispatches_without_store(
        self,
        command: str,
        extra_argv: list[str],
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Given a CLI invocation with a store-independent subcommand
        When main() is called
        Then the corresponding handler runs without creating a store
        """
        # Given: sys.argv set to the subcommand, cwd at tmp_path with config
        monkeypatch.chdir(tmp_path)
        _write_test_config(tmp_path)
        monkeypatch.setattr("sys.argv", ["jobsearch_rag", command, *extra_argv])

        # When: main() dispatches (store-independent commands complete without store)
        main()

        # Then: observable output proves the handler ran
        captured = capsys.readouterr()
        assert captured.err == "", f"Expected no error output for {command}, got: {captured.err!r}"

    @pytest.mark.parametrize(
        ("command", "extra_argv"),
        [
            ("index", []),
            ("search", ["--board", "ziprecruiter"]),
            ("decide", ["job-42", "--verdict", "yes"]),
            ("decisions", ["show", "job-1"]),
            ("review", []),
            ("rescore", []),
            ("eval", []),
            ("reset", []),
        ],
        ids=["index", "search", "decide", "decisions", "review", "rescore", "eval", "reset"],
    )
    def test_store_dependent_command_receives_store_from_composition_root(
        self,
        command: str,
        extra_argv: list[str],
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Given a CLI invocation with a store-dependent subcommand
        When main() is called
        Then the composition root creates a store via config and the handler
             uses it (proven by the sentinel store's error reaching stderr)
        """
        # Given: config pointing to _SentinelStore, cwd at tmp_path
        monkeypatch.chdir(tmp_path)
        _write_test_config(tmp_path)
        monkeypatch.setattr("sys.argv", ["jobsearch_rag", command, *extra_argv])

        # When: main() creates _SentinelStore via factory, passes to handler,
        # handler calls a store method → raises → main() catches and prints
        main()

        # Then: stderr contains the sentinel message, proving the store reached
        # the handler through the composition root
        captured = capsys.readouterr()
        assert _SENTINEL_MSG in captured.err, (
            f"Expected sentinel message in stderr for {command}, "
            f"proving store was delivered to handler. Got: {captured.err!r}"
        )


def _write_test_config(tmp_path: Path) -> None:
    """
    Write a minimal settings.toml with store_class pointing to _SentinelStore.

    Also creates the global_rubric.toml required by config validation.
    Uses ``monkeypatch.chdir(tmp_path)`` so ``load_settings()`` finds the
    relative ``config/settings.toml`` path.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    store_class = f"{_SentinelStore.__module__}.{_SentinelStore.__qualname__}"
    (config_dir / "settings.toml").write_text(
        _MINIMAL_SETTINGS_TOML.format(store_class=store_class),
        encoding="utf-8",
    )
    (config_dir / "global_rubric.toml").write_text(
        "# empty rubric for tests\n",
        encoding="utf-8",
    )


_MINIMAL_SETTINGS_TOML = """\
resume_path = "data/resume.md"
archetypes_path = "config/role_archetypes.toml"
global_rubric_path = "config/global_rubric.toml"

[boards]
enabled = ["testboard"]
session_storage_dir = "."

[boards.testboard]
searches = ["https://testboard.com/search"]
max_pages = 1
headless = true
rate_limit_range = [1.5, 3.5]

[scoring]
archetype_weight = 0.5
fit_weight = 0.3
history_weight = 0.2
comp_weight = 0.15
negative_weight = 0.4
culture_weight = 0.2
base_salary = 220000
disqualify_on_llm_flag = true
min_score_threshold = 0.45
missing_comp_score = 0.5
chunk_overlap = 2000
dedup_similarity_threshold = 0.95
top_k_retrieval = 3
salary_floor = 10.0
salary_ceiling = 1000000.0
hours_per_year = 2080

[[scoring.comp_bands]]
ratio = 1.0
score = 1.0

[[scoring.comp_bands]]
ratio = 0.90
score = 0.7

[[scoring.comp_bands]]
ratio = 0.77
score = 0.4

[[scoring.comp_bands]]
ratio = 0.68
score = 0.0

[ollama]
base_url = "http://localhost:11434"
llm_model = "mistral:7b"
embed_model = "nomic-embed-text"
slow_llm_threshold_ms = 30000
classify_system_prompt = "You are a classifier."
max_retries = 1
base_delay = 0.0
max_embed_chars = 8000
head_ratio = 0.6
retryable_status_codes = [408, 429, 500, 502, 503, 504]

[output]
default_format = "markdown"
output_dir = "./output"
open_top_n = 5
jd_dir = "output/jds"
decisions_dir = "decisions"
log_dir = "logs"
eval_history_path = "data/eval_history.jsonl"
max_slug_length = 80

[chroma]
persist_dir = "./chroma"
distance_metric = "cosine"
sync_threshold = 1

[vectorstore]
persist_dir = "./chroma"
distance_metric = "cosine"
sync_threshold = 1
store_class = "{store_class}"

[security]
screen_prompt = "Review the following job description text."

[adapters]
cdp_timeout = 15.0
max_full_text_chars = 250000
viewport_width = 1440
viewport_height = 900

[adapters.browser_paths]
"""


# ---------------------------------------------------------------------------
# TestMainErrorDisplay
# ---------------------------------------------------------------------------


class TestMainErrorDisplay:
    """
    REQUIREMENT: main() formats errors with rich context for the operator

    WHO: The operator seeing a CLI failure in their terminal
    WHAT: (1) The system prints the actionable error type, message, and suggestion to stderr when main() catches an ActionableError with a suggestion.
          (2) The system prints only the actionable error type and message to stderr when main() catches an ActionableError without a suggestion.
          (3) The system prints "Unexpected error" and the exception message to stderr when main() catches a non-ActionableError exception.
    WHY: Unformatted tracebacks are unactionable — the operator needs
         the error type, message, and recovery suggestion at a glance

    MOCK BOUNDARY:
        Mock: handle_* functions (raise controlled exceptions),
              sys.argv (process state), sys.exit (process termination)
        Real: main(), build_parser(), error formatting logic
        Never: Patch print or stderr internals
    """

    def test_actionable_error_prints_type_and_message(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """
        Given a handler that raises an ActionableError with a suggestion
        When main() catches it
        Then error_type, message, and suggestion are printed to stderr
        """
        # Given: handle_boards raises an ActionableError
        err = ActionableError(
            error="Session expired",
            error_type=ErrorType.AUTHENTICATION,
            service="ziprecruiter",
            suggestion="Re-authenticate in headed mode",
        )
        with (
            patch("sys.argv", ["jobsearch_rag", "boards"]),
            patch.dict(HANDLERS, {"boards": MagicMock(side_effect=err)}),
            pytest.raises(SystemExit) as exc_info,
        ):
            # When: main() runs
            main()

        # Then: process exits with code 1
        assert exc_info.value.code == 1, f"Expected exit code 1, got: {exc_info.value.code}"

        # Then: stderr contains error type, message, and suggestion
        captured = capsys.readouterr().err
        assert "authentication" in captured, f"Expected error_type in stderr, got: {captured!r}"
        assert "Session expired" in captured, (
            f"Expected error message in stderr, got: {captured!r}"
        )
        assert "Re-authenticate in headed mode" in captured, (
            f"Expected suggestion in stderr, got: {captured!r}"
        )

    def test_actionable_error_without_suggestion_omits_suggestion_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """
        Given a handler that raises an ActionableError without a suggestion
        When main() catches it
        Then only error_type and message are printed (no Suggestion line)
        """
        # Given: ActionableError with no suggestion
        err = ActionableError(
            error="Config file not found",
            error_type=ErrorType.CONFIG,
            service="settings",
        )
        with (
            patch("sys.argv", ["jobsearch_rag", "boards"]),
            patch.dict(HANDLERS, {"boards": MagicMock(side_effect=err)}),
            pytest.raises(SystemExit),
        ):
            # When: main() runs
            main()

        # Then: no Suggestion line in stderr
        captured = capsys.readouterr().err
        assert "Config file not found" in captured, (
            f"Expected error message in stderr, got: {captured!r}"
        )
        assert "Suggestion" not in captured, (
            f"Expected no Suggestion line when suggestion is None, got: {captured!r}"
        )

    def test_generic_exception_prints_unexpected_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """
        Given a handler that raises a non-ActionableError exception
        When main() catches it
        Then 'Unexpected error' and the message are printed to stderr
        """
        # Given: handle_boards raises a generic RuntimeError
        with (
            patch("sys.argv", ["jobsearch_rag", "boards"]),
            patch.dict(
                HANDLERS,
                {"boards": MagicMock(side_effect=RuntimeError("Something broke"))},
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            # When: main() runs
            main()

        # Then: process exits with code 1
        assert exc_info.value.code == 1, f"Expected exit code 1, got: {exc_info.value.code}"

        # Then: stderr contains 'Unexpected error' message
        captured = capsys.readouterr().err
        assert "Unexpected error" in captured, (
            f"Expected 'Unexpected error' in stderr, got: {captured!r}"
        )
        assert "Something broke" in captured, (
            f"Expected exception message in stderr, got: {captured!r}"
        )


# ---------------------------------------------------------------------------
# TestMainModuleEntryPoint
# ---------------------------------------------------------------------------


class TestMainModuleEntryPoint:
    """
    REQUIREMENT: ``python -m jobsearch_rag`` invokes main() via the __main__ guard

    WHO: The operator running the package as a module
    WHAT: (1) Running the package as ``python -m jobsearch_rag <cmd>`` executes main()
    WHY: Without the guard, ``python -m jobsearch_rag`` would import
         definitions but never dispatch — the operator would see no output

    MOCK BOUNDARY:
        Mock: (none — boards subcommand is pure: registry read + print)
        Real: full subprocess execution of ``python -m jobsearch_rag boards``
        Never: (none)
    """

    def test_module_entry_point_invokes_main(self) -> None:
        """
        Given the package is invoked as ``python -m jobsearch_rag boards``
        When Python executes __main__.py with __name__ set to "__main__"
        Then main() dispatches the subcommand and the handler runs
        """
        # Given: the boards subcommand requires no config or external services
        # When: the package is executed as a module in a fresh process
        result = subprocess.run(
            [sys.executable, "-m", "jobsearch_rag", "boards"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Then: main() dispatched to handle_boards(), printing adapter names
        assert result.returncode == 0, (
            f"Expected exit code 0, got {result.returncode}; stderr: {result.stderr}"
        )
        assert "Registered adapters:" in result.stdout, (
            f"Expected adapter listing in stdout, got: {result.stdout!r}"
        )
