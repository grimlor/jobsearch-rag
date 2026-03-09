"""CLI entry point dispatch and error display tests.

Spec classes:
    TestMainDispatch — main() routes each subcommand to its handler
    TestMainErrorDisplay — main() formats errors for the operator
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jobsearch_rag.__main__ import main
from jobsearch_rag.errors import ActionableError, ErrorType

# ---------------------------------------------------------------------------
# TestMainDispatch
# ---------------------------------------------------------------------------


class TestMainDispatch:
    """REQUIREMENT: main() dispatches each CLI subcommand to the correct handler.

    WHO: The operator invoking ``python -m jobsearch_rag <command>``
    WHAT: Each subcommand string (boards, index, search, decide, review,
          export, rescore, login, reset) is routed to its corresponding
          handle_* function with the parsed args namespace
    WHY: The shim is the only coupling between argparse and handler
         functions — incorrect wiring silently runs the wrong command

    MOCK BOUNDARY:
        Mock: handle_* functions (CLI handler I/O), sys.argv (process state)
        Real: main(), build_parser() argument parsing
        Never: Patch build_parser internals or argparse
    """

    @pytest.mark.parametrize(
        ("command", "handler_name", "extra_argv"),
        [
            ("boards", "handle_boards", []),
            ("index", "handle_index", []),
            ("search", "handle_search", ["--board", "ziprecruiter"]),
            ("decide", "handle_decide", ["job-42", "--verdict", "yes"]),
            ("review", "handle_review", []),
            ("export", "handle_export", []),
            ("rescore", "handle_rescore", []),
            ("login", "handle_login", ["--board", "ziprecruiter"]),
            ("reset", "handle_reset", []),
        ],
        ids=[
            "boards",
            "index",
            "search",
            "decide",
            "review",
            "export",
            "rescore",
            "login",
            "reset",
        ],
    )
    def test_subcommand_dispatches_to_correct_handler(
        self,
        command: str,
        handler_name: str,
        extra_argv: list[str],
    ) -> None:
        """
        GIVEN a CLI invocation with a subcommand
        When main() is called
        Then the corresponding handle_* function is invoked.
        """
        # Given: sys.argv set to the subcommand
        mock_handler = MagicMock()
        with (
            patch("sys.argv", ["jobsearch_rag", command, *extra_argv]),
            patch(f"jobsearch_rag.__main__.{handler_name}", mock_handler),
        ):
            # When: main() dispatches
            main()

        # Then: the correct handler was called
        mock_handler.assert_called_once()


# ---------------------------------------------------------------------------
# TestMainErrorDisplay
# ---------------------------------------------------------------------------


class TestMainErrorDisplay:
    """REQUIREMENT: main() formats errors with rich context for the operator.

    WHO: The operator seeing a CLI failure in their terminal
    WHAT: ActionableError prints error_type and message to stderr
          with suggestion when present; generic exceptions print
          'Unexpected error'; both paths exit with code 1
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
        GIVEN a handler that raises an ActionableError with a suggestion
        When main() catches it
        Then error_type, message, and suggestion are printed to stderr.
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
            patch("jobsearch_rag.__main__.handle_boards", side_effect=err),
            patch("sys.exit") as mock_exit,
        ):
            # When: main() runs
            main()

        # Then: stderr contains error type, message, and suggestion
        captured = capsys.readouterr().err
        assert "authentication" in captured, f"Expected error_type in stderr, got: {captured!r}"
        assert "Session expired" in captured, (
            f"Expected error message in stderr, got: {captured!r}"
        )
        assert "Re-authenticate in headed mode" in captured, (
            f"Expected suggestion in stderr, got: {captured!r}"
        )
        mock_exit.assert_called_once_with(1)

    def test_actionable_error_without_suggestion_omits_suggestion_line(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """
        GIVEN a handler that raises an ActionableError without a suggestion
        When main() catches it
        Then only error_type and message are printed (no Suggestion line).
        """
        # Given: ActionableError with no suggestion
        err = ActionableError(
            error="Config file not found",
            error_type=ErrorType.CONFIG,
            service="settings",
        )
        with (
            patch("sys.argv", ["jobsearch_rag", "boards"]),
            patch("jobsearch_rag.__main__.handle_boards", side_effect=err),
            patch("sys.exit"),
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
        GIVEN a handler that raises a non-ActionableError exception
        When main() catches it
        Then 'Unexpected error' and the message are printed to stderr.
        """
        # Given: handle_boards raises a generic RuntimeError
        with (
            patch("sys.argv", ["jobsearch_rag", "boards"]),
            patch(
                "jobsearch_rag.__main__.handle_boards",
                side_effect=RuntimeError("Something broke"),
            ),
            patch("sys.exit") as mock_exit,
        ):
            # When: main() runs
            main()

        # Then: stderr contains 'Unexpected error' message
        captured = capsys.readouterr().err
        assert "Unexpected error" in captured, (
            f"Expected 'Unexpected error' in stderr, got: {captured!r}"
        )
        assert "Something broke" in captured, (
            f"Expected exception message in stderr, got: {captured!r}"
        )
        mock_exit.assert_called_once_with(1)
