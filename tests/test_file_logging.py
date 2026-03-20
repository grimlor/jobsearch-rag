"""File logging tests — persistent log files for post-run diagnosis.

Maps to BDD spec: TestFileLogging

Tests verify that run logs are persisted to disk under data/logs/ with
timestamped filenames, configurable log level, and without suppressing
stderr output.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from jobsearch_rag.logging import configure_file_logging, logger

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class TestFileLogging:
    """
    REQUIREMENT: Run logs are persisted to disk for post-run diagnosis.

    WHO: The operator diagnosing a failed or unexpected pipeline run
    WHAT: (1) The system creates a log file under the specified logs directory when file logging is enabled and a message is logged.
          (2) The system names the log file with the `jobsearch-rag_YYYY-MM-DDTHH-MM-SS.log` timestamp pattern.
          (3) The system writes the logged warning message and its level marker to the log file.
          (4) The system creates the log directory automatically when the configured path does not already exist.
          (5) The system keeps the existing stderr `StreamHandler` active when file logging is enabled.
          (6) The system writes a debug message and its `DEBUG` level marker to the log file when the log level is configured as `DEBUG`.
    WHY: Without persistent file logs, transient console output is lost
         after the terminal closes, making post-run diagnosis impossible

    MOCK BOUNDARY:
        Mock:  nothing — uses tmp_path for real filesystem I/O
        Real:  configure_file_logging, logger, file handlers, log directory
        Never: Patch logging internals — exercise the real logging stack
    """

    def test_run_creates_log_file_in_data_logs_directory(self, tmp_path: Path) -> None:
        """
        Given a temporary logs directory
        When file logging is enabled and a message is logged
        Then a log file appears under the specified logs directory
        """
        # Given: a temporary logs directory
        log_dir = tmp_path / "logs"
        handler = configure_file_logging(log_dir=str(log_dir))
        try:
            # When: a message is logged
            logger.info("test message")

            # Then: exactly one log file is created
            log_files = list(log_dir.glob("*.log"))
            assert len(log_files) == 1, (
                f"Expected 1 log file, found {len(log_files)}: {[f.name for f in log_files]}"
            )
        finally:
            logger.removeHandler(handler)

    def test_log_file_name_includes_timestamp(self, tmp_path: Path) -> None:
        """
        Given file logging is enabled in a temporary directory
        When a message is logged
        Then the log filename follows jobsearch-rag_YYYY-MM-DDTHH-MM-SS.log
        """
        # Given: a temporary logs directory
        log_dir = tmp_path / "logs"
        handler = configure_file_logging(log_dir=str(log_dir))
        try:
            # When: a message is logged
            logger.info("timestamp check")

            # Then: the filename matches the timestamp pattern
            log_files = list(log_dir.glob("*.log"))
            assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"
            name = log_files[0].name
            assert re.match(r"jobsearch-rag_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.log$", name), (
                f"Log filename '{name}' does not match timestamp pattern"
            )
        finally:
            logger.removeHandler(handler)

    def test_log_file_contains_same_messages_as_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """
        Given file logging is enabled in a temporary directory
        When a warning is logged with file logging enabled
        Then the file contains the message text and log level
        """
        # Given: file logging enabled in a temporary directory
        log_dir = tmp_path / "logs"
        handler = configure_file_logging(log_dir=str(log_dir))
        try:
            # When: a warning message is logged
            logger.warning("duplicate check message")

            # Then: the log file contains the message and level
            log_files = list(log_dir.glob("*.log"))
            assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"
            content = log_files[0].read_text()
            assert "duplicate check message" in content, (
                f"Expected 'duplicate check message' in log content. Got: {content[:200]}"
            )
            assert "WARNING" in content, (
                f"Expected 'WARNING' level in log content. Got: {content[:200]}"
            )
        finally:
            logger.removeHandler(handler)

    def test_log_directory_is_created_if_absent(self, tmp_path: Path) -> None:
        """
        Given a log directory path that does not exist
        When file logging is configured
        Then the directory is created automatically
        """
        # Given: a deeply nested directory that doesn't exist
        log_dir = tmp_path / "nested" / "deep" / "logs"
        assert not log_dir.exists(), (
            f"Expected log_dir to not exist before configuration, but it does: {log_dir}"
        )

        # When: file logging is configured
        handler = configure_file_logging(log_dir=str(log_dir))
        try:
            logger.info("auto-create dir")

            # Then: the directory was created and contains a log file
            assert log_dir.exists(), f"Expected log_dir to be created automatically: {log_dir}"
            log_files = list(log_dir.glob("*.log"))
            assert len(log_files) == 1, (
                f"Expected 1 log file after auto-creation, found {len(log_files)}"
            )
        finally:
            logger.removeHandler(handler)

    def test_stderr_output_is_not_suppressed_when_file_logging_enabled(
        self, tmp_path: Path
    ) -> None:
        """
        Given file logging is enabled alongside stderr output
        When file logging is enabled
        Then the existing stderr StreamHandler remains active
        """
        # Given: file logging enabled
        log_dir = tmp_path / "logs"
        handler = configure_file_logging(log_dir=str(log_dir))
        try:
            # When: inspecting the logger's handlers
            handler_types = [type(h) for h in logger.handlers]

            # Then: a StreamHandler (stderr) is still present
            assert logging.StreamHandler in handler_types, (
                "stderr StreamHandler was removed when file logging was enabled. "
                f"Active handlers: {logger.handlers}"
            )
        finally:
            logger.removeHandler(handler)

    def test_log_level_is_configurable(self, tmp_path: Path) -> None:
        """
        Given file logging configured with DEBUG level
        When a debug message is logged
        Then the file contains the debug message and DEBUG level marker
        """
        # Given: file logging at DEBUG level
        log_dir = tmp_path / "logs"
        handler = configure_file_logging(log_dir=str(log_dir), level=logging.DEBUG)
        try:
            # When: a debug message is logged
            logger.debug("debug-level message")

            # Then: the log file captures the debug output
            log_files = list(log_dir.glob("*.log"))
            assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"
            content = log_files[0].read_text()
            assert "debug-level message" in content, (
                f"Expected 'debug-level message' in log content. Got: {content[:200]}"
            )
            assert "DEBUG" in content, (
                f"Expected 'DEBUG' level in log content. Got: {content[:200]}"
            )
        finally:
            logger.removeHandler(handler)
