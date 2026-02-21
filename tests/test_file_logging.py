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
    """Run logs are persisted to disk for post-run diagnosis."""

    def test_run_creates_log_file_in_data_logs_directory(self, tmp_path: Path) -> None:
        """A log file appears under the specified logs directory after
        file logging is enabled and a message is logged."""
        log_dir = tmp_path / "logs"
        handler = configure_file_logging(log_dir=str(log_dir))
        try:
            logger.info("test message")
            log_files = list(log_dir.glob("*.log"))
            assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"
        finally:
            logger.removeHandler(handler)

    def test_log_file_name_includes_timestamp(self, tmp_path: Path) -> None:
        """Log file names follow the pattern jobsearch-rag_YYYY-MM-DDTHH-MM-SS.log
        so that multiple runs produce distinct, chronologically sortable files."""
        log_dir = tmp_path / "logs"
        handler = configure_file_logging(log_dir=str(log_dir))
        try:
            logger.info("timestamp check")
            log_files = list(log_dir.glob("*.log"))
            assert len(log_files) == 1
            name = log_files[0].name
            # Expect: jobsearch-rag_2026-02-18T12-30-45.log
            assert re.match(r"jobsearch-rag_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.log$", name), (
                f"Log filename '{name}' does not match timestamp pattern"
            )
        finally:
            logger.removeHandler(handler)

    def test_log_file_contains_same_messages_as_stderr(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The file log contains the same formatted messages that appear
        on stderr, so the operator can reconstruct a run post-hoc."""
        log_dir = tmp_path / "logs"
        handler = configure_file_logging(log_dir=str(log_dir))
        try:
            logger.warning("duplicate check message")
            log_files = list(log_dir.glob("*.log"))
            assert len(log_files) == 1
            content = log_files[0].read_text()
            assert "duplicate check message" in content
            assert "WARNING" in content
        finally:
            logger.removeHandler(handler)

    def test_log_directory_is_created_if_absent(self, tmp_path: Path) -> None:
        """The log directory is created automatically when it doesn't
        exist — the operator does not need to mkdir manually."""
        log_dir = tmp_path / "nested" / "deep" / "logs"
        assert not log_dir.exists()
        handler = configure_file_logging(log_dir=str(log_dir))
        try:
            logger.info("auto-create dir")
            assert log_dir.exists()
            assert len(list(log_dir.glob("*.log"))) == 1
        finally:
            logger.removeHandler(handler)

    def test_stderr_output_is_not_suppressed_when_file_logging_enabled(
        self, tmp_path: Path
    ) -> None:
        """File logging is additive — the existing stderr handler must
        remain active so the operator still sees output in real time."""
        log_dir = tmp_path / "logs"
        handler = configure_file_logging(log_dir=str(log_dir))
        try:
            # Verify the stderr handler still exists on the logger
            handler_types = [type(h) for h in logger.handlers]
            assert logging.StreamHandler in handler_types, (
                "stderr StreamHandler was removed when file logging was enabled"
            )
        finally:
            logger.removeHandler(handler)

    def test_log_level_is_configurable(self, tmp_path: Path) -> None:
        """The file handler respects a configurable log level so the
        operator can capture DEBUG detail for diagnosis without changing
        the stderr output level."""
        log_dir = tmp_path / "logs"
        handler = configure_file_logging(log_dir=str(log_dir), level=logging.DEBUG)
        try:
            logger.debug("debug-level message")
            log_files = list(log_dir.glob("*.log"))
            assert len(log_files) == 1
            content = log_files[0].read_text()
            assert "debug-level message" in content
            assert "DEBUG" in content
        finally:
            logger.removeHandler(handler)
