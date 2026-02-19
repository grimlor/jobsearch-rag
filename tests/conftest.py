"""Global test configuration — guards against writing to real output/.

This conftest makes the real ``output/`` directory read-only for the
duration of the test session.  Any test that accidentally writes there
(because it forgot to override ``output_dir``) will get an immediate
``PermissionError`` instead of silently contaminating real data.

Tests that need an output directory should use ``tmp_path`` or
``tempfile.TemporaryDirectory()`` and pass the path via
``OutputConfig(output_dir=...)``.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

_PROJECT_OUTPUT = Path(__file__).resolve().parent.parent / "output"


@pytest.fixture(autouse=True, scope="session")
def _guard_real_output_dir():
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
            try:
                d.chmod(mode)
            except OSError:
                pass  # Best-effort restore
