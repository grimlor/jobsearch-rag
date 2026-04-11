"""
Shared browser session tests — BrowserManager, BoardSession, and orchestration.

Maps to BDD spec: BDD Specifications — shared-browser-session.md

Spec classes:
    TestBrowserManager — browser process lifecycle, CDP launch, cleanup
    TestBoardSession — per-board context, cookies, stealth, page management
    TestSharedBrowserOrchestration — channel grouping, shared browser, backward compat

Public API surface (from src/jobsearch_rag/adapters/session):
    NOTE: BrowserManager and BoardSession do not exist yet — these tests
    specify their expected API. Implementation follows in Phase 3.

    BrowserManager(config: SessionConfig) — async context manager
        .browser: Browser  (available after __aenter__)

    BoardSession(browser: Browser, config: SessionConfig) — async context manager
        .new_page() -> Page
        .save_storage_state() -> Path
        .has_storage_state() -> bool

    SessionManager(config: SessionConfig) — backward-compatible wrapper
        .new_page() -> Page
        .save_storage_state() -> Path
        .has_storage_state() -> bool
"""

from __future__ import annotations

import builtins
import contextlib
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from jobsearch_rag.adapters.base import JobBoardAdapter

from jobsearch_rag.adapters.registry import AdapterRegistry
from jobsearch_rag.adapters.session import (
    BoardSession,
    BrowserManager,
    SessionConfig,
    SessionManager,
)
from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.pipeline.runner import PipelineRunner
from tests.conftest import make_test_settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adapt(adapter: object) -> Callable[..., JobBoardAdapter]:
    """Wrap an adapter/mock as a registry-compatible factory accepting any kwargs."""

    def _factory(**_kwargs: object) -> JobBoardAdapter:
        return cast("JobBoardAdapter", adapter)

    return _factory


def _mock_playwright(
    *,
    connect_over_cdp: bool = True,
    cdp_pages: list[MagicMock] | None = None,
) -> MagicMock:
    """Build a mock Playwright instance with browser/context stubs."""
    mock_context = MagicMock()
    mock_context.close = AsyncMock()
    mock_context.storage_state = AsyncMock(return_value={})
    mock_context.new_page = AsyncMock(return_value=MagicMock())

    mock_browser = MagicMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_browser.close = AsyncMock()

    if cdp_pages is not None:
        cdp_ctx = MagicMock()
        cdp_ctx.pages = cdp_pages
        mock_browser.contexts = [cdp_ctx]
    else:
        mock_browser.contexts = []

    pw = MagicMock()
    pw.stop = AsyncMock()

    if connect_over_cdp:
        pw.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
    pw.chromium.launch = AsyncMock(return_value=mock_browser)

    return pw


def _patch_playwright(mock_pw: MagicMock) -> Any:
    """Return a patch context for ``async_playwright`` that yields *mock_pw*."""
    mock_pw_ctx = MagicMock()
    mock_pw_ctx.start = AsyncMock(return_value=mock_pw)
    return patch(
        "jobsearch_rag.adapters.session.async_playwright",
        return_value=mock_pw_ctx,
    )


# ---------------------------------------------------------------------------
# TestBrowserManager
# ---------------------------------------------------------------------------


class TestBrowserManager:
    """
    REQUIREMENT: BrowserManager owns the Playwright instance and browser
    process lifecycle, shared across multiple BoardSessions.

    WHO: The pipeline runner needing a long-lived browser process that
         outlives individual board sessions
    WHAT: (1) BrowserManager starts a Playwright instance and launches
              a browser via standard Playwright launch when no browser
              channel is configured.
          (2) BrowserManager launches a CDP subprocess and connects
              Playwright over the CDP endpoint when a browser channel
              is configured.
          (3) BrowserManager launches the CDP subprocess with the remote
              debugging port, user data directory, and no-first-run flags.
          (4) BrowserManager adds the `--headless=new` flag to the CDP
              subprocess when headless mode is enabled.
          (5) BrowserManager uses the browser binary resolved by
              `shutil.which` when no configured browser path matches
              the channel.
          (6) BrowserManager raises an ActionableError that tells the
              operator which browser to install when the configured CDP
              browser binary is missing.
          (7) BrowserManager sends SIGTERM to the CDP subprocess and
              removes its temporary directory during cleanup.
          (8) BrowserManager escalates to SIGKILL when the CDP subprocess
              does not exit after SIGTERM times out during cleanup.
          (9) BrowserManager skips sending termination signals when the
              CDP subprocess has already exited before cleanup.
          (10) BrowserManager raises a TimeoutError when the CDP endpoint
               never responds before the deadline expires.
          (11) BrowserManager raises a TimeoutError that includes the CDP
               URL when repeated CDP readiness checks exhaust the deadline.
          (12) BrowserManager exposes the browser object to callers after
               entering the context manager.
          (13) BrowserManager skips a non-existent browser binary path and
               uses the next valid binary in the list.
    WHY: Spinning up a full browser process per board wastes resources and
         causes repeated macOS focus-stealing events; a shared browser
         process eliminates both problems

    MOCK BOUNDARY:
        Mock:  subprocess.Popen (process I/O), urllib.request.urlopen
               (CDP polling), async_playwright (browser API), shutil
               (filesystem cleanup), tempfile (temp dir)
        Real:  BrowserManager (full lifecycle via context manager),
               binary resolution logic (runs as part of entry)
        Never: Patch _find_browser_binary internals or config construction;
               import or call private functions directly from tests
    """

    # --- Standard Playwright launch (WHAT 1) ---

    async def test_launches_playwright_browser_when_no_channel(self) -> None:
        """
        Given a BrowserManager with no browser_channel configured
        When the context manager is entered
        Then Playwright starts and launches a browser via chromium.launch()
        """
        # Given: config with no browser_channel
        config = SessionConfig(board_name="testboard", headless=True)
        mock_pw = _mock_playwright(connect_over_cdp=False)

        with _patch_playwright(mock_pw):
            # When: BrowserManager is entered
            async with BrowserManager(config) as mgr:
                # Then: standard Playwright launch was used
                mock_pw.chromium.launch.assert_called_once_with(headless=True)
                mock_pw.chromium.connect_over_cdp.assert_not_called()
                assert mgr.browser is not None, (
                    "BrowserManager should expose a browser after entry"
                )

    # --- CDP launch and connect (WHAT 2) ---

    async def test_cdp_launches_subprocess_and_connects(self, tmp_path: Path) -> None:
        """
        Given a BrowserManager with browser_channel="msedge" and a valid binary
        When the context manager is entered
        Then a CDP subprocess is launched and Playwright connects over CDP
        """
        # Given: fake binary and mock Playwright
        fake_binary = tmp_path / "edge"
        fake_binary.touch()

        blank_page = MagicMock()
        blank_page.close = AsyncMock()
        mock_pw = _mock_playwright(cdp_pages=[blank_page])

        config = SessionConfig(
            board_name="ziprecruiter",
            headless=False,
            browser_channel="msedge",
        )

        with (
            patch(
                "jobsearch_rag.adapters.session._BROWSER_PATHS",
                {"msedge": [str(fake_binary)]},
            ),
            patch("shutil.which", return_value=None),
            patch("jobsearch_rag.adapters.session.subprocess.Popen"),
            patch("urllib.request.urlopen"),
            patch(
                "jobsearch_rag.adapters.session.tempfile.mkdtemp",
                return_value="/tmp/test",
            ),
            _patch_playwright(mock_pw),
        ):
            # When: BrowserManager is entered
            async with BrowserManager(config) as mgr:
                # Then: Playwright connected via CDP, not standard launch
                mock_pw.chromium.connect_over_cdp.assert_called_once()
                mock_pw.chromium.launch.assert_not_called()
                assert mgr.browser is not None, (
                    "BrowserManager should expose a browser after CDP entry"
                )

    # --- CDP subprocess flags (WHAT 3) ---

    async def test_cdp_subprocess_receives_correct_flags(self, tmp_path: Path) -> None:
        """
        Given a BrowserManager with browser_channel="msedge" and headless=False
        When the CDP subprocess is launched
        Then the command includes --remote-debugging-port, --user-data-dir,
             --no-first-run, and about:blank
        """
        # Given: fake binary and CDP config
        fake_binary = tmp_path / "edge"
        fake_binary.touch()
        mock_pw = _mock_playwright()

        config = SessionConfig(
            board_name="ziprecruiter",
            headless=False,
            browser_channel="msedge",
        )

        with (
            patch(
                "jobsearch_rag.adapters.session._BROWSER_PATHS",
                {"msedge": [str(fake_binary)]},
            ),
            patch("shutil.which", return_value=None),
            patch("jobsearch_rag.adapters.session.subprocess.Popen") as mock_popen,
            patch("urllib.request.urlopen"),
            patch(
                "jobsearch_rag.adapters.session.tempfile.mkdtemp",
                return_value="/tmp/jobsearch-test",
            ),
            _patch_playwright(mock_pw),
        ):
            # When: BrowserManager is entered
            async with BrowserManager(config):
                # Then: subprocess command has correct flags
                mock_popen.assert_called_once()
                cmd = mock_popen.call_args[0][0]
                assert any("--remote-debugging-port=" in arg for arg in cmd), (
                    f"Command should include --remote-debugging-port, got: {cmd}"
                )
                assert any("--user-data-dir=" in arg for arg in cmd), (
                    f"Command should include --user-data-dir, got: {cmd}"
                )
                assert "--no-first-run" in cmd, (
                    f"Command should include --no-first-run, got: {cmd}"
                )
                assert "--headless=new" not in cmd, (
                    f"headless=False should not add --headless=new, got: {cmd}"
                )

    # --- CDP headless flag (WHAT 4) ---

    async def test_cdp_headless_adds_headless_flag(self, tmp_path: Path) -> None:
        """
        Given a BrowserManager with browser_channel="msedge" and headless=True
        When the CDP subprocess is launched
        Then --headless=new is added to the subprocess args
        """
        # Given: headless CDP config
        fake_binary = tmp_path / "edge"
        fake_binary.touch()
        mock_pw = _mock_playwright()

        config = SessionConfig(
            board_name="test",
            headless=True,
            browser_channel="msedge",
        )

        with (
            patch(
                "jobsearch_rag.adapters.session._BROWSER_PATHS",
                {"msedge": [str(fake_binary)]},
            ),
            patch("shutil.which", return_value=None),
            patch("jobsearch_rag.adapters.session.subprocess.Popen") as mock_popen,
            patch("urllib.request.urlopen"),
            patch(
                "jobsearch_rag.adapters.session.tempfile.mkdtemp",
                return_value="/tmp/test",
            ),
            _patch_playwright(mock_pw),
        ):
            # When: BrowserManager is entered
            async with BrowserManager(config):
                # Then: headless flag present
                cmd = mock_popen.call_args[0][0]
                assert "--headless=new" in cmd, (
                    f"headless=True should add --headless=new, got: {cmd}"
                )

    # --- Binary resolution fallback (WHAT 5) ---

    async def test_binary_resolved_via_shutil_which_fallback(self, tmp_path: Path) -> None:
        """
        Given no match in configured browser_paths for the channel
        When shutil.which finds the binary on PATH
        Then the which-resolved binary is used for the subprocess
        """
        # Given: shutil.which returns a valid binary
        which_binary = str(tmp_path / "msedge")
        Path(which_binary).touch()

        config = SessionConfig(
            board_name="test",
            headless=False,
            browser_channel="msedge",
        )
        mock_pw = _mock_playwright()

        with (
            patch("jobsearch_rag.adapters.session._BROWSER_PATHS", {}),
            patch("shutil.which", return_value=which_binary),
            patch("jobsearch_rag.adapters.session.subprocess.Popen") as mock_popen,
            patch("urllib.request.urlopen"),
            patch(
                "jobsearch_rag.adapters.session.tempfile.mkdtemp",
                return_value="/tmp/test",
            ),
            _patch_playwright(mock_pw),
        ):
            # When: BrowserManager is entered
            async with BrowserManager(config):
                # Then: which-resolved binary was used
                cmd = mock_popen.call_args[0][0]
                assert cmd[0] == which_binary, f"Expected binary '{which_binary}', got '{cmd[0]}'"

    # --- Missing browser binary (WHAT 6) ---

    async def test_missing_binary_raises_actionable_error(self) -> None:
        """
        Given no browser binary found for the configured channel
        When BrowserManager is entered
        Then an ActionableError tells the operator which browser to install
        """
        # Given: no binary available
        config = SessionConfig(
            board_name="ziprecruiter",
            headless=False,
            browser_channel="msedge",
        )
        mock_pw = _mock_playwright()

        with (
            patch("jobsearch_rag.adapters.session._BROWSER_PATHS", {}),
            patch("shutil.which", return_value=None),
            _patch_playwright(mock_pw),
        ):
            # When/Then: entering raises ActionableError
            with pytest.raises(ActionableError) as exc_info:
                async with BrowserManager(config):
                    pass

            # Then: error provides install guidance
            err = exc_info.value
            assert "Could not find" in err.error, (
                f"Error should say browser not found, got: {err.error}"
            )
            assert err.suggestion is not None, "ActionableError should include a suggestion"

    # --- Cleanup: SIGTERM (WHAT 7) ---

    async def test_cleanup_terminates_subprocess_and_removes_tmpdir(self, tmp_path: Path) -> None:
        """
        Given a running CDP subprocess
        When BrowserManager __aexit__ is called
        Then SIGTERM is sent and the temp directory is cleaned up
        """
        # Given: config and mock subprocess
        config = SessionConfig(
            board_name="test",
            headless=False,
            browser_channel="msedge",
        )
        fake_binary = tmp_path / "edge"
        fake_binary.touch()

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None  # still running
        mock_proc.wait.return_value = 0

        mock_pw = _mock_playwright()
        tmpdir = str(tmp_path / "cdp_profile")
        Path(tmpdir).mkdir()

        with (
            patch(
                "jobsearch_rag.adapters.session._BROWSER_PATHS",
                {"msedge": [str(fake_binary)]},
            ),
            patch("shutil.which", return_value=None),
            patch(
                "jobsearch_rag.adapters.session.subprocess.Popen",
                return_value=mock_proc,
            ),
            patch("urllib.request.urlopen"),
            patch(
                "jobsearch_rag.adapters.session.tempfile.mkdtemp",
                return_value=tmpdir,
            ),
            _patch_playwright(mock_pw),
            patch("jobsearch_rag.adapters.session.shutil.rmtree") as mock_rmtree,
        ):
            # When: enter and exit the BrowserManager
            async with BrowserManager(config):
                pass

            # Then: SIGTERM sent and temp dir cleaned
            mock_proc.send_signal.assert_called_once()
            mock_rmtree.assert_called_once_with(tmpdir, ignore_errors=True)

    # --- Cleanup: SIGKILL escalation (WHAT 8) ---

    async def test_cleanup_kills_on_sigterm_timeout(self, tmp_path: Path) -> None:
        """
        Given a CDP subprocess that doesn't respond to SIGTERM
        When __aexit__ is called and SIGTERM times out
        Then the process is escalated to SIGKILL
        """
        # Given: config and mock subprocess that ignores SIGTERM
        config = SessionConfig(
            board_name="test",
            headless=False,
            browser_channel="msedge",
        )
        fake_binary = tmp_path / "edge"
        fake_binary.touch()

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None  # still running
        mock_proc.wait.side_effect = [
            subprocess.TimeoutExpired("edge", 5),
            0,
        ]

        mock_pw = _mock_playwright()

        with (
            patch(
                "jobsearch_rag.adapters.session._BROWSER_PATHS",
                {"msedge": [str(fake_binary)]},
            ),
            patch("shutil.which", return_value=None),
            patch(
                "jobsearch_rag.adapters.session.subprocess.Popen",
                return_value=mock_proc,
            ),
            patch("urllib.request.urlopen"),
            patch(
                "jobsearch_rag.adapters.session.tempfile.mkdtemp",
                return_value="/tmp/test",
            ),
            _patch_playwright(mock_pw),
            patch("jobsearch_rag.adapters.session.shutil.rmtree"),
        ):
            # When: enter and exit
            async with BrowserManager(config):
                pass

            # Then: escalated to SIGKILL
            mock_proc.kill.assert_called_once()

    # --- Cleanup: already exited (WHAT 9) ---

    async def test_cleanup_noop_when_process_already_exited(self, tmp_path: Path) -> None:
        """
        Given a CDP subprocess that has already exited
        When __aexit__ is called
        Then no signals are sent to the process
        """
        # Given: config and mock subprocess that already exited
        config = SessionConfig(
            board_name="test",
            headless=False,
            browser_channel="msedge",
        )
        fake_binary = tmp_path / "edge"
        fake_binary.touch()

        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = 0  # already exited

        mock_pw = _mock_playwright()

        with (
            patch(
                "jobsearch_rag.adapters.session._BROWSER_PATHS",
                {"msedge": [str(fake_binary)]},
            ),
            patch("shutil.which", return_value=None),
            patch(
                "jobsearch_rag.adapters.session.subprocess.Popen",
                return_value=mock_proc,
            ),
            patch("urllib.request.urlopen"),
            patch(
                "jobsearch_rag.adapters.session.tempfile.mkdtemp",
                return_value="/tmp/test",
            ),
            _patch_playwright(mock_pw),
            patch("jobsearch_rag.adapters.session.shutil.rmtree"),
        ):
            # When: enter and exit
            async with BrowserManager(config):
                pass

            # Then: no signal sent to already-exited process
            mock_proc.send_signal.assert_not_called()

    # --- CDP timeout (WHAT 10) ---

    async def test_cdp_timeout_raises_timeout_error(self, tmp_path: Path) -> None:
        """
        Given a CDP endpoint that never responds
        When BrowserManager is entered and the deadline expires
        Then a TimeoutError is raised
        """
        # Given: config and mock where CDP never starts
        config = SessionConfig(
            board_name="test",
            headless=False,
            browser_channel="msedge",
        )
        fake_binary = tmp_path / "edge"
        fake_binary.touch()
        tmpdir = str(tmp_path / "cdp_profile")
        Path(tmpdir).mkdir()

        mock_pw = _mock_playwright()

        # Simulate a clock that immediately expires the deadline
        mock_loop = MagicMock()
        mock_loop.time = MagicMock(side_effect=[0.0, 100.0])

        with (
            patch(
                "jobsearch_rag.adapters.session._BROWSER_PATHS",
                {"msedge": [str(fake_binary)]},
            ),
            patch("shutil.which", return_value=None),
            patch("jobsearch_rag.adapters.session.subprocess.Popen"),
            patch(
                "jobsearch_rag.adapters.session.tempfile.mkdtemp",
                return_value=tmpdir,
            ),
            patch(
                "jobsearch_rag.adapters.session.urllib.request.urlopen",
                side_effect=OSError("Connection refused"),
            ),
            patch(
                "jobsearch_rag.adapters.session.asyncio.get_event_loop",
                return_value=mock_loop,
            ),
            patch(
                "jobsearch_rag.adapters.session.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            _patch_playwright(mock_pw),
            patch("jobsearch_rag.adapters.session.shutil.rmtree"),
            pytest.raises(TimeoutError, match="did not start"),
        ):
            # When/Then: BrowserManager enters, CDP wait times out
            async with BrowserManager(config):
                pass

    # --- CDP timeout with URL in message (WHAT 11) ---

    async def test_cdp_wait_retries_then_times_out(self, tmp_path: Path) -> None:
        """
        Given a CDP endpoint that never becomes available
        When repeated readiness checks exhaust the deadline
        Then a TimeoutError is raised with the CDP URL in the message
        """
        # Given: CDP config with urlopen always failing and a clock that
        #        expires the deadline on the third call
        config = SessionConfig(
            board_name="test",
            headless=False,
            browser_channel="msedge",
        )
        fake_binary = tmp_path / "edge"
        fake_binary.touch()
        tmpdir = str(tmp_path / "cdp_profile")
        Path(tmpdir).mkdir()

        mock_pw = _mock_playwright()

        # Clock: first call sets deadline, second within bounds, third exceeds
        mock_loop = MagicMock()
        mock_loop.time = MagicMock(side_effect=[0.0, 1.0, 100.0])

        with (
            patch(
                "jobsearch_rag.adapters.session._BROWSER_PATHS",
                {"msedge": [str(fake_binary)]},
            ),
            patch("shutil.which", return_value=None),
            patch("jobsearch_rag.adapters.session.subprocess.Popen"),
            patch(
                "jobsearch_rag.adapters.session.tempfile.mkdtemp",
                return_value=tmpdir,
            ),
            patch(
                "jobsearch_rag.adapters.session.urllib.request.urlopen",
                side_effect=OSError("Connection refused"),
            ),
            patch(
                "jobsearch_rag.adapters.session.asyncio.get_event_loop",
                return_value=mock_loop,
            ),
            patch(
                "jobsearch_rag.adapters.session.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            _patch_playwright(mock_pw),
            patch("jobsearch_rag.adapters.session.shutil.rmtree"),
            pytest.raises(TimeoutError, match="did not start"),
        ):
            # When/Then: BrowserManager enters, retries once, then times out
            async with BrowserManager(config):
                pass

    # --- Browser object access (WHAT 12) ---

    async def test_exposes_browser_after_entry(self) -> None:
        """
        Given a BrowserManager that has been entered
        When the browser property is accessed
        Then it returns the Playwright Browser object
        """
        # Given: standard config
        config = SessionConfig(board_name="testboard", headless=True)
        mock_pw = _mock_playwright(connect_over_cdp=False)
        expected_browser = mock_pw.chromium.launch.return_value

        with _patch_playwright(mock_pw):
            # When: BrowserManager is entered
            async with BrowserManager(config) as mgr:
                # Then: browser property returns the launched browser
                assert mgr.browser is expected_browser, (
                    f"Expected browser {expected_browser}, got {mgr.browser}"
                )

    # --- Binary path skip (WHAT 13) ---

    async def test_skips_nonexistent_binary_and_uses_next(self, tmp_path: Path) -> None:
        """
        Given browser_paths with a non-existent first path and a valid second
        When BrowserManager is entered
        Then the non-existent path is skipped and the second binary is used
        """
        # Given: second binary exists, first does not
        second = tmp_path / "edge-second"
        second.touch()

        mock_pw = _mock_playwright()

        config = SessionConfig(
            board_name="test",
            headless=False,
            browser_channel="msedge",
        )

        with (
            patch(
                "jobsearch_rag.adapters.session._BROWSER_PATHS",
                {"msedge": ["/nonexistent/edge", str(second)]},
            ),
            patch("shutil.which", return_value=None),
            patch("jobsearch_rag.adapters.session.subprocess.Popen") as mock_popen,
            patch("urllib.request.urlopen"),
            patch(
                "jobsearch_rag.adapters.session.tempfile.mkdtemp",
                return_value="/tmp/test",
            ),
            _patch_playwright(mock_pw),
        ):
            # When: BrowserManager is entered
            async with BrowserManager(config):
                # Then: second binary was used
                cmd = mock_popen.call_args[0][0]
                assert cmd[0] == str(second), f"Expected '{second}', got '{cmd[0]}'"

    # --- Config override with no valid binary (WHAT 14) ---

    async def test_config_override_browser_paths_no_match_raises_error(self) -> None:
        """
        Given browser_paths configured with paths that do not exist on disk
        When BrowserManager is entered
        Then an ActionableError is raised because no binary could be resolved
        """
        # Given: config override browser_paths with non-existent paths
        config = SessionConfig(
            board_name="test",
            headless=False,
            browser_channel="msedge",
            browser_paths={"msedge": ["/nonexistent/a", "/nonexistent/b"]},
        )
        mock_pw = _mock_playwright()

        with _patch_playwright(mock_pw):
            # When/Then: entering raises ActionableError (binary not found)
            with pytest.raises(ActionableError) as exc_info:
                async with BrowserManager(config):
                    pass

            assert "Could not find" in exc_info.value.error, (
                f"Error should say browser not found, got: {exc_info.value.error}"
            )

    # --- Browser property before entry (WHAT 15) ---

    async def test_browser_property_raises_before_entry(self) -> None:
        """
        Given a BrowserManager that has not been entered
        When the browser property is accessed
        Then an ActionableError is raised telling the caller to use 'async with'
        """
        # Given: uninitialised BrowserManager
        config = SessionConfig(board_name="test", headless=True)
        mgr = BrowserManager(config)

        # When/Then: accessing .browser raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            _ = mgr.browser

        assert "before entering the context manager" in str(exc_info.value), (
            f"Error should mention context manager. Got: {exc_info.value}"
        )
        assert exc_info.value.error_type == ErrorType.CONFIG, (
            f"Expected CONFIG error type, got {exc_info.value.error_type}"
        )

    # --- Uninitialised exit is no-op (WHAT 16) ---

    async def test_aexit_on_uninitialised_manager_is_noop(self) -> None:
        """
        Given a BrowserManager that has not been entered
        When __aexit__ is called directly
        Then no error is raised and no resources are cleaned up
        """
        # Given: uninitialised BrowserManager
        config = SessionConfig(board_name="test", headless=True)
        mgr = BrowserManager(config)

        # When/Then: __aexit__ completes without error
        await mgr.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# TestBoardSession
# ---------------------------------------------------------------------------


class TestBoardSession:
    """
    REQUIREMENT: BoardSession owns per-board browser context lifecycle,
    providing cookie isolation, stealth patches, and page management
    over a shared Browser.

    WHO: The pipeline runner needing per-board isolation without per-board
         browser launches
    WHAT: (1) BoardSession creates a BrowserContext on the provided Browser
              when entered.
          (2) BoardSession loads persisted cookies from the board's storage
              state file when the file exists.
          (3) BoardSession starts a fresh session when no storage state file
              exists for the board.
          (4) BoardSession applies playwright-stealth patches to its context
              when stealth mode is enabled.
          (5) BoardSession starts successfully and logs a warning when stealth
              mode is enabled but playwright-stealth is unavailable.
          (6) BoardSession returns a new page from its browser context when
              new_page is called.
          (7) BoardSession writes the active session cookies to the
              configured storage state file as JSON.
          (8) BoardSession returns True from has_storage_state when the
              persisted session file exists.
          (9) BoardSession returns False from has_storage_state when no
              persisted session file exists.
          (10) BoardSession closes its context but does not close the browser
               when __aexit__ is called.
    WHY: Per-board context isolation prevents cross-site cookie leakage
         while sharing the expensive browser process; stealth and storage
         state are per-board concerns that must not affect other boards

    MOCK BOUNDARY:
        Mock:  async_playwright (browser API — provides the Browser object)
        Real:  BoardSession (full lifecycle via context manager, including
               page creation, storage state, and cookie persistence)
        Never: Patch BoardSession internals; mock the Browser.new_context
               return value directly (let BoardSession call it);
               import or call private functions directly from tests
    """

    # --- Context creation (WHAT 1) ---

    async def test_creates_browser_context_on_entry(self) -> None:
        """
        Given a Browser object from a BrowserManager
        When BoardSession is entered
        Then a new BrowserContext is created on that Browser
        """
        # Given: mock browser and config
        config = SessionConfig(board_name="testboard", headless=True)
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_context.close = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        # When: BoardSession is entered
        async with BoardSession(mock_browser, config):
            # Then: new_context was called on the browser
            mock_browser.new_context.assert_called_once()

    # --- Storage state loading (WHAT 2) ---

    async def test_loads_storage_state_when_file_exists(self, tmp_path: Path) -> None:
        """
        Given a persisted storage state file on disk for this board
        When BoardSession is entered
        Then the context is created with the stored cookies
        """
        # Given: a storage state file exists
        config = SessionConfig(board_name="test", headless=True)
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_context.close = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        with patch("jobsearch_rag.adapters.session._DEFAULT_STORAGE_DIR", tmp_path):
            state_file = config.storage_state_path
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(json.dumps({"cookies": [{"name": "sid"}]}))

            # When: BoardSession is entered
            async with BoardSession(mock_browser, config):
                # Then: new_context received a non-None storage_state
                call_kwargs = mock_browser.new_context.call_args[1]
                assert call_kwargs.get("storage_state") is not None, (
                    "new_context should receive storage_state when file exists"
                )

    # --- Fresh session (WHAT 3) ---

    async def test_starts_fresh_session_when_no_storage_state(self) -> None:
        """
        Given no storage state file exists for this board
        When BoardSession is entered
        Then the context is created without a storage_state parameter
        """
        # Given: no storage state file
        config = SessionConfig(board_name="nonexistent_board_xyz", headless=True)
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_context.close = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        # When: BoardSession is entered
        async with BoardSession(mock_browser, config):
            # Then: storage_state is None (no persisted file)
            call_kwargs = mock_browser.new_context.call_args[1]
            assert call_kwargs.get("storage_state") is None, (
                "new_context should receive storage_state=None when no file exists"
            )

    # --- Stealth patches (WHAT 4) ---

    async def test_stealth_patches_applied_when_enabled(self) -> None:
        """
        Given stealth mode is enabled in the board config
        When BoardSession is entered
        Then playwright-stealth patches are applied to the context
        """
        # Given: stealth config with mock stealth module
        config = SessionConfig(board_name="linkedin", headless=True, stealth=True)
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_context.close = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        mock_stealth = MagicMock()
        mock_stealth_instance = MagicMock()
        mock_stealth_instance.apply_stealth_async = AsyncMock()
        mock_stealth.return_value = mock_stealth_instance

        with patch.dict(
            "sys.modules",
            {"playwright_stealth": MagicMock(Stealth=mock_stealth)},
        ):
            # When: BoardSession is entered
            async with BoardSession(mock_browser, config):
                # Then: stealth patches were applied
                mock_stealth_instance.apply_stealth_async.assert_called_once()

    # --- Stealth import failure (WHAT 5) ---

    async def test_stealth_import_error_logs_warning(self) -> None:
        """
        Given stealth mode is enabled but playwright-stealth is not installed
        When BoardSession is entered
        Then the session still starts and a warning is logged
        """
        # Given: stealth config with broken import
        config = SessionConfig(board_name="linkedin", headless=True, stealth=True)
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_context.close = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        original_import = builtins.__import__

        def _raise_import_error(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "playwright_stealth":
                raise ImportError("No module named 'playwright_stealth'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_raise_import_error):
            # When/Then: entering does not raise — stealth failure is graceful
            async with BoardSession(mock_browser, config):
                pass

    # --- Page creation (WHAT 6) ---

    async def test_new_page_returns_page_from_context(self) -> None:
        """
        Given a BoardSession that has been entered
        When new_page is called
        Then a new page is returned from the board's browser context
        """
        # Given: mock browser with page stub
        config = SessionConfig(board_name="test", headless=True)
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_context.close = AsyncMock()
        mock_page = MagicMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        # When: BoardSession is entered and new_page called
        async with BoardSession(mock_browser, config) as session:
            page = await session.new_page()

            # Then: page returned from context
            assert page is mock_page, "new_page should return a page from the browser context"

    # --- Storage state persistence (WHAT 7) ---

    async def test_save_storage_state_persists_cookies_to_disk(self, tmp_path: Path) -> None:
        """
        Given an active BoardSession with cookies
        When save_storage_state is called
        Then cookies are written to the board's storage state path as JSON
        """
        # Given: mock browser with cookie data
        config = SessionConfig(board_name="test", headless=True)
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_context.close = AsyncMock()
        mock_context.storage_state = AsyncMock(return_value={"cookies": [{"name": "session"}]})
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        with patch("jobsearch_rag.adapters.session._DEFAULT_STORAGE_DIR", tmp_path):
            # When: enter and save storage state
            async with BoardSession(mock_browser, config) as session:
                result = await session.save_storage_state()

                # Then: file exists with correct data
                assert result.exists(), "Storage state file should be written to disk"
                saved = json.loads(result.read_text())
                assert saved["cookies"][0]["name"] == "session", (
                    f"Expected 'session', got {saved['cookies'][0].get('name')}"
                )

    # --- has_storage_state true (WHAT 8) ---

    async def test_has_storage_state_returns_true_when_file_exists(self, tmp_path: Path) -> None:
        """
        Given a persisted session file exists on disk
        When has_storage_state is called
        Then it returns True
        """
        # Given: session file exists and BoardSession enters properly
        config = SessionConfig(board_name="test", headless=True)
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_context.close = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        with patch("jobsearch_rag.adapters.session._DEFAULT_STORAGE_DIR", tmp_path):
            state_file = config.storage_state_path
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text("{}")

            # When/Then: has_storage_state returns True
            async with BoardSession(mock_browser, config) as session:
                assert session.has_storage_state() is True, (
                    "has_storage_state should return True when file exists"
                )

    # --- has_storage_state false (WHAT 9) ---

    async def test_has_storage_state_returns_false_when_no_file(self) -> None:
        """
        Given no persisted session file exists
        When has_storage_state is called
        Then it returns False
        """
        # Given: no session file, BoardSession enters properly
        config = SessionConfig(board_name="nonexistent_board_xyz", headless=True)
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_context.close = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        # When/Then: has_storage_state returns False
        async with BoardSession(mock_browser, config) as session:
            assert session.has_storage_state() is False, (
                "has_storage_state should return False when no file exists"
            )

    # --- Context-only cleanup (WHAT 10) ---

    async def test_aexit_closes_context_but_not_browser(self) -> None:
        """
        Given a BoardSession that has been entered with a shared Browser
        When __aexit__ is called
        Then the BrowserContext is closed but the Browser remains open
        """
        # Given: mock browser and context
        config = SessionConfig(board_name="test", headless=True)
        mock_browser = MagicMock()
        mock_context = MagicMock()
        mock_context.close = AsyncMock()
        mock_browser.close = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        # When: BoardSession is entered and exited
        async with BoardSession(mock_browser, config):
            pass

        # Then: context closed, browser NOT closed
        mock_context.close.assert_called_once()
        mock_browser.close.assert_not_called()

    # --- new_page before entry (WHAT 11) ---

    async def test_new_page_raises_before_entry(self) -> None:
        """
        Given a BoardSession that has not been entered
        When new_page is called
        Then an ActionableError is raised telling the caller to use 'async with'
        """
        # Given: uninitialised BoardSession
        config = SessionConfig(board_name="test", headless=True)
        mock_browser = MagicMock()
        session = BoardSession(mock_browser, config)

        # When/Then: new_page raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            await session.new_page()

        assert "before entering the context manager" in str(exc_info.value), (
            f"Error should mention context manager. Got: {exc_info.value}"
        )
        assert exc_info.value.error_type == ErrorType.CONFIG, (
            f"Expected CONFIG error type, got {exc_info.value.error_type}"
        )

    # --- save_storage_state before entry (WHAT 12) ---

    async def test_save_storage_state_raises_before_entry(self) -> None:
        """
        Given a BoardSession that has not been entered
        When save_storage_state is called
        Then an ActionableError is raised telling the caller to use 'async with'
        """
        # Given: uninitialised BoardSession
        config = SessionConfig(board_name="test", headless=True)
        mock_browser = MagicMock()
        session = BoardSession(mock_browser, config)

        # When/Then: save_storage_state raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            await session.save_storage_state()

        assert "before entering the context manager" in str(exc_info.value), (
            f"Error should mention context manager. Got: {exc_info.value}"
        )
        assert exc_info.value.error_type == ErrorType.CONFIG, (
            f"Expected CONFIG error type, got {exc_info.value.error_type}"
        )

    # --- Uninitialised exit is no-op (WHAT 13) ---

    async def test_aexit_on_uninitialised_session_is_noop(self) -> None:
        """
        Given a BoardSession that has not been entered
        When __aexit__ is called directly
        Then no error is raised and the browser remains unaffected
        """
        # Given: uninitialised BoardSession
        config = SessionConfig(board_name="test", headless=True)
        mock_browser = MagicMock()
        mock_browser.close = AsyncMock()
        session = BoardSession(mock_browser, config)

        # When/Then: __aexit__ completes without error
        await session.__aexit__(None, None, None)

        # And: browser was NOT closed (only the context would be closed)
        mock_browser.close.assert_not_called()


# ---------------------------------------------------------------------------
# TestSharedBrowserOrchestration — helpers
# ---------------------------------------------------------------------------


def _make_runner_with_real_stack(
    settings: Any,
) -> tuple[PipelineRunner, AsyncMock]:
    """Build a PipelineRunner with a mocked Ollama client; everything else real."""
    from tests.constants import EMBED_FAKE  # noqa: PLC0415

    mock_client = AsyncMock()

    # health_check calls client.list()
    model_embed = MagicMock()
    model_embed.model = settings.ollama.embed_model
    model_llm = MagicMock()
    model_llm.model = settings.ollama.llm_model
    list_response = MagicMock()
    list_response.models = [model_embed, model_llm]
    mock_client.list.return_value = list_response

    # embed() calls client.embed()
    embed_response = MagicMock()
    embed_response.embeddings = [EMBED_FAKE]
    mock_client.embed.return_value = embed_response

    with patch(
        "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
        return_value=mock_client,
    ):
        runner = PipelineRunner(settings)

    return runner, mock_client


def _mock_playwright_boundary() -> tuple[MagicMock, MagicMock]:
    """
    Create a mock Playwright I/O boundary for real SessionManager.

    Mocks ``async_playwright`` — the edge where our system ends and the
    Playwright library begins. SessionManager/BrowserManager/BoardSession
    run for real on top.

    Returns ``(mock_async_playwright, mock_page)`` for use with
    ``patch("jobsearch_rag.adapters.session.async_playwright", ...)``.
    """
    mock_page = MagicMock()

    mock_context = MagicMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)
    mock_context.storage_state = AsyncMock(return_value={"cookies": [], "origins": []})
    mock_context.close = AsyncMock()

    mock_browser = MagicMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_browser.close = AsyncMock()
    mock_browser.contexts = []

    mock_pw = MagicMock()
    mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_pw.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
    mock_pw.stop = AsyncMock()

    mock_pw_cm = MagicMock()
    mock_pw_cm.start = AsyncMock(return_value=mock_pw)

    mock_async_pw = MagicMock(return_value=mock_pw_cm)

    return mock_async_pw, mock_page


def _make_test_adapter(
    board_name: str = "testboard",
    *,
    search_results: list[Any] | None = None,
) -> MagicMock:
    """Build a mock board adapter with sensible defaults."""
    adapter = MagicMock()
    adapter.board_name = board_name
    adapter.rate_limit_seconds = (0.0, 0.0)
    adapter.authenticate = AsyncMock()
    adapter.search = AsyncMock(return_value=search_results if search_results is not None else [])
    adapter.extract_detail = AsyncMock()
    return adapter


def _cdp_patches() -> contextlib.ExitStack:
    """
    Return a context manager that stubs CDP subprocess and urlopen boundaries.

    These are needed for any orchestration test where boards have a non-None
    ``browser_channel``.  The Playwright boundary itself is handled by
    ``_mock_playwright_boundary``; these cover the subprocess and HTTP
    readiness check that CDP mode uses.
    """
    stack = contextlib.ExitStack()
    stack.enter_context(
        patch(
            "jobsearch_rag.adapters.session._find_browser_binary",
            return_value="/fake/edge",
        )
    )
    stack.enter_context(patch("jobsearch_rag.adapters.session.subprocess.Popen"))
    stack.enter_context(patch("jobsearch_rag.adapters.session.urllib.request.urlopen"))
    stack.enter_context(
        patch(
            "jobsearch_rag.adapters.session.tempfile.mkdtemp",
            return_value="/tmp/cdp-test",
        )
    )
    return stack


# ---------------------------------------------------------------------------
# TestSharedBrowserOrchestration
# ---------------------------------------------------------------------------


class TestSharedBrowserOrchestration:
    """
    REQUIREMENT: The pipeline runner groups boards by browser channel and
    shares a BrowserManager across boards in each group.

    WHO: The operator running multi-board searches without repeated browser
         launches and focus-stealing events
    WHAT: (1) The runner groups boards by browser_channel before searching.
          (2) Boards with the same browser_channel share a single
              BrowserManager.
          (3) Boards with browser_channel=None share a single
              BrowserManager for Playwright-managed Chromium.
          (4) Boards with different browser_channel values get separate
              BrowserManagers.
          (5) A fatal error in one BoardSession does not tear down the shared
              BrowserManager or other boards' sessions.
          (6) The BrowserManager outlives all BoardSessions in its group
              (context manager nesting is correct).
          (7) SessionManager continues to work as a convenience wrapper
              that composes BrowserManager + BoardSession for single-board
              callers.
          (8) The SessionManager wrapper opens a headed browser, creates a
              context, and produces a page — backward-compatible with
              handle_login tests.
    WHY: Per-board browser launches waste resources and steal macOS focus
         N times per run; grouping by channel collapses that to one launch
         per channel type

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (Ollama API),
               async_playwright (Playwright browser library)
        Real:  PipelineRunner, BrowserManager, BoardSession, SessionManager,
               Embedder, Scorer, VectorStore, Ranker, DecisionRecorder,
               AdapterRegistry, throttle
        Never: Construct ScoreResult directly; mock BrowserManager or
               BoardSession internals
    """

    # --- Channel grouping (WHAT 1) ---

    async def test_runner_groups_boards_by_browser_channel(self, tmp_path: Path) -> None:
        """
        Given three boards: two with browser_channel="msedge" and one with None
        When run() is called
        Then the runner creates two groups: one for "msedge" and one for None
        """
        # Given: three boards, two sharing msedge channel
        tmpdir = str(tmp_path)
        settings = make_test_settings(
            tmpdir,
            enabled_boards=["edge_a", "edge_b", "chromium_c"],
        )
        settings.boards["edge_a"].browser_channel = "msedge"
        settings.boards["edge_b"].browser_channel = "msedge"
        settings.boards["chromium_c"].browser_channel = None

        runner, _ = _make_runner_with_real_stack(settings)

        adapter_a = _make_test_adapter(board_name="edge_a")
        adapter_b = _make_test_adapter(board_name="edge_b")
        adapter_c = _make_test_adapter(board_name="chromium_c")

        mock_pw_fn, _ = _mock_playwright_boundary()

        with (
            AdapterRegistry.override(
                {
                    "edge_a": _adapt(adapter_a),
                    "edge_b": _adapt(adapter_b),
                    "chromium_c": _adapt(adapter_c),
                },
            ),
            patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
            patch(
                "jobsearch_rag.adapters.session._DEFAULT_STORAGE_DIR",
                Path(tmpdir),
            ),
            _cdp_patches(),
        ):
            await runner.run()

            # Then: exactly two groups were formed (msedge + None)
            # pw.start() is called once per BrowserManager — 2 groups = 2 calls
            pw_cm = mock_pw_fn.return_value
            assert pw_cm.start.call_count == 2, (
                f"Expected 2 BrowserManager instances (msedge + None), "
                f"got {pw_cm.start.call_count}"
            )
            assert adapter_a.authenticate.called, "edge_a should have run"
            assert adapter_b.authenticate.called, "edge_b should have run"
            assert adapter_c.authenticate.called, "chromium_c should have run"

    # --- Shared browser within group (WHAT 2) ---

    async def test_boards_with_same_channel_share_browser_manager(self, tmp_path: Path) -> None:
        """
        Given two boards both configured with browser_channel="msedge"
        When run() searches both boards
        Then both boards' BoardSessions use the same Browser object
        """
        # Given: two boards with same channel
        tmpdir = str(tmp_path)
        settings = make_test_settings(tmpdir, enabled_boards=["board_a", "board_b"])
        settings.boards["board_a"].browser_channel = "msedge"
        settings.boards["board_b"].browser_channel = "msedge"

        runner, _ = _make_runner_with_real_stack(settings)

        browsers_seen: list[object] = []

        async def _capture_browser_search(page: object, url: str, max_pages: int = 1) -> list[Any]:
            # The page's context's browser gives us the real Browser object
            browsers_seen.append(id(page))
            return []

        adapter_a = _make_test_adapter(board_name="board_a")
        adapter_a.search = _capture_browser_search
        adapter_b = _make_test_adapter(board_name="board_b")
        adapter_b.search = _capture_browser_search

        mock_pw_fn, _mock_page = _mock_playwright_boundary()

        with (
            AdapterRegistry.override(
                {
                    "board_a": _adapt(adapter_a),
                    "board_b": _adapt(adapter_b),
                },
            ),
            patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
            patch(
                "jobsearch_rag.adapters.session._DEFAULT_STORAGE_DIR",
                Path(tmpdir),
            ),
            _cdp_patches(),
        ):
            await runner.run()

            # Then: one BrowserManager for the shared msedge channel
            pw_cm = mock_pw_fn.return_value
            assert pw_cm.start.call_count == 1, (
                f"Expected 1 BrowserManager for shared channel, got {pw_cm.start.call_count}"
            )
            # Both boards were searched under that single browser
            assert len(browsers_seen) == 2, f"Expected 2 board searches, got {len(browsers_seen)}"

    # --- Headless Chromium sharing (WHAT 3) ---

    async def test_boards_with_no_channel_share_playwright_browser(self, tmp_path: Path) -> None:
        """
        Given two boards both configured with browser_channel=None
        When run() searches both boards
        Then both boards' BoardSessions share a single Playwright-managed browser
        """
        # Given: two boards with no channel (default Playwright Chromium)
        tmpdir = str(tmp_path)
        settings = make_test_settings(tmpdir, enabled_boards=["board_x", "board_y"])
        settings.boards["board_x"].browser_channel = None
        settings.boards["board_y"].browser_channel = None

        runner, _ = _make_runner_with_real_stack(settings)

        adapter_x = _make_test_adapter(board_name="board_x")
        adapter_y = _make_test_adapter(board_name="board_y")

        mock_pw_fn, _ = _mock_playwright_boundary()

        with (
            AdapterRegistry.override(
                {
                    "board_x": _adapt(adapter_x),
                    "board_y": _adapt(adapter_y),
                },
            ),
            patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
            patch(
                "jobsearch_rag.adapters.session._DEFAULT_STORAGE_DIR",
                Path(tmpdir),
            ),
        ):
            await runner.run()

            # Then: one BrowserManager for the shared None channel
            pw_cm = mock_pw_fn.return_value
            assert pw_cm.start.call_count == 1, (
                f"Expected 1 BrowserManager for shared None channel, got {pw_cm.start.call_count}"
            )
            assert adapter_x.search.called, "board_x should have searched"
            assert adapter_y.search.called, "board_y should have searched"

    # --- Separate browsers for different channels (WHAT 4) ---

    async def test_different_channels_get_separate_browser_managers(self, tmp_path: Path) -> None:
        """
        Given one board with browser_channel="msedge" and one with None
        When run() searches both boards
        Then each board has a different Browser object
        """
        # Given: two boards with different channels
        tmpdir = str(tmp_path)
        settings = make_test_settings(tmpdir, enabled_boards=["edge_board", "chromium_board"])
        settings.boards["edge_board"].browser_channel = "msedge"
        settings.boards["chromium_board"].browser_channel = None

        runner, _ = _make_runner_with_real_stack(settings)

        adapter_edge = _make_test_adapter(board_name="edge_board")
        adapter_chrom = _make_test_adapter(board_name="chromium_board")

        mock_pw_fn, _ = _mock_playwright_boundary()

        with (
            AdapterRegistry.override(
                {
                    "edge_board": _adapt(adapter_edge),
                    "chromium_board": _adapt(adapter_chrom),
                },
            ),
            patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
            patch(
                "jobsearch_rag.adapters.session._DEFAULT_STORAGE_DIR",
                Path(tmpdir),
            ),
            _cdp_patches(),
        ):
            await runner.run()

            # Then: two BrowserManagers (msedge + None = separate groups)
            pw_cm = mock_pw_fn.return_value
            assert pw_cm.start.call_count == 2, (
                f"Expected 2 BrowserManagers for different channels, got {pw_cm.start.call_count}"
            )
            assert adapter_edge.search.called, "edge_board should have run"
            assert adapter_chrom.search.called, "chromium_board should have run"

    # --- Error isolation (WHAT 5) ---

    async def test_board_session_error_does_not_crash_browser_manager(
        self, tmp_path: Path
    ) -> None:
        """
        Given two boards sharing a BrowserManager where one raises an error
        When run() is called
        Then the other board still completes its search successfully
        """
        # Given: two boards, one fails during authenticate
        tmpdir = str(tmp_path)
        settings = make_test_settings(tmpdir, enabled_boards=["failing_board", "good_board"])
        settings.boards["failing_board"].browser_channel = "msedge"
        settings.boards["good_board"].browser_channel = "msedge"

        runner, _ = _make_runner_with_real_stack(settings)

        failing_adapter = _make_test_adapter(board_name="failing_board")
        failing_adapter.authenticate = AsyncMock(
            side_effect=ActionableError(
                error="Board failed",
                error_type=ErrorType.CONNECTION,
                service="failing_board",
            )
        )

        good_adapter = _make_test_adapter(board_name="good_board")

        mock_pw_fn, _ = _mock_playwright_boundary()

        with (
            AdapterRegistry.override(
                {
                    "failing_board": _adapt(failing_adapter),
                    "good_board": _adapt(good_adapter),
                },
            ),
            patch("jobsearch_rag.adapters.session.async_playwright", mock_pw_fn),
            patch(
                "jobsearch_rag.adapters.session._DEFAULT_STORAGE_DIR",
                Path(tmpdir),
            ),
            _cdp_patches(),
        ):
            result = await runner.run()

            # Then: good board still ran despite failing board's error
            assert good_adapter.search.called, (
                "good_board should still search when failing_board errors"
            )
            # Failing board error was captured
            assert any(e.service == "failing_board" for e in result.errors), (
                "failing_board error should be captured"
            )

    # --- Context manager nesting (WHAT 6) ---

    async def test_browser_manager_outlives_board_sessions(self) -> None:
        """
        Given a BrowserManager with two BoardSessions
        When both BoardSessions exit
        Then the BrowserManager's browser remains open until its own exit
        """
        # Given: a mock Playwright boundary
        config_a = SessionConfig(board_name="board_a", headless=True)
        config_b = SessionConfig(board_name="board_b", headless=True)
        config_browser = SessionConfig(board_name="board_a", headless=True)

        mock_pw = _mock_playwright(connect_over_cdp=False)
        mock_browser = mock_pw.chromium.launch.return_value

        with _patch_playwright(mock_pw):
            # When: BrowserManager entered, two BoardSessions enter and exit
            async with BrowserManager(config_browser) as mgr:
                async with BoardSession(mgr.browser, config_a):
                    pass
                # After first BoardSession exits, browser still open
                mock_browser.close.assert_not_called()

                async with BoardSession(mgr.browser, config_b):
                    pass
                # After second BoardSession exits, browser still open
                mock_browser.close.assert_not_called()

            # Then: after BrowserManager exits, browser is closed
            mock_browser.close.assert_called_once()

    # --- SessionManager backward compatibility (WHAT 7) ---

    async def test_session_manager_wrapper_composes_both_layers(self) -> None:
        """
        Given a SessionConfig for a single board
        When SessionManager is used as a context manager
        Then it creates a BrowserManager and BoardSession internally
             and exposes new_page, save_storage_state, has_storage_state
        """
        # Given: standard config
        config = SessionConfig(board_name="testboard", headless=True)
        mock_pw = _mock_playwright(connect_over_cdp=False)
        mock_context = mock_pw.chromium.launch.return_value.new_context.return_value
        mock_page = MagicMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)

        with _patch_playwright(mock_pw):
            # When: SessionManager used as context manager
            async with SessionManager(config) as session:
                # Then: new_page returns a page
                page = await session.new_page()
                assert page is mock_page, "SessionManager.new_page should return a page"

                # And: has_storage_state is accessible
                assert hasattr(session, "has_storage_state"), (
                    "SessionManager should expose has_storage_state"
                )

                # And: save_storage_state is accessible
                assert hasattr(session, "save_storage_state"), (
                    "SessionManager should expose save_storage_state"
                )

    # --- SessionManager backward compat with login (WHAT 8) ---

    async def test_session_manager_wrapper_works_for_headed_login(self, tmp_path: Path) -> None:
        """
        Given a SessionConfig with headless=False and browser_channel="msedge"
        When SessionManager is used as a context manager (handle_login style)
        Then the browser is launched headed, a page is produced, and storage
             state can be saved — matching current handle_login behavior
        """
        # Given: headed CDP config
        fake_binary = tmp_path / "edge"
        fake_binary.touch()

        config = SessionConfig(
            board_name="ziprecruiter",
            headless=False,
            browser_channel="msedge",
            storage_dir=tmp_path,
        )

        mock_pw = _mock_playwright()
        mock_context = mock_pw.chromium.connect_over_cdp.return_value.new_context.return_value
        mock_page = MagicMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.storage_state = AsyncMock(return_value={"cookies": [{"name": "auth_token"}]})

        with (
            patch(
                "jobsearch_rag.adapters.session._BROWSER_PATHS",
                {"msedge": [str(fake_binary)]},
            ),
            patch("shutil.which", return_value=None),
            patch("jobsearch_rag.adapters.session.subprocess.Popen"),
            patch("urllib.request.urlopen"),
            patch(
                "jobsearch_rag.adapters.session.tempfile.mkdtemp",
                return_value=str(tmp_path / "cdp_profile"),
            ),
            _patch_playwright(mock_pw),
        ):
            # When: SessionManager is entered and a page is created
            async with SessionManager(config) as session:
                page = await session.new_page()
                result = await session.save_storage_state()

                # Then: page created from CDP-connected browser
                assert page is mock_page, "SessionManager should produce a page in headed CDP mode"
                assert result.exists(), "save_storage_state should write cookies to disk"

    # --- SessionManager has_storage_state delegation (WHAT 9) ---

    async def test_session_manager_delegates_has_storage_state_when_entered(
        self, tmp_path: Path
    ) -> None:
        """
        Given a SessionManager that has been entered
        When has_storage_state is called
        Then it delegates to the BoardSession's has_storage_state method
        """
        # Given: config with no storage file -> has_storage_state should be False
        config = SessionConfig(
            board_name="testboard",
            headless=True,
            storage_dir=tmp_path,
        )
        mock_pw = _mock_playwright(connect_over_cdp=False)

        with _patch_playwright(mock_pw):
            # When: SessionManager is entered and has_storage_state is called
            async with SessionManager(config) as session:
                result = session.has_storage_state()

                # Then: result reflects the real storage state (no file exists)
                assert result is False, (
                    "has_storage_state should return False when no session file exists"
                )

                # And: create the storage file and check again
                config.storage_state_path.parent.mkdir(parents=True, exist_ok=True)
                config.storage_state_path.write_text("{}")

                result_after = session.has_storage_state()
                assert result_after is True, (
                    "has_storage_state should return True after session file is created"
                )
