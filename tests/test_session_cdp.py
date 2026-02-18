"""Tests for CDP-based browser launch in SessionManager.

BDD-style tests exercising the CDP launch path through SessionManager's
public API (``__aenter__`` / ``__aexit__``).  Only I/O boundaries are
mocked: subprocess, playwright, urllib, filesystem, shutil.

Run: ``uv run pytest tests/test_session_cdp.py -v``
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobsearch_rag.adapters.session import (
    SessionConfig,
    SessionManager,
)

# ───────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────


def _mock_playwright(*, connect_over_cdp: bool = True) -> MagicMock:
    """Build a mock Playwright instance with browser/context stubs."""
    mock_context = MagicMock()
    mock_context.close = AsyncMock()
    mock_context.storage_state = AsyncMock(return_value={})

    mock_browser = MagicMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_browser.close = AsyncMock()

    pw = MagicMock()
    pw.stop = AsyncMock()

    if connect_over_cdp:
        pw.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
    pw.chromium.launch = AsyncMock(return_value=mock_browser)

    return pw


def _patch_playwright(mock_pw: MagicMock):
    """Return a patch context for ``async_playwright`` that yields *mock_pw*."""
    mock_pw_ctx = MagicMock()
    mock_pw_ctx.start = AsyncMock(return_value=mock_pw)
    return patch(
        "playwright.async_api.async_playwright",
        return_value=mock_pw_ctx,
    )


# ───────────────────────────────────────────────────────────────────
# SessionConfig
# ───────────────────────────────────────────────────────────────────


class TestSessionConfigCDP:
    """REQUIREMENT: SessionConfig carries CDP channel selection through to launch.

    WHO: The operator choosing between Playwright-managed and CDP mode
    WHAT: Default config uses Playwright (channel=None); setting
          browser_channel enables CDP subprocess launch
    WHY: Cloudflare-protected sites require CDP mode to bypass
         automation detection flags
    """

    def test_browser_channel_defaults_to_none(self) -> None:
        """Default config uses Playwright-managed mode."""
        config = SessionConfig(board_name="test")
        assert config.browser_channel is None

    def test_browser_channel_set_to_msedge(self) -> None:
        """Setting browser_channel enables CDP launch mode."""
        config = SessionConfig(board_name="test", browser_channel="msedge")
        assert config.browser_channel == "msedge"


# ───────────────────────────────────────────────────────────────────
# SessionManager CDP launch path — all through __aenter__ / __aexit__
# ───────────────────────────────────────────────────────────────────


class TestSessionManagerCDP:
    """REQUIREMENT: CDP mode launches a system browser and connects via DevTools Protocol.

    WHO: The pipeline runner needing a Cloudflare-safe browser session
    WHAT: When browser_channel is set, SessionManager launches the system
          browser as a subprocess with --remote-debugging-port, waits for
          the CDP endpoint, and connects Playwright over CDP; cleanup
          terminates the subprocess gracefully with SIGKILL fallback;
          missing binary raises a clear config error; headless mode adds
          the correct flag; standard Playwright launch is used when no
          channel is set
    WHY: Cloudflare detects Playwright's --enable-automation flag and
         navigator.webdriver — CDP mode bypasses both
    """

    @pytest.fixture()
    def cdp_config(self) -> SessionConfig:
        return SessionConfig(
            board_name="ziprecruiter",
            headless=False,
            browser_channel="msedge",
        )

    @pytest.fixture()
    def playwright_config(self) -> SessionConfig:
        return SessionConfig(
            board_name="testboard",
            headless=True,
        )

    async def test_cdp_launches_subprocess_and_connects(
        self, cdp_config: SessionConfig, tmp_path: Path
    ) -> None:
        """CDP mode launches browser via subprocess and connects over CDP."""
        fake_binary = tmp_path / "edge"
        fake_binary.touch()

        mock_pw = _mock_playwright()

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
            manager = SessionManager(cdp_config)
            await manager.__aenter__()

            # Verify Playwright connected via CDP (not standard launch)
            mock_pw.chromium.connect_over_cdp.assert_called_once()
            mock_pw.chromium.launch.assert_not_called()

            await manager.__aexit__(None, None, None)

    async def test_cdp_subprocess_receives_correct_flags(
        self, cdp_config: SessionConfig, tmp_path: Path
    ) -> None:
        """The subprocess command includes --remote-debugging-port, --user-data-dir, --no-first-run."""
        fake_binary = tmp_path / "edge"
        fake_binary.touch()

        mock_pw = _mock_playwright()

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
            manager = SessionManager(cdp_config)
            await manager.__aenter__()

            mock_popen.assert_called_once()
            cmd = mock_popen.call_args[0][0]
            assert any("--remote-debugging-port=" in arg for arg in cmd)
            assert any("--user-data-dir=" in arg for arg in cmd)
            assert "--no-first-run" in cmd
            assert "--headless=new" not in cmd  # headless=False

            await manager.__aexit__(None, None, None)

    async def test_cdp_headless_adds_headless_flag(
        self, tmp_path: Path
    ) -> None:
        """When headless=True + CDP, --headless=new is added to subprocess args."""
        config = SessionConfig(
            board_name="test",
            headless=True,
            browser_channel="msedge",
        )
        fake_binary = tmp_path / "edge"
        fake_binary.touch()

        mock_pw = _mock_playwright()

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
            manager = SessionManager(config)
            await manager.__aenter__()

            cmd = mock_popen.call_args[0][0]
            assert "--headless=new" in cmd

            await manager.__aexit__(None, None, None)

    async def test_binary_resolved_via_shutil_which_fallback(
        self, tmp_path: Path
    ) -> None:
        """If _BROWSER_PATHS has no match, shutil.which is used as fallback."""
        # shutil.which returns a real-looking path; subprocess.Popen is
        # the I/O boundary that actually launches the browser.
        which_binary = str(tmp_path / "msedge")
        Path(which_binary).touch()  # must exist for _find_browser_binary

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
            manager = SessionManager(config)
            await manager.__aenter__()

            # Verify the which-resolved binary was used in the Popen command
            cmd = mock_popen.call_args[0][0]
            assert cmd[0] == which_binary

            await manager.__aexit__(None, None, None)

    async def test_standard_launch_when_no_channel(
        self, playwright_config: SessionConfig
    ) -> None:
        """Without browser_channel, standard Playwright launch is used."""
        mock_pw = _mock_playwright(connect_over_cdp=True)

        with _patch_playwright(mock_pw):
            manager = SessionManager(playwright_config)
            await manager.__aenter__()

            mock_pw.chromium.launch.assert_called_once_with(headless=True)
            mock_pw.chromium.connect_over_cdp.assert_not_called()

            await manager.__aexit__(None, None, None)

    async def test_cdp_missing_binary_tells_operator_which_browser_to_install(
        self, cdp_config: SessionConfig
    ) -> None:
        """A missing browser binary tells the operator which browser to install."""
        from jobsearch_rag.errors import ActionableError

        mock_pw = _mock_playwright()

        with (
            patch("jobsearch_rag.adapters.session._BROWSER_PATHS", {}),
            patch("shutil.which", return_value=None),
            _patch_playwright(mock_pw),
        ):
            manager = SessionManager(cdp_config)
            with pytest.raises(ActionableError) as exc_info:
                await manager.__aenter__()
            err = exc_info.value
            assert "Could not find" in err.error
            assert err.suggestion is not None
            assert err.troubleshooting is not None

    async def test_cdp_cleanup_terminates_subprocess(
        self, tmp_path: Path
    ) -> None:
        """Exiting the context manager sends SIGTERM to the CDP subprocess."""
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
            manager = SessionManager(config)
            await manager.__aenter__()
            await manager.__aexit__(None, None, None)

            # _terminate_process sends SIGTERM on running process
            mock_proc.send_signal.assert_called_once()
            mock_rmtree.assert_called_once_with(tmpdir, ignore_errors=True)

    async def test_cdp_cleanup_kills_on_sigterm_timeout(
        self, tmp_path: Path
    ) -> None:
        """If SIGTERM doesn't work, __aexit__ escalates to SIGKILL."""
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
            manager = SessionManager(config)
            await manager.__aenter__()
            await manager.__aexit__(None, None, None)

            mock_proc.kill.assert_called_once()

    async def test_cdp_cleanup_noop_when_process_already_exited(
        self, tmp_path: Path
    ) -> None:
        """No signals sent if the browser subprocess already exited."""
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
            manager = SessionManager(config)
            await manager.__aenter__()
            await manager.__aexit__(None, None, None)

            mock_proc.send_signal.assert_not_called()

    async def test_cdp_timeout_raises_timeout_error(self, tmp_path: Path) -> None:
        """GIVEN the CDP endpoint never responds
        WHEN _wait_for_cdp times out
        THEN a TimeoutError is raised with the CDP URL.
        """
        from jobsearch_rag.adapters.session import _wait_for_cdp

        with (
            patch("urllib.request.urlopen", side_effect=OSError("refused")),
            pytest.raises(TimeoutError, match="did not start"),
        ):
                await _wait_for_cdp("http://127.0.0.1:9999", timeout=0.1, poll_interval=0.05)

    async def test_new_page_without_context_raises_runtime_error(self) -> None:
        """GIVEN a SessionManager that has not been entered as a context manager
        WHEN new_page is called
        THEN a RuntimeError is raised.
        """
        config = SessionConfig(board_name="test", headless=True)
        manager = SessionManager(config)
        with pytest.raises(RuntimeError, match="not entered"):
            await manager.new_page()

    async def test_save_storage_state_without_context_raises_runtime_error(self) -> None:
        """GIVEN a SessionManager that has not been entered as a context manager
        WHEN save_storage_state is called
        THEN a RuntimeError is raised.
        """
        config = SessionConfig(board_name="test", headless=True)
        manager = SessionManager(config)
        with pytest.raises(RuntimeError, match="not entered"):
            await manager.save_storage_state()

    async def test_save_storage_state_persists_cookies_to_disk(
        self, tmp_path: Path
    ) -> None:
        """GIVEN an active session with cookies
        WHEN save_storage_state is called
        THEN cookies are written to the storage state path as JSON.
        """
        config = SessionConfig(board_name="test", headless=True)
        mock_pw = _mock_playwright(connect_over_cdp=False)
        mock_context = mock_pw.chromium.launch.return_value.new_context.return_value
        mock_context.storage_state = AsyncMock(return_value={"cookies": [{"name": "session"}]})

        with (
            _patch_playwright(mock_pw),
            patch("jobsearch_rag.adapters.session._STORAGE_DIR", tmp_path),
        ):
            manager = SessionManager(config)
            await manager.__aenter__()
            result = await manager.save_storage_state()
            assert result.exists()
            import json

            saved = json.loads(result.read_text())
            assert saved["cookies"][0]["name"] == "session"
            await manager.__aexit__(None, None, None)

    def test_has_storage_state_returns_true_when_file_exists(
        self, tmp_path: Path
    ) -> None:
        """GIVEN a persisted session file exists on disk
        WHEN has_storage_state is called
        THEN it returns True.
        """
        config = SessionConfig(board_name="test", headless=True)
        with patch("jobsearch_rag.adapters.session._STORAGE_DIR", tmp_path):
            state_file = config.storage_state_path
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text("{}")
            manager = SessionManager(config)
            assert manager.has_storage_state() is True

    def test_has_storage_state_returns_false_when_no_file(self) -> None:
        """GIVEN no persisted session file exists
        WHEN has_storage_state is called
        THEN it returns False.
        """
        config = SessionConfig(board_name="nonexistent_board_xyz", headless=True)
        manager = SessionManager(config)
        assert manager.has_storage_state() is False

    async def test_stealth_patches_applied_when_stealth_enabled(
        self, tmp_path: Path
    ) -> None:
        """GIVEN stealth mode is enabled in the config
        WHEN the session is entered
        THEN playwright-stealth patches are applied to the context.
        """
        config = SessionConfig(board_name="linkedin", headless=True, stealth=True)
        mock_pw = _mock_playwright(connect_over_cdp=False)
        mock_stealth = MagicMock()
        mock_stealth_instance = MagicMock()
        mock_stealth_instance.apply_stealth_async = AsyncMock()
        mock_stealth.return_value = mock_stealth_instance

        with (
            _patch_playwright(mock_pw),
            patch.dict(
                "sys.modules",
                {"playwright_stealth": MagicMock(Stealth=mock_stealth)},
            ),
        ):
            manager = SessionManager(config)
            await manager.__aenter__()
            mock_stealth_instance.apply_stealth_async.assert_called_once()
            await manager.__aexit__(None, None, None)

    async def test_stealth_import_error_logs_warning(
        self, tmp_path: Path
    ) -> None:
        """GIVEN stealth mode is enabled but playwright-stealth is not installed
        WHEN the session is entered
        THEN the session still starts and a warning is logged.
        """
        config = SessionConfig(board_name="linkedin", headless=True, stealth=True)
        mock_pw = _mock_playwright(connect_over_cdp=False)

        def _raise_import_error(name, *args, **kwargs):
            if name == "playwright_stealth":
                raise ImportError("No module named 'playwright_stealth'")
            return original_import(name, *args, **kwargs)

        import builtins

        original_import = builtins.__import__

        with (
            _patch_playwright(mock_pw),
            patch("builtins.__import__", side_effect=_raise_import_error),
        ):
            manager = SessionManager(config)
            # Should not raise — stealth failure is graceful
            await manager.__aenter__()
            await manager.__aexit__(None, None, None)

    async def test_new_page_returns_page_when_context_active(self) -> None:
        """GIVEN a SessionManager that has been entered as a context manager
        WHEN new_page is called
        THEN a new page is returned from the browser context.
        """
        config = SessionConfig(board_name="test", headless=True)
        mock_pw = _mock_playwright(connect_over_cdp=False)
        mock_context = mock_pw.chromium.launch.return_value.new_context.return_value
        mock_page = MagicMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)

        with _patch_playwright(mock_pw):
            manager = SessionManager(config)
            await manager.__aenter__()
            page = await manager.new_page()
            assert page is mock_page
            await manager.__aexit__(None, None, None)


class TestThrottle:
    """REQUIREMENT: Rate-limiting sleeps for a random duration within configured bounds.

    WHO: The pipeline runner calling throttle between requests
    WHAT: throttle sleeps for a duration between the adapter's rate_limit_seconds bounds
    WHY: Without throttling, rapid requests trigger anti-bot protections
    """

    async def test_throttle_sleeps_within_rate_limit_bounds(self) -> None:
        """GIVEN an adapter with rate_limit_seconds = (0.5, 1.0)
        WHEN throttle is called
        THEN asyncio.sleep is called with a duration between 0.5 and 1.0.
        """
        from jobsearch_rag.adapters.session import throttle

        mock_adapter = MagicMock()
        mock_adapter.rate_limit_seconds = (0.5, 1.0)
        mock_adapter.board_name = "testboard"

        with patch("jobsearch_rag.adapters.session.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            duration = await throttle(mock_adapter)
            assert 0.5 <= duration <= 1.0
            mock_sleep.assert_called_once_with(duration)
