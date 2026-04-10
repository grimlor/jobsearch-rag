"""
Playwright session manager with storage_state persistence and throttling.

Two-layer architecture:

**BrowserManager** — owns the Playwright instance and browser process.
One per ``browser_channel`` per run.  Shared across multiple boards.

**BoardSession** — owns a ``BrowserContext`` over a shared ``Browser``.
One per board.  Provides cookie isolation, stealth patches, and page
management without launching another browser process.

**SessionManager** — backward-compatible convenience wrapper that
composes ``BrowserManager`` + ``BoardSession`` for single-board callers.

When ``browser_channel`` is set (e.g. ``msedge``), the browser manager
launches the real system browser as a subprocess and connects to it via
the Chrome DevTools Protocol (CDP).  This avoids Playwright's automation
flags (``--enable-automation``, ``navigator.webdriver``) that Cloudflare
bot detection uses to block automated browsers.
"""

from __future__ import annotations

import asyncio
import json
import random
import shutil
import signal
import socket
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from playwright.async_api import async_playwright

from jobsearch_rag.errors import ActionableError
from jobsearch_rag.logging import logger

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page, Playwright

    from jobsearch_rag.adapters.base import JobBoardAdapter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_STORAGE_DIR = Path("data")


# Known browser binary paths on macOS
_BROWSER_PATHS: dict[str, list[str]] = {
    "msedge": [
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ],
    "chrome": [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    ],
    "chromium": [
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ],
}


@dataclass
class SessionConfig:
    """Per-board browser session configuration."""

    board_name: str
    headless: bool = True
    user_agent: str | None = None
    viewport_width: int = 1440
    viewport_height: int = 900
    stealth: bool = False
    overnight: bool = False
    browser_channel: str | None = None
    storage_dir: Path = _DEFAULT_STORAGE_DIR
    browser_paths: dict[str, list[str]] | None = None
    cdp_timeout: float | None = None

    @property
    def storage_state_path(self) -> Path:
        """Path to the cookie/session JSON for this board."""
        return self.storage_dir / f"{self.board_name}_session.json"


# ---------------------------------------------------------------------------
# CDP helpers
# ---------------------------------------------------------------------------


def _find_browser_binary(
    channel: str, config_paths: dict[str, list[str]] | None = None
) -> str | None:
    """
    Resolve a browser channel name to an executable path.

    When *config_paths* is provided, only those paths are checked —
    neither ``_BROWSER_PATHS`` nor ``shutil.which()`` are consulted.
    When *config_paths* is ``None``, falls back to the module-level
    ``_BROWSER_PATHS`` dict and then ``shutil.which()``.
    """
    if config_paths is not None:
        for path in config_paths.get(channel, []):
            if Path(path).exists():
                return path
        return None

    for path in _BROWSER_PATHS.get(channel, []):
        if Path(path).exists():
            return path

    # Fallback: try finding the binary on $PATH
    which = shutil.which(channel)
    if which:
        return which

    return None


def _find_free_port() -> int:
    """Ask the OS for a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


async def _wait_for_cdp(
    cdp_url: str,
    *,
    timeout: float | None = None,
    poll_interval: float = 0.3,
) -> None:
    """Poll until the CDP ``/json/version`` endpoint responds."""
    if timeout is None:
        timeout = 15.0
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        try:
            urllib.request.urlopen(f"{cdp_url}/json/version", timeout=2)
            logger.debug("CDP endpoint ready at %s", cdp_url)
            return
        except OSError:
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError(
                    f"CDP endpoint at {cdp_url} did not start within {timeout}s"
                ) from None
            await asyncio.sleep(poll_interval)


def _terminate_process(proc: subprocess.Popen[bytes]) -> None:
    """Gracefully stop the CDP browser subprocess."""
    if proc.poll() is not None:
        return  # already exited
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        logger.warning("Browser subprocess did not exit cleanly — killing")
        proc.kill()
        proc.wait(timeout=3)


# ---------------------------------------------------------------------------
# Throttle
# ---------------------------------------------------------------------------


async def throttle(
    adapter: JobBoardAdapter,
    rate_limit_range: tuple[float, float] | None = None,
) -> float:
    """
    Sleep for a random duration within the given rate limit range.

    If *rate_limit_range* is ``None``, falls back to the adapter's
    ``rate_limit_seconds`` property for backward compatibility.

    Returns the actual duration slept (useful for assertions).
    """
    lo, hi = rate_limit_range if rate_limit_range is not None else adapter.rate_limit_seconds
    duration = random.uniform(lo, hi)
    logger.debug("Throttle: sleeping %.2fs for %s", duration, adapter.board_name)
    await asyncio.sleep(duration)
    return duration


# ---------------------------------------------------------------------------
# Browser manager — owns Playwright instance and browser process
# ---------------------------------------------------------------------------


class BrowserManager:
    """
    Owns the Playwright instance and browser process lifecycle.

    Shared across multiple ``BoardSession`` instances that use the same
    ``browser_channel``.

    Two launch modes:

    **Playwright-managed** (when ``browser_channel`` is ``None``):
      ``playwright.chromium.launch()`` — Playwright starts its bundled
      Chromium.

    **CDP mode** (when ``browser_channel`` is set, e.g. ``"msedge"``):
      The real system browser is launched as a subprocess with
      ``--remote-debugging-port``.  Playwright then connects via
      ``connect_over_cdp()``.
    """

    def __init__(self, config: SessionConfig) -> None:
        """Initialize with browser session configuration."""
        self._config = config
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._cdp_process: subprocess.Popen[bytes] | None = None
        self._cdp_tmpdir: str | None = None

    @property
    def browser(self) -> Browser:
        """The Playwright Browser object (available after ``__aenter__``)."""
        if self._browser is None:
            raise ActionableError.config(
                field_name="BrowserManager",
                reason="Browser accessed before entering the context manager",
                suggestion="Use 'async with BrowserManager(config) as mgr' before accessing .browser",
            )
        return self._browser

    async def __aenter__(self) -> BrowserManager:
        """Start Playwright and launch (or connect to) the browser."""
        self._playwright = await async_playwright().start()

        if self._config.browser_channel:
            await self._launch_cdp()
        else:
            await self._launch_playwright()

        return self

    async def _launch_playwright(self) -> None:
        """Launch browser via Playwright (standard mode)."""
        assert self._playwright is not None
        self._browser = await self._playwright.chromium.launch(
            headless=self._config.headless,
        )

    async def _launch_cdp(self) -> None:
        """Launch system browser as subprocess and connect via CDP."""
        assert self._playwright is not None
        channel = self._config.browser_channel or "msedge"

        binary = _find_browser_binary(channel, self._config.browser_paths)
        if not binary:
            raise ActionableError.config(
                field_name="browser_channel",
                reason=f"Could not find '{channel}' browser binary",
                suggestion=(
                    f"Install {channel} or set browser_channel to an installed browser "
                    "(msedge, chrome, chromium)"
                ),
            )

        port = _find_free_port()

        self._cdp_tmpdir = tempfile.mkdtemp(prefix=f"jobsearch-{channel}-")

        cmd = [
            binary,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={self._cdp_tmpdir}",
            "--no-first-run",
            "--no-default-browser-check",
        ]

        if self._config.headless:
            cmd.append("--headless=new")

        cmd.append("about:blank")

        logger.info(
            "Launching %s via CDP on port %d for %s",
            channel,
            port,
            self._config.board_name,
        )
        self._cdp_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        cdp_url = f"http://localhost:{port}"
        await _wait_for_cdp(cdp_url, timeout=self._config.cdp_timeout)

        self._browser = await self._playwright.chromium.connect_over_cdp(cdp_url)

        for context in self._browser.contexts:
            for page in context.pages:
                await page.close()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Close browser, Playwright, and clean up CDP resources."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

        if self._cdp_process:
            _terminate_process(self._cdp_process)
            self._cdp_process = None
        if self._cdp_tmpdir:
            shutil.rmtree(self._cdp_tmpdir, ignore_errors=True)
            self._cdp_tmpdir = None


# ---------------------------------------------------------------------------
# Board session — per-board context over a shared browser
# ---------------------------------------------------------------------------


class BoardSession:
    """
    Per-board browser context over a shared ``Browser``.

    Provides cookie isolation, stealth patches, and page management.

    Usage::

        async with BrowserManager(config) as mgr:
            async with BoardSession(mgr.browser, board_config) as session:
                page = await session.new_page()
                ...
                await session.save_storage_state()
    """

    def __init__(self, browser: Browser, config: SessionConfig) -> None:
        """Initialize with a shared browser and per-board configuration."""
        self._browser = browser
        self._config = config
        self._context: BrowserContext | None = None

    async def __aenter__(self) -> BoardSession:
        """Create a browser context with optional cookie loading and stealth."""
        storage_path = self._config.storage_state_path
        storage_state: str | None = None
        if storage_path.exists():
            logger.info("Loading session state from %s", storage_path)
            storage_state = str(storage_path)
        else:
            logger.info(
                "No existing session state for %s — fresh session",
                self._config.board_name,
            )

        self._context = await self._browser.new_context(
            viewport={
                "width": self._config.viewport_width,
                "height": self._config.viewport_height,
            },
            user_agent=self._config.user_agent if self._config.user_agent else None,
            storage_state=storage_state,
        )

        if self._config.stealth:
            try:
                from playwright_stealth import Stealth  # pyright: ignore[reportMissingTypeStubs] # optional dependency, no stubs available  # noqa: I001, PLC0415

                await Stealth().apply_stealth_async(self._context)
                logger.info("Stealth patches applied for %s", self._config.board_name)
            except ImportError:
                logger.warning("playwright-stealth not installed — stealth mode unavailable")

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Close the browser context (the browser itself remains open)."""
        if self._context:
            await self._context.close()

    async def new_page(self) -> Page:
        """Create a new page in this board's browser context."""
        if self._context is None:
            raise ActionableError.config(
                field_name="BoardSession",
                reason="new_page() called before entering the context manager",
                suggestion="Use 'async with BoardSession(browser, config) as session' before calling .new_page()",
            )
        return await self._context.new_page()

    async def save_storage_state(self) -> Path:
        """Persist cookies/session to disk for reuse on next run."""
        if self._context is None:
            raise ActionableError.config(
                field_name="BoardSession",
                reason="save_storage_state() called before entering the context manager",
                suggestion="Use 'async with BoardSession(browser, config) as session' before calling .save_storage_state()",
            )

        path = self._config.storage_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        state = await self._context.storage_state()
        path.write_text(json.dumps(state, indent=2))
        logger.info("Session state saved to %s", path)
        return path

    def has_storage_state(self) -> bool:
        """Check whether a persisted session exists for this board."""
        return self._config.storage_state_path.exists()


# ---------------------------------------------------------------------------
# Session manager — backward-compatible wrapper
# ---------------------------------------------------------------------------


class SessionManager:
    """
    Backward-compatible wrapper composing BrowserManager + BoardSession.

    For single-board callers (e.g. ``handle_login``) that want the
    original one-liner interface::

        async with SessionManager(config) as session:
            page = await session.new_page()
            await session.save_storage_state()

    For multi-board callers, use ``BrowserManager`` + ``BoardSession``
    directly to share the browser across boards.
    """

    def __init__(self, config: SessionConfig) -> None:
        """Initialize with browser session configuration."""
        self.config = config
        self._browser_manager: BrowserManager | None = None
        self._board_session: BoardSession | None = None

    async def __aenter__(self) -> SessionManager:
        """Start browser (via BrowserManager) and board context (via BoardSession)."""
        self._browser_manager = BrowserManager(self.config)
        await self._browser_manager.__aenter__()

        self._board_session = BoardSession(self._browser_manager.browser, self.config)
        await self._board_session.__aenter__()

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Close board context, then browser."""
        if self._board_session:
            await self._board_session.__aexit__(exc_type, exc_val, exc_tb)
        if self._browser_manager:
            await self._browser_manager.__aexit__(exc_type, exc_val, exc_tb)

    async def new_page(self) -> Page:
        """Create a new page in the managed browser context."""
        if self._board_session is None:
            raise ActionableError.config(
                field_name="SessionManager",
                reason="new_page() called before entering the context manager",
                suggestion="Use 'async with SessionManager(config) as session' before calling .new_page()",
            )
        return await self._board_session.new_page()

    async def save_storage_state(self) -> Path:
        """Persist cookies/session to disk for reuse on next run."""
        if self._board_session is None:
            raise ActionableError.config(
                field_name="SessionManager",
                reason="save_storage_state() called before entering the context manager",
                suggestion="Use 'async with SessionManager(config) as session' before calling .save_storage_state()",
            )
        return await self._board_session.save_storage_state()

    def has_storage_state(self) -> bool:
        """Check whether a persisted session exists for this board."""
        if self._board_session is None:
            # Allow checking before entering — use config directly
            return self.config.storage_state_path.exists()
        return self._board_session.has_storage_state()
