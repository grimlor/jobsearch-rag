"""Playwright session manager with storage_state persistence and throttling.

Owns browser lifecycle, cookie persistence, and human-like rate limiting.
Adapters receive a ``Page`` — they never launch browsers themselves.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from jobsearch_rag.logging import logger

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page, Playwright

    from jobsearch_rag.adapters.base import JobBoardAdapter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_STORAGE_DIR = Path("data")


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

    @property
    def storage_state_path(self) -> Path:
        """Path to the cookie/session JSON for this board."""
        return _STORAGE_DIR / f"{self.board_name}_session.json"


# ---------------------------------------------------------------------------
# Throttle
# ---------------------------------------------------------------------------


async def throttle(adapter: JobBoardAdapter) -> float:
    """Sleep for a random duration within the adapter's rate limit range.

    Returns the actual duration slept (useful for assertions).
    """
    lo, hi = adapter.rate_limit_seconds
    duration = random.uniform(lo, hi)
    logger.debug("Throttle: sleeping %.2fs for %s", duration, adapter.board_name)
    await asyncio.sleep(duration)
    return duration


# ---------------------------------------------------------------------------
# Session manager
# ---------------------------------------------------------------------------


class SessionManager:
    """Manages Playwright browser sessions with storage_state persistence.

    Usage::

        async with SessionManager(config) as session:
            page = await session.new_page()
            await adapter.authenticate(page)
            await session.save_storage_state()
            results = await adapter.search(page, query)
    """

    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    async def __aenter__(self) -> SessionManager:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless,
        )

        # Load existing session cookies if available
        storage_path = self.config.storage_state_path
        storage_state: str | None = None
        if storage_path.exists():
            logger.info("Loading session state from %s", storage_path)
            storage_state = str(storage_path)
        else:
            logger.info(
                "No existing session state for %s — fresh session",
                self.config.board_name,
            )

        self._context = await self._browser.new_context(
            viewport={"width": self.config.viewport_width, "height": self.config.viewport_height},
            user_agent=self.config.user_agent if self.config.user_agent else None,
            storage_state=storage_state,
        )

        # Apply stealth patches if requested (LinkedIn)
        if self.config.stealth:
            try:
                from playwright_stealth import Stealth

                await Stealth().apply_stealth_async(self._context)
                logger.info("Stealth patches applied for %s", self.config.board_name)
            except ImportError:
                logger.warning("playwright-stealth not installed — stealth mode unavailable")

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def new_page(self) -> Page:
        """Create a new page in the managed browser context."""
        if self._context is None:
            msg = "SessionManager not entered — use 'async with'"
            raise RuntimeError(msg)
        return await self._context.new_page()

    async def save_storage_state(self) -> Path:
        """Persist cookies/session to disk for reuse on next run."""
        if self._context is None:
            msg = "SessionManager not entered — use 'async with'"
            raise RuntimeError(msg)

        path = self.config.storage_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        state = await self._context.storage_state()
        path.write_text(json.dumps(state, indent=2))
        logger.info("Session state saved to %s", path)
        return path

    def has_storage_state(self) -> bool:
        """Check whether a persisted session exists for this board."""
        return self.config.storage_state_path.exists()



