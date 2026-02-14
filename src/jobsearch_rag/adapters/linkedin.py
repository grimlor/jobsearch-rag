"""LinkedIn adapter — overnight mode, stealth, throttled."""

from __future__ import annotations

from playwright.async_api import Page

from jobsearch_rag.adapters.base import JobBoardAdapter, JobListing
from jobsearch_rag.adapters.registry import AdapterRegistry


@AdapterRegistry.register
class LinkedInAdapter(JobBoardAdapter):
    """Browser automation adapter for LinkedIn.

    Requires headed mode and ``playwright-stealth`` fingerprint patches.
    Run via ``--board linkedin --overnight`` for safe, throttled operation.
    """

    @property
    def board_name(self) -> str:
        return "linkedin"

    @property
    def rate_limit_seconds(self) -> tuple[float, float]:
        return (8.0, 20.0)

    async def authenticate(self, page: Page) -> None:
        raise NotImplementedError

    async def search(
        self,
        page: Page,
        query: str,
        max_pages: int = 3,
    ) -> list[JobListing]:
        raise NotImplementedError

    async def extract_detail(
        self,
        page: Page,
        listing: JobListing,
    ) -> JobListing:
        raise NotImplementedError
