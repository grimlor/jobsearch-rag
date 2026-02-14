"""WeWorkRemotely adapter — auth, search pagination, JD extraction."""

from __future__ import annotations

from playwright.async_api import Page

from jobsearch_rag.adapters.base import JobBoardAdapter, JobListing
from jobsearch_rag.adapters.registry import AdapterRegistry


@AdapterRegistry.register
class WeWorkRemotelyAdapter(JobBoardAdapter):
    """Browser automation adapter for WeWorkRemotely."""

    @property
    def board_name(self) -> str:
        return "weworkremotely"

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
