"""ZipRecruiter adapter — auth, search pagination, JD extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from jobsearch_rag.adapters.base import JobBoardAdapter, JobListing
from jobsearch_rag.adapters.registry import AdapterRegistry

if TYPE_CHECKING:
    from playwright.async_api import Page


@AdapterRegistry.register
class ZipRecruiterAdapter(JobBoardAdapter):
    """Browser automation adapter for ZipRecruiter."""

    @property
    def board_name(self) -> str:
        return "ziprecruiter"

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
