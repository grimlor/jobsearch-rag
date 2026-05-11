"""Protocol defining the structural interface for job board adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from playwright.async_api import Page

    from jobsearch_rag.adapters.base import JobListing


@runtime_checkable
class JobBoardPort(Protocol):
    """
    Structural interface for job board integration.

    Callers depend on this Protocol rather than the concrete
    ``JobBoardAdapter`` ABC.  Any object providing these members
    satisfies the contract — inheritance is not required.
    """

    @property
    def board_name(self) -> str:
        """Unique identifier string for this board."""
        ...

    @property
    def rate_limit_seconds(self) -> tuple[float, float]:
        """(min, max) seconds to sleep between page loads."""
        ...

    async def authenticate(self, page: Page) -> None:
        """Establish an authenticated session."""
        ...

    async def search(
        self,
        page: Page,
        query: str,
        max_pages: int,
    ) -> list[JobListing]:
        """Navigate search results and return shallow listings."""
        ...

    async def extract_detail(
        self,
        page: Page,
        listing: JobListing,
    ) -> JobListing:
        """Navigate to listing URL and populate full_text."""
        ...
