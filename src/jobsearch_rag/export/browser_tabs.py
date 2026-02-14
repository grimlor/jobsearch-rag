"""Browser tab opener for top results."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jobsearch_rag.pipeline.ranker import RankedListing


class BrowserTabOpener:
    """Opens the top-N ranked listing URLs in the default browser."""

    def open(self, listings: list[RankedListing], top_n: int = 5) -> None:
        """Open the first *top_n* listing URLs in browser tabs."""
        raise NotImplementedError
