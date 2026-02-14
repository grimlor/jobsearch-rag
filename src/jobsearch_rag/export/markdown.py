"""Markdown table export."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jobsearch_rag.pipeline.ranker import RankedListing


class MarkdownExporter:
    """Renders ranked listings as a human-readable Markdown report."""

    def export(self, listings: list[RankedListing], output_path: str) -> None:
        """Write a Markdown file with run summary and ranked listing table."""
        raise NotImplementedError
