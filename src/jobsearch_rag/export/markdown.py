"""Markdown table export."""

from __future__ import annotations

from jobsearch_rag.pipeline.ranker import RankedListing


class MarkdownExporter:
    """Renders ranked listings as a human-readable Markdown report."""

    def export(self, listings: list[RankedListing], output_path: str) -> None:
        """Write a Markdown file with run summary and ranked listing table."""
        raise NotImplementedError
