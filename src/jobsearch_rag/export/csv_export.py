"""CSV export."""

from __future__ import annotations

from jobsearch_rag.pipeline.ranker import RankedListing


class CSVExporter:
    """Renders ranked listings as a CSV file suitable for spreadsheet import."""

    def export(self, listings: list[RankedListing], output_path: str) -> None:
        """Write a CSV with header row. Full JD text is excluded."""
        raise NotImplementedError
