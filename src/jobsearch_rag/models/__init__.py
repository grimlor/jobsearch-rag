"""Domain dataclasses shared across port boundaries."""

from __future__ import annotations

from jobsearch_rag.models.inference import InferenceMetrics
from jobsearch_rag.models.store import DocumentRecord, QueryMatch, QueryResult

__all__ = [
    "DocumentRecord",
    "InferenceMetrics",
    "QueryMatch",
    "QueryResult",
]
