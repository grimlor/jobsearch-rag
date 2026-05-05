"""Domain dataclasses for vector storage operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """A document with its embedding and optional metadata."""

    id: str
    document: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class QueryMatch:
    """A single similarity-query result with its distance score."""

    id: str
    document: str
    distance: float
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Aggregated results from a similarity query."""

    matches: list[QueryMatch]
