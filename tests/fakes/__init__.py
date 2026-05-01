"""Fake implementations of port protocols for testing."""

from __future__ import annotations

from tests.fakes.embedder import FakeEmbedder
from tests.fakes.store import FakeVectorStore

__all__ = ["FakeEmbedder", "FakeVectorStore"]
