"""ChromaDB collection management."""

from __future__ import annotations


class VectorStore:
    """Manages ChromaDB collections for resume, archetypes, and decisions."""

    def __init__(self, persist_dir: str) -> None:
        self.persist_dir = persist_dir

    def get_or_create_collection(self, name: str) -> object:
        """Return the named ChromaDB collection, creating if necessary."""
        raise NotImplementedError

    def collection_count(self, name: str) -> int:
        """Return the document count for collection *name*."""
        raise NotImplementedError

    def reset_collection(self, name: str) -> None:
        """Drop and recreate the named collection."""
        raise NotImplementedError
