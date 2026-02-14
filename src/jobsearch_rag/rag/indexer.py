"""Resume and archetype ingestion pipeline."""

from __future__ import annotations


class Indexer:
    """Chunks and indexes resume and role archetypes into ChromaDB."""

    async def index_resume(self, resume_path: str) -> int:
        """Chunk resume by section heading and index into the ``resume`` collection.

        Re-indexing replaces previous content (not appends).
        Returns the number of chunks indexed.
        """
        raise NotImplementedError

    async def index_archetypes(self, archetypes_path: str) -> int:
        """Load archetypes from TOML and index into the ``role_archetypes`` collection.

        Each archetype produces one ChromaDB document.
        Returns the number of archetypes indexed.
        """
        raise NotImplementedError
