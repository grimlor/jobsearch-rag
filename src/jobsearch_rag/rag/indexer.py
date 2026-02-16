"""Resume and archetype ingestion pipeline.

The Indexer coordinates :class:`VectorStore` and :class:`Embedder` to
prepare documents for semantic scoring:

1. **Resume indexing** — splits ``resume.md`` on ``##`` headings, embeds
   each section, and stores it in the ``resume`` collection.  Nested
   ``###`` sub-headings stay with their parent section so context is
   preserved.

2. **Archetype indexing** — loads ``role_archetypes.toml``, normalizes
   whitespace in each description, embeds it, and stores one document
   per archetype in the ``role_archetypes`` collection.

Both operations are **idempotent** — re-indexing resets the collection
first, so documents are replaced rather than accumulated.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.logging import logger

if TYPE_CHECKING:
    from jobsearch_rag.rag.embedder import Embedder
    from jobsearch_rag.rag.store import VectorStore

# Regex: match a line starting with ## (but not ###) as a section heading
_SECTION_HEADING_RE = re.compile(r"^## .+", re.MULTILINE)


def _slugify(text: str) -> str:
    """Convert a heading like '## Core Strengths' into 'core-strengths'."""
    text = text.lstrip("#").strip()
    text = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return text.strip("-")


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces, strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def _chunk_resume(content: str) -> list[tuple[str, str, str]]:
    """Split resume markdown into ``(id, heading, body)`` tuples.

    The ``# Title`` block is skipped.  Each ``## Heading`` starts a new
    chunk; ``### Sub-headings`` are included in their parent chunk.
    """
    # Find all ## heading positions
    matches = list(_SECTION_HEADING_RE.finditer(content))
    if not matches:
        return []

    chunks: list[tuple[str, str, str]] = []
    for i, match in enumerate(matches):
        heading = match.group(0)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        slug = _slugify(heading)
        doc_id = f"resume-{slug}"
        chunks.append((doc_id, heading, body))

    return chunks


class Indexer:
    """Chunks and indexes resume and role archetypes into ChromaDB.

    Usage::

        indexer = Indexer(store=vector_store, embedder=embedder)
        n_resume = await indexer.index_resume("data/resume.md")
        n_arch   = await indexer.index_archetypes("config/role_archetypes.toml")
    """

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    # -- Resume ingestion ----------------------------------------------------

    async def index_resume(self, resume_path: str) -> int:
        """Chunk resume by ``##`` headings and index into the ``resume`` collection.

        Re-indexing resets the collection first (replaces, not appends).
        Returns the number of chunks indexed.

        Raises :class:`~jobsearch_rag.errors.ActionableError`:
          - CONFIG if the resume file doesn't exist
        """
        path = Path(resume_path)
        if not path.exists():
            raise ActionableError.config(
                field_name="resume_path",
                reason=f"Resume file not found: {resume_path}",
                suggestion=f"Create {resume_path} or update the path in settings.toml",
            )

        content = path.read_text(encoding="utf-8")
        chunks = _chunk_resume(content)

        # Reset collection for idempotent re-indexing
        self._store.reset_collection("resume")

        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, str]] = []

        for doc_id, heading, body in chunks:
            embedding = await self._embedder.embed(body)
            ids.append(doc_id)
            documents.append(body)
            embeddings.append(embedding)
            metadatas.append({
                "source": "resume",
                "section": heading.lstrip("#").strip(),
            })

        if ids:
            self._store.add_documents(
                collection_name="resume",
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        logger.info("Indexed %d resume chunks from %s", len(ids), resume_path)
        return len(ids)

    # -- Archetype ingestion -------------------------------------------------

    async def index_archetypes(self, archetypes_path: str) -> int:
        """Load archetypes from TOML and index into the ``role_archetypes`` collection.

        Each archetype produces one ChromaDB document with the name stored
        as metadata and the normalized description as the document text.
        Re-indexing resets the collection first.
        Returns the number of archetypes indexed.

        Raises :class:`~jobsearch_rag.errors.ActionableError`:
          - CONFIG if the file doesn't exist
          - PARSE if the TOML is malformed
          - VALIDATION if no archetypes are found
        """
        path = Path(archetypes_path)
        if not path.exists():
            raise ActionableError.config(
                field_name="archetypes_path",
                reason=f"Archetypes file not found: {archetypes_path}",
                suggestion=f"Create {archetypes_path} or update the path in settings.toml",
            )

        raw = path.read_text(encoding="utf-8")

        try:
            data = tomllib.loads(raw)
        except tomllib.TOMLDecodeError as exc:
            raise ActionableError.parse(
                board="role_archetypes",
                selector="TOML syntax",
                raw_error=str(exc),
                suggestion=f"Fix TOML syntax in {archetypes_path}",
            ) from None

        archetypes = data.get("archetypes", [])
        if not archetypes:
            raise ActionableError(
                error=f"No archetypes found in {archetypes_path}",
                error_type=ErrorType.VALIDATION,
                service="role_archetypes",
                suggestion="Add at least one [[archetypes]] entry",
            )

        # Reset collection for idempotent re-indexing
        self._store.reset_collection("role_archetypes")

        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, str]] = []

        for arch in archetypes:
            name = arch["name"]
            description = _normalize_whitespace(arch["description"])
            slug = _slugify(name)
            doc_id = f"archetype-{slug}"

            embedding = await self._embedder.embed(description)
            ids.append(doc_id)
            documents.append(description)
            embeddings.append(embedding)
            metadatas.append({"name": name, "source": "role_archetypes"})

        self._store.add_documents(
            collection_name="role_archetypes",
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "Indexed %d archetypes from %s", len(ids), archetypes_path
        )
        return len(ids)

