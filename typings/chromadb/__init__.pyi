"""
Type stub overrides for ``chromadb``.

ChromaDB's shipped types use narrow aliases (``Embeddings``, ``Metadata``,
``Include``, ``Where``) that reject standard Python types our code naturally
produces (``list[list[float]]``, ``list[dict[str, Any]]``, ``list[str]``,
``dict[str, Any]``).  These stubs widen the signatures to accept the
standard types, eliminating scattered ``# type: ignore[arg-type]`` pragmas.

Re-evaluate when chromadb releases updated type stubs.
"""

from collections.abc import Sequence
from typing import Any

from . import errors as errors

class Collection:
    name: str
    def add(
        self,
        ids: list[str] | None = ...,
        embeddings: Sequence[Sequence[float | int]] | None = ...,
        metadatas: list[dict[str, Any]] | None = ...,
        documents: list[str] | None = ...,
    ) -> None: ...
    def upsert(
        self,
        ids: list[str] | None = ...,
        embeddings: Sequence[Sequence[float | int]] | None = ...,
        metadatas: list[dict[str, Any]] | None = ...,
        documents: list[str] | None = ...,
    ) -> None: ...
    def get(
        self,
        ids: list[str] | None = ...,
        where: dict[str, Any] | None = ...,
        include: list[str] | None = ...,
    ) -> dict[str, Any]: ...
    def query(
        self,
        query_embeddings: Sequence[Sequence[float | int]] | None = ...,
        query_texts: list[str] | None = ...,
        n_results: int = ...,
        where: dict[str, Any] | None = ...,
        include: list[str] | None = ...,
    ) -> dict[str, Any]: ...
    def delete(self, ids: list[str] | None = ...) -> None: ...
    def count(self) -> int: ...

class PersistentClient:
    def __init__(self, path: str | None = ..., **kwargs: Any) -> None: ...
    def get_or_create_collection(self, name: str, **kwargs: Any) -> Collection: ...
    def get_collection(self, name: str, **kwargs: Any) -> Collection: ...
    def delete_collection(self, name: str) -> None: ...
    def list_collections(self) -> list[Collection]: ...
    def clear_system_cache(self) -> None: ...
