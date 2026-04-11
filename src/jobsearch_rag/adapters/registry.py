"""Adapter registry — IoC loader and factory for job board adapters."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from jobsearch_rag.adapters.base import JobBoardAdapter

_T = TypeVar("_T", bound="JobBoardAdapter")


class AdapterRegistry:
    """
    Decorator-based registry that maps board name strings to adapter classes.

    Usage::

        @AdapterRegistry.register
        class ZipRecruiterAdapter(JobBoardAdapter):
            @property
            def board_name(self) -> str:
                return "ziprecruiter"
            ...
    """

    _registry: ClassVar[dict[str, type[JobBoardAdapter]]] = {}

    @classmethod
    def register(cls, adapter_class: type[_T]) -> type[_T]:
        """Class decorator — registers an adapter by its ``board_name``."""
        instance = adapter_class.__new__(adapter_class)
        cls._registry[instance.board_name] = adapter_class  # type: ignore[assignment]  # _T is a subtype of JobBoardAdapter
        return adapter_class

    @classmethod
    def get(cls, board_name: str, **kwargs: Any) -> JobBoardAdapter:
        """
        Return a new instance of the adapter registered under *board_name*.

        Any *kwargs* are forwarded to the adapter constructor.  Adapters
        that do not accept them will receive no arguments (the caller
        should only pass kwargs that the target adapter understands).
        """
        if board_name not in cls._registry:
            msg = f"No adapter registered for board: '{board_name}'"
            raise ValueError(msg)
        return cls._registry[board_name](**kwargs)

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return all registered board name strings."""
        return list(cls._registry.keys())

    @classmethod
    @contextmanager
    def override(
        cls,
        factories: dict[str, Callable[..., JobBoardAdapter]],
        *,
        clear: bool = False,
    ) -> Iterator[None]:
        """
        Context manager that temporarily replaces registry entries.

        Saves the current registry state, applies *factories* (merging by
        default), yields, then restores the original state on exit.

        Parameters
        ----------
        factories:
            Board-name → factory mappings to add/overwrite.
        clear:
            If ``True``, remove all pre-existing entries before merging.
            Useful for test isolation when no production adapters should
            be visible.

        """
        saved = dict(cls._registry)
        if clear:
            cls._registry.clear()
        cls._registry.update(factories)  # type: ignore[arg-type]  # test factories may return mock subtypes
        try:
            yield
        finally:
            cls._registry.clear()
            cls._registry.update(saved)
