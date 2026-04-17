"""
Observable class decorator for embedding port implementations.

Adds structured call tracing (``embed_call``, ``classify_call`` log events)
and :class:`MetricGroup` accumulation to any
:class:`~jobsearch_rag.ports.EmbeddingPort` implementation.

Usage::

    @observable
    class OllamaEmbedder:
        ...

    @observable
    class FakeEmbedder:
        ...
"""

from __future__ import annotations

import functools
import time
from typing import Any, TypeVar, cast

from jobsearch_rag.logging import log_event

_T = TypeVar("_T", bound=type)


class MetricKey:
    """IDE-checkable metric name constants."""

    EMBED_CALLS = "embed_calls"
    EMBED_TOKENS_TOTAL = "embed_tokens_total"
    LLM_CALLS = "llm_calls"
    LLM_TOKENS_TOTAL = "llm_tokens_total"
    LLM_LATENCY_MS_TOTAL = "llm_latency_ms_total"
    SLOW_LLM_CALLS = "slow_llm_calls"


class MetricGroup:
    """Accumulates float metrics for one source component."""

    def __init__(self) -> None:
        """Initialise an empty metric group."""
        self._data: dict[str, float] = {}

    def increment(self, key: str, amount: float = 1.0) -> None:
        """Increment *key* by *amount* (default 1.0)."""
        self._data[key] = self._data.get(key, 0.0) + amount

    def set(self, key: str, value: float) -> None:
        """Set *key* to *value*."""
        self._data[key] = value

    def get(self, key: str) -> float:
        """Return value for *key*, or 0.0 if missing."""
        return self._data.get(key, 0.0)

    def as_dict(self) -> dict[str, int | float]:
        """
        Return a shallow copy of all accumulated metrics.

        Whole-number values are returned as ``int`` so that JSON
        round-trips preserve the integer representation.
        """
        return {k: int(v) if v == int(v) else v for k, v in self._data.items()}


def observable(cls: _T) -> _T:
    """
    Class decorator adding call tracing and MetricGroup accumulation.

    Wraps ``embed()`` and ``classify()`` to:

    - Emit ``log_event("embed_call", ...)`` / ``log_event("classify_call", ...)``
    - Accumulate metrics in a :class:`MetricGroup` instance
    - Detect slow LLM calls via ``self._slow_llm_threshold_ms``

    Token counts are read from ``self._last_embed_tokens`` /
    ``self._last_classify_tokens`` published by the underlying implementation
    after each call.
    """
    original_init: Any = cls.__init__  # type: ignore[misc]
    original_embed: Any = cls.embed  # type: ignore[attr-defined]
    original_classify: Any = cls.classify  # type: ignore[attr-defined]

    @functools.wraps(original_init)
    def _init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        mg = MetricGroup()
        for key in (
            MetricKey.EMBED_CALLS,
            MetricKey.EMBED_TOKENS_TOTAL,
            MetricKey.LLM_CALLS,
            MetricKey.LLM_TOKENS_TOTAL,
            MetricKey.LLM_LATENCY_MS_TOTAL,
            MetricKey.SLOW_LLM_CALLS,
        ):
            mg.set(key, 0)
        self._metrics = mg

    async def _embed(self: Any, text: str) -> list[float]:
        t0 = time.monotonic()
        result = cast("list[float]", await original_embed(self, text))
        latency_ms = (time.monotonic() - t0) * 1000

        tokens: int = getattr(self, "_last_embed_tokens", 0)
        model: str = getattr(self, "embed_model", "unknown")

        metrics: MetricGroup = self._metrics
        metrics.increment(MetricKey.EMBED_CALLS)
        metrics.increment(MetricKey.EMBED_TOKENS_TOTAL, float(tokens))

        log_event(
            "embed_call",
            model=model,
            input_chars=len(text),
            latency_ms=int(latency_ms),
            tokens=tokens,
        )
        return result

    async def _classify(self: Any, prompt: str) -> str:
        t0 = time.monotonic()
        result = cast("str", await original_classify(self, prompt))
        latency_ms = int((time.monotonic() - t0) * 1000)

        tokens: int = getattr(self, "_last_classify_tokens", 0)
        model: str = getattr(self, "llm_model", "unknown")
        threshold: float = getattr(self, "_slow_llm_threshold_ms", 999_999_999)

        metrics: MetricGroup = self._metrics
        metrics.increment(MetricKey.LLM_CALLS)
        metrics.increment(MetricKey.LLM_TOKENS_TOTAL, float(tokens))
        metrics.increment(MetricKey.LLM_LATENCY_MS_TOTAL, latency_ms)
        if latency_ms > threshold:
            metrics.increment(MetricKey.SLOW_LLM_CALLS)

        log_event(
            "classify_call",
            model=model,
            input_chars=len(prompt),
            latency_ms=int(latency_ms),
            tokens=tokens,
        )
        return result

    @property  # type: ignore[misc]
    def _metrics_property(self: Any) -> MetricGroup:
        return self._metrics

    cls.__init__ = _init  # type: ignore[misc]
    cls.embed = _embed  # type: ignore[attr-defined]
    cls.classify = _classify  # type: ignore[attr-defined]
    cls.metrics = _metrics_property  # type: ignore[attr-defined]

    return cls
