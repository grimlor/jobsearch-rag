"""Inference timing and token statistics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InferenceMetrics:
    """Accumulated inference timing and token statistics."""

    total_embed_calls: int = 0
    total_classify_calls: int = 0
    total_embed_time_s: float = 0.0
    total_classify_time_s: float = 0.0
