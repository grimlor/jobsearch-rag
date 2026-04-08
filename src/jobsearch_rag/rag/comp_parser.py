"""
Compensation parsing and scoring.

Extracts salary ranges from job description text via regex, normalizes
hourly rates to annual (x2080), and computes a continuous comp_score
relative to a configurable base_salary target.

No LLM involvement — all extraction is pure regex on raw text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jobsearch_rag.config import CompBand


@dataclass
class CompResult:
    """Parsed compensation data from a job description."""

    comp_min: float
    comp_max: float
    comp_source: str
    comp_text: str


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches dollar amounts: $180,000  $180000  $180k  $180K  $95.50
_DOLLAR = r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*([kK])?"

# Hourly rate indicators (after the number)
_HOURLY_SUFFIX = r"(?:/\s*(?:hr|hour|h)\b)"

# Annual indicators (after the range or single value)
_ANNUAL_SUFFIX = r"(?:\s*(?:per\s+(?:year|annum)|annually|a\s+year|/\s*(?:yr|year))\b)?"

# Range connectors: " - ", "-", " to "
_RANGE_SEP = r"\s*(?:-|\N{EN DASH}|to)\s*"

# Full range pattern: $NNN - $NNN with optional hourly/annual suffix
_RANGE_PATTERN = re.compile(
    _DOLLAR + _RANGE_SEP + _DOLLAR + r"\s*" + _HOURLY_SUFFIX + "?" + r"\s*" + _ANNUAL_SUFFIX,
    re.IGNORECASE,
)

# Single value: $NNN with optional hourly/annual suffix
_SINGLE_PATTERN = re.compile(
    _DOLLAR + r"\s*" + _HOURLY_SUFFIX + "?" + r"\s*" + _ANNUAL_SUFFIX,
    re.IGNORECASE,
)

# Hourly detection: checks if the matched text contains /hr, /hour, etc.
_HOURLY_RE = re.compile(r"/\s*(?:hr|hour|h)\b", re.IGNORECASE)

# Context words that indicate a number is NOT a salary
_FALSE_POSITIVE_CONTEXT = re.compile(
    r"(?:employees?|people|staff|team\s+members?|offices?|"
    r"(?:billion|million|trillion)\b|"
    r"\bARR\b|\brevenue\b|\bfunding\b|\braised\b)",
    re.IGNORECASE,
)

_HOURS_PER_YEAR = 2080


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_dollar(raw_digits: str, k_suffix: str | None) -> float:
    """Convert a captured dollar string to a float."""
    value = float(raw_digits.replace(",", ""))
    if k_suffix:
        value *= 1_000
    return value


def _is_hourly(match_text: str) -> bool:
    """Check whether the matched text indicates an hourly rate."""
    return bool(_HOURLY_RE.search(match_text))


def _is_salary_range(value: float) -> bool:
    """
    Heuristic: a salary number should be in a reasonable range.

    Annual salaries: $20,000 - $1,000,000
    Hourly rates: $10 - $500 (before annualization)
    """
    return 10.0 <= value <= 1_000_000.0


def _has_false_positive_context(text: str, match_start: int, match_end: int) -> bool:
    """Check whether the match is surrounded by false-positive context words."""
    # Look at a window of ~80 chars around the match
    window_start = max(0, match_start - 80)
    window_end = min(len(text), match_end + 80)
    window = text[window_start:window_end]
    return bool(_FALSE_POSITIVE_CONTEXT.search(window))


def parse_compensation(text: str, source: str = "employer") -> CompResult | None:
    """
    Extract compensation data from job description text.

    Returns ``None`` if no salary information is found or if detected
    numbers are likely false positives (employee counts, revenue, etc.).

    Parameters
    ----------
    text:
        Raw JD text to parse.
    source:
        Origin of the salary data — ``"employer"`` (JD body) or
        ``"estimated"`` (board-generated estimate).

    """
    # Try range patterns first (more specific)
    for m in _RANGE_PATTERN.finditer(text):
        low = _parse_dollar(m.group(1), m.group(2))
        high = _parse_dollar(m.group(3), m.group(4))

        if _has_false_positive_context(text, m.start(), m.end()):
            continue

        hourly = _is_hourly(m.group(0))

        if hourly:
            low *= _HOURS_PER_YEAR
            high *= _HOURS_PER_YEAR
        elif not _is_salary_range(low) or not _is_salary_range(high):
            continue

        return CompResult(
            comp_min=low,
            comp_max=high,
            comp_source=source,
            comp_text=m.group(0).strip(),
        )

    # Try single-value patterns
    for m in _SINGLE_PATTERN.finditer(text):
        value = _parse_dollar(m.group(1), m.group(2))

        if _has_false_positive_context(text, m.start(), m.end()):
            continue

        hourly = _is_hourly(m.group(0))

        if hourly:
            value *= _HOURS_PER_YEAR
        elif not _is_salary_range(value):
            continue

        return CompResult(
            comp_min=value,
            comp_max=value,
            comp_source=source,
            comp_text=m.group(0).strip(),
        )

    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def compute_comp_score(
    comp_max: float | None,
    base_salary: float,
    *,
    breakpoints: list[CompBand] | None = None,
    default_score: float = 0.5,
) -> float:
    """
    Compute a continuous compensation score in [0.0, 1.0].

    Uses piecewise-linear interpolation between the given breakpoints
    (or built-in defaults).

    Parameters
    ----------
    comp_max:
        Maximum compensation from the JD, or None if missing.
    base_salary:
        The user's target base salary.
    breakpoints:
        Optional list of CompBand(ratio, score) pairs in descending
        ratio order.  Falls back to the hardcoded 4-point curve when
        None.
    default_score:
        Score returned when ``comp_max`` is None (missing data).

    Returns
    -------
    float
        Score in [0.0, 1.0].

    """
    if comp_max is None:
        return default_score

    ratio = comp_max / base_salary

    if breakpoints is not None:
        return _interpolate(ratio, breakpoints)

    # Legacy hardcoded curve (used when no breakpoints provided)
    if ratio >= 1.0:
        return 1.0
    if ratio >= 0.90:
        return 0.7 + (ratio - 0.90) / 0.10 * 0.2
    if ratio >= 0.77:
        return 0.4 + (ratio - 0.77) / 0.13 * 0.3
    if ratio >= 0.68:
        return 0.0 + (ratio - 0.68) / 0.09 * 0.4
    return 0.0


def _interpolate(ratio: float, bands: list[CompBand]) -> float:
    """
    Piecewise-linear interpolation over CompBand breakpoints.

    Requires *bands* sorted by ratio descending (highest first) with at
    least two entries.  Config-time validation guarantees this.
    """
    # Clamp to the outer breakpoints
    if ratio >= bands[0].ratio:
        return bands[0].score
    if ratio <= bands[-1].ratio:
        return bands[-1].score
    # Find the first enclosing segment and interpolate.
    upper, lower = next((upper, lower) for upper, lower in pairwise(bands) if ratio >= lower.ratio)
    t = (ratio - lower.ratio) / (upper.ratio - lower.ratio)
    return lower.score + t * (upper.score - lower.score)
