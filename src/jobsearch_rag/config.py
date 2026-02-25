"""Configuration loading and validation.

Loads ``settings.toml`` and validates all fields at startup, before any
browser sessions are opened or expensive work begins.  A mid-run config
failure after 10 minutes of browser work is far more costly than a
startup validation failure.

The validated config is exposed as a :class:`Settings` dataclass with
typed fields for each section: ``boards``, ``scoring``, ``ollama``,
``output``, and ``chroma``.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from jobsearch_rag.errors import ActionableError
from jobsearch_rag.rag.comp_parser import DEFAULT_COMP_BANDS, CompBand

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BoardConfig:
    """Per-board configuration from ``[boards.<name>]``."""

    name: str
    searches: list[str]
    max_pages: int = 3
    headless: bool = True
    browser_channel: str | None = None


@dataclass
class ScoringConfig:
    """Scoring weights and thresholds from ``[scoring]``."""

    archetype_weight: float = 0.5
    fit_weight: float = 0.3
    history_weight: float = 0.2
    comp_weight: float = 0.15
    negative_weight: float = 0.4
    culture_weight: float = 0.2
    base_salary: float = 220_000
    disqualify_on_llm_flag: bool = True
    min_score_threshold: float = 0.45
    comp_bands: list[CompBand] = field(default_factory=lambda: list(DEFAULT_COMP_BANDS))
    missing_comp_score: float = 0.5


@dataclass
class OllamaConfig:
    """Ollama connection settings from ``[ollama]``."""

    base_url: str = "http://localhost:11434"
    llm_model: str = "mistral:7b"
    embed_model: str = "nomic-embed-text"


@dataclass
class OutputConfig:
    """Output settings from ``[output]``."""

    default_format: str = "markdown"
    output_dir: str = "./output"
    open_top_n: int = 5


@dataclass
class ChromaConfig:
    """ChromaDB settings from ``[chroma]``."""

    persist_dir: str = "./data/chroma_db"


@dataclass
class Settings:
    """Top-level validated configuration."""

    enabled_boards: list[str]
    overnight_boards: list[str]
    boards: dict[str, BoardConfig]
    scoring: ScoringConfig
    ollama: OllamaConfig
    output: OutputConfig
    chroma: ChromaConfig
    resume_path: str = "data/resume.md"
    archetypes_path: str = "config/role_archetypes.toml"
    global_rubric_path: str = "config/global_rubric.toml"


# ---------------------------------------------------------------------------
# Default settings path
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS_PATH = Path("config/settings.toml")


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> Settings:
    """Load and validate settings from a TOML file.

    Raises :class:`~jobsearch_rag.errors.ActionableError`:
      - CONFIG if the file is missing or a required field is absent
      - VALIDATION if field values are out of range
      - PARSE if the TOML is malformed

    Returns a fully validated :class:`Settings` instance.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise ActionableError.config(
            field_name="settings_path",
            reason=f"Settings file not found: {filepath}",
            suggestion=f"Create {filepath} or copy from config/settings.toml.example",
        )

    raw_text = filepath.read_text(encoding="utf-8")
    try:
        data = tomllib.loads(raw_text)
    except tomllib.TOMLDecodeError as exc:
        raise ActionableError.parse(
            board="settings",
            selector="TOML syntax",
            raw_error=str(exc),
            suggestion=f"Fix TOML syntax in {filepath}",
        ) from None

    return _validate(data, filepath)


# ---------------------------------------------------------------------------
# Typed TOML extraction helpers
# ---------------------------------------------------------------------------
# tomllib.loads() returns dict[str, Any], but these helpers let us keep
# dict[str, object] in our signatures (no Any — per CONTRIBUTING.md) while
# safely narrowing the dynamic TOML values through isinstance guards.


def _get_float(section: dict[str, object], key: str, default: float) -> float:
    """Extract a float from a TOML section, falling back to *default*."""
    raw = section.get(key, default)
    if isinstance(raw, (int, float)):
        return float(raw)
    return default


def _get_int(section: dict[str, object], key: str, default: int) -> int:
    """Extract an int from a TOML section, falling back to *default*."""
    raw = section.get(key, default)
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    return default


def _get_opt_str(section: dict[str, object], key: str) -> str | None:
    """Extract an optional string, returning ``None`` for missing or empty."""
    raw = section.get(key)
    if raw is None:
        return None
    result = str(raw)
    return result or None


def _section_or_empty(data: dict[str, object], name: str) -> dict[str, object]:
    """Return an optional TOML section, defaulting to empty dict."""
    raw = data.get(name)
    if isinstance(raw, dict):
        return cast("dict[str, object]", raw)
    return {}


def _validate(data: dict[str, object], filepath: Path) -> Settings:
    """Validate raw TOML data and return a Settings instance."""

    # -- boards section ------------------------------------------------------
    boards_section = _require_section(data, "boards", filepath)
    raw_enabled = _require_field(boards_section, "enabled", "boards", filepath)
    if not isinstance(raw_enabled, list) or not raw_enabled:
        raise ActionableError.config(
            field_name="boards.enabled",
            reason="boards.enabled must be a non-empty list of board names",
            suggestion="Add at least one board name to [boards].enabled",
        )
    enabled_boards: list[str] = [str(b) for b in cast("list[object]", raw_enabled)]

    raw_overnight = boards_section.get("overnight_boards")
    overnight_boards: list[str] = (
        [str(b) for b in cast("list[object]", raw_overnight)]
        if isinstance(raw_overnight, list)
        else []
    )

    # Validate each enabled board has a config section
    board_configs: dict[str, BoardConfig] = {}
    for board_name in enabled_boards:
        if board_name not in boards_section:
            raise ActionableError.config(
                field_name=f"boards.{board_name}",
                reason=f"Board '{board_name}' is in [boards].enabled but has no [boards.{board_name}] section",
                suggestion=f"Add a [boards.{board_name}] section with 'searches' and 'max_pages'",
            )
        raw_board = boards_section[board_name]
        if not isinstance(raw_board, dict):
            raise ActionableError.config(
                field_name=f"boards.{board_name}",
                reason=f"[boards.{board_name}] must be a table, not {type(raw_board).__name__}",
                suggestion=f"Define [boards.{board_name}] as a TOML table",
            )
        board_data = cast("dict[str, object]", raw_board)
        searches_raw = board_data.get("searches")
        board_configs[board_name] = BoardConfig(
            name=board_name,
            searches=(
                [str(s) for s in cast("list[object]", searches_raw)]
                if isinstance(searches_raw, list)
                else []
            ),
            max_pages=_get_int(board_data, "max_pages", 3),
            headless=bool(board_data.get("headless", True)),
            browser_channel=_get_opt_str(board_data, "browser_channel"),
        )

    # Also parse overnight board configs if they have sections
    for board_name in overnight_boards:
        if board_name in boards_section and board_name not in board_configs:
            raw_board = boards_section[board_name]
            if isinstance(raw_board, dict):
                board_data = cast("dict[str, object]", raw_board)
                searches_raw = board_data.get("searches")
                board_configs[board_name] = BoardConfig(
                    name=board_name,
                    searches=(
                        [str(s) for s in cast("list[object]", searches_raw)]
                        if isinstance(searches_raw, list)
                        else []
                    ),
                    max_pages=_get_int(board_data, "max_pages", 2),
                    headless=bool(board_data.get("headless", False)),
                    browser_channel=_get_opt_str(board_data, "browser_channel"),
                )

    # -- scoring section -----------------------------------------------------
    scoring_data = _section_or_empty(data, "scoring")

    scoring = ScoringConfig(
        archetype_weight=_get_float(scoring_data, "archetype_weight", 0.5),
        fit_weight=_get_float(scoring_data, "fit_weight", 0.3),
        history_weight=_get_float(scoring_data, "history_weight", 0.2),
        comp_weight=_get_float(scoring_data, "comp_weight", 0.15),
        negative_weight=_get_float(scoring_data, "negative_weight", 0.4),
        culture_weight=_get_float(scoring_data, "culture_weight", 0.2),
        base_salary=_get_float(scoring_data, "base_salary", 220_000),
        disqualify_on_llm_flag=bool(scoring_data.get("disqualify_on_llm_flag", True)),
        min_score_threshold=_get_float(scoring_data, "min_score_threshold", 0.45),
        comp_bands=_parse_comp_bands(scoring_data),
        missing_comp_score=_get_float(scoring_data, "missing_comp_score", 0.5),
    )

    # Validate weight ranges
    for weight_name in (
        "archetype_weight",
        "fit_weight",
        "history_weight",
        "comp_weight",
        "negative_weight",
        "culture_weight",
    ):
        value = getattr(scoring, weight_name)
        if value < 0.0:
            raise ActionableError.validation(
                field_name=f"scoring.{weight_name}",
                reason=f"is {value} — must be >= 0.0",
                suggestion=f"Set [scoring].{weight_name} to a value between 0.0 and 1.0",
            )
        if value > 1.0:
            raise ActionableError.validation(
                field_name=f"scoring.{weight_name}",
                reason=f"is {value} — must be <= 1.0",
                suggestion=f"Set [scoring].{weight_name} to a value between 0.0 and 1.0",
            )

    # Validate base_salary
    if scoring.base_salary <= 0:
        raise ActionableError.validation(
            field_name="scoring.base_salary",
            reason=f"is {scoring.base_salary} — must be > 0",
            suggestion="Set [scoring].base_salary to a positive number",
        )

    # Validate comp_bands
    _validate_comp_bands(scoring.comp_bands)

    # -- ollama section ------------------------------------------------------
    ollama_data = _section_or_empty(data, "ollama")

    base_url = str(ollama_data.get("base_url", "http://localhost:11434"))
    if not base_url.startswith(("http://", "https://")):
        raise ActionableError.validation(
            field_name="ollama.base_url",
            reason=f"'{base_url}' is missing a scheme (http:// or https://)",
            suggestion="Set [ollama].base_url to a URL starting with http:// or https://",
        )

    ollama = OllamaConfig(
        base_url=base_url,
        llm_model=str(ollama_data.get("llm_model", "mistral:7b")),
        embed_model=str(ollama_data.get("embed_model", "nomic-embed-text")),
    )

    # -- output section ------------------------------------------------------
    output_data = _section_or_empty(data, "output")

    output = OutputConfig(
        default_format=str(output_data.get("default_format", "markdown")),
        output_dir=str(output_data.get("output_dir", "./output")),
        open_top_n=_get_int(output_data, "open_top_n", 5),
    )

    # -- chroma section ------------------------------------------------------
    chroma_data = _section_or_empty(data, "chroma")

    chroma = ChromaConfig(
        persist_dir=str(chroma_data.get("persist_dir", "./data/chroma_db")),
    )

    # -- shared file paths --------------------------------------------------
    resume_path = str(data.get("resume_path", "data/resume.md"))
    archetypes_path = str(data.get("archetypes_path", "config/role_archetypes.toml"))
    global_rubric_path = str(data.get("global_rubric_path", "config/global_rubric.toml"))

    if not Path(global_rubric_path).exists():
        raise ActionableError.config(
            field_name="global_rubric_path",
            reason=f"Global rubric file not found: {global_rubric_path}",
            suggestion=(
                "Create the file or update global_rubric_path in settings.toml "
                "to a valid TOML file path"
            ),
        )

    return Settings(
        enabled_boards=list(enabled_boards),
        overnight_boards=list(overnight_boards),
        boards=board_configs,
        scoring=scoring,
        ollama=ollama,
        output=output,
        chroma=chroma,
        resume_path=resume_path,
        archetypes_path=archetypes_path,
        global_rubric_path=global_rubric_path,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_section(data: dict[str, object], name: str, filepath: Path) -> dict[str, object]:
    """Return a required top-level section, or raise CONFIG error."""
    section = data.get(name)
    if section is None or not isinstance(section, dict):
        raise ActionableError.config(
            field_name=name,
            reason=f"Required section [{name}] is missing from {filepath}",
            suggestion=f"Add a [{name}] section to {filepath}",
        )
    return cast("dict[str, object]", section)


def _require_field(
    section: dict[str, object], field_name: str, section_name: str, filepath: Path
) -> object:
    """Return a required field within a section, or raise CONFIG error."""
    value = section.get(field_name)
    if value is None:
        raise ActionableError.config(
            field_name=f"{section_name}.{field_name}",
            reason=f"Required field '{field_name}' is missing from [{section_name}] in {filepath}",
            suggestion=f"Add '{field_name}' to the [{section_name}] section in {filepath}",
        )
    return value


def _parse_comp_bands(scoring_data: dict[str, object]) -> list[CompBand]:
    """Parse ``[[scoring.comp_bands]]`` array-of-tables from TOML data.

    Returns :data:`DEFAULT_COMP_BANDS` when the key is absent.

    Raises :class:`~jobsearch_rag.errors.ActionableError`:
      - PARSE if an entry is not a table or is missing ``ratio`` / ``score``
    """
    raw_bands = scoring_data.get("comp_bands")
    if raw_bands is None:
        return list(DEFAULT_COMP_BANDS)

    if not isinstance(raw_bands, list):
        raise ActionableError.parse(
            board="settings",
            selector="scoring.comp_bands",
            raw_error=f"Expected a list of tables, got {type(raw_bands).__name__}",
            suggestion=(
                "Define comp_bands as an array of tables: "
                "[[scoring.comp_bands]] with ratio and score keys"
            ),
        )

    band_entries = cast("list[object]", raw_bands)
    bands: list[CompBand] = []
    for idx, raw_entry in enumerate(band_entries):
        if not isinstance(raw_entry, dict):
            raise ActionableError.parse(
                board="settings",
                selector=f"scoring.comp_bands[{idx}]",
                raw_error=f"Expected a table, got {type(raw_entry).__name__}",
                suggestion="Each [[scoring.comp_bands]] entry must be a TOML table with ratio and score",
            )
        entry = cast("dict[str, object]", raw_entry)
        if "ratio" not in entry or "score" not in entry:
            raise ActionableError.parse(
                board="settings",
                selector=f"scoring.comp_bands[{idx}]",
                raw_error=f"Missing required key(s): {{'ratio', 'score'}} - got {set(entry.keys())}",
                suggestion="Each [[scoring.comp_bands]] entry must have both 'ratio' and 'score'",
            )
        bands.append(CompBand(
            ratio=_get_float(entry, "ratio", 0.0),
            score=_get_float(entry, "score", 0.0),
        ))

    return bands


def _validate_comp_bands(bands: list[CompBand]) -> None:
    """Validate that comp bands are well-formed.

    Rules:
    - At least one band is required.
    - Ratios must be in descending order (highest first).
    - Ratios must be >= 0.
    - Scores must be in [0.0, 1.0].

    Raises :class:`~jobsearch_rag.errors.ActionableError` (VALIDATION)
    on the first violation found.
    """
    if not bands:
        raise ActionableError.validation(
            field_name="scoring.comp_bands",
            reason="comp_bands list is empty — at least one band is required",
            suggestion="Add at least one [[scoring.comp_bands]] entry with ratio and score",
        )

    for idx, band in enumerate(bands):
        if band.ratio < 0:
            raise ActionableError.validation(
                field_name=f"scoring.comp_bands[{idx}].ratio",
                reason=f"is {band.ratio} — must be >= 0",
                suggestion="Set ratio to a non-negative number (e.g. 0.90)",
            )
        if band.score < 0.0 or band.score > 1.0:
            raise ActionableError.validation(
                field_name=f"scoring.comp_bands[{idx}].score",
                reason=f"is {band.score} — must be between 0.0 and 1.0",
                suggestion="Set score to a value in [0.0, 1.0]",
            )

    for i in range(1, len(bands)):
        if bands[i].ratio >= bands[i - 1].ratio:
            raise ActionableError.validation(
                field_name="scoring.comp_bands",
                reason=(
                    f"bands are not in descending ratio order — "
                    f"band[{i - 1}].ratio={bands[i - 1].ratio} is not > band[{i}].ratio={bands[i].ratio}"
                ),
                suggestion="Order [[scoring.comp_bands]] entries from highest ratio to lowest",
            )
