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
from dataclasses import dataclass
from pathlib import Path

from jobsearch_rag.errors import ActionableError

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
    base_salary: float = 220_000
    disqualify_on_llm_flag: bool = True
    min_score_threshold: float = 0.45


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


def _validate(data: dict[str, object], filepath: Path) -> Settings:
    """Validate raw TOML data and return a Settings instance."""

    # -- boards section ------------------------------------------------------
    boards_section = _require_section(data, "boards", filepath)
    enabled_boards = _require_field(boards_section, "enabled", "boards", filepath)
    if not isinstance(enabled_boards, list) or not enabled_boards:
        raise ActionableError.config(
            field_name="boards.enabled",
            reason="boards.enabled must be a non-empty list of board names",
            suggestion="Add at least one board name to [boards].enabled",
        )

    overnight_boards: list[str] = boards_section.get("overnight_boards", [])  # type: ignore[assignment]

    # Validate each enabled board has a config section
    board_configs: dict[str, BoardConfig] = {}
    for board_name in enabled_boards:
        if board_name not in boards_section:
            raise ActionableError.config(
                field_name=f"boards.{board_name}",
                reason=f"Board '{board_name}' is in [boards].enabled but has no [boards.{board_name}] section",
                suggestion=f"Add a [boards.{board_name}] section with 'searches' and 'max_pages'",
            )
        board_data = boards_section[board_name]
        if not isinstance(board_data, dict):
            raise ActionableError.config(
                field_name=f"boards.{board_name}",
                reason=f"[boards.{board_name}] must be a table, not {type(board_data).__name__}",
                suggestion=f"Define [boards.{board_name}] as a TOML table",
            )
        searches = board_data.get("searches", [])
        board_configs[board_name] = BoardConfig(
            name=board_name,
            searches=list(searches),
            max_pages=int(board_data.get("max_pages", 3)),
            headless=bool(board_data.get("headless", True)),
            browser_channel=board_data.get("browser_channel") or None,
        )

    # Also parse overnight board configs if they have sections
    for board_name in overnight_boards:
        if board_name in boards_section and board_name not in board_configs:
            board_data = boards_section[board_name]
            if isinstance(board_data, dict):
                searches = board_data.get("searches", [])
                board_configs[board_name] = BoardConfig(
                    name=board_name,
                    searches=list(searches),
                    max_pages=int(board_data.get("max_pages", 2)),
                    headless=bool(board_data.get("headless", False)),
                    browser_channel=board_data.get("browser_channel") or None,
                )

    # -- scoring section -----------------------------------------------------
    scoring_data = data.get("scoring", {})
    if not isinstance(scoring_data, dict):
        scoring_data = {}

    scoring = ScoringConfig(
        archetype_weight=float(scoring_data.get("archetype_weight", 0.5)),
        fit_weight=float(scoring_data.get("fit_weight", 0.3)),
        history_weight=float(scoring_data.get("history_weight", 0.2)),
        comp_weight=float(scoring_data.get("comp_weight", 0.15)),
        base_salary=float(scoring_data.get("base_salary", 220_000)),
        disqualify_on_llm_flag=bool(scoring_data.get("disqualify_on_llm_flag", True)),
        min_score_threshold=float(scoring_data.get("min_score_threshold", 0.45)),
    )

    # Validate weight ranges
    for weight_name in ("archetype_weight", "fit_weight", "history_weight", "comp_weight"):
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

    # -- ollama section ------------------------------------------------------
    ollama_data = data.get("ollama", {})
    if not isinstance(ollama_data, dict):
        ollama_data = {}

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
    output_data = data.get("output", {})
    if not isinstance(output_data, dict):
        output_data = {}

    output = OutputConfig(
        default_format=str(output_data.get("default_format", "markdown")),
        output_dir=str(output_data.get("output_dir", "./output")),
        open_top_n=int(output_data.get("open_top_n", 5)),
    )

    # -- chroma section ------------------------------------------------------
    chroma_data = data.get("chroma", {})
    if not isinstance(chroma_data, dict):
        chroma_data = {}

    chroma = ChromaConfig(
        persist_dir=str(chroma_data.get("persist_dir", "./data/chroma_db")),
    )

    return Settings(
        enabled_boards=list(enabled_boards),
        overnight_boards=list(overnight_boards),
        boards=board_configs,
        scoring=scoring,
        ollama=ollama,
        output=output,
        chroma=chroma,
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
    return section


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
