"""
Configuration loading and validation.

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
from typing import Any, cast

from jobsearch_rag.errors import ActionableError

# Type alias for TOML-parsed dicts — values are heterogeneous.
_TOMLDict = dict[str, Any]

# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CompBand:
    """A single breakpoint in the piecewise-linear comp score curve."""

    ratio: float
    score: float


@dataclass
class DisqualifierConfig:
    """Disqualifier prompt settings from ``[disqualifier]``."""

    system_prompt: str | None = None


@dataclass
class SecurityConfig:
    """Security screening settings from ``[security]``."""

    screen_prompt: str


@dataclass
class BoardConfig:
    """Per-board configuration from ``[boards.<name>]``."""

    name: str
    searches: list[str]
    max_pages: int
    headless: bool
    rate_limit_range: tuple[float, float]
    browser_channel: str | None = None
    throttle_max_retries: int | None = None
    throttle_base_delay: float | None = None
    login_url: str | None = None
    stealth: bool = False


@dataclass
class ScoringConfig:
    """Scoring weights and thresholds from ``[scoring]``."""

    archetype_weight: float
    fit_weight: float
    history_weight: float
    comp_weight: float
    negative_weight: float
    culture_weight: float
    base_salary: float
    disqualify_on_llm_flag: bool
    min_score_threshold: float
    comp_bands: list[CompBand]
    missing_comp_score: float
    chunk_overlap: int
    dedup_similarity_threshold: float


@dataclass
class OllamaConfig:
    """Ollama connection settings from ``[ollama]``."""

    base_url: str
    llm_model: str
    embed_model: str
    slow_llm_threshold_ms: int
    classify_system_prompt: str
    max_retries: int
    base_delay: float
    max_embed_chars: int
    head_ratio: float
    retryable_status_codes: list[int]


@dataclass
class OutputConfig:
    """Output settings from ``[output]``."""

    default_format: str
    output_dir: str
    open_top_n: int
    jd_dir: str
    decisions_dir: str
    log_dir: str
    eval_history_path: str


@dataclass
class AdaptersConfig:
    """Browser and CDP settings from ``[adapters]``."""

    browser_paths: dict[str, list[str]]
    cdp_timeout: float


@dataclass
class ChromaConfig:
    """ChromaDB settings from ``[chroma]``."""

    persist_dir: str


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
    security: SecurityConfig
    resume_path: str
    archetypes_path: str
    global_rubric_path: str
    session_storage_dir: str
    adapters: AdaptersConfig
    disqualifier: DisqualifierConfig | None = None

    def __post_init__(self) -> None:
        """Apply defaults for optional config sections."""
        if self.disqualifier is None:
            self.disqualifier = DisqualifierConfig()


# ---------------------------------------------------------------------------
# Default settings path
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS_PATH = Path("config/settings.toml")


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------


def load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> Settings:
    """
    Load and validate settings from a TOML file.

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


def _validate(data: _TOMLDict, filepath: Path) -> Settings:
    """Validate raw TOML data and return a Settings instance."""
    # -- boards section ------------------------------------------------------
    boards_section = _require_section(data, "boards", filepath)
    enabled_boards: list[str] = _require_list(boards_section, "enabled", "boards", filepath)

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
        board_data = _require_table(boards_section, str(board_name), "boards", filepath)
        searches: list[str] = board_data.get("searches", [])  # type: ignore[assignment]
        rl_raw = _require_field(board_data, "rate_limit_range", f"boards.{board_name}")
        rl_list: list[float] = [float(v) for v in rl_raw]
        board_configs[board_name] = BoardConfig(
            name=board_name,
            searches=searches,
            max_pages=int(board_data.get("max_pages", 3)),
            headless=bool(board_data.get("headless", True)),
            rate_limit_range=(rl_list[0], rl_list[1]),
            browser_channel=board_data.get("browser_channel") or None,
            throttle_max_retries=(
                int(board_data["throttle_max_retries"])
                if "throttle_max_retries" in board_data
                else None
            ),
            throttle_base_delay=(
                float(board_data["throttle_base_delay"])
                if "throttle_base_delay" in board_data
                else None
            ),
            login_url=board_data.get("login_url") or None,
            stealth=bool(board_data.get("stealth", False)),
        )

    # Also parse overnight board configs if they have sections
    for board_name in overnight_boards:
        if board_name in boards_section and board_name not in board_configs:
            board_data_raw = boards_section[board_name]
            if isinstance(board_data_raw, dict):
                bd = cast("_TOMLDict", board_data_raw)
                searches_ov: list[str] = bd.get("searches", [])  # type: ignore[assignment]
                rl_raw_ov = _require_field(bd, "rate_limit_range", f"boards.{board_name}")
                rl_list_ov: list[float] = [float(v) for v in rl_raw_ov]
                board_configs[board_name] = BoardConfig(
                    name=board_name,
                    searches=searches_ov,
                    max_pages=int(bd.get("max_pages", 2)),
                    headless=bool(bd.get("headless", False)),
                    rate_limit_range=(rl_list_ov[0], rl_list_ov[1]),
                    browser_channel=bd.get("browser_channel") or None,
                    throttle_max_retries=(
                        int(bd["throttle_max_retries"]) if "throttle_max_retries" in bd else None
                    ),
                    throttle_base_delay=(
                        float(bd["throttle_base_delay"]) if "throttle_base_delay" in bd else None
                    ),
                    login_url=bd.get("login_url") or None,
                    stealth=bool(bd.get("stealth", False)),
                )

    # -- scoring section -----------------------------------------------------
    scoring_data = _require_section(data, "scoring", filepath)

    scoring = ScoringConfig(
        archetype_weight=float(_require_field(scoring_data, "archetype_weight", "scoring")),
        fit_weight=float(_require_field(scoring_data, "fit_weight", "scoring")),
        history_weight=float(_require_field(scoring_data, "history_weight", "scoring")),
        comp_weight=float(_require_field(scoring_data, "comp_weight", "scoring")),
        negative_weight=float(_require_field(scoring_data, "negative_weight", "scoring")),
        culture_weight=float(_require_field(scoring_data, "culture_weight", "scoring")),
        base_salary=float(_require_field(scoring_data, "base_salary", "scoring")),
        disqualify_on_llm_flag=bool(
            _require_field(scoring_data, "disqualify_on_llm_flag", "scoring")
        ),
        min_score_threshold=float(_require_field(scoring_data, "min_score_threshold", "scoring")),
        comp_bands=_require_comp_bands(scoring_data),
        missing_comp_score=float(_require_field(scoring_data, "missing_comp_score", "scoring")),
        chunk_overlap=int(_require_field(scoring_data, "chunk_overlap", "scoring")),
        dedup_similarity_threshold=float(
            _require_field(scoring_data, "dedup_similarity_threshold", "scoring")
        ),
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

    # Validate missing_comp_score range
    if not 0.0 <= scoring.missing_comp_score <= 1.0:
        raise ActionableError.validation(
            field_name="scoring.missing_comp_score",
            reason=f"is {scoring.missing_comp_score} — must be in [0.0, 1.0]",
            suggestion="Set [scoring].missing_comp_score to a value between 0.0 and 1.0",
        )

    # Validate comp_bands
    bands = scoring.comp_bands
    if len(bands) < 2:
        raise ActionableError.validation(
            field_name="scoring.comp_bands",
            reason=f"has {len(bands)} entry — at least 2 breakpoints are required",
            suggestion="Add at least 2 [[scoring.comp_bands]] entries with ratio and score",
        )
    for i in range(1, len(bands)):
        if bands[i].ratio >= bands[i - 1].ratio:
            raise ActionableError.validation(
                field_name="scoring.comp_bands",
                reason=(
                    f"ratio {bands[i].ratio} at index {i} is not less than "
                    f"ratio {bands[i - 1].ratio} at index {i - 1} — "
                    f"ratios must be monotonically decreasing"
                ),
                suggestion="Order [[scoring.comp_bands]] entries with ratios in decreasing order",
            )

    # -- ollama section ------------------------------------------------------
    ollama_data = _require_section(data, "ollama", filepath)

    base_url = str(_require_field(ollama_data, "base_url", "ollama"))
    if not base_url.startswith(("http://", "https://")):
        raise ActionableError.validation(
            field_name="ollama.base_url",
            reason=f"'{base_url}' is missing a scheme (http:// or https://)",
            suggestion="Set [ollama].base_url to a URL starting with http:// or https://",
        )

    classify_system_prompt = str(_require_field(ollama_data, "classify_system_prompt", "ollama"))

    # Validate classify_system_prompt is not empty
    if not classify_system_prompt:
        raise ActionableError.validation(
            field_name="ollama.classify_system_prompt",
            reason="is empty — must be a non-empty string",
            suggestion="Set [ollama].classify_system_prompt to a non-empty prompt string",
        )

    ollama = OllamaConfig(
        base_url=base_url,
        llm_model=str(_require_field(ollama_data, "llm_model", "ollama")),
        embed_model=str(_require_field(ollama_data, "embed_model", "ollama")),
        slow_llm_threshold_ms=int(_require_field(ollama_data, "slow_llm_threshold_ms", "ollama")),
        classify_system_prompt=classify_system_prompt,
        max_retries=int(_require_field(ollama_data, "max_retries", "ollama")),
        base_delay=float(_require_field(ollama_data, "base_delay", "ollama")),
        max_embed_chars=int(_require_field(ollama_data, "max_embed_chars", "ollama")),
        head_ratio=float(_require_field(ollama_data, "head_ratio", "ollama")),
        retryable_status_codes=[
            int(c) for c in _require_field(ollama_data, "retryable_status_codes", "ollama")
        ],
    )

    # -- output section ------------------------------------------------------
    output_data = _require_section(data, "output", filepath)

    output = OutputConfig(
        default_format=str(_require_field(output_data, "default_format", "output")),
        output_dir=str(_require_field(output_data, "output_dir", "output")),
        open_top_n=int(_require_field(output_data, "open_top_n", "output")),
        jd_dir=str(_require_field(output_data, "jd_dir", "output")),
        decisions_dir=str(_require_field(output_data, "decisions_dir", "output")),
        log_dir=str(_require_field(output_data, "log_dir", "output")),
        eval_history_path=str(_require_field(output_data, "eval_history_path", "output")),
    )

    # -- chroma section ------------------------------------------------------
    chroma_data = _require_section(data, "chroma", filepath)

    chroma = ChromaConfig(
        persist_dir=str(_require_field(chroma_data, "persist_dir", "chroma")),
    )

    # -- shared file paths --------------------------------------------------
    resume_path = str(_require_field(data, "resume_path", "top-level"))
    archetypes_path = str(_require_field(data, "archetypes_path", "top-level"))
    global_rubric_path = str(_require_field(data, "global_rubric_path", "top-level"))

    if not Path(global_rubric_path).exists():
        raise ActionableError.config(
            field_name="global_rubric_path",
            reason=f"Global rubric file not found: {global_rubric_path}",
            suggestion=(
                "Create the file or update global_rubric_path in settings.toml "
                "to a valid TOML file path"
            ),
        )

    # -- disqualifier section ------------------------------------------------
    disqualifier_data = _get_table(data, "disqualifier")
    dq_prompt_raw = disqualifier_data.get("system_prompt")
    if dq_prompt_raw is not None:
        dq_prompt_str = str(dq_prompt_raw)
        if not dq_prompt_str:
            raise ActionableError.validation(
                field_name="disqualifier.system_prompt",
                reason="is empty — must be a non-empty string or omitted to trigger synthesis",
                suggestion=(
                    "Remove [disqualifier].system_prompt to use archetype synthesis, "
                    "or provide a non-empty prompt string"
                ),
            )
    disqualifier = DisqualifierConfig(
        system_prompt=str(dq_prompt_raw) if dq_prompt_raw is not None else None,
    )

    # -- security section ----------------------------------------------------
    security_data = _require_section(data, "security", filepath)
    screen_prompt = str(_require_field(security_data, "screen_prompt", "security"))
    if not screen_prompt:
        raise ActionableError.validation(
            field_name="security.screen_prompt",
            reason="is empty — must be a non-empty string",
            suggestion="Set [security].screen_prompt to a non-empty prompt string",
        )
    security = SecurityConfig(screen_prompt=screen_prompt)

    # -- adapters section ----------------------------------------------------
    adapters_data = _require_section(data, "adapters", filepath)

    browser_paths_raw = adapters_data.get("browser_paths")
    if browser_paths_raw is None or not isinstance(browser_paths_raw, dict):
        raise ActionableError.config(
            field_name="adapters.browser_paths",
            reason="Required field 'browser_paths' is missing from [adapters]",
            suggestion=(
                "Add [adapters.browser_paths] section with channel entries "
                "like msedge, chrome, chromium"
            ),
        )
    browser_paths: dict[str, list[str]] = {}
    browser_paths_table = cast("_TOMLDict", browser_paths_raw)
    for channel_name, paths in browser_paths_table.items():
        if not isinstance(paths, list):
            raise ActionableError.config(
                field_name=f"adapters.browser_paths.{channel_name}",
                reason=f"paths must be a list, got {type(paths).__name__}",
                suggestion=f"Set [adapters.browser_paths].{channel_name} to a list of path strings",
            )
        browser_paths[str(channel_name)] = [str(p) for p in cast("list[Any]", paths)]

    cdp_timeout_raw = _require_field(adapters_data, "cdp_timeout", "adapters")
    cdp_timeout = float(cdp_timeout_raw)
    if cdp_timeout <= 0.0:
        raise ActionableError.validation(
            field_name="adapters.cdp_timeout",
            reason=f"is {cdp_timeout} — must be > 0.0",
            suggestion="Set [adapters].cdp_timeout to a positive number (e.g. 15.0)",
        )

    adapters = AdaptersConfig(
        browser_paths=browser_paths,
        cdp_timeout=cdp_timeout,
    )

    # -- session_storage_dir (from [boards]) ----------------------------------
    session_storage_dir = str(_require_field(boards_section, "session_storage_dir", "boards"))

    return Settings(
        enabled_boards=list(enabled_boards),
        overnight_boards=list(overnight_boards),
        boards=board_configs,
        scoring=scoring,
        ollama=ollama,
        output=output,
        chroma=chroma,
        adapters=adapters,
        disqualifier=disqualifier,
        security=security,
        resume_path=resume_path,
        archetypes_path=archetypes_path,
        global_rubric_path=global_rubric_path,
        session_storage_dir=session_storage_dir,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_section(data: _TOMLDict, name: str, filepath: Path) -> _TOMLDict:
    """Return a required top-level section, or raise CONFIG error."""
    section = data.get(name)
    if section is None or not isinstance(section, dict):
        raise ActionableError.config(
            field_name=name,
            reason=f"Required section [{name}] is missing from {filepath}",
            suggestion=f"Add a [{name}] section to {filepath}",
        )
    return cast("_TOMLDict", section)


def _require_list(
    section: _TOMLDict, field_name: str, section_name: str, filepath: Path
) -> list[str]:
    """Return a required list field, or raise CONFIG error."""
    value = section.get(field_name)
    if not isinstance(value, list) or not value:
        raise ActionableError.config(
            field_name=f"{section_name}.{field_name}",
            reason=f"{section_name}.{field_name} must be a non-empty list",
            suggestion=f"Add at least one item to [{section_name}].{field_name}",
        )
    return cast("list[str]", value)


def _require_table(section: _TOMLDict, key: str, section_name: str, filepath: Path) -> _TOMLDict:
    """Return a required sub-table, or raise CONFIG error."""
    value = section.get(key)
    if not isinstance(value, dict):
        raise ActionableError.config(
            field_name=f"{section_name}.{key}",
            reason=f"[{section_name}.{key}] must be a table",
            suggestion=f"Define [{section_name}.{key}] as a TOML table",
        )
    return cast("_TOMLDict", value)


def _get_table(data: _TOMLDict, name: str) -> _TOMLDict:
    """Return an optional section as a dict (empty if missing/invalid)."""
    section = data.get(name, {})
    if isinstance(section, dict):
        return cast("_TOMLDict", section)
    return {}


def _require_field(section: _TOMLDict, field_name: str, section_name: str) -> Any:
    """Return a required field value, or raise CONFIG error if missing."""
    value = section.get(field_name)
    if value is None:
        raise ActionableError.config(
            field_name=f"{section_name}.{field_name}",
            reason=f"Required field '{field_name}' is missing from [{section_name}]",
            suggestion=f"Add {field_name} to [{section_name}] in settings.toml",
        )
    return value


def _require_comp_bands(scoring_data: _TOMLDict) -> list[CompBand]:
    """Parse required ``[[scoring.comp_bands]]`` array into CompBand list."""
    raw_bands = scoring_data.get("comp_bands")
    if raw_bands is None or not isinstance(raw_bands, list):
        raise ActionableError.config(
            field_name="scoring.comp_bands",
            reason="Required field 'comp_bands' is missing from [scoring]",
            suggestion="Add [[scoring.comp_bands]] entries with ratio and score to settings.toml",
        )
    bands: list[CompBand] = []
    for b in raw_bands:  # type: ignore[reportUnknownVariableType]
        if isinstance(b, dict):
            entry = cast("_TOMLDict", b)
            bands.append(CompBand(ratio=float(entry["ratio"]), score=float(entry["score"])))
    return bands


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def synthesize_disqualifier_prompt(archetypes_path: str | Path) -> str:
    """
    Build a disqualifier system prompt from ``role_archetypes.toml``.

    Reads the archetype names and ``signals_negative`` fields to produce
    a prompt that instructs the LLM to disqualify roles matching those
    negative signals.

    Raises :class:`~jobsearch_rag.errors.ActionableError` (CONFIG) if the
    file has no archetypes.
    """
    filepath = Path(archetypes_path)
    raw_text = filepath.read_text(encoding="utf-8")
    data = tomllib.loads(raw_text)
    archetypes: list[_TOMLDict] = data.get("archetypes", [])  # type: ignore[assignment]

    if not archetypes:
        raise ActionableError.config(
            field_name="archetypes",
            reason=f"No archetypes found in {filepath}",
            suggestion=f"Add at least one [[archetypes]] entry to {filepath}",
        )

    # Collect archetype names and negative signals
    role_names = [str(a.get("name", "unnamed")) for a in archetypes]
    negative_signals: list[str] = []
    for arch in archetypes:
        for sig in arch.get("signals_negative", []):
            negative_signals.append(str(sig))

    roles_str = ", ".join(role_names)
    negatives_str = "\n".join(f"- {s}" for s in negative_signals)

    return (
        f"You are a role-fit classifier for the following target roles: {roles_str}.\n"
        f"Analyse the following job description and decide whether it is structurally\n"
        f"unsuitable for a candidate targeting these roles.\n\n"
        f"Disqualify if the role matches any of these negative signals:\n"
        f"{negatives_str}\n\n"
        f"Respond ONLY with a JSON object (no markdown fences):\n"
        f'{{"disqualified": true/false, "reason": "short explanation or null"}}'
    )
