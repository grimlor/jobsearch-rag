"""BDD specs for configuration loading and validation.

Covers: TestSettingsValidation
Spec doc: BDD Specifications — configuration.md
"""

# Public API surface (from src/jobsearch_rag/config.py):
#   load_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> Settings
#   Settings(enabled_boards, overnight_boards, boards, scoring, ollama, output, chroma,
#            resume_path="data/resume.md", archetypes_path="config/role_archetypes.toml",
#            global_rubric_path="config/global_rubric.toml")
#   ScoringConfig(archetype_weight=0.5, fit_weight=0.3, history_weight=0.2,
#                 comp_weight=0.15, negative_weight=0.4, culture_weight=0.2,
#                 base_salary=220_000, disqualify_on_llm_flag=True, min_score_threshold=0.45,
#                 comp_bands=DEFAULT_COMP_BANDS, missing_comp_score=0.5)
#   CompBand(ratio: float, score: float)
#   DEFAULT_COMP_BANDS: list[CompBand] — [(1.0, 1.0), (0.90, 0.7), (0.77, 0.4), (0.68, 0.0)]
#   OllamaConfig(base_url="http://localhost:11434", llm_model="mistral:7b",
#                embed_model="nomic-embed-text")
#   OutputConfig(default_format="markdown", output_dir="./output", open_top_n=5)
#   ChromaConfig(persist_dir="./data/chroma_db")
#   BoardConfig(name, searches, max_pages=3, headless=True, browser_channel=None)
#   ActionableError — raised on config/validation/parse errors with:
#     .error (str), .error_type (ErrorType), .suggestion (str|None),
#     .troubleshooting (Troubleshooting|None with .steps list)

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jobsearch_rag.config import Settings, load_settings
from jobsearch_rag.errors import ActionableError

if TYPE_CHECKING:
    from pathlib import Path


def _write_valid_toml(tmp_path: Path) -> Path:
    """Write a minimal valid TOML to tmp_path and return the settings file path.

    Also creates the global rubric file that the validator checks for on disk.
    """
    rubric_path = tmp_path / "global_rubric.toml"
    rubric_path.write_text("[rubric]\n", encoding="utf-8")

    toml_path = tmp_path / "settings.toml"
    toml_path.write_text(
        f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]
max_pages = 1

global_rubric_path = "{rubric_path}"
""",
        encoding="utf-8",
    )
    return toml_path


class TestSettingsValidation:
    """
    REQUIREMENT: Configuration errors tell the operator exactly what to fix
    before any expensive work begins.

    WHO: The operator who misconfigured settings.toml
    WHAT: Each validation failure is caught at startup with an error that
          names the problematic field, explains what is wrong, and provides
          step-by-step recovery guidance including which file to open and
          what to change; all config errors include a suggestion and
          troubleshooting steps; optional fields use documented defaults
    WHY: A mid-run config failure after 10 minutes of browser work is far
         more costly than a startup validation failure. An error that says
         "validation failed" without naming the field is nearly as costly
         as a mid-run crash

    MOCK BOUNDARY:
        Mock:  nothing — Settings loading is pure TOML parsing
        Real:  Settings loaded from TOML files written to tmp_path;
               ActionableError instances produced by the validator
        Never: Mock Settings or the TOML parser; write real TOML content
               to tmp_path to trigger each validation path, so the error
               message is verified against real validator output
    """

    def test_missing_boards_section_names_the_field_and_advises_adding_it(
        self, tmp_path: Path
    ) -> None:
        """
        When settings.toml has no [boards] section
        Then an ActionableError is raised whose message names 'boards'
        and whose suggestion tells the operator to add the section
        """
        # Given: a TOML file without a [boards] section
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f'global_rubric_path = "{rubric}"\n',
            encoding="utf-8",
        )

        # When / Then: load_settings raises ActionableError naming the field
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "boards" in error_msg.lower(), (
            f"Error should name the 'boards' field. Got: {error_msg}"
        )
        assert exc_info.value.suggestion is not None, (
            f"Error should include a suggestion. Got suggestion=None for: {error_msg}"
        )

    def test_weight_above_range_names_the_field_and_valid_range(
        self, tmp_path: Path
    ) -> None:
        """
        Given a scoring weight set to 1.5 (above the valid 0.0-1.0 range)
        When settings are loaded
        Then an ActionableError names the weight field and states the valid range
        """
        # Given: archetype_weight = 1.5 (above range)
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]

[scoring]
archetype_weight = 1.5

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then: validation error names the field
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "archetype_weight" in error_msg, (
            f"Error should name 'archetype_weight'. Got: {error_msg}"
        )
        assert exc_info.value.suggestion is not None, (
            "Error should include a suggestion with the valid range"
        )
        assert "0.0" in exc_info.value.suggestion or "1.0" in exc_info.value.suggestion, (
            f"Suggestion should mention the valid range. Got: {exc_info.value.suggestion}"
        )

    def test_weight_below_range_names_the_field_and_valid_range(
        self, tmp_path: Path
    ) -> None:
        """
        Given a scoring weight set to -0.1 (below the valid 0.0-1.0 range)
        When settings are loaded
        Then an ActionableError names the weight field and states the valid range
        """
        # Given: fit_weight = -0.1 (below range)
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]

[scoring]
fit_weight = -0.1

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then: validation error names the field
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "fit_weight" in error_msg, (
            f"Error should name 'fit_weight'. Got: {error_msg}"
        )
        assert exc_info.value.suggestion is not None, (
            "Error should include a suggestion with the valid range"
        )

    def test_missing_board_config_names_the_board_and_section_to_add(
        self, tmp_path: Path
    ) -> None:
        """
        Given a board listed in enabled but with no [boards.<name>] section
        When settings are loaded
        Then an ActionableError names the missing board and tells the operator
        which section to add
        """
        # Given: "missing_board" in enabled list but no config section for it
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["missing_board"]

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "missing_board" in error_msg, (
            f"Error should name the missing board 'missing_board'. Got: {error_msg}"
        )

    def test_ollama_url_without_scheme_suggests_adding_http_prefix(
        self, tmp_path: Path
    ) -> None:
        """
        Given ollama.base_url is "localhost:11434" (missing http:// scheme)
        When settings are loaded
        Then an ActionableError suggests adding the http:// prefix
        """
        # Given: base_url without scheme
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]

[ollama]
base_url = "localhost:11434"

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "base_url" in error_msg, (
            f"Error should name 'base_url'. Got: {error_msg}"
        )
        suggestion = exc_info.value.suggestion or ""
        assert "http://" in suggestion or "https://" in suggestion, (
            f"Suggestion should mention adding http:// prefix. Got: {suggestion}"
        )

    def test_all_config_errors_include_suggestion_and_recovery_steps(
        self, tmp_path: Path
    ) -> None:
        """
        When any configuration error is raised
        Then the error includes both a suggestion string and troubleshooting steps
        """
        # Given: a TOML that triggers a config error (missing boards section)
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text("# empty config\n", encoding="utf-8")

        # When: load_settings raises
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        # Then: suggestion and troubleshooting are populated
        err = exc_info.value
        assert err.suggestion is not None, (
            f"Every config error should have a suggestion. Got: {err}"
        )
        assert err.troubleshooting is not None, (
            f"Every config error should have troubleshooting. Got: {err}"
        )
        assert len(err.troubleshooting.steps) > 0, (
            f"Troubleshooting should have at least one step. Got: {err.troubleshooting}"
        )

    def test_valid_settings_load_without_error(self, tmp_path: Path) -> None:
        """
        When a valid settings.toml is loaded
        Then a Settings instance is returned without raising
        """
        # Given: a valid TOML file
        toml_path = _write_valid_toml(tmp_path)

        # When: settings are loaded
        settings = load_settings(toml_path)

        # Then: a Settings instance is returned
        assert isinstance(settings, Settings), (
            f"Expected a Settings instance, got {type(settings).__name__}"
        )
        assert "testboard" in settings.enabled_boards, (
            f"Expected 'testboard' in enabled_boards, got {settings.enabled_boards}"
        )

    def test_optional_fields_use_documented_defaults_when_absent(
        self, tmp_path: Path
    ) -> None:
        """
        When optional sections (scoring, ollama, output, chroma) are absent
        Then Settings uses the documented default values
        """
        # Given: a minimal valid TOML with no optional sections
        toml_path = _write_valid_toml(tmp_path)

        # When: settings are loaded
        settings = load_settings(toml_path)

        # Then: defaults are applied
        assert settings.ollama.base_url == "http://localhost:11434", (
            f"ollama.base_url should default to 'http://localhost:11434', "
            f"got {settings.ollama.base_url!r}"
        )
        assert settings.ollama.llm_model == "mistral:7b", (
            f"ollama.llm_model should default to 'mistral:7b', "
            f"got {settings.ollama.llm_model!r}"
        )
        assert settings.output.default_format == "markdown", (
            f"output.default_format should default to 'markdown', "
            f"got {settings.output.default_format!r}"
        )

    def test_comp_weight_defaults_to_zero_point_fifteen(
        self, tmp_path: Path
    ) -> None:
        """
        When the scoring section omits comp_weight
        Then it defaults to 0.15
        """
        # Given: valid TOML without comp_weight specified
        toml_path = _write_valid_toml(tmp_path)

        # When: settings are loaded
        settings = load_settings(toml_path)

        # Then: comp_weight is 0.15
        assert settings.scoring.comp_weight == pytest.approx(0.15), (
            f"comp_weight should default to 0.15, got {settings.scoring.comp_weight}"
        )

    def test_culture_weight_defaults_to_zero_point_two(
        self, tmp_path: Path
    ) -> None:
        """
        When the scoring section omits culture_weight
        Then it defaults to 0.2
        """
        # Given: valid TOML without culture_weight specified
        toml_path = _write_valid_toml(tmp_path)

        # When: settings are loaded
        settings = load_settings(toml_path)

        # Then: culture_weight is 0.2
        assert settings.scoring.culture_weight == pytest.approx(0.2), (
            f"culture_weight should default to 0.2, got {settings.scoring.culture_weight}"
        )

    def test_negative_weight_defaults_to_zero_point_four(
        self, tmp_path: Path
    ) -> None:
        """
        When the scoring section omits negative_weight
        Then it defaults to 0.4
        """
        # Given: valid TOML without negative_weight specified
        toml_path = _write_valid_toml(tmp_path)

        # When: settings are loaded
        settings = load_settings(toml_path)

        # Then: negative_weight is 0.4
        assert settings.scoring.negative_weight == pytest.approx(0.4), (
            f"negative_weight should default to 0.4, got {settings.scoring.negative_weight}"
        )

    def test_base_salary_defaults_to_220000(self, tmp_path: Path) -> None:
        """
        When the scoring section omits base_salary
        Then it defaults to 220,000
        """
        # Given: valid TOML without base_salary specified
        toml_path = _write_valid_toml(tmp_path)

        # When: settings are loaded
        settings = load_settings(toml_path)

        # Then: base_salary is 220,000
        assert settings.scoring.base_salary == pytest.approx(220_000.0), (
            f"base_salary should default to 220000, got {settings.scoring.base_salary}"
        )

    def test_negative_base_salary_names_the_field_and_constraint(
        self, tmp_path: Path
    ) -> None:
        """
        Given base_salary is set to -50000 (negative)
        When settings are loaded
        Then an ActionableError names the base_salary field and states it must be positive
        """
        # Given: negative base_salary
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]

[scoring]
base_salary = -50000

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "base_salary" in error_msg, (
            f"Error should name 'base_salary'. Got: {error_msg}"
        )

    def test_culture_weight_above_range_names_field_and_valid_range(
        self, tmp_path: Path
    ) -> None:
        """
        Given culture_weight is set to 1.5 (above the 0.0-1.0 range)
        When settings are loaded
        Then an ActionableError names the culture_weight field
        """
        # Given: culture_weight above range
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]

[scoring]
culture_weight = 1.5

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "culture_weight" in error_msg, (
            f"Error should name 'culture_weight'. Got: {error_msg}"
        )

    def test_negative_weight_above_range_names_field_and_valid_range(
        self, tmp_path: Path
    ) -> None:
        """
        Given negative_weight is set to 2.0 (above the 0.0-1.0 range)
        When settings are loaded
        Then an ActionableError names the negative_weight field
        """
        # Given: negative_weight above range
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]

[scoring]
negative_weight = 2.0

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "negative_weight" in error_msg, (
            f"Error should name 'negative_weight'. Got: {error_msg}"
        )

    def test_missing_global_rubric_path_produces_actionable_error(
        self, tmp_path: Path
    ) -> None:
        """
        Given global_rubric_path points to a file that does not exist
        When settings are loaded
        Then an ActionableError is raised with a message about the missing rubric
        """
        # Given: TOML with a nonexistent global_rubric_path
        #        (top-level key must appear before any [section] header in TOML)
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            """\
global_rubric_path = "/nonexistent/path/rubric.toml"

[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]
""",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "rubric" in error_msg.lower(), (
            f"Error should mention the rubric file. Got: {error_msg}"
        )
        assert exc_info.value.suggestion is not None, (
            "Error should include a suggestion for creating or fixing the rubric path"
        )

    def test_comp_bands_default_when_absent_from_config(
        self, tmp_path: Path
    ) -> None:
        """
        When the scoring section omits comp_bands
        Then it defaults to the documented default bands
        """
        # Given: valid TOML without comp_bands specified
        toml_path = _write_valid_toml(tmp_path)

        # When: settings are loaded
        settings = load_settings(toml_path)

        # Then: comp_bands matches the documented defaults
        bands = settings.scoring.comp_bands
        assert len(bands) == 4, (
            f"Default comp_bands should have 4 entries, got {len(bands)}"
        )
        assert bands[0].ratio == pytest.approx(1.0), (
            f"First band ratio should be 1.0, got {bands[0].ratio}"
        )
        assert bands[0].score == pytest.approx(1.0), (
            f"First band score should be 1.0, got {bands[0].score}"
        )
        assert bands[-1].ratio == pytest.approx(0.68), (
            f"Last band ratio should be 0.68, got {bands[-1].ratio}"
        )
        assert bands[-1].score == pytest.approx(0.0), (
            f"Last band score should be 0.0, got {bands[-1].score}"
        )

    def test_missing_comp_score_defaults_to_zero_point_five(
        self, tmp_path: Path
    ) -> None:
        """
        When the scoring section omits missing_comp_score
        Then it defaults to 0.5
        """
        # Given: valid TOML without missing_comp_score specified
        toml_path = _write_valid_toml(tmp_path)

        # When: settings are loaded
        settings = load_settings(toml_path)

        # Then: missing_comp_score is 0.5
        assert settings.scoring.missing_comp_score == pytest.approx(0.5), (
            f"missing_comp_score should default to 0.5, "
            f"got {settings.scoring.missing_comp_score}"
        )

    def test_comp_bands_must_be_descending_by_ratio(
        self, tmp_path: Path
    ) -> None:
        """
        Given comp_bands with non-descending ratios
        When settings are loaded
        Then an ActionableError names comp_bands and states they must be descending
        """
        # Given: comp_bands in ascending order (invalid)
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]

[scoring]
comp_bands = [
  {{ ratio = 0.68, score = 0.0 }},
  {{ ratio = 0.77, score = 0.4 }},
  {{ ratio = 0.90, score = 0.7 }},
  {{ ratio = 1.00, score = 1.0 }},
]

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "comp_bands" in error_msg, (
            f"Error should name 'comp_bands'. Got: {error_msg}"
        )

    def test_comp_band_ratio_out_of_range_produces_error(
        self, tmp_path: Path
    ) -> None:
        """
        Given a comp_band with ratio < 0
        When settings are loaded
        Then an ActionableError names comp_bands and the invalid ratio
        """
        # Given: comp_band with negative ratio
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]

[scoring]
comp_bands = [
  {{ ratio = 1.00, score = 1.0 }},
  {{ ratio = -0.10, score = 0.0 }},
]

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "comp_bands" in error_msg, (
            f"Error should name 'comp_bands'. Got: {error_msg}"
        )

    def test_comp_band_score_out_of_range_produces_error(
        self, tmp_path: Path
    ) -> None:
        """
        Given a comp_band with score > 1.0
        When settings are loaded
        Then an ActionableError names comp_bands and the invalid score
        """
        # Given: comp_band with score above 1.0
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]

[scoring]
comp_bands = [
  {{ ratio = 1.00, score = 1.5 }},
  {{ ratio = 0.68, score = 0.0 }},
]

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "comp_bands" in error_msg, (
            f"Error should name 'comp_bands'. Got: {error_msg}"
        )

    def test_missing_settings_file_names_path_and_suggests_creating_it(
        self, tmp_path: Path
    ) -> None:
        """
        Given a path to a settings file that does not exist
        When settings are loaded
        Then an ActionableError names the path and suggests creating the file
        """
        # Given: nonexistent file path
        nonexistent = tmp_path / "does_not_exist.toml"

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(nonexistent)

        error_msg = str(exc_info.value)
        assert "does_not_exist.toml" in error_msg, (
            f"Error should name the missing file. Got: {error_msg}"
        )
        assert exc_info.value.suggestion is not None, (
            f"Error should include a suggestion. Got suggestion=None for: {error_msg}"
        )

    def test_malformed_toml_produces_parse_error_with_syntax_detail(
        self, tmp_path: Path
    ) -> None:
        """
        Given a settings file with invalid TOML syntax
        When settings are loaded
        Then an ActionableError of type PARSE is raised with syntax detail
        """
        # Given: broken TOML syntax
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            "[boards\nenabled = broken\n",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert exc_info.value.suggestion is not None, (
            f"Parse error should include a suggestion. Got: {error_msg}"
        )

    def test_boards_enabled_as_empty_list_produces_actionable_error(
        self, tmp_path: Path
    ) -> None:
        """
        Given boards.enabled is an empty list
        When settings are loaded
        Then an ActionableError names boards.enabled
        """
        # Given: enabled = []
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = []

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "boards.enabled" in error_msg, (
            f"Error should name 'boards.enabled'. Got: {error_msg}"
        )

    def test_board_section_must_be_table_not_scalar(
        self, tmp_path: Path
    ) -> None:
        """
        Given a board section that is a scalar instead of a table
        When settings are loaded
        Then an ActionableError names the board and says it must be a table
        """
        # Given: board config as scalar
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["badboard"]
badboard = "not a table"

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "badboard" in error_msg, (
            f"Error should name the board 'badboard'. Got: {error_msg}"
        )
        assert "table" in error_msg.lower(), (
            f"Error should mention 'table'. Got: {error_msg}"
        )

    def test_overnight_boards_are_parsed_when_config_sections_exist(
        self, tmp_path: Path
    ) -> None:
        """
        Given an overnight board with its own [boards.<name>] section
        When settings are loaded
        Then the overnight board config is parsed into settings.boards
        """
        # Given: overnight board with config section
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["dayboard"]
overnight_boards = ["nightboard"]

[boards.dayboard]
searches = ["https://dayboard.com/search"]

[boards.nightboard]
searches = ["https://nightboard.com/search"]
max_pages = 5

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When
        settings = load_settings(toml_path)

        # Then: overnight board is in boards dict
        assert "nightboard" in settings.boards, (
            f"Expected 'nightboard' in settings.boards. Got: {list(settings.boards.keys())}"
        )
        assert settings.boards["nightboard"].max_pages == 5, (
            f"Expected max_pages=5 for nightboard. Got: {settings.boards['nightboard'].max_pages}"
        )

    def test_non_dict_scoring_section_defaults_to_empty(
        self, tmp_path: Path
    ) -> None:
        """
        Given scoring section is a non-dict value
        When settings are loaded
        Then scoring defaults are used (no crash)
        """
        # Given: scoring = "not a dict"
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
scoring = "not a dict"
global_rubric_path = "{rubric}"

[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]
""",
            encoding="utf-8",
        )

        # When
        settings = load_settings(toml_path)

        # Then: scoring uses defaults
        assert settings.scoring.archetype_weight == pytest.approx(0.5), (
            f"Expected default archetype_weight=0.5. Got: {settings.scoring.archetype_weight}"
        )

    def test_non_dict_ollama_section_defaults_to_empty(
        self, tmp_path: Path
    ) -> None:
        """
        Given ollama section is a non-dict value
        When settings are loaded
        Then ollama defaults are used (no crash)
        """
        # Given: ollama = "not a dict"
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
ollama = "not a dict"
global_rubric_path = "{rubric}"

[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]
""",
            encoding="utf-8",
        )

        # When
        settings = load_settings(toml_path)

        # Then: ollama uses defaults
        assert settings.ollama.base_url == "http://localhost:11434", (
            f"Expected default base_url. Got: {settings.ollama.base_url}"
        )

    def test_non_dict_output_section_defaults_to_empty(
        self, tmp_path: Path
    ) -> None:
        """
        Given output section is a non-dict value
        When settings are loaded
        Then output defaults are used (no crash)
        """
        # Given: output = "not a dict"
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
output = "not a dict"
global_rubric_path = "{rubric}"

[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]
""",
            encoding="utf-8",
        )

        # When
        settings = load_settings(toml_path)

        # Then: output uses defaults
        assert settings.output.default_format == "markdown", (
            f"Expected default format='markdown'. Got: {settings.output.default_format}"
        )

    def test_non_dict_chroma_section_defaults_to_empty(
        self, tmp_path: Path
    ) -> None:
        """
        Given chroma section is a non-dict value
        When settings are loaded
        Then chroma defaults are used (no crash)
        """
        # Given: chroma = "not a dict"
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
chroma = "not a dict"
global_rubric_path = "{rubric}"

[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]
""",
            encoding="utf-8",
        )

        # When
        settings = load_settings(toml_path)

        # Then: chroma uses defaults
        assert settings.chroma.persist_dir == "./data/chroma_db", (
            f"Expected default persist_dir. Got: {settings.chroma.persist_dir}"
        )

    def test_boards_section_without_enabled_key_produces_error(
        self, tmp_path: Path
    ) -> None:
        """
        Given a [boards] section exists but has no 'enabled' key
        When settings are loaded
        Then an ActionableError names boards.enabled as the missing field
        """
        # Given: [boards] without enabled
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
overnight_boards = ["someboard"]

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "enabled" in error_msg, (
            f"Error should name the missing 'enabled' field. Got: {error_msg}"
        )

    def test_comp_bands_as_non_list_produces_parse_error(
        self, tmp_path: Path
    ) -> None:
        """
        Given comp_bands is a string instead of a list
        When settings are loaded
        Then an ActionableError mentions comp_bands and the expected type
        """
        # Given: comp_bands as string
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]

[scoring]
comp_bands = "not a list"

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "comp_bands" in error_msg, (
            f"Error should name 'comp_bands'. Got: {error_msg}"
        )

    def test_comp_band_entry_must_be_table_not_scalar(
        self, tmp_path: Path
    ) -> None:
        """
        Given comp_bands contains scalar entries instead of tables
        When settings are loaded
        Then an ActionableError names the entry and says it must be a table
        """
        # Given: comp_bands with scalar entries
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]

[scoring]
comp_bands = [1, 2, 3]

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "comp_bands" in error_msg, (
            f"Error should name 'comp_bands'. Got: {error_msg}"
        )
        assert "table" in error_msg.lower(), (
            f"Error should mention 'table'. Got: {error_msg}"
        )

    def test_comp_band_entry_missing_required_keys_produces_error(
        self, tmp_path: Path
    ) -> None:
        """
        Given a comp_band entry with ratio but no score
        When settings are loaded
        Then an ActionableError names the entry and lists missing keys
        """
        # Given: comp_band entry missing 'score' key
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]

[scoring]
comp_bands = [
  {{ ratio = 1.0 }},
]

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "comp_bands" in error_msg, (
            f"Error should name 'comp_bands'. Got: {error_msg}"
        )

    def test_empty_comp_bands_list_produces_validation_error(
        self, tmp_path: Path
    ) -> None:
        """
        Given comp_bands is an empty list
        When settings are loaded
        Then an ActionableError says at least one band is required
        """
        # Given: comp_bands = []
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]

[scoring]
comp_bands = []

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When / Then
        with pytest.raises(ActionableError) as exc_info:
            load_settings(toml_path)

        error_msg = str(exc_info.value)
        assert "comp_bands" in error_msg, (
            f"Error should name 'comp_bands'. Got: {error_msg}"
        )
        assert "empty" in error_msg.lower() or "at least one" in error_msg.lower(), (
            f"Error should mention the list is empty. Got: {error_msg}"
        )

    def test_non_numeric_weight_value_falls_back_to_default(
        self, tmp_path: Path
    ) -> None:
        """
        Given a scoring weight is set to a non-numeric string in TOML
        When settings are loaded
        Then the weight falls back to its documented default value
        """
        # Given: archetype_weight = "not_a_number" (string instead of float)
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]

[scoring]
archetype_weight = "not_a_number"

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When: settings are loaded
        settings = load_settings(toml_path)

        # Then: archetype_weight falls back to default 0.5
        assert settings.scoring.archetype_weight == pytest.approx(0.5), (
            f"Expected default 0.5 for non-numeric weight. "
            f"Got {settings.scoring.archetype_weight}"
        )

    def test_boolean_integer_field_falls_back_to_default(
        self, tmp_path: Path
    ) -> None:
        """
        Given an integer field is set to a boolean in TOML
        When settings are loaded
        Then the field falls back to its documented default
        (because Python booleans are int subclasses, the guard rejects them)
        """
        # Given: max_pages = true (boolean instead of int)
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]
max_pages = true

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When: settings are loaded
        settings = load_settings(toml_path)

        # Then: max_pages falls back to default 3
        board = settings.boards["testboard"]
        assert board.max_pages == 3, (
            f"Expected default 3 for boolean max_pages. Got {board.max_pages}"
        )

    def test_empty_optional_string_treated_as_none(
        self, tmp_path: Path
    ) -> None:
        """
        Given an optional string field is set to an empty string in TOML
        When settings are loaded
        Then the field is treated as None
        """
        # Given: browser_channel = ""
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text("[rubric]\n", encoding="utf-8")
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text(
            f"""\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://testboard.com/search"]
browser_channel = ""

global_rubric_path = "{rubric}"
""",
            encoding="utf-8",
        )

        # When: settings are loaded
        settings = load_settings(toml_path)

        # Then: browser_channel is None
        board = settings.boards["testboard"]
        assert board.browser_channel is None, (
            f"Expected None for empty optional string. Got {board.browser_channel!r}"
        )
