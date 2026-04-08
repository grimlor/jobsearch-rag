"""
Configuration validation tests.

Maps to BDD spec: TestSettingsValidation

Tests verify that configuration errors are caught at startup with
actionable messages, valid configs load successfully, and optional
fields use documented defaults.

Spec classes:
    TestSettingsValidation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from jobsearch_rag.config import load_settings
from jobsearch_rag.errors import ActionableError, ErrorType

# A minimal valid settings TOML for tests
_VALID_SETTINGS = """\
resume_path = "data/resume.md"
archetypes_path = "config/role_archetypes.toml"
global_rubric_path = "config/global_rubric.toml"

[boards]
enabled = ["testboard"]
session_storage_dir = "data"

[boards.testboard]
searches = ["https://example.org/search"]
max_pages = 2
headless = true
rate_limit_range = [1.5, 3.5]

[scoring]
archetype_weight = 0.5
fit_weight = 0.3
history_weight = 0.2
comp_weight = 0.15
negative_weight = 0.4
culture_weight = 0.2
base_salary = 220000
disqualify_on_llm_flag = true
min_score_threshold = 0.45
missing_comp_score = 0.5
chunk_overlap = 2000
dedup_similarity_threshold = 0.95

[[scoring.comp_bands]]
ratio = 1.0
score = 1.0

[[scoring.comp_bands]]
ratio = 0.90
score = 0.7

[[scoring.comp_bands]]
ratio = 0.77
score = 0.4

[[scoring.comp_bands]]
ratio = 0.68
score = 0.0

[ollama]
base_url = "http://localhost:11434"
llm_model = "mistral:7b"
embed_model = "nomic-embed-text"
slow_llm_threshold_ms = 30000
classify_system_prompt = "You are a job listing classifier. Respond concisely with your classification."
max_retries = 3
base_delay = 1.0
max_embed_chars = 8000
head_ratio = 0.6
retryable_status_codes = [408, 429, 500, 502, 503, 504]

[output]
default_format = "markdown"
output_dir = "./output"
open_top_n = 5
jd_dir = "output/jds"
decisions_dir = "data/decisions"
log_dir = "data/logs"
eval_history_path = "data/eval_history.jsonl"

[chroma]
persist_dir = "./data/chroma_db"

[adapters]
cdp_timeout = 15.0

[adapters.browser_paths]
msedge = ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]

[security]
screen_prompt = "Review the following job description text."
"""


def _write_settings(tmpdir: str, content: str) -> Path:
    """Write settings content to a temp file and return the path."""
    path = Path(tmpdir) / "settings.toml"
    path.write_text(content, encoding="utf-8")
    return path


class TestSettingsValidation:
    """
    REQUIREMENT: Configuration errors tell the operator exactly what to fix.

    WHO: The operator who misconfigured settings.toml
    WHAT: (1) The system reports that the `boards` section is missing and tells the operator how to add it.
          (2) The system rejects an out-of-range `archetype_weight` value and states the valid range with guidance.
          (3) The system rejects an out-of-range `fit_weight` value and states the valid range with guidance.
          (4) The system reports the missing enabled board configuration and tells the operator to add that board section.
          (5) The system rejects an Ollama URL without a scheme and suggests adding the `http://` prefix.
          (6) The system loads a valid settings file and makes all parsed fields accessible.
          (7) The system applies documented default values for optional fields that are absent.
          (8) The system parses `browser_channel` from a board configuration and exposes it on `BoardConfig`.
          (9) The system defaults `comp_weight` to `0.15` when it is absent.
          (10) The system defaults `base_salary` to `220000` when it is absent.
          (11) The system rejects a negative `base_salary` value and states that it must be positive.
          (12) The system reports that the settings file is not found and tells the operator to create it.
          (13) The system raises a parse error for malformed TOML syntax and suggests fixing the file.
          (14) The system rejects an empty `boards.enabled` list and tells the operator to add board names.
          (15) The system reports a board section that is not a TOML table and tells the operator to define it as a table.
          (16) The system parses overnight board configurations into `BoardConfig` entries.
          (17) The system uses default scoring values when the `scoring` section is not a table.
          (18) The system uses default Ollama values when the `ollama` section is not a table.
          (19) The system uses default output values when the `output` section is not a table.
          (20) The system uses default Chroma values when the `chroma` section is not a table.
          (21) The system reports that the `enabled` field is missing from `boards` and tells the operator to add it.
          (22) The system defaults `negative_weight` to `0.4` when it is absent.
          (23) The system rejects an out-of-range `negative_weight` value above the limit and states the valid range with guidance.
          (24) The system rejects an out-of-range `negative_weight` value below the limit and states the valid range with guidance.
          (25) The system defaults `global_rubric_path` to `config/global_rubric.toml` when it is absent.
          (26) The system reports a nonexistent `global_rubric_path` file and provides recovery guidance.
          (27) The system defaults `culture_weight` to `0.2` when it is absent.
          (28) The system rejects an out-of-range `culture_weight` value above the limit and states the valid range with guidance.
          (29) The system rejects an out-of-range `culture_weight` value below the limit and states the valid range with guidance.
          (30) The system uses the enabled board's config when an overnight board is also in enabled boards.
          (31) The system skips an overnight board whose section is not a TOML table.
          (32) The system defaults `disqualifier.system_prompt` to `None` when the `[disqualifier]` section is absent.
          (33) The system defaults `security.screen_prompt` to the documented default when the `[security]` section is absent.
          (34) The system defaults `ollama.classify_system_prompt` to the documented default when the field is absent.
          (35) The system rejects an empty `[disqualifier] system_prompt` and tells the operator to remove the key or provide content.
          (36) The system rejects an empty `[security] screen_prompt` and tells the operator to remove the key or provide content.
          (37) The system rejects an empty `[ollama] classify_system_prompt` and tells the operator to remove the key or provide content.
          (38) The system ignores a `[disqualifier]` value that is not a TOML table and defaults to no disqualifier config.
    WHY: A mid-run config failure after 10 minutes of browser work is
         far more costly than a startup validation failure — and an error
         that says "validation failed" without telling the operator what
         to fix is nearly as costly as a mid-run crash

    MOCK BOUNDARY:
        Mock:  nothing — uses real filesystem via tempfile.TemporaryDirectory
        Real:  load_settings, TOML parsing, validation, ActionableError
        Never: Patch config internals — exercise the real validation stack
    """

    def test_missing_boards_section_names_the_field_and_tells_operator_to_add_it(self) -> None:
        """
        Given a settings file missing the [boards] section
        When load_settings is called
        Then the error names 'boards' and includes recovery guidance
        """
        # Given: settings TOML with no [boards] section
        bad_toml = """\
[scoring]
archetype_weight = 0.5
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When: the settings are loaded
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            # Then: error identifies the missing section with recovery guidance
            err = exc_info.value
            assert err.error_type == ErrorType.CONFIG, (
                f"Expected CONFIG error, got {err.error_type}"
            )
            assert "boards" in err.error.lower(), f"Error should name 'boards'. Got: {err.error}"
            assert err.suggestion is not None, "Missing suggestion"
            assert "boards" in err.suggestion.lower(), (
                f"Suggestion should mention 'boards'. Got: {err.suggestion}"
            )
            assert err.troubleshooting is not None, "Missing troubleshooting"
            assert len(err.troubleshooting.steps) > 0, "Troubleshooting has no steps"

    def test_weight_above_range_names_the_field_and_valid_range(self) -> None:
        """
        Given archetype_weight is set to 1.5 (above valid range)
        When load_settings is called
        Then the error names 'archetype_weight' and includes guidance
        """
        # Given: archetype_weight exceeds valid range [0.0, 1.0]
        bad_toml = _VALID_SETTINGS.replace("archetype_weight = 0.5", "archetype_weight = 1.5")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When: the settings are loaded
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            # Then: error identifies the out-of-range field
            err = exc_info.value
            assert err.error_type == ErrorType.VALIDATION, (
                f"Expected VALIDATION error, got {err.error_type}"
            )
            assert "archetype_weight" in err.error, (
                f"Error should name 'archetype_weight'. Got: {err.error}"
            )
            assert err.suggestion is not None, "Missing suggestion"
            assert err.troubleshooting is not None, "Missing troubleshooting"

    def test_weight_below_range_names_the_field_and_valid_range(self) -> None:
        """
        Given fit_weight is set to -0.1 (below valid range)
        When load_settings is called
        Then the error names 'fit_weight' and includes guidance
        """
        # Given: fit_weight is negative (below valid range [0.0, 1.0])
        bad_toml = _VALID_SETTINGS.replace("fit_weight = 0.3", "fit_weight = -0.1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When: the settings are loaded
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            # Then: error identifies the out-of-range field
            err = exc_info.value
            assert err.error_type == ErrorType.VALIDATION, (
                f"Expected VALIDATION error, got {err.error_type}"
            )
            assert "fit_weight" in err.error, f"Error should name 'fit_weight'. Got: {err.error}"
            assert err.suggestion is not None, "Missing suggestion"
            assert err.troubleshooting is not None, "Missing troubleshooting"

    def test_missing_board_config_names_the_board_and_section_to_add(self) -> None:
        """
        Given a board listed in enabled that has no config section
        When load_settings is called
        Then the error names the missing board and tells operator to add it
        """
        # Given: 'nonexistent_board' in enabled but no [boards.nonexistent_board]
        bad_toml = """\
[boards]
enabled = ["nonexistent_board"]

[scoring]
archetype_weight = 0.5
fit_weight = 0.3
history_weight = 0.2

[ollama]
base_url = "http://localhost:11434"

[chroma]
persist_dir = "./data/chroma_db"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When: the settings are loaded
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            # Then: error names the missing board
            err = exc_info.value
            assert err.error_type == ErrorType.CONFIG, (
                f"Expected CONFIG error, got {err.error_type}"
            )
            assert "nonexistent_board" in err.error, (
                f"Error should name 'nonexistent_board'. Got: {err.error}"
            )
            assert err.suggestion is not None, "Missing suggestion"
            assert "nonexistent_board" in err.suggestion, (
                f"Suggestion should name the board. Got: {err.suggestion}"
            )
            assert err.troubleshooting is not None, "Missing troubleshooting"

    def test_ollama_url_without_scheme_suggests_adding_http_prefix(self) -> None:
        """
        Given the Ollama URL lacks an http:// scheme
        When load_settings is called
        Then the error mentions 'scheme' and suggests adding the prefix
        """
        # Given: base_url missing scheme
        bad_toml = _VALID_SETTINGS.replace(
            'base_url = "http://localhost:11434"',
            'base_url = "localhost:11434"',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When: the settings are loaded
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            # Then: error identifies the scheme issue
            err = exc_info.value
            assert err.error_type == ErrorType.VALIDATION, (
                f"Expected VALIDATION error, got {err.error_type}"
            )
            assert "scheme" in err.error.lower(), (
                f"Error should mention 'scheme'. Got: {err.error}"
            )
            assert err.suggestion is not None, "Missing suggestion"
            assert err.troubleshooting is not None, "Missing troubleshooting"

    def test_valid_settings_load_without_error(self) -> None:
        """
        Given a complete, valid settings.toml file
        When the settings are loaded
        Then all fields are parsed and accessible
        """
        # Given: a complete, valid settings file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, _VALID_SETTINGS)

            # When: settings are loaded
            settings = load_settings(path)

            # Then: key fields are accessible
            assert settings.enabled_boards == ["testboard"], (
                f"Expected ['testboard'], got {settings.enabled_boards}"
            )
            assert settings.scoring.archetype_weight == 0.5, (
                f"Expected archetype_weight=0.5, got {settings.scoring.archetype_weight}"
            )
            assert settings.ollama.base_url == "http://localhost:11434", (
                f"Expected base_url='http://localhost:11434', got {settings.ollama.base_url}"
            )
            assert settings.chroma.persist_dir == "./data/chroma_db", (
                f"Expected persist_dir='./data/chroma_db', got {settings.chroma.persist_dir}"
            )

    def test_minimal_toml_raises_actionable_error_for_missing_required_section(self) -> None:
        """
        Given a minimal settings file with only boards config
        When load_settings is called
        Then ActionableError is raised for the first missing required section
        """
        # Given: minimal TOML with only boards config
        minimal_toml = """\
resume_path = "data/resume.md"
archetypes_path = "config/role_archetypes.toml"
global_rubric_path = "config/global_rubric.toml"

[boards]
enabled = ["testboard"]
session_storage_dir = "data"

[boards.testboard]
searches = ["https://example.org/search"]
rate_limit_range = [1.5, 3.5]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, minimal_toml)

            # When / Then: load raises ActionableError for missing required section
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "scoring" in str(exc_info.value).lower(), (
                f"Error should name missing section. Got: {exc_info.value}"
            )

    def test_browser_channel_is_parsed_from_board_config(self) -> None:
        """
        Given a board config with browser_channel = 'msedge'
        When load_settings is called
        Then the browser_channel value is accessible on BoardConfig
        """
        # Given: settings with browser_channel in the board section
        channel_toml = _VALID_SETTINGS.replace(
            "headless = true",
            'headless = true\nbrowser_channel = "msedge"',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, channel_toml)

            # When: settings are loaded
            settings = load_settings(path)

            # Then: browser_channel is parsed
            assert settings.boards["testboard"].browser_channel == "msedge", (
                f"Expected browser_channel='msedge', got {settings.boards['testboard'].browser_channel}"
            )

    def test_missing_comp_weight_raises_actionable_error(self) -> None:
        """
        Given comp_weight is not specified in settings.toml
        When load_settings is called
        Then ActionableError is raised naming the missing field
        """
        # Given: TOML without comp_weight
        bad_toml = _VALID_SETTINGS.replace("comp_weight = 0.15\n", "")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When / Then: load raises ActionableError
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "comp_weight" in str(exc_info.value).lower(), (
                f"Error should name missing field. Got: {exc_info.value}"
            )

    def test_missing_base_salary_raises_actionable_error(self) -> None:
        """
        Given base_salary is not specified in settings.toml
        When load_settings is called
        Then ActionableError is raised naming the missing field
        """
        # Given: TOML without base_salary
        bad_toml = _VALID_SETTINGS.replace("base_salary = 220000\n", "")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When / Then: load raises ActionableError
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "base_salary" in str(exc_info.value).lower(), (
                f"Error should name missing field. Got: {exc_info.value}"
            )

    def test_negative_base_salary_names_the_field_and_constraint(self) -> None:
        """
        Given base_salary is set to -50000
        When load_settings is called
        Then the error names 'base_salary' and the positive-number constraint
        """
        # Given: negative base_salary in scoring section
        bad_toml = _VALID_SETTINGS.replace(
            "base_salary = 220000",
            "base_salary = -50000",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When: the settings are loaded
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            # Then: error identifies the invalid field
            err = exc_info.value
            assert err.error_type == ErrorType.VALIDATION, (
                f"Expected VALIDATION error, got {err.error_type}"
            )
            assert "base_salary" in err.error, f"Error should name 'base_salary'. Got: {err.error}"
            assert err.suggestion is not None, "Missing suggestion"
            assert err.troubleshooting is not None, "Missing troubleshooting"

    def test_missing_file_tells_operator_to_create_settings(self) -> None:
        """
        Given the settings file does not exist
        When load_settings is called
        Then the error says 'not found' with creation guidance
        """
        # Given: a nonexistent settings path
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "nonexistent.toml"

            # When: load is attempted
            with pytest.raises(ActionableError) as exc_info:
                load_settings(missing)

            # Then: error identifies the missing file
            err = exc_info.value
            assert err.error_type == ErrorType.CONFIG, (
                f"Expected CONFIG error, got {err.error_type}"
            )
            assert "not found" in err.error.lower(), (
                f"Error should say 'not found'. Got: {err.error}"
            )
            assert err.suggestion is not None, "Missing suggestion"
            assert err.troubleshooting is not None, "Missing troubleshooting"

    def test_malformed_toml_identifies_syntax_error_and_suggests_fix(self) -> None:
        """
        Given the settings file contains invalid TOML syntax
        When load_settings is called
        Then a PARSE error is raised with a fix suggestion
        """
        # Given: syntactically broken TOML
        bad_toml = "[boards\nenabled = broken"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When: load is attempted
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            # Then: PARSE error with guidance
            err = exc_info.value
            assert err.error_type == ErrorType.PARSE, f"Expected PARSE error, got {err.error_type}"
            assert err.suggestion is not None, "Missing suggestion"
            assert err.troubleshooting is not None, "Missing troubleshooting"

    def test_empty_enabled_boards_tells_operator_to_add_board_names(self) -> None:
        """
        Given boards.enabled is an empty list
        When load_settings is called
        Then the error names 'boards.enabled' and tells operator to add names
        """
        # Given: empty enabled list
        bad_toml = _VALID_SETTINGS.replace('enabled = ["testboard"]', "enabled = []")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When: the settings are loaded
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            # Then: error names the empty field
            err = exc_info.value
            assert err.error_type == ErrorType.CONFIG, (
                f"Expected CONFIG error, got {err.error_type}"
            )
            assert "boards.enabled" in err.error, (
                f"Error should name 'boards.enabled'. Got: {err.error}"
            )
            assert err.suggestion is not None, "Missing suggestion"
            assert err.troubleshooting is not None, "Missing troubleshooting"

    def test_board_section_not_a_table_tells_operator_to_define_as_table(self) -> None:
        """
        Given a board value is a scalar instead of a TOML table
        When load_settings is called
        Then the error mentions 'table' and tells operator to restructure
        """
        # Given: board defined as a string instead of a table
        bad_toml = """\
[boards]
enabled = ["testboard"]
testboard = "not a table"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When: the settings are loaded
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            # Then: error mentions the table requirement
            err = exc_info.value
            assert err.error_type == ErrorType.CONFIG, (
                f"Expected CONFIG error, got {err.error_type}"
            )
            assert "table" in err.error.lower(), f"Error should mention 'table'. Got: {err.error}"
            assert err.suggestion is not None, "Missing suggestion"
            assert err.troubleshooting is not None, "Missing troubleshooting"

    def test_overnight_board_configs_are_parsed(self) -> None:
        """
        Given a settings file with overnight_boards that have config sections
        When load_settings is called
        Then overnight boards have parsed BoardConfig entries
        """
        # Given: settings with an overnight board config
        overnight_toml = _VALID_SETTINGS.replace(
            '[boards]\nenabled = ["testboard"]',
            '[boards]\nenabled = ["testboard"]\novernight_boards = ["nightboard"]',
        ).replace(
            "rate_limit_range = [1.5, 3.5]",
            "rate_limit_range = [1.5, 3.5]\n\n[boards.nightboard]\nsearches = "
            '["https://night.org/search"]\nmax_pages = 5\nheadless = false'
            "\nrate_limit_range = [2.0, 4.0]",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, overnight_toml)

            # When: settings are loaded
            settings = load_settings(path)

            # Then: overnight board config is accessible
            assert "nightboard" in settings.boards, (
                f"Expected 'nightboard' in boards. Got: {list(settings.boards.keys())}"
            )
            assert settings.boards["nightboard"].max_pages == 5, (
                f"Expected max_pages=5, got {settings.boards['nightboard'].max_pages}"
            )
            assert settings.boards["nightboard"].headless is False, (
                f"Expected headless=False, got {settings.boards['nightboard'].headless}"
            )

    def test_overnight_board_already_in_enabled_uses_enabled_config(self) -> None:
        """
        Given an overnight board that is also listed in enabled boards
        When load_settings is called
        Then the enabled board's config is used and no duplicate entry is created.
        """
        # Given: same board in both enabled and overnight lists
        overlap_toml = _VALID_SETTINGS.replace(
            '[boards]\nenabled = ["testboard"]',
            '[boards]\nenabled = ["testboard"]\novernight_boards = ["testboard"]',
        ).replace("max_pages = 2", "max_pages = 7")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, overlap_toml)

            # When: settings are loaded
            settings = load_settings(path)

            # Then: the enabled board config is preserved (max_pages from enabled config)
            assert "testboard" in settings.boards, (
                f"Expected 'testboard' in boards. Got: {list(settings.boards.keys())}"
            )
            assert settings.boards["testboard"].max_pages == 7, (
                f"Expected max_pages=7 from enabled config, got "
                f"{settings.boards['testboard'].max_pages}"
            )

    def test_overnight_board_non_dict_section_is_skipped(self) -> None:
        """
        Given an overnight board whose section in [boards] is a scalar, not a table
        When load_settings is called
        Then the non-dict section is skipped and no BoardConfig is created for it.
        """
        # Given: overnight board section is a string scalar
        bad_section_toml = _VALID_SETTINGS.replace(
            '[boards]\nenabled = ["testboard"]',
            '[boards]\nenabled = ["testboard"]\novernight_boards = ["nightboard"]\nnightboard = "not a table"',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_section_toml)

            # When: settings are loaded
            settings = load_settings(path)

            # Then: nightboard has no BoardConfig (non-dict section was skipped)
            assert "nightboard" not in settings.boards, (
                f"Non-dict section should be skipped. Got boards: {list(settings.boards.keys())}"
            )

    def test_scoring_section_non_dict_raises_actionable_error(self) -> None:
        """
        Given scoring is a scalar instead of a table
        When load_settings is called
        Then ActionableError is raised naming the missing section
        """
        # Given: scoring defined as a string
        bad_toml = """\
resume_path = "data/resume.md"
archetypes_path = "config/role_archetypes.toml"
global_rubric_path = "config/global_rubric.toml"
scoring = "not a dict"

[boards]
enabled = ["testboard"]
session_storage_dir = "data"

[boards.testboard]
searches = ["https://example.org/search"]
max_pages = 2
headless = true
rate_limit_range = [1.5, 3.5]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When / Then: load raises ActionableError
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "scoring" in str(exc_info.value).lower(), (
                f"Error should name missing section. Got: {exc_info.value}"
            )

    def test_ollama_section_non_dict_raises_actionable_error(self) -> None:
        """
        Given ollama is a scalar instead of a table
        When load_settings is called
        Then ActionableError is raised naming the missing section
        """
        # Given: replace [ollama] section with a scalar
        # Strip the entire [ollama] block and replace with scalar
        bad_toml = _VALID_SETTINGS.replace(
            '[ollama]\nbase_url = "http://localhost:11434"\n'
            'llm_model = "mistral:7b"\n'
            'embed_model = "nomic-embed-text"\n'
            "slow_llm_threshold_ms = 30000\n"
            'classify_system_prompt = "You are a job listing classifier. '
            'Respond concisely with your classification."\n'
            "max_retries = 3\n"
            "base_delay = 1.0\n"
            "max_embed_chars = 8000\n"
            "head_ratio = 0.6\n"
            "retryable_status_codes = [408, 429, 500, 502, 503, 504]\n",
            "",
        )
        bad_toml = 'ollama = "not a dict"\n' + bad_toml
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When / Then: load raises ActionableError
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "ollama" in str(exc_info.value).lower(), (
                f"Error should name missing section. Got: {exc_info.value}"
            )

    def test_output_section_non_dict_raises_actionable_error(self) -> None:
        """
        Given output is a scalar instead of a table
        When load_settings is called
        Then ActionableError is raised naming the missing section
        """
        # Given: replace [output] section with a scalar
        bad_toml = _VALID_SETTINGS.replace(
            '[output]\ndefault_format = "markdown"\noutput_dir = "./output"\nopen_top_n = 5\n'
            'jd_dir = "output/jds"\n'
            'decisions_dir = "data/decisions"\n'
            'log_dir = "data/logs"\n',
            "",
        )
        bad_toml = 'output = "not a dict"\n' + bad_toml
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When / Then: load raises ActionableError
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "output" in str(exc_info.value).lower(), (
                f"Error should name missing section. Got: {exc_info.value}"
            )

    def test_chroma_section_non_dict_raises_actionable_error(self) -> None:
        """
        Given chroma is a scalar instead of a table
        When load_settings is called
        Then ActionableError is raised naming the missing section
        """
        # Given: replace [chroma] section with a scalar
        bad_toml = _VALID_SETTINGS.replace(
            '[chroma]\npersist_dir = "./data/chroma_db"\n',
            "",
        )
        bad_toml = 'chroma = "not a dict"\n' + bad_toml
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When / Then: load raises ActionableError
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "chroma" in str(exc_info.value).lower(), (
                f"Error should name missing section. Got: {exc_info.value}"
            )

    def test_missing_enabled_field_tells_operator_which_field_to_add(self) -> None:
        """
        Given the 'enabled' field is missing from [boards]
        When load_settings is called
        Then the error names 'enabled' and tells operator to add it
        """
        # Given: boards section without 'enabled' field
        bad_toml = """\
[boards]
overnight_boards = []

[boards.testboard]
searches = ["https://example.org/search"]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When: the settings are loaded
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            # Then: error names the missing field
            err = exc_info.value
            assert err.error_type == ErrorType.CONFIG, (
                f"Expected CONFIG error, got {err.error_type}"
            )
            assert "enabled" in err.error.lower(), f"Error should name 'enabled'. Got: {err.error}"
            assert err.suggestion is not None, "Missing suggestion"
            assert err.troubleshooting is not None, "Missing troubleshooting"

    def test_missing_negative_weight_raises_actionable_error(self) -> None:
        """
        Given negative_weight is not specified in settings.toml
        When load_settings is called
        Then ActionableError is raised naming the missing field
        """
        # Given: TOML without negative_weight
        bad_toml = _VALID_SETTINGS.replace("negative_weight = 0.4\n", "")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When / Then: load raises ActionableError
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "negative_weight" in str(exc_info.value).lower(), (
                f"Error should name missing field. Got: {exc_info.value}"
            )

    def test_negative_weight_above_range_names_the_field_and_valid_range(self) -> None:
        """
        Given negative_weight is set to 1.5 (above valid range)
        When load_settings is called
        Then the error names 'negative_weight' and includes guidance
        """
        # Given: negative_weight exceeds valid range
        bad_toml = _VALID_SETTINGS.replace("negative_weight = 0.4", "negative_weight = 1.5")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When: the settings are loaded
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            # Then: error identifies the out-of-range field
            err = exc_info.value
            assert err.error_type == ErrorType.VALIDATION, (
                f"Expected VALIDATION error, got {err.error_type}"
            )
            assert "negative_weight" in err.error, (
                f"Error should name negative_weight. Got: {err.error}"
            )
            assert err.suggestion is not None, "Missing suggestion"
            assert err.troubleshooting is not None, "Missing troubleshooting"

    def test_negative_weight_below_range_names_the_field_and_valid_range(self) -> None:
        """
        Given negative_weight is set to -0.1 (below valid range)
        When load_settings is called
        Then the error names 'negative_weight' and includes guidance
        """
        # Given: negative_weight is below valid range
        bad_toml = _VALID_SETTINGS.replace("negative_weight = 0.4", "negative_weight = -0.1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When: the settings are loaded
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            # Then: error identifies the out-of-range field
            err = exc_info.value
            assert err.error_type == ErrorType.VALIDATION, (
                f"Expected VALIDATION error, got {err.error_type}"
            )
            assert "negative_weight" in err.error, (
                f"Error should name negative_weight. Got: {err.error}"
            )
            assert err.suggestion is not None, "Missing suggestion"
            assert err.troubleshooting is not None, "Missing troubleshooting"

    def test_missing_global_rubric_path_raises_actionable_error(self) -> None:
        """
        Given global_rubric_path is not specified
        When load_settings is called
        Then ActionableError is raised naming the missing field
        """
        # Given: TOML without global_rubric_path
        bad_toml = _VALID_SETTINGS.replace(
            'global_rubric_path = "config/global_rubric.toml"\n', ""
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When / Then: load raises ActionableError
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "global_rubric_path" in str(exc_info.value).lower(), (
                f"Error should name missing field. Got: {exc_info.value}"
            )

    def test_missing_global_rubric_path_names_field_and_creation_guidance(self) -> None:
        """
        Given global_rubric_path points to a nonexistent file
        When load_settings is called
        Then a CONFIG error names the field with recovery guidance
        """
        # Given: global_rubric_path pointing to a nonexistent file
        bad_toml = _VALID_SETTINGS.replace(
            'global_rubric_path = "config/global_rubric.toml"',
            'global_rubric_path = "/nonexistent/global_rubric.toml"',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When: the settings are loaded
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            # Then: error identifies the missing rubric file
            err = exc_info.value
            assert err.error_type == ErrorType.CONFIG, (
                f"Expected CONFIG error, got {err.error_type}"
            )
            assert "global_rubric_path" in err.error, (
                f"Error should name 'global_rubric_path'. Got: {err.error}"
            )
            assert err.suggestion is not None, "Missing suggestion"
            assert err.troubleshooting is not None, "Missing troubleshooting"

    def test_missing_culture_weight_raises_actionable_error(self) -> None:
        """
        Given culture_weight is not specified in settings.toml
        When load_settings is called
        Then ActionableError is raised naming the missing field
        """
        # Given: TOML without culture_weight
        bad_toml = _VALID_SETTINGS.replace("culture_weight = 0.2\n", "")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When / Then: load raises ActionableError
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "culture_weight" in str(exc_info.value).lower(), (
                f"Error should name missing field. Got: {exc_info.value}"
            )

    def test_culture_weight_above_range_names_the_field_and_valid_range(self) -> None:
        """
        Given culture_weight is set to 1.5 (above valid range)
        When load_settings is called
        Then the error names 'culture_weight' and includes guidance
        """
        # Given: culture_weight exceeds valid range
        bad_toml = _VALID_SETTINGS.replace(
            "culture_weight = 0.2",
            "culture_weight = 1.5",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When: the settings are loaded
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            # Then: error identifies the out-of-range field
            err = exc_info.value
            assert err.error_type == ErrorType.VALIDATION, (
                f"Expected VALIDATION error, got {err.error_type}"
            )
            assert "culture_weight" in err.error, (
                f"Error should name culture_weight. Got: {err.error}"
            )
            assert err.suggestion is not None, "Missing suggestion"
            assert err.troubleshooting is not None, "Missing troubleshooting"

    def test_culture_weight_below_range_names_the_field_and_valid_range(self) -> None:
        """
        Given culture_weight is set to -0.1 (below valid range)
        When load_settings is called
        Then the error names 'culture_weight' and includes guidance
        """
        # Given: culture_weight is below valid range
        bad_toml = _VALID_SETTINGS.replace(
            "culture_weight = 0.2",
            "culture_weight = -0.1",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When: the settings are loaded
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            # Then: error identifies the out-of-range field
            err = exc_info.value
            assert err.error_type == ErrorType.VALIDATION, (
                f"Expected VALIDATION error, got {err.error_type}"
            )
            assert "culture_weight" in err.error, (
                f"Error should name culture_weight. Got: {err.error}"
            )

    # --- Phase 8a additions (config externalization) ---

    def test_disqualifier_prompt_defaults_to_none_when_absent(self) -> None:
        """
        Given settings.toml has no [disqualifier] section
        When load_settings() is called
        Then settings.disqualifier.system_prompt is None
        """
        # Given: valid settings with no [disqualifier] section
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, _VALID_SETTINGS)

            # When: settings are loaded
            settings = load_settings(path)

            # Then: disqualifier.system_prompt is None
            assert settings.disqualifier is not None
            assert settings.disqualifier.system_prompt is None, (
                f"Expected None, got {settings.disqualifier.system_prompt!r}"
            )

    def test_missing_security_section_raises_actionable_error(self) -> None:
        """
        Given settings.toml has no [security] section
        When load_settings() is called
        Then ActionableError is raised naming the missing section
        """
        # Given: valid settings with no [security] section
        no_security = _VALID_SETTINGS.replace(
            '[security]\nscreen_prompt = "Review the following job description text."\n',
            "",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, no_security)

            # When / Then: load raises ActionableError
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "security" in str(exc_info.value).lower(), (
                f"Error should name missing section. Got: {exc_info.value}"
            )

    def test_missing_classify_system_prompt_raises_actionable_error(self) -> None:
        """
        Given settings.toml has no classify_system_prompt in [ollama]
        When load_settings() is called
        Then ActionableError is raised naming the missing field
        """
        # Given: valid settings with no classify_system_prompt
        no_prompt = _VALID_SETTINGS.replace(
            'classify_system_prompt = "You are a job listing classifier. Respond concisely with your classification."\n',
            "",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, no_prompt)

            # When / Then: load raises ActionableError
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "classify_system_prompt" in str(exc_info.value).lower(), (
                f"Error should name missing field. Got: {exc_info.value}"
            )

    def test_non_dict_disqualifier_section_defaults_to_none(self) -> None:
        """
        Given settings.toml has `disqualifier` set to a non-table value
        When load_settings() is called
        Then settings.disqualifier defaults to no custom system_prompt
        """
        # Given: disqualifier is a string instead of a table
        bad_toml = _VALID_SETTINGS + '\ndisqualifier = "not a table"\n'
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When: settings are loaded
            settings = load_settings(path)

            # Then: disqualifier falls back to default (no custom prompt)
            assert settings.disqualifier is not None
            assert settings.disqualifier.system_prompt is None, (
                f"Expected None, got {settings.disqualifier.system_prompt!r}"
            )

    def test_empty_disqualifier_prompt_override_raises_actionable_error(self) -> None:
        """
        Given settings.toml has [disqualifier] system_prompt = ""
        When load_settings() is called
        Then ActionableError is raised naming the field and suggesting removal or content
        """
        # Given: empty disqualifier prompt override
        bad_toml = _VALID_SETTINGS + '\n[disqualifier]\nsystem_prompt = ""\n'
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When / Then: load raises ActionableError
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "disqualifier" in str(exc_info.value).lower(), (
                f"Error should mention disqualifier. Got: {exc_info.value}"
            )

    def test_empty_screen_prompt_raises_actionable_error(self) -> None:
        """
        Given settings.toml has [security] screen_prompt = ""
        When load_settings() is called
        Then ActionableError is raised naming the field and suggesting removal
        """
        # Given: empty screen prompt
        bad_toml = _VALID_SETTINGS.replace(
            'screen_prompt = "Review the following job description text."',
            'screen_prompt = ""',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When / Then: load raises ActionableError
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "screen_prompt" in str(exc_info.value).lower(), (
                f"Error should mention screen_prompt. Got: {exc_info.value}"
            )

    def test_empty_classify_system_prompt_raises_actionable_error(self) -> None:
        """
        Given settings.toml has [ollama] classify_system_prompt = ""
        When load_settings() is called
        Then ActionableError is raised naming the field and suggesting removal
        """
        # Given: empty classify system prompt
        bad_toml = _VALID_SETTINGS.replace(
            'classify_system_prompt = "You are a job listing classifier. Respond concisely with your classification."',
            'classify_system_prompt = ""',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)

            # When / Then: load raises ActionableError
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "classify_system_prompt" in str(exc_info.value).lower(), (
                f"Error should mention classify_system_prompt. Got: {exc_info.value}"
            )


# ---------------------------------------------------------------------------
# TestCommittedConfigCompleteness
# ---------------------------------------------------------------------------

# Complete valid TOML with ALL required fields (no code-side defaults).
_COMPLETE_SETTINGS = """\
resume_path = "data/resume.md"
archetypes_path = "config/role_archetypes.toml"
global_rubric_path = "config/global_rubric.toml"

[boards]
enabled = ["testboard"]
session_storage_dir = "data"

[boards.testboard]
searches = ["https://example.org/search"]
max_pages = 2
headless = true
rate_limit_range = [1.5, 3.5]

[scoring]
archetype_weight = 0.5
fit_weight = 0.3
history_weight = 0.2
comp_weight = 0.15
negative_weight = 0.4
culture_weight = 0.2
base_salary = 220000
disqualify_on_llm_flag = true
min_score_threshold = 0.45
missing_comp_score = 0.5
chunk_overlap = 2000
dedup_similarity_threshold = 0.95

[[scoring.comp_bands]]
ratio = 1.0
score = 1.0

[[scoring.comp_bands]]
ratio = 0.90
score = 0.7

[[scoring.comp_bands]]
ratio = 0.77
score = 0.4

[[scoring.comp_bands]]
ratio = 0.68
score = 0.0

[ollama]
base_url = "http://localhost:11434"
llm_model = "mistral:7b"
embed_model = "nomic-embed-text"
slow_llm_threshold_ms = 30000
classify_system_prompt = "You are a job listing classifier. Respond concisely with your classification."
max_retries = 3
base_delay = 1.0
max_embed_chars = 8000
head_ratio = 0.6
retryable_status_codes = [408, 429, 500, 502, 503, 504]

[output]
default_format = "markdown"
output_dir = "./output"
open_top_n = 5
jd_dir = "output/jds"
decisions_dir = "data/decisions"
log_dir = "data/logs"
eval_history_path = "data/eval_history.jsonl"

[chroma]
persist_dir = "./data/chroma_db"

[adapters]
cdp_timeout = 15.0

[adapters.browser_paths]
msedge = ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]

[security]
screen_prompt = "Review the following job description text."
"""


class TestCommittedConfigCompleteness:
    """
    REQUIREMENT: The committed config/settings.toml is the single source of
    truth for all tunable values — no field may rely on a code-side default.

    WHO: Any operator or developer who needs to understand or change a
         configuration value
    WHAT: (1) The committed settings.toml contains every required scoring field
          (2) The committed settings.toml contains every required ollama field
          (3) The committed settings.toml contains every required output field
          (4) The committed settings.toml contains the required chroma field
          (5) The committed settings.toml contains the required security screen_prompt
          (6) The committed settings.toml contains comp_bands with at least 2 breakpoints
          (7) The committed settings.toml contains top-level path fields
          (8) load_settings() rejects a TOML missing any required scoring field
          (9) load_settings() rejects a TOML missing any required ollama field
          (10) load_settings() rejects a TOML missing a required output field
          (11) load_settings() rejects a TOML missing the chroma persist_dir
          (12) load_settings() rejects a TOML missing the security screen_prompt
          (13) load_settings() rejects a TOML missing comp_bands
          (14) load_settings() rejects a TOML missing a required top-level path
          (15) load_settings() rejects a TOML missing slow_llm_threshold_ms
    WHY: Dual maintenance of defaults in code and config is error-prone;
         a single authoritative config with test enforcement prevents drift

    MOCK BOUNDARY:
        Mock:  nothing — uses real filesystem via tempfile and real config/settings.toml
        Real:  load_settings, TOML parsing, validation, committed config file
        Never: Patch config internals
    """

    # -- Committed config assertions (items 1-7) ----------------------------

    def test_committed_config_contains_all_scoring_fields(self) -> None:
        """
        Given the committed config/settings.toml
        When parsed as TOML
        Then every required scoring field is present
        """
        # Given: the real committed settings file
        import tomllib  # noqa: PLC0415

        settings_path = Path(__file__).resolve().parent.parent / "config" / "settings.toml"
        data = tomllib.loads(settings_path.read_text(encoding="utf-8"))

        # When: the scoring section is examined
        scoring = data.get("scoring", {})

        # Then: every required field is present
        required = [
            "archetype_weight",
            "fit_weight",
            "history_weight",
            "comp_weight",
            "negative_weight",
            "culture_weight",
            "base_salary",
            "disqualify_on_llm_flag",
            "min_score_threshold",
            "missing_comp_score",
        ]
        for field in required:
            assert field in scoring, (
                f"Committed settings.toml is missing [scoring].{field} — "
                f"add it to config/settings.toml"
            )

    def test_committed_config_contains_all_ollama_fields(self) -> None:
        """
        Given the committed config/settings.toml
        When parsed as TOML
        Then every required ollama field is present
        """
        # Given: the real committed settings file
        import tomllib  # noqa: PLC0415

        settings_path = Path(__file__).resolve().parent.parent / "config" / "settings.toml"
        data = tomllib.loads(settings_path.read_text(encoding="utf-8"))

        # When: the ollama section is examined
        ollama = data.get("ollama", {})

        # Then: every required field is present
        required = [
            "base_url",
            "llm_model",
            "embed_model",
            "slow_llm_threshold_ms",
            "classify_system_prompt",
        ]
        for field in required:
            assert field in ollama, (
                f"Committed settings.toml is missing [ollama].{field} — "
                f"add it to config/settings.toml"
            )

    def test_committed_config_contains_all_output_fields(self) -> None:
        """
        Given the committed config/settings.toml
        When parsed as TOML
        Then every required output field is present
        """
        # Given: the real committed settings file
        import tomllib  # noqa: PLC0415

        settings_path = Path(__file__).resolve().parent.parent / "config" / "settings.toml"
        data = tomllib.loads(settings_path.read_text(encoding="utf-8"))

        # When: the output section is examined
        output = data.get("output", {})

        # Then: every required field is present
        required = ["default_format", "output_dir", "open_top_n"]
        for field in required:
            assert field in output, (
                f"Committed settings.toml is missing [output].{field} — "
                f"add it to config/settings.toml"
            )

    def test_committed_config_contains_chroma_persist_dir(self) -> None:
        """
        Given the committed config/settings.toml
        When parsed as TOML
        Then [chroma].persist_dir is present
        """
        # Given: the real committed settings file
        import tomllib  # noqa: PLC0415

        settings_path = Path(__file__).resolve().parent.parent / "config" / "settings.toml"
        data = tomllib.loads(settings_path.read_text(encoding="utf-8"))

        # When / Then
        chroma = data.get("chroma", {})
        assert "persist_dir" in chroma, "Committed settings.toml is missing [chroma].persist_dir"

    def test_committed_config_contains_security_screen_prompt(self) -> None:
        """
        Given the committed config/settings.toml
        When parsed as TOML
        Then [security].screen_prompt is present and non-empty
        """
        # Given: the real committed settings file
        import tomllib  # noqa: PLC0415

        settings_path = Path(__file__).resolve().parent.parent / "config" / "settings.toml"
        data = tomllib.loads(settings_path.read_text(encoding="utf-8"))

        # When / Then
        security = data.get("security", {})
        assert "screen_prompt" in security, (
            "Committed settings.toml is missing [security].screen_prompt"
        )
        assert security["screen_prompt"], (
            "Committed settings.toml has an empty [security].screen_prompt"
        )

    def test_committed_config_contains_comp_bands(self) -> None:
        """
        Given the committed config/settings.toml
        When parsed as TOML
        Then [scoring].comp_bands has at least 2 breakpoints
        """
        # Given: the real committed settings file
        import tomllib  # noqa: PLC0415

        settings_path = Path(__file__).resolve().parent.parent / "config" / "settings.toml"
        data = tomllib.loads(settings_path.read_text(encoding="utf-8"))

        # When / Then
        scoring = data.get("scoring", {})
        bands = scoring.get("comp_bands")
        assert bands is not None, "Committed settings.toml is missing [[scoring.comp_bands]]"
        assert len(bands) >= 2, (
            f"Committed settings.toml needs at least 2 comp_bands, got {len(bands)}"
        )

    def test_committed_config_contains_top_level_paths(self) -> None:
        """
        Given the committed config/settings.toml
        When parsed as TOML
        Then resume_path, archetypes_path, and global_rubric_path are present
        """
        # Given: the real committed settings file
        import tomllib  # noqa: PLC0415

        settings_path = Path(__file__).resolve().parent.parent / "config" / "settings.toml"
        data = tomllib.loads(settings_path.read_text(encoding="utf-8"))

        # When / Then
        for field in ("resume_path", "archetypes_path", "global_rubric_path"):
            assert field in data, f"Committed settings.toml is missing top-level {field}"

    # -- Missing field rejection (items 8-15) --------------------------------

    def test_missing_scoring_field_raises_actionable_error(self) -> None:
        """
        Given settings.toml omits a required scoring field (archetype_weight)
        When load_settings() is called
        Then ActionableError is raised naming the missing field
        """
        # Given: TOML without archetype_weight
        incomplete = _COMPLETE_SETTINGS.replace("archetype_weight = 0.5\n", "")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, incomplete)

            # When / Then
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "archetype_weight" in str(exc_info.value).lower(), (
                f"Error should name the missing field. Got: {exc_info.value}"
            )

    def test_missing_ollama_field_raises_actionable_error(self) -> None:
        """
        Given settings.toml omits a required ollama field (llm_model)
        When load_settings() is called
        Then ActionableError is raised naming the missing field
        """
        # Given: TOML without llm_model
        incomplete = _COMPLETE_SETTINGS.replace('llm_model = "mistral:7b"\n', "")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, incomplete)

            # When / Then
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "llm_model" in str(exc_info.value).lower(), (
                f"Error should name the missing field. Got: {exc_info.value}"
            )

    def test_missing_output_field_raises_actionable_error(self) -> None:
        """
        Given settings.toml omits a required output field (output_dir)
        When load_settings() is called
        Then ActionableError is raised naming the missing field
        """
        # Given: TOML without output_dir
        incomplete = _COMPLETE_SETTINGS.replace('output_dir = "./output"\n', "")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, incomplete)

            # When / Then
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "output_dir" in str(exc_info.value).lower(), (
                f"Error should name the missing field. Got: {exc_info.value}"
            )

    def test_missing_chroma_persist_dir_raises_actionable_error(self) -> None:
        """
        Given settings.toml omits [chroma].persist_dir
        When load_settings() is called
        Then ActionableError is raised naming the missing field
        """
        # Given: TOML without persist_dir
        incomplete = _COMPLETE_SETTINGS.replace('persist_dir = "./data/chroma_db"\n', "")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, incomplete)

            # When / Then
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "persist_dir" in str(exc_info.value).lower(), (
                f"Error should name the missing field. Got: {exc_info.value}"
            )

    def test_missing_security_screen_prompt_raises_actionable_error(self) -> None:
        """
        Given settings.toml has a [security] section but omits screen_prompt
        When load_settings() is called
        Then ActionableError is raised naming the missing field
        """
        # Given: TOML with [security] but no screen_prompt
        incomplete = _COMPLETE_SETTINGS.replace(
            'screen_prompt = "Review the following job description text."',
            "",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, incomplete)

            # When / Then
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "screen_prompt" in str(exc_info.value).lower(), (
                f"Error should name the missing field. Got: {exc_info.value}"
            )

    def test_missing_comp_bands_raises_actionable_error(self) -> None:
        """
        Given settings.toml omits all [[scoring.comp_bands]] entries
        When load_settings() is called
        Then ActionableError is raised naming the missing field
        """
        # Given: TOML with no comp_bands
        # Remove all comp_bands entries from _COMPLETE_SETTINGS
        import re  # noqa: PLC0415

        incomplete = re.sub(
            r"\[\[scoring\.comp_bands\]\]\nratio = [\d.]+\nscore = [\d.]+\n*",
            "",
            _COMPLETE_SETTINGS,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, incomplete)

            # When / Then
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "comp_bands" in str(exc_info.value).lower(), (
                f"Error should name the missing field. Got: {exc_info.value}"
            )

    def test_missing_top_level_path_raises_actionable_error(self) -> None:
        """
        Given settings.toml omits resume_path
        When load_settings() is called
        Then ActionableError is raised naming the missing field
        """
        # Given: TOML without resume_path
        incomplete = _COMPLETE_SETTINGS.replace('resume_path = "data/resume.md"\n', "")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, incomplete)

            # When / Then
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "resume_path" in str(exc_info.value).lower(), (
                f"Error should name the missing field. Got: {exc_info.value}"
            )

    def test_missing_slow_llm_threshold_raises_actionable_error(self) -> None:
        """
        Given settings.toml omits [ollama].slow_llm_threshold_ms
        When load_settings() is called
        Then ActionableError is raised naming the missing field
        """
        # Given: TOML without slow_llm_threshold_ms
        incomplete = _COMPLETE_SETTINGS.replace("slow_llm_threshold_ms = 30000\n", "")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, incomplete)

            # When / Then
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)

            assert "slow_llm_threshold_ms" in str(exc_info.value).lower(), (
                f"Error should name the missing field. Got: {exc_info.value}"
            )
