"""Configuration validation tests.

Maps to BDD spec: TestSettingsLoading
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from jobsearch_rag.config import load_settings
from jobsearch_rag.errors import ActionableError, ErrorType

# A minimal valid settings TOML for tests
_VALID_SETTINGS = """\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://example.org/search"]
max_pages = 2
headless = true

[scoring]
archetype_weight = 0.5
fit_weight = 0.3
history_weight = 0.2
disqualify_on_llm_flag = true
min_score_threshold = 0.45

[ollama]
base_url = "http://localhost:11434"
llm_model = "mistral:7b"
embed_model = "nomic-embed-text"

[output]
default_format = "markdown"
output_dir = "./output"
open_top_n = 5

[chroma]
persist_dir = "./data/chroma_db"
"""


def _write_settings(tmpdir: str, content: str) -> Path:
    """Write settings content to a temp file and return the path."""
    path = Path(tmpdir) / "settings.toml"
    path.write_text(content, encoding="utf-8")
    return path


class TestSettingsLoading:
    """REQUIREMENT: Invalid or missing configuration is caught at startup, not mid-run.

    WHO: The operator who misconfigured settings.toml
    WHAT: Missing required fields raise a descriptive ConfigError naming the field;
          weight values outside [0.0, 1.0] raise a validation error;
          referencing a board in [boards.enabled] with no corresponding
          [boards.<name>] section raises an error; Ollama URL with no scheme raises
          a validation error
    WHY: A mid-run config failure after 10 minutes of browser work is
         far more costly than a startup validation failure
    """

    def test_missing_required_field_raises_config_error_naming_field(self) -> None:
        """A missing required field raises a CONFIG error that names the field so the operator knows what to add."""
        # Missing [boards] section entirely
        bad_toml = """\
[scoring]
archetype_weight = 0.5
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            assert exc_info.value.error_type == ErrorType.CONFIG
            assert "boards" in exc_info.value.error.lower()

    def test_weight_above_one_raises_validation_error(self) -> None:
        """A scoring weight greater than 1.0 is rejected at startup to prevent unbounded score inflation."""
        bad_toml = _VALID_SETTINGS.replace("archetype_weight = 0.5", "archetype_weight = 1.5")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            assert exc_info.value.error_type == ErrorType.VALIDATION
            assert "archetype_weight" in exc_info.value.error

    def test_weight_below_zero_raises_validation_error(self) -> None:
        """A negative scoring weight is rejected at startup to prevent inverted scoring."""
        bad_toml = _VALID_SETTINGS.replace("fit_weight = 0.3", "fit_weight = -0.1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            assert exc_info.value.error_type == ErrorType.VALIDATION
            assert "fit_weight" in exc_info.value.error

    def test_enabled_board_with_no_config_section_raises_config_error(self) -> None:
        """A board listed in [boards.enabled] with no matching [boards.<name>] section is caught at load time."""
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
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            assert exc_info.value.error_type == ErrorType.CONFIG
            assert "nonexistent_board" in exc_info.value.error

    def test_ollama_url_without_scheme_raises_validation_error(self) -> None:
        """An Ollama URL missing http:// or https:// is rejected before the pipeline attempts a connection."""
        bad_toml = _VALID_SETTINGS.replace(
            'base_url = "http://localhost:11434"',
            'base_url = "localhost:11434"',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            assert exc_info.value.error_type == ErrorType.VALIDATION
            assert "scheme" in exc_info.value.error.lower()

    def test_valid_settings_load_without_error(self) -> None:
        """A well-formed settings.toml loads successfully and is usable by the pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, _VALID_SETTINGS)
            settings = load_settings(path)
            assert settings.enabled_boards == ["testboard"]
            assert settings.scoring.archetype_weight == 0.5
            assert settings.ollama.base_url == "http://localhost:11434"
            assert settings.chroma.persist_dir == "./data/chroma_db"

    def test_optional_fields_use_documented_defaults_when_absent(self) -> None:
        """Omitted optional fields fall back to documented defaults rather than None or zero."""
        minimal_toml = """\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://example.org/search"]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, minimal_toml)
            settings = load_settings(path)
            # Scoring defaults
            assert settings.scoring.archetype_weight == 0.5
            assert settings.scoring.fit_weight == 0.3
            assert settings.scoring.history_weight == 0.2
            assert settings.scoring.min_score_threshold == 0.45
            # Ollama defaults
            assert settings.ollama.base_url == "http://localhost:11434"
            assert settings.ollama.llm_model == "mistral:7b"
            # Output defaults
            assert settings.output.default_format == "markdown"
            assert settings.output.open_top_n == 5
            # Board defaults
            assert settings.boards["testboard"].browser_channel is None

    def test_browser_channel_is_parsed_from_board_config(self) -> None:
        """A browser_channel value in the board section is parsed and available on BoardConfig."""
        channel_toml = _VALID_SETTINGS.replace(
            "headless = true",
            'headless = true\nbrowser_channel = "msedge"',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, channel_toml)
            settings = load_settings(path)
            assert settings.boards["testboard"].browser_channel == "msedge"

    def test_comp_weight_defaults_to_zero_point_fifteen_when_absent(self) -> None:
        """comp_weight defaults to 0.15 when not specified in settings.toml."""
        minimal_toml = """\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://example.org/search"]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, minimal_toml)
            settings = load_settings(path)
            assert settings.scoring.comp_weight == 0.15

    def test_base_salary_defaults_to_220000_when_absent(self) -> None:
        """base_salary defaults to 220000 when not specified in settings.toml."""
        minimal_toml = """\
[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://example.org/search"]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, minimal_toml)
            settings = load_settings(path)
            assert settings.scoring.base_salary == 220_000

    def test_base_salary_must_be_positive_number(self) -> None:
        """A non-positive base_salary is rejected at startup."""
        bad_toml = _VALID_SETTINGS + "\nbase_salary = -50000\n"
        # Insert base_salary into [scoring] section
        bad_toml = _VALID_SETTINGS.replace(
            "min_score_threshold = 0.45",
            "min_score_threshold = 0.45\nbase_salary = -50000",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            assert exc_info.value.error_type == ErrorType.VALIDATION
            assert "base_salary" in exc_info.value.error

    def test_missing_file_raises_config_error(self) -> None:
        """GIVEN a settings path that does not exist
        WHEN load_settings is called
        THEN a CONFIG error is raised identifying the missing file.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "nonexistent.toml"
            with pytest.raises(ActionableError) as exc_info:
                load_settings(missing)
            assert exc_info.value.error_type == ErrorType.CONFIG
            assert "not found" in exc_info.value.error.lower()

    def test_malformed_toml_raises_parse_error(self) -> None:
        """GIVEN a settings file with invalid TOML syntax
        WHEN load_settings is called
        THEN a PARSE error is raised.
        """
        bad_toml = "[boards\nenabled = broken"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            assert exc_info.value.error_type == ErrorType.PARSE

    def test_empty_enabled_boards_raises_config_error(self) -> None:
        """GIVEN boards.enabled is an empty list
        WHEN load_settings is called
        THEN a CONFIG error is raised.
        """
        bad_toml = _VALID_SETTINGS.replace(
            'enabled = ["testboard"]', "enabled = []"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            assert exc_info.value.error_type == ErrorType.CONFIG
            assert "boards.enabled" in exc_info.value.error

    def test_board_section_not_a_table_raises_config_error(self) -> None:
        """GIVEN a board in enabled has a scalar value instead of a table
        WHEN load_settings is called
        THEN a CONFIG error about the board not being a table is raised.
        """
        bad_toml = """\
[boards]
enabled = ["testboard"]
testboard = "not a table"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            assert exc_info.value.error_type == ErrorType.CONFIG
            assert "table" in exc_info.value.error.lower()

    def test_overnight_board_configs_are_parsed(self) -> None:
        """GIVEN a settings file with overnight_boards that have config sections
        WHEN load_settings is called
        THEN overnight boards have parsed BoardConfig entries.
        """
        overnight_toml = """\
[boards]
enabled = ["testboard"]
overnight_boards = ["nightboard"]

[boards.testboard]
searches = ["https://example.org/search"]

[boards.nightboard]
searches = ["https://night.org/search"]
max_pages = 5
headless = false
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, overnight_toml)
            settings = load_settings(path)
            assert "nightboard" in settings.boards
            assert settings.boards["nightboard"].max_pages == 5
            assert settings.boards["nightboard"].headless is False

    def test_scoring_section_non_dict_uses_defaults(self) -> None:
        """GIVEN scoring is a scalar instead of a table
        WHEN load_settings is called
        THEN scoring uses default values.
        """
        bad_toml = """\
scoring = "not a dict"

[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://example.org/search"]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            settings = load_settings(path)
            assert settings.scoring.archetype_weight == 0.5

    def test_ollama_section_non_dict_uses_defaults(self) -> None:
        """GIVEN ollama is a scalar instead of a table
        WHEN load_settings is called
        THEN ollama uses default values.
        """
        bad_toml = """\
ollama = "not a dict"

[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://example.org/search"]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            settings = load_settings(path)
            assert settings.ollama.base_url == "http://localhost:11434"

    def test_output_section_non_dict_uses_defaults(self) -> None:
        """GIVEN output is a scalar instead of a table
        WHEN load_settings is called
        THEN output uses default values.
        """
        bad_toml = """\
output = "not a dict"

[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://example.org/search"]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            settings = load_settings(path)
            assert settings.output.default_format == "markdown"

    def test_chroma_section_non_dict_uses_defaults(self) -> None:
        """GIVEN chroma is a scalar instead of a table
        WHEN load_settings is called
        THEN chroma uses default values.
        """
        bad_toml = """\
chroma = "not a dict"

[boards]
enabled = ["testboard"]

[boards.testboard]
searches = ["https://example.org/search"]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            settings = load_settings(path)
            assert settings.chroma.persist_dir == "./data/chroma_db"

    def test_missing_enabled_field_raises_config_error(self) -> None:
        """GIVEN boards section exists but has no 'enabled' key
        WHEN load_settings is called
        THEN a CONFIG error naming the missing field is raised.
        """
        bad_toml = """\
[boards]
overnight_boards = []

[boards.testboard]
searches = ["https://example.org/search"]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            assert exc_info.value.error_type == ErrorType.CONFIG
            assert "enabled" in exc_info.value.error.lower()
