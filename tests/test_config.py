"""Configuration validation tests.

Maps to BDD spec: TestSettingsValidation
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


class TestSettingsValidation:
    """REQUIREMENT: Configuration errors tell the operator exactly what to fix.

    WHO: The operator who misconfigured settings.toml
    WHAT: Each validation failure is caught at startup (not mid-pipeline)
          and produces an actionable error that names the problematic field,
          explains what is wrong, and provides step-by-step recovery guidance
          including which file to open and what to change; all config errors
          include a suggestion and troubleshooting steps
    WHY: A mid-run config failure after 10 minutes of browser work is
         far more costly than a startup validation failure — and an error
         that says "validation failed" without telling the operator what
         to fix is nearly as costly as a mid-run crash
    """

    def test_missing_boards_section_names_the_field_and_tells_operator_to_add_it(self) -> None:
        """A missing [boards] section produces an error that names the field and tells the operator to add it."""
        bad_toml = """\
[scoring]
archetype_weight = 0.5
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            err = exc_info.value
            assert err.error_type == ErrorType.CONFIG
            assert "boards" in err.error.lower()
            assert err.suggestion is not None
            assert "boards" in err.suggestion.lower()
            assert err.troubleshooting is not None
            assert len(err.troubleshooting.steps) > 0

    def test_weight_above_range_names_the_field_and_valid_range(self) -> None:
        """A weight > 1.0 produces an error naming the field so the operator knows which value to fix."""
        bad_toml = _VALID_SETTINGS.replace("archetype_weight = 0.5", "archetype_weight = 1.5")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            err = exc_info.value
            assert err.error_type == ErrorType.VALIDATION
            assert "archetype_weight" in err.error
            assert err.suggestion is not None
            assert err.troubleshooting is not None

    def test_weight_below_range_names_the_field_and_valid_range(self) -> None:
        """A negative weight produces an error naming the field so the operator knows which value to fix."""
        bad_toml = _VALID_SETTINGS.replace("fit_weight = 0.3", "fit_weight = -0.1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            err = exc_info.value
            assert err.error_type == ErrorType.VALIDATION
            assert "fit_weight" in err.error
            assert err.suggestion is not None
            assert err.troubleshooting is not None

    def test_missing_board_config_names_the_board_and_section_to_add(self) -> None:
        """A board in [boards.enabled] with no config section produces an error naming the board and section to add."""
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
            err = exc_info.value
            assert err.error_type == ErrorType.CONFIG
            assert "nonexistent_board" in err.error
            assert err.suggestion is not None
            assert "nonexistent_board" in err.suggestion
            assert err.troubleshooting is not None

    def test_ollama_url_without_scheme_suggests_adding_http_prefix(self) -> None:
        """A URL without http:// scheme produces an error suggesting the operator add the prefix."""
        bad_toml = _VALID_SETTINGS.replace(
            'base_url = "http://localhost:11434"',
            'base_url = "localhost:11434"',
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            err = exc_info.value
            assert err.error_type == ErrorType.VALIDATION
            assert "scheme" in err.error.lower()
            assert err.suggestion is not None
            assert err.troubleshooting is not None

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

    def test_negative_base_salary_names_the_field_and_constraint(self) -> None:
        """A negative base_salary produces an error naming the field and the positive-number constraint."""
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
            err = exc_info.value
            assert err.error_type == ErrorType.VALIDATION
            assert "base_salary" in err.error
            assert err.suggestion is not None
            assert err.troubleshooting is not None

    def test_missing_file_tells_operator_to_create_settings(self) -> None:
        """A missing settings file produces an error suggesting the operator create or copy from example."""
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "nonexistent.toml"
            with pytest.raises(ActionableError) as exc_info:
                load_settings(missing)
            err = exc_info.value
            assert err.error_type == ErrorType.CONFIG
            assert "not found" in err.error.lower()
            assert err.suggestion is not None
            assert err.troubleshooting is not None

    def test_malformed_toml_identifies_syntax_error_and_suggests_fix(self) -> None:
        """Invalid TOML syntax produces a PARSE error with a suggestion to fix the syntax."""
        bad_toml = "[boards\nenabled = broken"
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            err = exc_info.value
            assert err.error_type == ErrorType.PARSE
            assert err.suggestion is not None
            assert err.troubleshooting is not None

    def test_empty_enabled_boards_tells_operator_to_add_board_names(self) -> None:
        """An empty boards.enabled list produces an error telling the operator to add board names."""
        bad_toml = _VALID_SETTINGS.replace(
            'enabled = ["testboard"]', "enabled = []"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            err = exc_info.value
            assert err.error_type == ErrorType.CONFIG
            assert "boards.enabled" in err.error
            assert err.suggestion is not None
            assert err.troubleshooting is not None

    def test_board_section_not_a_table_tells_operator_to_define_as_table(self) -> None:
        """A scalar board value (not a table) produces an error telling the operator to define it as a TOML table."""
        bad_toml = """\
[boards]
enabled = ["testboard"]
testboard = "not a table"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_settings(tmpdir, bad_toml)
            with pytest.raises(ActionableError) as exc_info:
                load_settings(path)
            err = exc_info.value
            assert err.error_type == ErrorType.CONFIG
            assert "table" in err.error.lower()
            assert err.suggestion is not None
            assert err.troubleshooting is not None

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

    def test_missing_enabled_field_tells_operator_which_field_to_add(self) -> None:
        """A missing 'enabled' field produces an error naming the field so the operator knows what to add."""
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
            err = exc_info.value
            assert err.error_type == ErrorType.CONFIG
            assert "enabled" in err.error.lower()
            assert err.suggestion is not None
            assert err.troubleshooting is not None
