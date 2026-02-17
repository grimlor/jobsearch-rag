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
