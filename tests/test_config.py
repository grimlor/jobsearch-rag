"""Configuration validation tests.

Maps to BDD spec: TestSettingsLoading
"""

from __future__ import annotations


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

    def test_missing_required_field_raises_config_error_naming_field(self) -> None: ...
    def test_weight_above_one_raises_validation_error(self) -> None: ...
    def test_weight_below_zero_raises_validation_error(self) -> None: ...
    def test_enabled_board_with_no_config_section_raises_config_error(self) -> None: ...
    def test_ollama_url_without_scheme_raises_validation_error(self) -> None: ...
    def test_valid_settings_load_without_error(self) -> None: ...
    def test_optional_fields_use_documented_defaults_when_absent(self) -> None: ...
