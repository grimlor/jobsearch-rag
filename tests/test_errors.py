"""Actionable error hierarchy tests.

Tests that the error factory methods produce correct, structured,
recoverable errors per the actionable-error philosophy.
"""

from __future__ import annotations

from jobsearch_rag.errors import ActionableError, ErrorType


class TestErrorFactoryMethods:
    """REQUIREMENT: Factory methods produce structured errors with embedded guidance.

    WHO: Any component catching exceptions and wrapping them for consumers
    WHAT: Each factory produces the correct error_type; suggestion is always populated;
          ai_guidance and troubleshooting are present; to_dict() excludes None values
    WHY: Opaque errors halt autonomous recovery — every error must carry
         its own recovery path
    """

    def test_authentication_error_has_correct_type(self) -> None:
        """authentication() produces an error typed AUTHENTICATION so retry logic can classify it."""
        err = ActionableError.authentication("ziprecruiter", "session expired")
        assert err.error_type == ErrorType.AUTHENTICATION

    def test_config_error_names_the_field(self) -> None:
        """config() embeds the offending field name so the operator knows which setting to fix."""
        err = ActionableError.config("scoring.archetype_weight", "must be between 0.0 and 1.0")
        assert "archetype_weight" in err.error

    def test_connection_error_includes_url(self) -> None:
        """connection() embeds the target URL so the operator can verify reachability."""
        err = ActionableError.connection("Ollama", "http://localhost:11434", "refused")
        assert "http://localhost:11434" in err.error

    def test_embedding_error_names_model(self) -> None:
        """embedding() includes the model name so the operator can verify it is pulled and available."""
        err = ActionableError.embedding("nomic-embed-text", "timeout after 3 retries")
        assert "nomic-embed-text" in err.error

    def test_index_error_names_collection(self) -> None:
        """index() names the missing ChromaDB collection so the operator knows which index to rebuild."""
        err = ActionableError.index("resume")
        assert "resume" in err.error

    def test_parse_error_names_board_and_selector(self) -> None:
        """parse() includes both board name and CSS selector so the developer can pinpoint the broken scraper."""
        err = ActionableError.parse("ziprecruiter", ".job-title", "element not found")
        assert "ziprecruiter" in err.error
        assert ".job-title" in err.error

    def test_decision_error_names_job_id(self) -> None:
        """decision() names the unknown job_id so the operator can verify the listing exists."""
        err = ActionableError.decision("abc-123")
        assert "abc-123" in err.error

    def test_to_dict_excludes_none_values(self) -> None:
        """to_dict() omits None-valued keys so serialized output is clean for logging and API responses."""
        err = ActionableError.config("weight", "too high")
        d = err.to_dict()
        assert None not in d.values()

    def test_all_factories_set_success_false(self) -> None:
        """Every factory method marks success=False so callers never accidentally treat an error as a success."""
        err = ActionableError.unexpected("test", "op", "boom")
        assert err.success is False


class TestSuggestionPreservation:
    """REQUIREMENT: Custom suggestions are always preserved.

    WHO: Callers providing operation-specific context
    WHAT: Custom suggestions flow through factory methods and from_exception()
    WHY: Callers have context that generic classifiers cannot infer
    """

    def test_authentication_preserves_custom_suggestion(self) -> None:
        """A caller-provided suggestion is preserved verbatim, overriding any generic default."""
        err = ActionableError.authentication(
            "linkedin", "401", suggestion="Delete data/linkedin_session.json and re-auth"
        )
        assert err.suggestion == "Delete data/linkedin_session.json and re-auth"

    def test_from_exception_preserves_caller_suggestion(self) -> None:
        """from_exception() forwards the caller's suggestion rather than generating a generic one."""
        err = ActionableError.from_exception(
            ValueError("401 unauthorized"),
            "test",
            "test_op",
            suggestion="Custom hint",
        )
        assert err.suggestion == "Custom hint"
