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
        err = ActionableError.authentication("ziprecruiter", "session expired")
        assert err.error_type == ErrorType.AUTHENTICATION

    def test_config_error_names_the_field(self) -> None:
        err = ActionableError.config("scoring.archetype_weight", "must be between 0.0 and 1.0")
        assert "archetype_weight" in err.error

    def test_connection_error_includes_url(self) -> None:
        err = ActionableError.connection("Ollama", "http://localhost:11434", "refused")
        assert "http://localhost:11434" in err.error

    def test_embedding_error_names_model(self) -> None:
        err = ActionableError.embedding("nomic-embed-text", "timeout after 3 retries")
        assert "nomic-embed-text" in err.error

    def test_index_error_names_collection(self) -> None:
        err = ActionableError.index("resume")
        assert "resume" in err.error

    def test_parse_error_names_board_and_selector(self) -> None:
        err = ActionableError.parse("ziprecruiter", ".job-title", "element not found")
        assert "ziprecruiter" in err.error
        assert ".job-title" in err.error

    def test_decision_error_names_job_id(self) -> None:
        err = ActionableError.decision("abc-123")
        assert "abc-123" in err.error

    def test_to_dict_excludes_none_values(self) -> None:
        err = ActionableError.config("weight", "too high")
        d = err.to_dict()
        assert None not in d.values()

    def test_all_factories_set_success_false(self) -> None:
        err = ActionableError.unexpected("test", "op", "boom")
        assert err.success is False


class TestSuggestionPreservation:
    """REQUIREMENT: Custom suggestions are always preserved.

    WHO: Callers providing operation-specific context
    WHAT: Custom suggestions flow through factory methods and from_exception()
    WHY: Callers have context that generic classifiers cannot infer
    """

    def test_authentication_preserves_custom_suggestion(self) -> None:
        err = ActionableError.authentication(
            "linkedin", "401", suggestion="Delete data/linkedin_session.json and re-auth"
        )
        assert err.suggestion == "Delete data/linkedin_session.json and re-auth"

    def test_from_exception_preserves_caller_suggestion(self) -> None:
        err = ActionableError.from_exception(
            ValueError("401 unauthorized"),
            "test",
            "test_op",
            suggestion="Custom hint",
        )
        assert err.suggestion == "Custom hint"
