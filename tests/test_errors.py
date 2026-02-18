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


class TestAIGuidanceToDict:
    """REQUIREMENT: AIGuidance.to_dict() includes only non-None optional fields.

    WHO: Logging and API consumers deserializing error guidance
    WHAT: Optional fields (command, discovery_tool, checks, steps) appear
          in the dict only when populated; mandatory action_required is always present
    WHY: Including None values creates noisy logs and confuses downstream
         tooling that treats presence as meaningful
    """

    def test_to_dict_includes_command_when_set(self) -> None:
        """GIVEN AIGuidance with command populated THEN to_dict includes 'command' key."""
        from jobsearch_rag.errors import AIGuidance

        g = AIGuidance(action_required="fix it", command="run fix")
        d = g.to_dict()
        assert d["command"] == "run fix"

    def test_to_dict_includes_discovery_tool_when_set(self) -> None:
        """GIVEN AIGuidance with discovery_tool populated THEN to_dict includes it."""
        from jobsearch_rag.errors import AIGuidance

        g = AIGuidance(action_required="fix it", discovery_tool="tool_x")
        d = g.to_dict()
        assert d["discovery_tool"] == "tool_x"

    def test_to_dict_includes_checks_when_set(self) -> None:
        """GIVEN AIGuidance with checks populated THEN to_dict includes 'checks' list."""
        from jobsearch_rag.errors import AIGuidance

        g = AIGuidance(action_required="fix it", checks=["check1", "check2"])
        d = g.to_dict()
        assert d["checks"] == ["check1", "check2"]

    def test_to_dict_includes_steps_when_set(self) -> None:
        """GIVEN AIGuidance with steps populated THEN to_dict includes 'steps' list."""
        from jobsearch_rag.errors import AIGuidance

        g = AIGuidance(action_required="fix it", steps=["step1", "step2"])
        d = g.to_dict()
        assert d["steps"] == ["step1", "step2"]

    def test_to_dict_excludes_none_optional_fields(self) -> None:
        """GIVEN AIGuidance with only action_required THEN to_dict has no extra keys."""
        from jobsearch_rag.errors import AIGuidance

        g = AIGuidance(action_required="fix it")
        d = g.to_dict()
        assert d == {"action_required": "fix it"}


class TestActionableErrorToDict:
    """REQUIREMENT: ActionableError.to_dict() includes troubleshooting and context when set.

    WHO: Error serialization consumers (logs, API responses, AI agents)
    WHAT: troubleshooting and context keys appear only when populated
    WHY: Structured errors enable automated recovery by downstream agents
    """

    def test_to_dict_includes_troubleshooting_when_set(self) -> None:
        """GIVEN an error with troubleshooting steps
        THEN to_dict includes the troubleshooting dict.
        """
        err = ActionableError.authentication("test_board", "session expired")
        assert err.troubleshooting is not None
        d = err.to_dict()
        assert "troubleshooting" in d
        assert "steps" in d["troubleshooting"]

    def test_to_dict_includes_context_when_set(self) -> None:
        """GIVEN an error with context dict populated
        THEN to_dict includes the context dict.
        """
        err = ActionableError(
            error="test error",
            error_type=ErrorType.UNEXPECTED,
            service="test",
            context={"key": "value"},
        )
        d = err.to_dict()
        assert d["context"] == {"key": "value"}


class TestValidationFactory:
    """REQUIREMENT: validation() factory produces VALIDATION-typed errors.

    WHO: Config validation and input parsing code
    WHAT: validation() creates an error with correct type, field name, and reason
    WHY: Validation errors need distinct routing from config or connection errors
    """

    def test_validation_factory_produces_validation_type(self) -> None:
        """validation() produces an error typed VALIDATION."""
        err = ActionableError.validation("email", "must contain @")
        assert err.error_type == ErrorType.VALIDATION
        assert "email" in err.error
        assert "must contain @" in err.error


class TestFromExceptionClassifier:
    """REQUIREMENT: from_exception() classifies exceptions by keyword patterns.

    WHO: Generic exception handlers wrapping unknown errors
    WHAT: Timeout keywords → connection error; connection refused → connection;
          not found → unexpected; unmatched → unexpected
    WHY: Auto-classification provides structured recovery paths for
         exceptions that weren't explicitly caught
    """

    def test_timeout_keyword_produces_connection_error(self) -> None:
        """An exception containing 'timeout' is classified as CONNECTION."""
        err = ActionableError.from_exception(
            RuntimeError("request timeout after 30s"),
            "ollama",
            "embed",
        )
        assert err.error_type == ErrorType.CONNECTION

    def test_timed_out_keyword_produces_connection_error(self) -> None:
        """An exception containing 'timed out' is classified as CONNECTION."""
        err = ActionableError.from_exception(
            RuntimeError("connection timed out"),
            "chromadb",
            "query",
        )
        assert err.error_type == ErrorType.CONNECTION

    def test_connection_refused_keyword_produces_connection_error(self) -> None:
        """An exception containing 'connection refused' is classified as CONNECTION."""
        err = ActionableError.from_exception(
            OSError("connection refused on port 11434"),
            "ollama",
            "health_check",
        )
        assert err.error_type == ErrorType.CONNECTION

    def test_not_found_keyword_produces_unexpected_error(self) -> None:
        """An exception containing 'not found' is classified as UNEXPECTED."""
        err = ActionableError.from_exception(
            FileNotFoundError("model not found"),
            "ollama",
            "pull",
        )
        assert err.error_type == ErrorType.UNEXPECTED

    def test_unmatched_keyword_produces_unexpected_error(self) -> None:
        """An exception with no matching keywords falls through to UNEXPECTED."""
        err = ActionableError.from_exception(
            ValueError("some random error"),
            "test",
            "op",
        )
        assert err.error_type == ErrorType.UNEXPECTED
