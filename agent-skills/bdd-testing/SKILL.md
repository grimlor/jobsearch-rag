# BDD Testing — How to Write Tests in This Repo

## When This Skill Applies

Whenever writing, modifying, or reviewing test files in this repository. This includes
creating tests for new features (Phase 2 of the feature workflow) and adding coverage
specs (Phase 4).

---

## Test Organization

Tests are organized by **consumer requirement**, not by code structure or persona.

```python
# ✅ Grouped by requirement
class TestSuggestionPreservation:
class TestErrorCategorization:
class TestScoreFusion:

# ❌ Grouped by code structure
class TestScorerModule:
class TestRankerModule:

# ❌ Grouped by persona
class TestDeveloperFeatures:
```

A single test file may contain multiple requirement classes. Group related requirements
in one file when they exercise the same module. The file-level docstring should explain
which BDD spec classes it covers.

---

## Class-Level Docstrings — REQUIREMENT / WHO / WHAT / WHY

Every test class MUST have a structured docstring:

```python
class TestAdapterRegistration:
    """
    REQUIREMENT: Adapters self-register and are discoverable by board name.

    WHO: The pipeline runner loading adapters from settings.toml
    WHAT: Registered adapters are retrievable by board name string;
          an unregistered board name produces an error that names the
          requested board and lists available options
    WHY: The runner must not know concrete adapter classes — IoC requires
         that board name is the only coupling between config and implementation
    """
```

| Field | Purpose | Question It Answers |
|-------|---------|---------------------|
| **REQUIREMENT** | One-line capability statement | What promise does this group verify? |
| **WHO** | Stakeholder or consumer | Who benefits when this is met? |
| **WHAT** | Concrete, testable behavior | What observable behavior proves it? |
| **WHY** | Business/operational justification | What goes wrong if it's missing? |

---

## Method-Level Docstrings — Scenario Format

This repo uses the **scenario ("When / Then")** format for individual test docstrings,
with persona context from the class-level WHO field:

```python
def test_unregistered_board_name_error_names_the_board_and_lists_available(self):
    """
    When an unregistered board name is requested
    Then the error message names the missing board and lists available options
    """
```

**Do not mix** user-story and scenario formats within this repository.

---

## Method Naming

Names read as behavior statements, not implementation descriptions:

```python
# ✅ Behavior-focused
def test_registered_adapter_is_retrievable_by_board_name(self): ...
def test_operations_team_gets_clear_errors_for_missing_config(self): ...
def test_comp_parser_normalizes_hourly_to_annual(self): ...

# ❌ Implementation-focused
def test_get_adapter_returns_class(self): ...
def test_config_validation_raises_exception(self): ...
def test_multiply_by_2080(self): ...
```

---

## Test Body Structure — Given / When / Then

Use comments to delineate the three phases:

```python
def test_score_fusion_uses_configured_weights(self):
    """
    When scoring with custom weights
    Then the final score reflects the configured weight distribution
    """
    # Given: Custom weights that differ from defaults
    weights = {"fit": 0.5, "archetype": 0.3, "history": 0.2}

    # When: Computing the final score
    result = ranker.compute_final_score(scores, weights=weights)

    # Then: Score reflects the weight distribution
    assert result == pytest.approx(expected, abs=0.01), (
        f"Expected {expected} with weights {weights}, got {result}"
    )
```

---

## Assertion Quality

Every assertion MUST include a diagnostic message:

```python
# ✅ Actionable message
assert err.error_type == ErrorType.AUTHENTICATION, (
    f"Expected AUTHENTICATION error type, got {err.error_type}. "
    f"Error message: {err.error}"
)

# ❌ Bare assertion — failure is opaque
assert err.error_type == ErrorType.AUTHENTICATION
```

---

## Mocking Rules

**Mock at I/O boundaries only:**
- Network calls (Ollama API, HTTP requests)
- Browser automation (Playwright)
- System resources (`webbrowser.open`, `builtins.input`)
- Time/randomness (`asyncio.sleep`, `random.uniform`) for speed and determinism

**Use real instances:**
- Filesystem with `tmp_path` or `tempfile.TemporaryDirectory()`
- ChromaDB with temp directories (VectorStore tests do this)
- All pure computation (scoring math, regex parsing, config validation)
- Dataclass instances — never `MagicMock` for data objects

**Never mock:**
- Internal helper functions within the module under test
- Pure computation logic
- Config loading (use real TOML with temp files)

---

## Test Markers

This repo uses pytest markers defined in `pyproject.toml`:

- `@pytest.mark.integration` — Tests requiring external services (Ollama, ChromaDB server)
- `@pytest.mark.live` — Tests requiring browser and live board access

Unit tests (no marker) should run fast with zero external dependencies.

---

## Error Testing

When testing error paths, verify the **message content**, not just the exception type:

```python
def test_unregistered_board_name_error_names_the_board_and_lists_available(self):
    """
    When an unregistered board name is requested
    Then the error names the missing board and lists available options
    """
    with pytest.raises(ActionableError) as exc_info:
        registry.get("nonexistent_board")

    assert "nonexistent_board" in str(exc_info.value), (
        f"Error should name the missing board. Got: {exc_info.value}"
    )
```

Errors in this repo follow the **ActionableError** pattern — factory methods, recovery
paths, AI guidance, and troubleshooting steps. See `src/jobsearch_rag/errors.py`.

---

## Failure-Mode Specs

Failure-mode specs are as important as happy-path specs. An unspecified failure is an
unhandled failure. For every feature, ask:

- What happens when input is missing or malformed?
- What happens when an external service is unavailable?
- What happens when configuration is invalid?
- What happens at boundary values?

---

## Coverage = Complete Specification

100% coverage means every line has a specification justifying it. After implementation:

```bash
pytest --cov=jobsearch_rag --cov-report=term-missing tests/
```

Every uncovered line triggers the question: "Which requirement is this line serving?"

---

## Reference Documents

For the full philosophy and rationale behind these practices:
- `Patterns & Practices/BDD_TESTING_PRINCIPLES.md`
- `Patterns & Practices/spec-first-bdd-testing-patterns.md`
- `Patterns & Practices/actionable-error-handling-patterns.md`
- `Patterns & Practices/actionable-error-philosophy.md`
