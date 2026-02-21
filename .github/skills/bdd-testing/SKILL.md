---
name: bdd-testing
description: "BDD test conventions for this repository. Use when writing, modifying, or reviewing test files, including creating tests for new features and adding coverage specs."
---

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

## Test Body Structure

Use Given / When / Then comments to delineate the three phases. See [test-patterns.md](references/test-patterns.md) for full examples of:

- Given / When / Then body structure
- Assertion quality (diagnostic messages required)
- Mocking rules (I/O boundaries only, real instances for computation)
- Test markers (`@pytest.mark.integration`, `@pytest.mark.live`)
- Error testing (verify message content, not just type)
- Failure-mode specs (coverage of error paths and edge cases)
- Coverage as complete specification

---

## Reference Documents

For the full philosophy and rationale behind these practices:
- `Patterns & Practices/BDD_TESTING_PRINCIPLES.md`
- `Patterns & Practices/spec-first-bdd-testing-patterns.md`
- `Patterns & Practices/actionable-error-handling-patterns.md`
- `Patterns & Practices/actionable-error-philosophy.md`
