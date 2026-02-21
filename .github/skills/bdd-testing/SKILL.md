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

## The Hierarchy

BDD in this repo has three levels. Each level answers a different question:

| Level | Form | Question Answered |
|---|---|---|
| **Test class** | REQUIREMENT / WHO / WHAT / WHY | What user story does this group prove? |
| **Test method** | Given / When / Then scenario | Under what specific conditions does the behavior occur? |
| **Test body** | Given / When / Then comments | How is the scenario implemented in code? |

The class captures the user story. The WHAT field enumerates which scenarios are needed
to prove it — if WHAT is well-written, the list of required test methods should follow
from it directly. Each method then specifies one of those scenarios in full.

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

## The Three-Part Contract

Every test method requires all three of the following. None substitutes for the others:

| Part | Purpose | Serves |
|---|---|---|
| **Method name** | The claim — behavior stated as a fact | Scanability; test output |
| **Given / When / Then docstring** | The scenario — explicit conditions and observable outcome | Precision; review; spec traceability |
| **Given / When / Then body comments** | The structure — setup, action, assertion delineated | Readability; maintenance |

A good name without a docstring leaves the scenario ambiguous. A docstring without
body comments buries the structure in undifferentiated code. All three are required
on every test method.

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

The WHAT field is the bridge between the user story and the test methods. Each clause
in WHAT should correspond to one or more test methods. If a test method cannot be
traced to a clause in WHAT, either the test is speculative or WHAT is incomplete.

---

## Method-Level Docstrings — Given / When / Then (REQUIRED)

Every test method MUST have a Given / When / Then docstring. This is the full scenario
form — not just When / Then.

**Given is required in the docstring** when the precondition is the distinguishing
condition of the scenario — when it is specifically what makes this test different from
the others in the class.

**Given may be omitted from the docstring** when the precondition is the default state
established by conftest fixtures and is the same for all tests in the class. In that
case the body comment `# Given:` still appears in the test body.

```python
# Given required — the non-trivial precondition is the point of the test
def test_extraction_error_on_one_listing_does_not_abort_others(self):
    """
    Given a batch where one listing raises an extraction error
    When the runner processes the batch
    Then the remaining listings are scored and returned
    """

# Given omitted — default fixture state, same for all tests in this class
def test_registered_adapter_is_retrievable_by_board_name(self):
    """
    When a registered board name is requested from the registry
    Then the correct adapter class is returned
    """
```

The method name and the docstring are not redundant — they serve different purposes.
The name is a scannable claim. The docstring makes the specific conditions and
observable outcome explicit, and provides the traceability link back to the BDD spec.

**Do not mix** user-story ("As a … I want … So that …") and scenario ("Given / When /
Then") formats within this repository. Use scenario format only.

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

## Test Body Structure — Given / When / Then (REQUIRED)

Every test method body MUST use Given / When / Then comments to delineate
the three phases. This is not optional annotation — it is the structure that
makes a test readable without executing it.

```python
def test_registered_adapter_is_retrievable_by_board_name(self):
    """
    When a registered board name is requested from the registry
    Then the correct adapter class is returned
    """
    # Given: an adapter registered under a known board name
    registry = AdapterRegistry()
    registry.register("ziprecruiter", ZipRecruiterAdapter)

    # When: the board name is looked up
    adapter_cls = registry.get("ziprecruiter")

    # Then: the correct adapter class is returned
    assert adapter_cls is ZipRecruiterAdapter, (
        f"Expected ZipRecruiterAdapter, got {adapter_cls}"
    )
```

See [test-patterns.md](references/test-patterns.md) for full examples of:

- Complete Given / When / Then structure with assertion diagnostic messages
- When to include Given in the docstring versus the body only
- Mocking rules (I/O boundaries only, real instances for computation)
- Mock anti-patterns and fixture consolidation
- `conftest.py` infrastructure — shared stubs and the output directory guard
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
