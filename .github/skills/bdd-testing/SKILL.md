---
name: bdd-testing
description: "BDD test conventions for this repository. Use when writing, modifying, or reviewing test files, including creating tests for new features and adding coverage specs."
---

# BDD Testing — How to Write Tests

## When This Skill Applies

Whenever writing, modifying, or reviewing test files. This includes
creating tests for new features (Phase 2 of the feature workflow) and adding coverage
specs (Phase 4).

---

## Foundational Principle — System Specification, Not Unit Testing

We are not testing units in isolation. We are **specifying a system**.

The system under test is the full call chain from the public entry point down to the
I/O boundary. Every layer of the system runs for real — Layer 3 composes Layer 2,
which composes Layer 1, all the way down to the process/network/hardware edge. The
only mock is at that edge.

This means:

- **Inter-layer interactions are the point.** If Layer 3 calls Layer 2 which calls
  Layer 1, all three run for real. Mocking Layer 1 from a Layer 3 test would hide
  integration defects — the very bugs that matter most.
- **The mock boundary is where the system ends**, not where a module ends. A function
  in our codebase is part of the system regardless of which module or layer it lives
  in. `subprocess.run` is not part of our system. `discover_repositories` is, even
  though it calls `subprocess.run`.
- **The exception is foundational pure functions.** Functions with no system
  dependencies (e.g., URL parsers, math, config validation) can be tested directly
  with no mocks at all — they have no I/O to mock and no layers to compose.

```python
# ✅ System specification — all layers run for real, only subprocess is mocked
def test_cached_context_produces_correct_repo_info(self, tmp_path):
    # Given: a real .git directory, subprocess returns a valid remote URL
    (tmp_path / ".git").mkdir()
    mock_git = MagicMock(returncode=0, stdout="https://github.com/Org/Repo\n")
    with patch("myapp.discovery.subprocess.run", return_value=mock_git):
        RepoContext.set(str(tmp_path))  # Layer 2 calls Layer 1 for real

    # When: Layer 3 entry point is called — exercises Layer 2 caching for real
    ctx = ProjectContext.from_repo()

    # Then: result reflects the full system's behavior
    assert ctx.organization == "Org"

# ❌ Unit testing — mocks our own code, hides integration bugs
def test_cached_context_produces_correct_repo_info(self):
    fake_info = {"name": "Repo", "organization": "Org", ...}
    with patch("myapp.project.RepoContext.get", return_value=fake_info):
        ctx = ProjectContext.from_repo()
    assert ctx.organization == "Org"  # proves nothing — you wrote both sides
```

The second test is a tautology: you hand-craft a dict and then verify you can read it
back. It cannot catch mismatches between what `RepoContext.get()` actually returns
and what `from_repo()` expects — which is exactly the class of bug that matters.

### What counts as an I/O boundary?

The boundary is where the **system** ends and the **environment** begins:

| I/O boundary (mock these) | Part of the system (never mock) |
|---|---|
| `subprocess.run` — spawns a process | `discover_repositories` — our function that calls subprocess |
| `requests.get` — HTTP call | `fetch_details` — our function that calls requests |
| `Connection.query` — database wire call | `RepoContext.get` — our caching logic over discovery |
| `os.getcwd` — process-level state | `os.path.exists` with `tmp_path` — use real filesystem instead |

If you find yourself mocking a function in your own codebase, stop. Trace the call
chain to the actual I/O operation and mock that instead. Use `tmp_path` for real
filesystem structure so that `os.path.exists`, `os.listdir`, and `os.path.isdir` all
run against real directories.

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
class TestPluginRegistration:
class TestErrorCategorization:
class TestScoreCalculation:

# ❌ Grouped by code structure
class TestProcessorModule:
class TestValidatorModule:

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
    REQUIREMENT: Adapters self-register and are discoverable by name.

    WHO: The pipeline runner loading adapters from configuration
    WHAT: Registered adapters are retrievable by name string;
          an unregistered name produces an error that names the
          requested adapter and lists available options
    WHY: The runner must not know concrete adapter classes — IoC requires
         that the name is the only coupling between config and implementation
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

## Mock Boundary Contract (REQUIRED per class)

Every test class must include a MOCK BOUNDARY declaration immediately after the
WHO/WHAT/WHY block. This is not optional annotation — it is the contract that
prevents the most common violations.

```python
class TestSemanticScoring:
    """
    REQUIREMENT: ...
    WHO: ...
    WHAT: ...
    WHY: ...

    MOCK BOUNDARY:
        Mock:  mock_client fixture (external API — the only I/O boundary)
        Real:  Scorer instance, database via store fixture, tmp_path filesystem
        Never: Construct Result directly — always obtain via scorer.score(item)
    """
```

The three lines answer:
- **Mock** — what is patched and which fixture to use
- **Real** — what runs for real (computation, filesystem, embedded DB)
- **Never** — what must not be constructed or mocked directly

If a test class has no I/O, the Mock line reads `Mock: nothing — this class tests
pure computation`. This is still required so the intent is explicit.

---

## Method-Level Docstrings — Given / When / Then (REQUIRED)

Every test method MUST have a Given / When / Then docstring.

**Always include Given in the docstring.** The only exception is when the
precondition is the default state established by conftest fixtures and adds no
distinguishing information. Even then, the `# Given:` body comment is always present.

```python
# Given required — the non-trivial precondition is the point of the test
def test_extraction_error_on_one_listing_does_not_abort_others(self):
    """
    Given a batch where one listing raises an extraction error
    When the runner processes the batch
    Then the remaining listings are scored and returned
    """

# Given included — explicit setup in the test body
def test_registered_adapter_is_retrievable_by_board_name(self):
    """
    Given an adapter registered under a known name
    When a registered name is requested from the registry
    Then the correct adapter class is returned
    """
```

Do not mix user-story ("As a … I want … So that …") and scenario ("Given / When /
Then") formats. Use scenario format only.

---

## Method Naming

Names read as behavior statements, not implementation descriptions:

```python
# ✅ Behavior-focused
def test_registered_adapter_is_retrievable_by_board_name(self): ...
def test_operations_team_gets_clear_errors_for_missing_config(self): ...
def test_parser_normalizes_hourly_to_annual(self): ...

# ❌ Implementation-focused
def test_get_adapter_returns_class(self): ...
def test_config_validation_raises_exception(self): ...
def test_multiply_by_2080(self): ...
```

---

## Test Body Structure — Given / When / Then (REQUIRED)

Every test method body MUST use Given / When / Then comments to delineate
the three phases.

```python
def test_registered_adapter_is_retrievable_by_board_name(self):
    """
    When a registered name is requested from the registry
    Then the correct adapter class is returned
    """
    # Given: an adapter registered under a known name
    registry = AdapterRegistry()
    registry.register("postgres", PostgresAdapter)

    # When: the name is looked up
    adapter_cls = registry.get("postgres")

    # Then: the correct adapter class is returned
    assert adapter_cls is PostgresAdapter, (
        f"Expected PostgresAdapter, got {adapter_cls}"
    )
```

---

## Assertion Quality (REQUIRED on every assertion)

Every assertion MUST include a diagnostic message. Bare assertions are prohibited.

```python
# ✅ Shows expected vs actual
assert result.score == pytest.approx(0.74, abs=0.05), (
    f"score out of range. Expected ~0.74, got {result.score:.4f}. "
    f"Full result: {result}"
)

# ✅ Loop assertions include the failing item
for i, chunk in enumerate(chunks):
    assert len(chunk) <= MAX_EMBED_CHARS, (
        f"Chunk {i} exceeds max length. "
        f"Expected <= {MAX_EMBED_CHARS}, got {len(chunk)}: ...{chunk[-60:]!r}"
    )

# ❌ Bare assertion — failure is opaque
assert result.is_valid
assert len(items) == 3
```

---

## Test Data

Test data should be representative — close enough to real-world values that
failures mean something. Placeholder strings like `"t"` for title or `"f"` for
full_text produce opaque failures and hide bugs where the implementation uses
the field content.

Magic numbers are acceptable when their meaning is stated:

```python
# ✅ Magic number explained in comment
expected_chunks = 3  # resume has 3 sections: Experience, Skills, Education

# ✅ Magic number explained in assertion message
assert len(results) == 3, (
    "Expected 3 results (Bronze/Silver/Gold tier) "
    f"but got {len(results)}: {[r.title for r in results]}"
)

# ❌ Unexplained magic number
assert result == 0.7
assert len(chunks) == 4
```

Extract constants only when a value appears multiple times or encodes a business
rule referenced by name in the production code.

---

## Coverage = Complete Specification

100% coverage means every line of production code has a spec justifying it.
After all spec tests pass, run:

```bash
pytest --cov=<package> --cov-report=term-missing tests/
```

Every uncovered line triggers the question: *"Which requirement is this line serving?"*

Three categories of requirements surface only at coverage time — they are real
requirements, not optional extras:

| Category | Description | Example |
|---|---|---|
| **Defensive guard code** | Protects against misuse — empty input, wrong types, boundary values | `if not full_text.strip(): raise ValidationError(...)` |
| **Graceful degradation** | Soft failures the system absorbs rather than raising | Missing `history` collection returns empty list, not error |
| **Conditional formatting** | Display logic that varies by state | Warning line only appears when `is_flagged=True` |

**"Pre-existing" is not a category.** Whether a line existed before your changes is irrelevant — if it is uncovered after your work, it is uncovered. The only valid dispositions are: real requirement (write the spec), dead code (remove it), or over-engineering (remove it). "It was already there" is not a disposition.

For each uncovered line: keep it and write the spec, or remove it if it has no
justifying requirement.

---

## Reading `src/` — Public API Discovery Only

Before writing any test for a module, read the relevant `src/` files to discover
the real public API: method signatures, return types, constructor parameters,
and which names are public vs. private (`_` prefix).

**This is the only permitted reason to read `src/` during test writing.**

```python
# ✅ Correct use of src/ knowledge — discovered service.process() returns Result
result = service.process(item)
assert result.score > 0.5, (...)

# ❌ Wrong — used src/ to find an internal function to mock
with patch("mypackage.internal._compute_chunks") as mock_chunks:
    ...
```

If a failure condition cannot be induced through public API inputs alone, that is
a signal the condition may be dead code — flag it in the deviation log rather
than patching around it.

---

## Public APIs Only — No Private Imports

Tests must **never** import `_`-prefixed names from production modules.

```python
# ❌ Testing private internals
from mypackage.pipeline.processor import _parse_header

# ✅ Testing through the public API
from mypackage.pipeline.processor import load_files
```

If a private function's logic seems worth testing directly, that is a signal it
should be promoted to its own module — not a justification to import it.

---

## Error Testing — Messages, Not Just Types

When testing error paths, verify message content, not just that an exception was raised:

```python
# ✅ Tests the message the operator actually sees
with pytest.raises(ValueError) as exc_info:
    registry.get("nonexistent")

assert "nonexistent" in str(exc_info.value), (
    f"Error should name the missing item. Got: {exc_info.value}"
)

# ❌ Only confirms an exception occurred
with pytest.raises(ValueError):
    registry.get("nonexistent")
```

Errors should include enough context for the operator to diagnose the problem
without consulting source code.

---

## Failure-Mode Specs Are Mandatory

Failure-mode specs are as important as happy-path specs. An unspecified failure is
an unhandled failure. For every feature, the spec must cover:

- Missing or malformed input
- External service unavailable
- Invalid configuration
- Boundary values
- Partial failure (one item in a batch fails, others continue)

---

## Reference Documents

For detailed implementation examples including mock boundary contracts, tautology
anti-patterns, assertion quality, test data, and conftest infrastructure:
- `.github/skills/bdd-testing/references/test-patterns.md`
