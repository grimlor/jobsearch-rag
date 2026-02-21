# BDD Test Patterns — Detailed Examples

## The Three-Part Contract — Complete Examples

Every test method requires a name, a Given/When/Then docstring, and Given/When/Then
body comments. The following contrasts the anti-pattern with both correct forms.

```python
# ❌ Wrong — name only, no docstring, no body structure
def test_registered_adapter_is_retrievable_by_board_name(self):
    registry = AdapterRegistry()
    registry.register("ziprecruiter", ZipRecruiterAdapter)
    result = registry.get("ziprecruiter")
    assert result is ZipRecruiterAdapter


# ✅ Correct — Given omitted from docstring (default fixture state is the precondition)
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


# ✅ Correct — Given required in docstring (the precondition is the point)
def test_extraction_error_on_one_listing_does_not_abort_others(self):
    """
    Given a batch where one listing raises an extraction error
    When the runner processes the batch
    Then the remaining listings are scored and returned
    """
    # Given: a batch where the first listing raises on extract_detail
    failing = make_listing(external_id="bad", full_text="")
    succeeding = make_listing(external_id="good")
    mock_adapter.extract_detail.side_effect = [
        ActionableError(...),
        succeeding,
    ]

    # When: the runner processes the batch
    result = await runner.run()

    # Then: the good listing is in the output and the failure is counted
    assert len(result.ranked_listings) == 1, (
        f"Expected 1 ranked listing, got {len(result.ranked_listings)}"
    )
    assert result.ranked_listings[0].listing.external_id == "good"
    assert result.failed_listings == 1, (
        f"Expected 1 failed listing, got {result.failed_listings}"
    )
```

**Rule:** Include Given in the docstring when the precondition is the distinguishing
condition of the scenario — when "given X" is specifically what makes this test
different from the others in the class. Omit it when the precondition is the default
state established by conftest fixtures. The `# Given:` body comment is always present.

---

## Test Body Structure — Given / When / Then

Given / When / Then comments are required on every test method body:

```python
def test_score_fusion_uses_configured_weights(self):
    """
    When scoring with custom weights
    Then the final score reflects the configured weight distribution
    """
    # Given: custom weights that differ from defaults
    weights = {"fit": 0.5, "archetype": 0.3, "history": 0.2}

    # When: computing the final score
    result = ranker.compute_final_score(scores, weights=weights)

    # Then: score reflects the weight distribution
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
    # Given: a registry with no boards registered
    registry = AdapterRegistry()

    # When: an unknown board name is requested
    with pytest.raises(ActionableError) as exc_info:
        registry.get("nonexistent_board")

    # Then: the error names the missing board
    assert "nonexistent_board" in str(exc_info.value), (
        f"Error should name the missing board. Got: {exc_info.value}"
    )
```

Errors in this repo follow the **ActionableError** pattern — factory methods,
recovery paths, AI guidance, and troubleshooting steps. See `src/jobsearch_rag/errors.py`.

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

## No Private-Function Imports

Tests must **never** import `_`-prefixed names from production modules.

Private functions are implementation details — testing them directly couples tests to
internal structure rather than observable behavior. When the implementation changes, the
tests break even though the public contract is intact.

```python
# ❌ Testing private internals — breaks encapsulation
from jobsearch_rag.pipeline.rescorer import _parse_jd_header, _extract_jd_body

def test_parse_jd_header_extracts_metadata(self):
    meta = _parse_jd_header(content)
    assert meta["title"] == "Staff Architect"

# ✅ Testing through the public API
from jobsearch_rag.pipeline.rescorer import load_jd_files

def test_loaded_listing_has_correct_metadata(self, jd_dir):
    listings = load_jd_files(jd_dir)
    assert listings[0].title == "Staff Architect"
```

If a private function has complex logic that seems worth testing directly, that is a
signal it should be inspected for promotion to its own module — not a justification
to import it directly.

---

## No Local Imports Inside Tests

All imports belong at the **top of the file**. Never place imports inside test methods,
helper functions, or fixtures.

```python
# ❌ Local import inside a helper
class TestRescoreWorkflow:
    def _make_ranker(self):
        from jobsearch_rag.pipeline.ranker import Ranker  # buried import
        return Ranker(...)

# ✅ Module-level import
from jobsearch_rag.pipeline.ranker import Ranker

class TestRescoreWorkflow:
    def _make_ranker(self):
        return Ranker(...)
```

---

## Mock Anti-Patterns

### ❌ Mocking the Registry (Computation Mock)

```python
# ❌ Wrong — AdapterRegistry is pure computation (dict lookup), not I/O
mock_registry = MagicMock()
mock_registry.get.return_value = mock_adapter
patch("...AdapterRegistry", mock_registry)

# ✅ Correct — register a mock adapter in a real registry
registry = AdapterRegistry()
registry.register("ziprecruiter", mock_adapter)
# mock_adapter's I/O methods (authenticate, search, extract_detail) are AsyncMock
```

The registry has no I/O and no side effects. Mocking it severs the contract
that board names resolve to adapter classes and hides misconfiguration.

### ❌ Patching Internal Parsing Functions

```python
# ❌ Wrong — card_to_listing is an internal parser, not an I/O boundary
patch("...card_to_listing", side_effect=ParseError)

# ✅ Correct — return mixed valid/invalid HTML from the Playwright locator mock
card_locator.all.return_value = [valid_card_mock, malformed_card_mock]
# Then assert output contains only the valid listing
```

When a card can't be parsed, the behavioral contract is "other cards still
return." Test that contract through the adapter's public interface with a page
mock that produces the failure condition, not by injecting it into the parser.

### ❌ Repeated Patch Blocks

```python
# ❌ Wrong — same 6-patch block repeated in every test method
def test_search_happy_path(self):
    with patch("...load_settings") as ms, \
         patch("...PipelineRunner") as mr, \
         patch("...webbrowser.open") as mw:
        ...

def test_search_no_results(self):
    with patch("...load_settings") as ms, \  # identical block
         patch("...PipelineRunner") as mr, \
         patch("...webbrowser.open") as mw:
        ...

# ✅ Correct — fixture in conftest.py handles shared setup
@pytest.fixture
def cli_search_mocks():
    with patch("...load_settings") as ms, \
         patch("...PipelineRunner") as mr, \
         patch("...webbrowser.open") as mw:
        yield SimpleNamespace(settings=ms, runner=mr, browser=mw)

def test_search_happy_path(self, cli_search_mocks):
    cli_search_mocks.runner.return_value.run = AsyncMock(return_value=result)
    ...
```

Shared setup belongs in conftest. Repeated inline blocks make it impossible to
tell whether variation between copies is intentional or drift.

---

## conftest.py — Infrastructure You Must Not Bypass

`conftest.py` provides two categories of infrastructure that all test files
depend on. Neither is optional.

### Shared I/O Stubs

The `embedder` fixture constructs an `Embedder` instance via `__new__` (bypassing
`__init__` to avoid Ollama client construction) and replaces `embed`, `classify`,
and `health_check` with `AsyncMock`. Use this fixture rather than creating a
local embedder mock in individual test files.

If a setup pattern recurs across two or more test classes — such as a runner with
mocked adapter I/O — add it to conftest as a fixture rather than reconstructing
it inline in each test method.

### Output Directory Guard

`conftest.py` redirects the application's output directory to a temporary path
for the duration of every test run. This prevents tests from writing to the real
`output/jds/`, `output/results.md`, and `output/results.csv` files produced by
live search runs.

**Do not bypass this guard.** Any test that exercises file output uses the
redirected path provided by the conftest fixture. Never hardcode or reference
the real `output/` directory in a test.
