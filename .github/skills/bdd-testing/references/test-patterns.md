# BDD Test Patterns — Detailed Examples

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

## No Private-Function Imports

Tests must **never** import `_`-prefixed names from production modules.

Private functions are implementation details — testing them directly couples tests to
internal structure rather than observable behavior. When the implementation changes, the
tests break even though the public contract is intact.

```python
# ❌ Testing private internals — breaks encapsulation
from jobsearch_rag.pipeline.rescorer import _parse_jd_header, _extract_jd_body

def test_parse_jd_header_extracts_metadata(self):
    meta = _parse_jd_header(content)  # coupled to internal function signature
    assert meta["title"] == "Staff Architect"

# ✅ Testing through the public API
from jobsearch_rag.pipeline.rescorer import load_jd_files

def test_loaded_listing_has_correct_metadata(self, jd_dir):
    listings = load_jd_files(jd_dir)
    assert listings[0].title == "Staff Architect"
```

**If a private function has complex logic worth testing directly**, that is a signal it
should be critically inspected for promotion to its own domain/module. However, it is 
not an immediate signal that it *should* be tested directly, but that tests should be 
written to test it thoroughly through the public API it is called from. This is the 
crux of black-box testing.

---

## No Local Imports Inside Tests

As with all modules, all imports belong at the **top of the file** — either as regular 
imports or inside an `if TYPE_CHECKING:` block (for annotations only). Never place 
imports inside test methods, helper functions, or fixtures.

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

**Why:** Local imports scatter dependencies, making it hard to see what a module 
actually touches. Module-level imports make the full dependency surface visible at a
glance and allow tools (linters, type checkers, import-graph analyzers) to reason about
the file correctly.
