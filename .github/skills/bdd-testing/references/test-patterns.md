# BDD Test Patterns — Detailed Examples

## The Three-Part Contract — Complete Examples

Every test method requires a name, a Given/When/Then docstring, and Given/When/Then
body comments. The following contrasts the anti-pattern with both correct forms.

```python
# ❌ Wrong — name only, no docstring, no body structure
def test_registered_adapter_is_retrievable_by_name(self):
    registry = AdapterRegistry()
    registry.register("postgres", PostgresAdapter)
    result = registry.get("postgres")
    assert result is PostgresAdapter


# ✅ Correct — Given included (explicit setup, not default fixture state)
def test_registered_adapter_is_retrievable_by_name(self):
    """
    Given an adapter registered under a known name
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


# ✅ Correct — Given required in docstring (the precondition is the point)
def test_extraction_error_on_one_listing_does_not_abort_others(self):
    """
    Given a batch where one listing raises an extraction error
    When the runner processes the batch
    Then the remaining listings are scored and returned
    """
    # Given: a batch where the first listing raises on extract_detail
    failing = make_item(external_id="bad", content="")
    succeeding = make_item(external_id="good")
    mock_adapter.extract_detail.side_effect = [
        ValueError(...),
        succeeding,
    ]

    # When: the runner processes the batch
    result = await runner.run()

    # Then: the good item is in the output and the failure is counted
    assert len(result.processed_items) == 1, (
        f"Expected 1 processed item, got {len(result.processed_items)}"
    )
    assert result.processed_items[0].item.external_id == "good"
    assert result.failed_items == 1, (
        f"Expected 1 failed item, got {result.failed_items}"
    )
```

**Rule:** Always include Given in the docstring. The only exception is when the
precondition is the default state established by conftest fixtures and adds no
distinguishing information — even then, the `# Given:` body comment is always present.

---

## Mock Boundary Contract — Full Examples

Every test class declares its mock boundary in the class docstring. Three lines: what
is mocked, what runs real, and what must never be constructed directly.

```python
class TestSemanticScoring:
    """
    REQUIREMENT: The scorer produces meaningful similarity scores for items.

    WHO: The pipeline runner invoking scorer.score() per item
    WHAT: relevance_score, category_score, quality_score, and penalty_score are
          derived from real database similarity queries against indexed content;
          an item matching target criteria scores higher than one that does not
    WHY: Scores are the core signal — if they are tautological or mocked, the
         entire pipeline produces meaningless output

    MOCK BOUNDARY:
        Mock:  mock_client fixture (external API — the only I/O boundary)
        Real:  Scorer instance, database via store fixture, tmp_path
        Never: Construct Result directly — always obtain via scorer.score(item)
    """


class TestScoreComputation:
    """
    REQUIREMENT: Score computation scales continuously relative to baseline.

    WHO: The ranker computing final_score via score fusion
    WHAT: normalized_score is 1.0 at or above baseline; grades linearly below; 0.5 when absent
    WHY: Computed scores are a key signal — wrong values silently skew ranking without errors

    MOCK BOUNDARY:
        Mock:  nothing — this class tests pure computation
        Real:  Scorer.compute_score() called directly with float inputs
        Never: Mock the scorer itself to return a preset score
    """


class TestAdapterRegistration:
    """
    REQUIREMENT: Adapters self-register and are discoverable by name.

    WHO: The pipeline runner loading adapters from configuration
    WHAT: Registered adapters are retrievable by name string;
          an unregistered name produces an error that names the adapter
    WHY: The runner must not know concrete adapter classes — the name is
         the only coupling between config and implementation

    MOCK BOUNDARY:
        Mock:  nothing — AdapterRegistry is pure computation (dict lookup)
        Real:  AdapterRegistry, PostgresAdapter class reference
        Never: Mock the registry itself; mock_adapter's I/O methods are AsyncMock
               but the registry must be a real instance
    """
```

---

## Assertion Quality — Full Examples

Every assertion requires a diagnostic message. The message must show enough context
that a failing test is self-explanatory without running a debugger.

```python
# ✅ Shows expected vs actual with context
assert result.score == pytest.approx(0.74, abs=0.05), (
    f"score out of range. Expected ~0.74, got {result.score:.4f}. "
    f"Full result: {result}"
)

# ✅ Loop assertions include the item that failed
for i, chunk in enumerate(chunks):
    assert len(chunk) <= MAX_EMBED_CHARS, (
        f"Chunk {i} exceeds max length. "
        f"Expected <= {MAX_EMBED_CHARS}, got {len(chunk)}: ...{chunk[-60:]!r}"
    )

# ✅ Multi-assertion blocks give each assertion its own message
assert result.amount == 180000.0, (
    f"amount mismatch. Expected 180000.0, got {result.amount}. "
    f"Parsed from: {result.raw_text!r}"
)
assert result.source == "verified", (
    f"source mismatch. Expected 'verified', got {result.source!r}"
)

# ✅ Collection checks name the expected contents
assert set(result.keys()) == {"score", "category_score", "final_score"}, (
    f"Unexpected keys in result. Got: {sorted(result.keys())}"
)

# ❌ Bare assertion — failure is opaque
assert result.is_valid
assert len(items) == 3
assert "error" in response
```

**When pytest.approx is required:**
Floating-point comparisons from similarity scores or weighted sums must always
use `pytest.approx`. A tolerance of `abs=0.01` is appropriate for final scores;
`abs=0.05` is appropriate for component scores where external computation introduces
small variation.

---

## Test Data — Representative Values

Test data should be close enough to real-world values that failures mean something.
Placeholder strings like `"t"` for title or `"f"` for full_text produce opaque
failures and hide bugs where the implementation uses the field content.

```python
# ✅ Realistic — failures are interpretable
order = Order(
    source="web",
    external_id="ord_8821234",
    title="Premium Subscription - Annual",
    customer="Acme Corp",
    region="US-West",
    url="https://example.com/orders/ord_8821234",
    description=(
        "Annual premium subscription for the enterprise platform, "
        "including priority support, advanced analytics, and "
        "custom integrations. Billed annually with a 15% discount "
        "over monthly pricing."
    ),
    price_min=18000.0,
    price_max=22000.0,
    price_source="catalog",
    price_text="$18,000 - $22,000/yr",
)

# ❌ Meaningless placeholders — failures are opaque
order = Order(
    source="s", external_id="1", title="t",
    customer="c", region="r", url="u", description="d"
)
```

**Magic numbers** are acceptable when their meaning is stated in either an inline
comment or the assertion message:

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

Extract a named constant only when the same value appears in multiple tests or
when it encodes a business rule referenced by name in the production code.

---

## Mocking Rules

**Mock at I/O boundaries only:**
- Network calls (external APIs, HTTP requests)
- Browser automation (Playwright, Selenium)
- System resources (`webbrowser.open`, `builtins.input`)
- Time/randomness (`asyncio.sleep`, `random.uniform`) for speed and determinism

**Use real instances for everything else:**
- Filesystem operations: use `tmp_path` or `tempfile.TemporaryDirectory()`
- Embedded databases: use real instance with `tmp_path`
- All pure computation: math, regex parsing, config validation, dataclass construction
- Config loading: use real config files written to `tmp_path`, not a mock

**Never mock:**
- The subject under test — if you mock `Scorer` in `TestSemanticScoring`, you are
  not testing `Scorer`
- Internal helper functions within the module under test
- Dataclass constructors — build real instances with real field values
- Pure computation logic of any kind

The mock boundary contract in each test class docstring is the authoritative
statement of what is mocked for that class.

---

## Tautology Tests — The Most Dangerous Anti-Pattern

A tautology test always passes regardless of production code behavior. It typically
appears when a test constructs expected output directly and asserts on the constructed
object rather than invoking the system under test.

```python
# ❌ Tautology — Result is constructed here, not by Scorer
# This test passes even if Scorer.score() is deleted from the codebase
def test_score_above_threshold_for_matching_item(self):
    result = Result(score=0.8, category_score=0.7, final_score=0.75)
    assert result.score > 0.5, (
        f"Expected score > 0.5, got {result.score}"
    )

# ✅ Real — Scorer is invoked, database is queried, score is earned
def test_score_above_threshold_for_matching_item(self, scorer, indexed_data):
    """
    Given data indexed in the store and an item matching the indexed content
    When scorer.score() is called
    Then the score exceeds 0.5
    """
    # Given: an item that mirrors the indexed content
    item = make_item(description=MATCHING_DESCRIPTION)

    # When: the scorer evaluates the item
    result = scorer.score(item)

    # Then: score reflects genuine similarity
    assert result.score > 0.5, (
        f"Expected score > 0.5 for matching item, got {result.score:.4f}"
    )
```

**How to recognize a tautology:**
- The When step does not call any production code
- The Given and Then operate entirely on objects created in the test body
- Deleting the module under test would not cause the test to fail

---

## Coverage — Three Categories That Surface Late

After all spec tests pass, run coverage and examine uncovered lines. Three categories
of requirements routinely surface only at coverage time. All three are real
requirements — do not delete the code, write the spec.

**Defensive guard code** — protects against API misuse:
```python
# Production code: what happens when full_text is empty?
if not item.description.strip():
    raise ValidationError("description cannot be empty")
# → Write: test_empty_description_raises_validation_error
```

**Graceful degradation** — soft failures the system absorbs rather than raising:
```python
# Production code: history collection may not exist yet
try:
    collection = store.get_collection("history")
except CollectionNotFoundError:
    return []  # ← this line
# → Write: test_missing_history_collection_returns_empty_list
```

**Conditional formatting branches** — display logic that varies by state:
```python
# Production code: warning only appears when flagged
if result.flagged:
    lines.append(f"⚠️  Flagged: {result.flag_reason}")  # ← this line
# → Write: test_flagged_item_includes_warning_in_display
```

For each uncovered line, ask: is this a real requirement (write the spec), dead code
(remove it), or over-engineering (remove it and simplify)?

**"Pre-existing" is not a category.** Whether a line existed before your changes is irrelevant — if it is uncovered after your work, it is uncovered. The only valid dispositions are: real requirement (write the spec), dead code (remove it), or over-engineering (remove it). "It was already there" is not a disposition.

---

## Error Testing — Messages, Not Just Types

Verify message content, not just exception type. Future changes to error message
wording will break tests that check for the right message — that is the point.

```python
# ✅ Checks what the operator actually sees
with pytest.raises(ValueError) as exc_info:
    registry.get("nonexistent")

assert "nonexistent" in str(exc_info.value), (
    f"Error should name the missing item. Got: {exc_info.value}"
)
assert "available" in str(exc_info.value).lower(), (
    f"Error should list available options. Got: {exc_info.value}"
)

# ❌ Only confirms an exception occurred — message could be anything
with pytest.raises(ValueError):
    registry.get("nonexistent")

# ❌ Constructs the error manually and asserts on the construction — always true
err = ValueError(f"No adapter for 'nonexistent'")
assert "nonexistent" in str(err)
```

The third anti-pattern is particularly insidious — it looks like it tests error
content but only confirms that Python string formatting works.

---

## Test Markers

Use pytest markers defined in `pyproject.toml` to categorize tests:

- `@pytest.mark.integration` — Tests requiring external services (databases, APIs)
- `@pytest.mark.live` — Tests requiring live network access

Unit tests (no marker) should run fast with zero external dependencies.

---

## No Private-Function Imports

Tests must **never** import `_`-prefixed names from production modules.

```python
# ❌ Testing private internals — breaks encapsulation
from myapp.pipeline.processor import _parse_header, _extract_body

def test_parse_header_extracts_metadata(self):
    meta = _parse_header(content)
    assert meta["title"] == "Premium Subscription"

# ✅ Testing through the public API
from myapp.pipeline.processor import load_files

def test_loaded_item_has_correct_metadata(self, data_dir):
    items = load_files(data_dir)
    assert items[0].title == "Premium Subscription", (
        f"Expected 'Premium Subscription', got {items[0].title!r}"
    )
```

If a private function has complex logic that seems worth testing directly, that is a
signal it should be promoted to its own module — not a justification to import it.

---

## No Local Imports Inside Tests

All imports belong at the top of the file:

```python
# ❌ Local import inside a helper
class TestProcessingWorkflow:
    def _make_ranker(self):
        from myapp.pipeline.ranker import Ranker  # buried
        return Ranker(...)

# ✅ Module-level import
from myapp.pipeline.ranker import Ranker

class TestProcessingWorkflow:
    def _make_ranker(self):
        return Ranker(...)
```

---

## Mock Anti-Patterns

### ❌ Mocking the Subject Under Test

```python
# ❌ Wrong — mocking Scorer in a test that claims to test scoring
mock_scorer = MagicMock()
mock_scorer.score.return_value = Result(score=0.8, ...)
result = mock_scorer.score(item)
assert result.score == 0.8  # tautology
```

This is the most common and most damaging violation. It produces 100% green tests
for code that may be entirely broken.

### ❌ Mocking Pure Computation (Registry)

```python
# ❌ Wrong — AdapterRegistry is pure computation, not I/O
mock_registry = MagicMock()
mock_registry.get.return_value = mock_adapter

# ✅ Correct — register a mock adapter in a real registry
registry = AdapterRegistry()
registry.register("postgres", mock_adapter)
```

The registry has no I/O and no side effects. Mocking it hides misconfiguration.

### ❌ Direct Construction Bypassing the SUT

```python
# ❌ Wrong — inserts directly into database, bypassing AuditLogger.record()
collection = logger._store.get_or_create_collection("history")
collection.add(ids=["item-1"], documents=["Premium Subscription at Acme"], metadatas=[{}])
results = collection.get(ids=["item-1"])
assert len(results["documents"]) == 1  # tautology — logger.record() never called

# ✅ Correct — calls the real logger, then verifies via public query
logger.record(item, verdict="approved", reason="Strong match")
history = logger.get_recent(limit=10)
assert any(entry.external_id == item.external_id for entry in history), (
    f"Expected {item.external_id} in history, got: {[e.external_id for e in history]}"
)
```

### ❌ Accessing Private Store Attributes

```python
# ❌ Wrong — _store and _registry are implementation details
collection = logger._store.get_or_create_collection("history")
original = dict(AdapterRegistry._registry)

# ✅ Correct — test through public methods only
logger.record(item, verdict="approved")
runner.run()
```

### ❌ Patching Internal Parsing Functions

```python
# ❌ Wrong — parse_item is an internal parser, not an I/O boundary
patch("...parse_item", side_effect=ParseError)

# ✅ Correct — return mixed valid/invalid data from the I/O mock
data_source.fetch_all.return_value = [valid_data, malformed_data]
# Then assert output contains only the valid item
```

### ❌ Repeated Patch Blocks

```python
# ❌ Wrong — same block repeated in every test method
def test_search_happy_path(self):
    with patch("...load_config") as mc, patch("...Runner") as mr:
        ...

def test_search_no_results(self):
    with patch("...load_config") as mc, patch("...Runner") as mr:
        ...

# ✅ Correct — extract to conftest fixture
@pytest.fixture
def cli_mocks():
    with patch("...load_config") as mc, patch("...Runner") as mr:
        yield SimpleNamespace(config=mc, runner=mr)
```

Shared setup belongs in conftest. Repeated inline blocks hide whether variation
between copies is intentional or drift.

---

## conftest.py — Infrastructure You Must Not Bypass

`conftest.py` provides two categories of infrastructure that all test files
depend on. Neither is optional.

### Shared I/O Stubs

When an I/O client (API client, database connection, etc.) is used across
multiple test classes, create a conftest fixture that constructs the client
via `__new__` (bypassing `__init__` to avoid real connection setup) and
replaces I/O methods with `AsyncMock` or `MagicMock`. Use this fixture
rather than creating local mocks in individual test files.

If a setup pattern recurs across two or more test classes — such as a runner with
mocked adapter I/O — add it to conftest as a fixture rather than reconstructing
it inline in each test method.

### Output Directory Guard

If your application writes files to an output directory, `conftest.py` should
redirect it to a temporary path for the duration of every test run. This
prevents tests from writing to real output locations.

**Do not bypass this guard.** Any test that exercises file output uses the
redirected path provided by the conftest fixture. Never hardcode or reference
the real output directory in a test.
