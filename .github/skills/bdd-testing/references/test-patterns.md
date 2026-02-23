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

## Mock Boundary Contract — Full Examples

Every test class declares its mock boundary in the class docstring. Three lines: what
is mocked, what runs real, and what must never be constructed directly.

```python
class TestSemanticScoring:
    """
    REQUIREMENT: The scorer produces meaningful similarity scores for job listings.

    WHO: The pipeline runner invoking scorer.score() per listing
    WHAT: fit_score, archetype_score, culture_score, and negative_score are
          derived from real ChromaDB similarity queries against indexed content;
          a JD matching archetype signals scores higher than one that does not
    WHY: Scores are the core signal — if they are tautological or mocked, the
         entire pipeline produces meaningless output

    MOCK BOUNDARY:
        Mock:  mock_embedder fixture (Ollama HTTP — the only I/O boundary)
        Real:  Scorer instance, ChromaDB via vector_store fixture, tmp_path
        Never: Construct ScoreResult directly — always obtain via scorer.score(listing)
    """


class TestCompScoring:
    """
    REQUIREMENT: Compensation scoring scales continuously relative to base salary.

    WHO: The ranker computing final_score via score fusion
    WHAT: comp_score is 1.0 at or above base; grades linearly below; 0.5 when absent
    WHY: Comp is a taste signal — wrong scores silently skew ranking without errors

    MOCK BOUNDARY:
        Mock:  nothing — this class tests pure computation
        Real:  Scorer.compute_comp_score() called directly with float inputs
        Never: Mock the scorer itself to return a preset comp_score
    """


class TestAdapterRegistration:
    """
    REQUIREMENT: Adapters self-register and are discoverable by board name.

    WHO: The pipeline runner loading adapters from settings.toml
    WHAT: Registered adapters are retrievable by board name string;
          an unregistered board name produces an error that names the board
    WHY: The runner must not know concrete adapter classes — board name is
         the only coupling between config and implementation

    MOCK BOUNDARY:
        Mock:  nothing — AdapterRegistry is pure computation (dict lookup)
        Real:  AdapterRegistry, ZipRecruiterAdapter class reference
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
assert result.fit_score == pytest.approx(0.74, abs=0.05), (
    f"fit_score out of range. Expected ~0.74, got {result.fit_score:.4f}. "
    f"Full scores: {result}"
)

# ✅ Loop assertions include the item that failed
for i, chunk in enumerate(chunks):
    assert len(chunk) <= MAX_EMBED_CHARS, (
        f"Chunk {i} exceeds max length. "
        f"Expected <= {MAX_EMBED_CHARS}, got {len(chunk)}: ...{chunk[-60:]!r}"
    )

# ✅ Multi-assertion blocks give each assertion its own message
assert result.comp_min == 180000.0, (
    f"comp_min mismatch. Expected 180000.0, got {result.comp_min}. "
    f"Parsed from: {result.comp_text!r}"
)
assert result.comp_source == "employer", (
    f"comp_source mismatch. Expected 'employer', got {result.comp_source!r}"
)

# ✅ Collection checks name the expected contents
assert set(result.keys()) == {"fit_score", "archetype_score", "final_score"}, (
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
`abs=0.05` is appropriate for component scores where the embedding model introduces
small variation.

---

## Test Data — Representative Values

Test data should be close enough to real-world values that failures mean something.
Placeholder strings like `"t"` for title or `"f"` for full_text produce opaque
failures and hide bugs where the implementation uses the field content.

```python
# ✅ Realistic — failures are interpretable
listing = JobListing(
    board="ziprecruiter",
    external_id="zr_8821234",
    title="Staff Platform Architect",
    company="Acme Corp",
    location="Remote (USA)",
    url="https://www.ziprecruiter.com/jobs/acme-corp/staff-architect",
    full_text=(
        "We are seeking a Staff Platform Architect to define technical strategy "
        "for our distributed infrastructure platform serving 200M users. "
        "You will own RFC processes, drive cross-team alignment, and mentor "
        "senior engineers across the organization."
    ),
    comp_min=180000.0,
    comp_max=220000.0,
    comp_source="employer",
    comp_text="$180,000 - $220,000/yr",
)

# ❌ Meaningless placeholders — failures are opaque
listing = JobListing(
    board="b", external_id="1", title="t",
    company="c", location="l", url="u", full_text="f"
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
- Network calls (Ollama API, HTTP requests)
- Browser automation (Playwright)
- System resources (`webbrowser.open`, `builtins.input`)
- Time/randomness (`asyncio.sleep`, `random.uniform`) for speed and determinism

**Use real instances for everything else:**
- Filesystem operations: use `tmp_path` or `tempfile.TemporaryDirectory()`
- ChromaDB: use real instance with `tmp_path` (VectorStore fixture does this)
- All pure computation: scoring math, regex parsing, config validation, dataclass construction
- Config loading: use real TOML written to `tmp_path`, not a mock

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
# ❌ Tautology — ScoreResult is constructed here, not by Scorer
# This test passes even if Scorer.score() is deleted from the codebase
def test_fit_score_above_threshold_for_matching_jd(self):
    result = ScoreResult(fit_score=0.8, archetype_score=0.7, final_score=0.75)
    assert result.fit_score > 0.5, (
        f"Expected fit_score > 0.5, got {result.fit_score}"
    )

# ✅ Real — Scorer is invoked, ChromaDB is queried, score is earned
def test_fit_score_above_threshold_for_matching_jd(self, scorer, indexed_resume):
    """
    Given a resume indexed in ChromaDB and a JD describing matching experience
    When scorer.score() is called
    Then fit_score exceeds 0.5
    """
    # Given: a JD that mirrors the indexed resume's content
    listing = make_listing(full_text=MATCHING_JD_TEXT)

    # When: the scorer evaluates the listing
    result = scorer.score(listing)

    # Then: fit_score reflects genuine semantic similarity
    assert result.fit_score > 0.5, (
        f"Expected fit_score > 0.5 for matching JD, got {result.fit_score:.4f}"
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
if not listing.full_text.strip():
    raise ValidationError("full_text cannot be empty")
# → Write: test_empty_full_text_raises_validation_error
```

**Graceful degradation** — soft failures the system absorbs rather than raising:
```python
# Production code: decisions collection may not exist yet
try:
    collection = store.get_collection("decisions")
except CollectionNotFoundError:
    return []  # ← this line
# → Write: test_missing_decisions_collection_returns_empty_list
```

**Conditional formatting branches** — display logic that varies by state:
```python
# Production code: disqualification warning only appears when disqualified
if result.disqualified:
    lines.append(f"⚠️  Disqualified: {result.disqualification_reason}")  # ← this line
# → Write: test_disqualified_listing_includes_warning_in_display
```

For each uncovered line, ask: is this a real requirement (write the spec), dead code
(remove it), or over-engineering (remove it and simplify)?

---

## Error Testing — Messages, Not Just Types

Verify message content, not just exception type. Future changes to error message
wording will break tests that check for the right message — that is the point.

```python
# ✅ Checks what the operator actually sees
with pytest.raises(ActionableError) as exc_info:
    registry.get("nonexistent_board")

assert "nonexistent_board" in str(exc_info.value), (
    f"Error should name the missing board. Got: {exc_info.value}"
)
assert "available" in str(exc_info.value).lower(), (
    f"Error should list available boards. Got: {exc_info.value}"
)

# ❌ Only confirms an exception occurred — message could be anything
with pytest.raises(ActionableError):
    registry.get("nonexistent_board")

# ❌ Constructs the error manually and asserts on the construction — always true
err = ActionableError(error=f"No adapter for 'nonexistent_board'")
assert "nonexistent_board" in str(err)
```

The third anti-pattern is particularly insidious — it looks like it tests error
content but only confirms that Python string formatting works.

---

## Test Markers

This repo uses pytest markers defined in `pyproject.toml`:

- `@pytest.mark.integration` — Tests requiring external services (Ollama, ChromaDB server)
- `@pytest.mark.live` — Tests requiring browser and live board access

Unit tests (no marker) should run fast with zero external dependencies.

---

## No Private-Function Imports

Tests must **never** import `_`-prefixed names from production modules.

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
    assert listings[0].title == "Staff Architect", (
        f"Expected 'Staff Architect', got {listings[0].title!r}"
    )
```

If a private function has complex logic that seems worth testing directly, that is a
signal it should be promoted to its own module — not a justification to import it.

---

## No Local Imports Inside Tests

All imports belong at the top of the file:

```python
# ❌ Local import inside a helper
class TestRescoreWorkflow:
    def _make_ranker(self):
        from jobsearch_rag.pipeline.ranker import Ranker  # buried
        return Ranker(...)

# ✅ Module-level import
from jobsearch_rag.pipeline.ranker import Ranker

class TestRescoreWorkflow:
    def _make_ranker(self):
        return Ranker(...)
```

---

## Mock Anti-Patterns

### ❌ Mocking the Subject Under Test

```python
# ❌ Wrong — mocking Scorer in a test that claims to test scoring
mock_scorer = MagicMock()
mock_scorer.score.return_value = ScoreResult(fit_score=0.8, ...)
result = mock_scorer.score(listing)
assert result.fit_score == 0.8  # tautology
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
registry.register("ziprecruiter", mock_adapter)
```

The registry has no I/O and no side effects. Mocking it hides misconfiguration.

### ❌ Direct Construction Bypassing the SUT

```python
# ❌ Wrong — inserts directly into ChromaDB, bypassing DecisionRecorder.record()
collection = recorder._store.get_or_create_collection("history")
collection.add(ids=["job-1"], documents=["Staff Architect at Acme"], metadatas=[{}])
results = collection.get(ids=["job-1"])
assert len(results["documents"]) == 1  # tautology — recorder.record() never called

# ✅ Correct — calls the real recorder, then verifies via public query
recorder.record(listing, verdict="yes", reason="Strong archetype match")
history = recorder.get_recent(limit=10)
assert any(item.external_id == listing.external_id for item in history), (
    f"Expected {listing.external_id} in history, got: {[i.external_id for i in history]}"
)
```

### ❌ Accessing Private Store Attributes

```python
# ❌ Wrong — _store and _registry are implementation details
collection = recorder._store.get_or_create_collection("history")
original = dict(AdapterRegistry._registry)

# ✅ Correct — test through public methods only
recorder.record(listing, verdict="yes")
runner.run()
```

### ❌ Patching Internal Parsing Functions

```python
# ❌ Wrong — card_to_listing is an internal parser, not an I/O boundary
patch("...card_to_listing", side_effect=ParseError)

# ✅ Correct — return mixed valid/invalid HTML from the Playwright locator mock
card_locator.all.return_value = [valid_card_mock, malformed_card_mock]
# Then assert output contains only the valid listing
```

### ❌ Repeated Patch Blocks

```python
# ❌ Wrong — same block repeated in every test method
def test_search_happy_path(self):
    with patch("...load_settings") as ms, patch("...PipelineRunner") as mr:
        ...

def test_search_no_results(self):
    with patch("...load_settings") as ms, patch("...PipelineRunner") as mr:
        ...

# ✅ Correct — extract to conftest fixture
@pytest.fixture
def cli_search_mocks():
    with patch("...load_settings") as ms, patch("...PipelineRunner") as mr:
        yield SimpleNamespace(settings=ms, runner=mr)
```

Shared setup belongs in conftest. Repeated inline blocks hide whether variation
between copies is intentional or drift.

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
`output/jds/`, `output/results.md`, and `output/results.csv` files.

**Do not bypass this guard.** Any test that exercises file output uses the
redirected path provided by the conftest fixture. Never hardcode or reference
the real `output/` directory in a test.
