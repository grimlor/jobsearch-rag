---
name: bdd-feedback-loop
description: "Feedback loop procedure for implementing BDD test modules. Use when implementing a spec doc — covering one test module from spec to Pylance-clean, self-audited output."
---

# BDD Feedback Loop — Test Implementation Procedure

## When This Skill Applies

Whenever implementing tests from a BDD spec document. Each iteration of this loop
covers one test module: read the spec, implement, verify, audit, log, and hand off.

Do not proceed to the next module if unresolved failures remain from Steps 4 or 5.

---

## The Loop (Per Module)

### Step 1 — Read the Spec Doc

Read the module's spec doc in full before writing any code.

The spec is the authoritative source of truth. It defines:
- Which test classes to write
- The REQUIREMENT / WHO / WHAT / WHY for each class
- The MOCK BOUNDARY contract for each class
- The scenario signatures (Given / When / Then) for each test method

Do not invent test classes or scenarios not present in the spec. Do not silently
correct what appears to be a spec error — flag it in the deviation log (Step 6)
and implement what the spec says.

---

### Step 2 — Discover the Public API

Read the relevant `src/` files for the module under test. Extract:
- Public method signatures (name, parameters, return type)
- Public class constructors (required and optional parameters)
- Public names (no `_` prefix)
- Return types — specifically whether methods return dataclasses, primitives,
  or raise exceptions

**This is the only permitted reason to read `src/` during test implementation.**

Do not use `src/` to find internal functions to mock. If a failure condition
cannot be induced through public API inputs alone, note it in the deviation log
rather than patching around it.

Record the discovered API surface as a brief comment block at the top of the
test file, for traceability:

```python
# Public API surface (from src/jobsearch_rag/rag/scorer.py):
#   Scorer(embedder: Embedder, store: VectorStore, config: Settings)
#   scorer.score(listing: JobListing) -> ScoreResult
#   scorer.compute_comp_score(comp_max: float | None, base: float) -> float
```

---

### Step 3 — Implement the Tests

Implement each test class and method exactly as specified in the spec doc.

**File creation is always incremental. Do not attempt to write the entire test
file in a single tool call — doing so will cause a timeout and produce no output.**

The required sequence for every module:
1. Create the file with the header (module docstring, API surface comment,
   imports, and any shared helpers) using `create_file`. No test classes yet.
2. Add one test class at a time using `replace_string_in_file`.
   A single tool call must contain exactly one test class — no more.
3. Repeat step 2 until all test classes from the spec are in the file.

For each class:
1. Copy the REQUIREMENT / WHO / WHAT / WHY docstring from the spec
2. Copy the MOCK BOUNDARY contract from the spec
3. Implement each scenario from the spec's Given / When / Then signatures

For each method:
1. Use the scenario signature from the spec as the docstring
2. Write Given / When / Then body comments
3. Use real instances per the MOCK BOUNDARY (Step 2 revealed the constructors)
4. Assert on SUT output, not on objects you constructed yourself
5. Include a diagnostic message on every assertion

Refer to `.github/skills/bdd-testing/SKILL.md` and
`.github/skills/bdd-testing/references/test-patterns.md` for all conventions.

---

### Step 4 — Run Pylance

After implementing each test file, run the `get_errors` tool on the test file.
Do not use `pyright` or `mypy` at the terminal — this violates the tool-usage skill.

For each reported error, attempt to resolve it. If an error cannot be resolved
after three attempts, log it as a deviation (Step 6) and continue to the next
error. Once all errors have been iterated, if any remain unresolved, do not
proceed to the next module — the module is blocked pending human review.

Common issues to fix:
- Missing imports (module not imported at top of file)
- Wrong argument types passed to constructors or methods
- Incompatible return type assignments
- Undefined names (typos in fixture names, method names)
- `AsyncMock` vs `MagicMock` mismatches on async methods

---

### Step 5 — Self-Audit Against BDD Principles

Read the completed test file and work through each checklist item below. For each
violation found, attempt to resolve it. If a violation cannot be resolved after
three attempts, log it as a deviation (Step 6) and continue to the next item.
Once all items have been iterated, if any violations remain unresolved, do not
proceed to the next module — the module is blocked pending human review.

**Tautology check — the most important:**
For every test method, ask: *if I deleted the module under test entirely, would
this test still pass?* If yes, it is a tautology. The When step must invoke
production code. The Then step must assert on what that production code returned.

Checklist — work through every item:

- [ ] Every test method's When step calls production code (no tautologies)
- [ ] No test constructs the expected output and asserts on the constructed object
- [ ] No test accesses `_`-prefixed attributes or methods on the SUT (`._store`, `._registry`, etc.)
- [ ] No test imports `_`-prefixed names from production modules
- [ ] Mock boundaries match the class-level MOCK BOUNDARY contract exactly
- [ ] Every assertion includes a diagnostic message
- [ ] All Given / When / Then body comments are present
- [ ] No local imports inside test methods or helper functions
- [ ] `pytest.approx` used for all float comparisons
- [ ] Error tests verify message content, not just exception type
- [ ] No `assert exc_info.value is not None` — this always passes inside `pytest.raises`

---

### Step 6 — Log Deviations

After Steps 4 and 5, record every item that could not be resolved. A deviation is
anything that prevented full compliance with the spec or with the BDD principles.

Append to the module's deviation log section in the orchestration doc:

```
## Deviations — test_scorer.py

### [DEVIATION] TestSemanticScoring.test_culture_score_penalizes_negative_signals
Could not induce negative_score > 0.3 through public API alone. The production
code path requires at least 3 documents in the negative_signals collection, but
the vector_store fixture only seeds 1. The test currently seeds manually via
store.add() — this bypasses the indexer but is the only path available.
Recommendation: add a multi-document fixture to conftest, or expose a
batch-seed method on VectorStore.

### [DEVIATION] TestCompScoring — entire class
compute_comp_score() is not exposed as a public method on Scorer. It appears
to be internal. All comp scoring tests currently call the private method
_compute_comp_score() directly, violating Principle 9.
Recommendation: either promote to public API or test exclusively through
scorer.score() with real comp data in the listing.
```

A deviation log entry must include:
- The specific test or class affected
- What the spec requires
- Why full compliance was not achievable
- A concrete recommendation for resolution

Vague entries ("couldn't make it work") are not acceptable. The log is the
handoff artifact — it must give the next person enough context to act without
re-investigation.

---

### Step 7 — Proceed to Next Module

If Steps 4 and 5 are both clean (or all remaining failures are logged in Step 6),
proceed to the next module in the orchestration doc.

If any unresolved failures exist that were not logged, stop and complete Step 6
before proceeding.

**Logged deviations do not authorize proceeding.** A deviation that cannot be resolved after three attempts — including deviations caused by spec-mandated mock boundaries — requires a human decision before the module is considered complete. Do not advance to the next module. Present the unresolved deviations and wait.

The orchestration doc defines the module order. Do not reorder modules without
updating the orchestration doc.

---

## Spec Immutability

The spec doc is an input to this loop, not an output. If the spec appears to be
wrong:

- **Minor wording issues** — correct silently
- **A scenario that seems incomplete** — implement what is written, note the gap
  in the deviation log
- **A scenario that is impossible to implement** — implement the closest compliant
  approximation, log the deviation with full explanation
- **A genuine error in a REQUIREMENT or MOCK BOUNDARY** — do not silently correct
  it; log the deviation and stop work on that class until the spec is updated

The spec encodes domain knowledge and behavioral contracts that were authored with
full system understanding. A test that contradicts the spec is more likely wrong
than the spec is.

---

## Deviation Log Format

The orchestration doc contains a `## Deviation Log` section. Each module gets its
own subsection. Use this format:

```markdown
## Deviation Log

### test_scorer.py
- [DEVIATION] TestSemanticScoring.test_culture_score_... — <one-line summary>
  <explanation and recommendation>

### test_config.py
- [CLEAN] No deviations.
```

Mark clean modules explicitly. A missing entry is ambiguous — it could mean clean
or could mean the loop was not completed.
