---
name: test-rewrite
description: "Session charter for the BDD test suite regeneration. Use at the start of any session involving writing, regenerating, or reviewing test files for JobSearch-RAG."
---

# Test Rewrite — Session Charter

## Purpose of This Session

This is a **test regeneration session**, not a feature implementation session.
The goal is to produce a test suite derived entirely from behavioral specifications,
with no coupling to implementation internals.

---

## Authoritative Sources

**What to test** comes from:

```
AI Systems Portfolio/Portfolio Projects/JobSearch-RAG/BDD Specifications.md
```

Read this document in full before writing any test code. All test class names,
method names, and scenario descriptions are derived from it exclusively.

**Fixture infrastructure** comes from:

```
tests/conftest.py
```

Read this file before writing any test file. It provides shared fixtures that
all test files build on. Do not reconstruct what it already provides.

**Do not read `src/`** — implementation files are closed for this session.

---

## What conftest.py Provides

`conftest.py` contains two categories of infrastructure that all test files
depend on. Both are mandatory.

**Shared I/O stubs** — the `embedder` fixture constructs an `Embedder` via
`__new__` (bypassing `__init__` to avoid Ollama client construction) and
replaces `embed`, `classify`, and `health_check` with `AsyncMock`. Use it —
do not create local embedder mocks in individual test files. Any setup pattern
that recurs across two or more test classes belongs here as a fixture, not
inline in test methods.

**Output directory guard** — conftest redirects the application's output
directory to a temporary path for the duration of every test run. This prevents
tests from writing to the real `output/jds/`, `output/results.md`, and
`output/results.csv` files produced by live search runs. Do not bypass this
guard. Any test that exercises file output uses the redirected path provided by
the conftest fixture. Never hardcode or reference the real `output/` directory
in a test.

---

## Active Skills for This Session

| Skill | Role |
|---|---|
| `bdd-testing` | Test organization, docstring format, mocking rules, anti-patterns |
| `plan-updates` | Update `BDD Specifications.md` and `Project Plan.md` on completion |
| `tool-usage` | Tool vs. terminal decisions, script handling |
| `feature-workflow` | **Not active** — this session has no new features |

Read `bdd-testing/SKILL.md` and its `references/test-patterns.md` before
writing any test. Pay particular attention to the Mock Anti-Patterns and
conftest sections.

---

## Never Import These Symbols

These are internal helpers. Test them through the public API they serve —
never directly.

| Symbol | Test Through Instead |
|---|---|
| `is_throttle_response` | `ZipRecruiterAdapter.search()` |
| `check_linkedin_detection` | `runner.run()` or `SessionManager` |
| `_wait_for_cloudflare` | `adapter.search()` |
| Any `_`-prefixed name | Its public caller |
| Module-level constants | Behavior that depends on them |

For the full allowed-imports table and mocking boundary rules, see the
"API Surface and Mocking Constraints" section in `BDD Specifications.md`.

---

## Session Start Checklist

Before writing any code:

- [ ] Read `BDD Specifications.md` in full
- [ ] Read `tests/conftest.py` in full
- [ ] Read `bdd-testing/SKILL.md`
- [ ] Read `bdd-testing/references/test-patterns.md`
- [ ] Confirm `src/` files are closed
