---
name: feature-workflow
description: "Spec-before-code feature development workflow. Use when the user requests a new feature, enhancement, or non-trivial change — anything that adds or modifies behavior, including requests phrased as add, implement, build, create, or start implementation."
---

# Feature Workflow — Spec Before Code

## When This Skill Applies

Whenever the user requests a new feature, enhancement, or non-trivial change — anything
that adds or modifies behavior. This includes requests phrased as "add …", "implement …",
"build …", "create …", "I need …", or "Start implementation".

This skill does **NOT** apply to:
- Bug fixes with a clear cause and obvious one-line fix
- Typo corrections, formatting, or comment-only changes
- Pure refactoring that preserves existing behavior (tests already exist)
- Questions, explanations, or research tasks

---

## The Problem This Solves

AI agents default to writing code immediately. This produces rework, scope creep,
and implementations that solve *a* problem but not *the user's* problem. The user's
established workflow requires specification before implementation — every time.

---

## The Required Workflow

Every feature request MUST proceed through these phases in order.
**Do not skip phases. Do not combine phases. Do not start implementation before tests exist.**

### Phase 1 — Planning (User Stories / Spec)

**Goal:** "Are we building the right thing?"

1. **Ask clarifying questions** — Do not assume. Identify ambiguity and resolve it.
2. **Write user stories or scenarios** that describe the feature from the consumer's
   perspective (user, downstream module, AI agent — whoever benefits).
3. **Create or update the spec** — Add scenarios to the BDD Specifications document
   (`BDD Specifications.md` in the Obsidian vault) following the format already
   established there. Each spec class needs:
   - `REQUIREMENT:` one-line capability statement
   - `WHO:` the stakeholder
   - `WHAT:` concrete, testable behavior
   - `WHY:` what goes wrong if this requirement is missing
4. **Present the plan to the user for review** before proceeding.

Reference: `Patterns & Practices/plan-first-agentic-development.md`

### Phase 2 — BDD Test Specification

**Goal:** "How do we know it works?"

1. **Create test classes** from the specs written in Phase 1.
2. **Follow BDD testing principles** — see the `bdd-testing` skill for details.
3. **Tests must fail** — Run the tests to confirm they fail. If they pass,
   either the behavior already exists or the tests aren't testing anything.
4. **Include failure-mode specs** — An unspecified failure is an unhandled failure.
   Test error paths, edge cases, and boundary conditions.

Reference: `Patterns & Practices/spec-first-bdd-testing-patterns.md`

### Phase 3 — Implementation

**Goal:** "Build it."

1. **Write code to make the failing tests pass.** The tests are the specification —
   implementation is done when all tests pass.
2. **Follow existing code patterns** — ActionableError for errors, factory methods,
   async patterns, etc. Check existing modules for conventions.
3. **Do not add behavior that isn't specified by a test.** If you discover a need
   during implementation, go back to Phase 2 and add the spec first.

### Phase 4 — Coverage Verification

**Goal:** "Is the specification complete?"

1. **Run tests with coverage:** `pytest --cov=jobsearch_rag --cov-report=term-missing`
2. **Every uncovered line is an unspecified requirement.** For each:
   - Is this a real requirement? → Write the spec, then keep the code.
   - Is this dead code? → Remove it.
   - Is this over-engineering? → Remove it and simplify.
3. **Target: 100% coverage.** Not as a vanity metric — as proof that every line of
   code has a specification justifying its existence.

Three categories routinely surface only at coverage time:
- **Defensive guard code** — misuse protection
- **Graceful degradation paths** — soft failures the system absorbs
- **Conditional formatting branches** — display logic that varies by state

### Phase 5 — Plan Status Update

**Goal:** "Record what was done."

1. **Update the Project Plan** (`Project Plan.md` in the Obsidian vault) —
   check off completed items, add new line items if scope expanded.
2. **Update the BDD Specifications** if any specs were added or modified during
   implementation (Phase 3 discoveries).

---

## Critical Rules

- **NEVER start writing production code before test specs exist and fail.**
- **NEVER treat "Start implementation" as permission to skip planning.** If the
  user says "Start implementation" and there are no specs yet, Phase 1 is the
  starting point. If specs exist but tests don't, Phase 2 is the starting point.
- **Present each phase's output to the user** before moving to the next phase.
- **Use the todo list** to track progress through phases — this gives the user
  visibility into where you are in the workflow.
