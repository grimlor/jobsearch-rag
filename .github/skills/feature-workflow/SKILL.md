---
name: feature-workflow
description: "Spec-before-code feature development workflow. Use when the user requests a new feature, enhancement, or non-trivial change — anything that adds or modifies behavior, including requests phrased as add, implement, build, create, refactor, or start implementation."
---

# Feature Workflow — Spec Before Code

## When This Skill Applies

Whenever the user requests a new feature, enhancement, or non-trivial change — anything
that adds or modifies behavior. This includes requests phrased as "add …", "implement …",
"build …", "create …", "refactor …", "I need …", or "Start implementation".

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

### Phase 1 — Spec Gate (Planning)

**Goal:** "Are we building the right thing?" and "Does a reviewed spec exist?"

Before writing any implementation code, answer these three questions:

1. **Does a spec document exist for this feature?**
   Look in the project's spec location (BDD Specifications doc, architecture docs,
   or a feature spec file). A spec exists if it describes WHAT the feature does,
   WHO uses it, and what the public API surface looks like.

2. **Is the spec complete enough to implement against?**
   A minimum viable spec must include:
   - WHAT the feature does (behavior, not implementation)
   - WHAT it explicitly does NOT do (scope boundary)
   - The public API surface: method signatures, parameters, return types
   - At least one scenario per major behavior path (Given / When / Then)

3. **Has the human reviewed the spec?**
   A spec written in the same session as implementation has not been reviewed.
   If the spec was just created, stop and wait for human sign-off before
   proceeding to implementation.

**If the answer to any of these is no — stop. Create or complete the spec first.**

#### If No Spec Exists

1. **Ask clarifying questions** — Do not assume. Identify ambiguity and resolve it.
2. **Write user stories or scenarios** that describe the feature from the consumer's
   perspective (user, downstream module, AI agent — whoever benefits).
3. **Create the spec** using this structure:

   ```markdown
   # Spec — <Feature Name>

   ## Overview
   One paragraph: what this feature does and why it exists.

   ## Out of Scope
   Explicit list of what this feature does NOT do.

   ## Public API Surface
   # New symbols this feature will expose:
   #   ClassName(param: Type, ...) -> ReturnType
   #   module_function(param: Type) -> ReturnType

   ## Behaviors
   ### <BehaviorName>
   REQUIREMENT: <one sentence>
   WHO: <who depends on this behavior>
   WHAT: <what the behavior does, observable from outside>
   WHY: <why is this requirement important? what happens if it's not met?>

   MOCK BOUNDARY:
       Mock:  <what to stub>
       Real:  <what must be real>
       Never: <what must never be mocked>

   Scenarios:
   - Given <precondition> / When <action> / Then <outcome>
   ```

4. **Present the spec to the user for review** and wait for explicit approval
   before proceeding to Phase 2.

#### If the Spec Exists but Has Gaps

Gaps discovered during any phase:
1. Stop at the point where the gap was discovered
2. Add the missing behavior to the spec document
3. Present the gap and proposed spec addition to the human
4. Wait for approval before continuing

Do not silently fill gaps with undocumented behavior.

### Phase 2 — BDD Test Specification

**Goal:** "How do we know it works?"

1. **Create test classes** from the specs written in Phase 1.
2. **Follow BDD testing principles** — see the `bdd-testing` skill for conventions.
3. **Tests must fail** — Run the tests to confirm they fail. Refer to tool-usage skill. 
   If they pass, either the behavior already exists or the tests aren't testing anything.
4. **Include failure-mode specs** — An unspecified failure is an unhandled failure.
   Test error paths, edge cases, and boundary conditions.

### Phase 3 — Implementation

**Goal:** "Build it."

1. **Write code to make the failing tests pass.** The tests are the specification —
   implementation is done when all tests pass.
2. **Follow existing code patterns** — Check existing modules for conventions
   (error handling patterns, factory methods, async patterns, etc.).
3. **Do not add behavior that isn't specified by a test.** If you discover a need
   during implementation, go back to Phase 2 and add the spec first.

### Phase 4 — Coverage Verification

**Goal:** "Is the specification complete?"

1. **Run tests with coverage** for the project's source package. 
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

**"Pre-existing" is not a category.** Whether a line existed before your changes is irrelevant — if it is uncovered after your work, it is uncovered. The only valid dispositions are: real requirement (write the spec), dead code (remove it), or over-engineering (remove it). "It was already there" is not a disposition.

### Phase 5 — Plan Status Update

**Goal:** "Record what was done."

1. **Update the project's plan document** — check off completed items, add new
   line items if scope expanded.
2. **Update BDD Specifications** if any specs were added or modified during
   implementation (Phase 3 discoveries).
3. See the `plan-updates` skill for detailed rules on tracking progress.

---

## Critical Rules

- **NEVER start writing production code before test specs exist and fail.**
- **NEVER treat "Start implementation" as permission to skip planning.** If the
  user says "Start implementation" and there are no specs yet, Phase 1 is the
  starting point. If specs exist but tests don't, Phase 2 is the starting point.
- **Present each phase's output to the user** before moving to the next phase.
- **Use the todo list** to track progress through phases — this gives the user
  visibility into where you are in the workflow.

---

## Relationship to Other Skills

- `feature-workflow` (this skill) governs the full lifecycle — spec through plan update
- `bdd-testing` governs test quality — referenced from Phase 2 and the `bdd-feedback-loop`
- `bdd-feedback-loop` governs per-module test implementation — used during Phase 2
- `plan-updates` governs progress tracking — used during Phase 5
- `tool-usage` is cross-cutting — applies at every phase

The flow: **spec gate → human review → tests → implementation → gaps → spec update → continue**
