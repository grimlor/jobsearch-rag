---
name: plan-updates
description: "Progress tracking in project artifacts. Use after completing implementation work (Phase 5 of the feature workflow), or when the user asks to update project status."
---

# Plan Updates — Tracking Progress in Project Artifacts

## When This Skill Applies

After completing implementation work (Phase 5 of the feature workflow), or whenever
the user asks to update project status. Also applies when reviewing what has been
done vs. what remains.

---

## Artifacts to Update

### 1. Project Plan

**Location:** The project's plan document — typically a Markdown file with checkbox
lists organized by phase. Look for it in the `.copilot/work plans/` directory,
project root, or ask the user for the location.

Each phase has a heading like `### Phase N — Description` followed by checkboxes:

```markdown
### Phase 4c — Operational Resilience
- [x] File logging — persist logs alongside stderr output
- [x] Throttle detection — recognize rate limiting signals
- [ ] Some future item not yet implemented
```

**Rules:**
- Check off items (`- [x]`) only when implementation is complete AND tests pass
- Add new line items when scope expands during implementation
- Append the phase marker (e.g., `✅`) to the phase heading when all items are checked
- If a new phase is needed, follow the existing naming convention (`Phase Na`, `Phase Nb`, etc.)
- Include the BDD spec file and test count when checking off test-related items
  (e.g., `BDD specs: test_comp_parser.py (22 tests), test_comp_scoring.py (15 tests)`)
- Brief descriptions should explain WHAT was built and WHERE in the codebase

### 2. BDD Specifications

**Location:** The project's BDD specifications document — a canonical listing of all
behavioral contracts. Look for it alongside the project plan, or in a `specs/` or
`tests/` directory.

The BDD Specifications document contains `TestClass` definitions with method signatures
(no bodies — just `...`).

**Rules:**
- Add new spec classes during Phase 1 (planning) of the feature workflow
- If implementation reveals requirements not in the spec, add them here
- Keep the document in sync with actual test files — if a test class exists in code,
  its spec should exist here
- Follow the existing format: class docstring with REQUIREMENT/WHO/WHAT/WHY,
  then method signatures with `...` bodies
- Group specs under the appropriate section heading for the project's structure

---

## When NOT to Update

- Do not update plan status for work that is still in progress
- Do not check off items speculatively ("this should work")
- Do not modify the plan structure (section ordering, narrative text) unless asked
- Do not update the plan for trivial changes that don't correspond to plan items

---

## Update Workflow

1. Identify which plan items were completed
2. Check off completed items in the project plan
3. Add any new items that were discovered during implementation
4. Update BDD Specifications if specs were added or modified
5. Briefly confirm to the user what was updated
