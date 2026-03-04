---
name: skill-compliance
description: "ALWAYS READ FIRST. Before performing any task, read all relevant skills and confirm to the user which skills were loaded. Use at the start of every task, before any other skill."
---

# Skill Compliance — Pre-Task Checklist

## Purpose

This skill exists to ensure all relevant skills are read and acknowledged
before any work begins. It is not optional. It applies to every task.

---

## Procedure

### Step 1 — Identify Relevant Skills

Before writing any code, editing any file, or executing any command, scan
the task description and identify which skills apply. The available skills
are in `.github/skills/`. Read the directory listing first.

### Step 2 — Read Each Relevant Skill

Read each identified skill's `SKILL.md` in full. Do not skim. Do not
summarize from memory. Use the file read tool to load the complete content.

At minimum, the following skills apply to almost every task:
- `tool-usage` — applies whenever you use any tool or terminal command
- `bdd-testing` — applies whenever tests are involved
- `bdd-feedback-loop` — applies whenever implementing from a spec doc

### Step 3 — Confirm to the User

Before beginning any work, post a message to the user that includes:

1. The task as you understand it (one sentence)
2. The skills you loaded (list each by name)
3. Any skills you considered but determined were not relevant, with a
   one-line reason for exclusion

Example:

> **Task:** Implement test coverage for the scoring module.
> **Skills loaded:** tool-usage, bdd-testing, bdd-feedback-loop
> **Skills excluded:** feature-workflow (no new feature branch needed),
> plan-updates (no plan doc changes required)

Do not begin work until this confirmation is posted.

### Step 4 — Proceed

After posting the confirmation, begin the task using the procedures
defined in the loaded skills.

If at any point during the task you realize an additional skill applies
that you did not load, stop, read it, and post an amended confirmation
before continuing.

---

## Why This Exists

Skills are reference material, not constraints. The agent must actively
choose to read and follow them. This skill makes that choice explicit
and observable, so the user can verify compliance before work begins
rather than discovering non-compliance after the fact.
