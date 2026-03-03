```skill
---
name: conventional-commits
description: "Commit message format rules. Use whenever staging, committing, or describing changes — including when the user asks to commit, when preparing a PR, or when writing a changelog entry."
---

# Conventional Commits — Message Format

## When This Skill Applies

Whenever writing a commit message, preparing a PR title, or describing changes
for a changelog. This includes interactive commits, automated commits, and
squash-merge titles.

---

## Format

Every commit message follows [Conventional Commits v1.0.0](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Type (required)

| Type | Meaning | Version bump |
|------|---------|-------------|
| `feat` | New feature or capability | **minor** (0.1.0 → 0.2.0) |
| `fix` | Bug fix | **patch** (0.1.0 → 0.1.1) |
| `docs` | Documentation only | none |
| `style` | Formatting, whitespace, semicolons | none |
| `refactor` | Code change that neither fixes nor adds | none |
| `perf` | Performance improvement | none |
| `test` | Adding or correcting tests | none |
| `build` | Build system, dependencies, CI config | none |
| `ci` | CI pipeline changes | none |
| `chore` | Maintenance tasks (tooling, config) | none |
| `revert` | Reverts a previous commit | depends on reverted type |

### Breaking Changes

A breaking change triggers a **major** bump (0.2.0 → 1.0.0). Signal it with either:

- `!` after the type/scope: `feat!: redesign session format`
- A `BREAKING CHANGE:` footer in the body:

```
feat: redesign session format

BREAKING CHANGE: Session JSON schema now uses `evaluations` instead of `results`.
Existing session files must be migrated.
```

### Scope (optional)

Scope narrows the area of change. Use the module or subsystem name:

```
feat(scraper): add ZipRecruiter board support
fix(scoring): handle missing rubric fields gracefully
test(indexing): add RAG retrieval assertions
ci(release): add python-semantic-release workflow
```

Common scopes for this project:
- `scraper` — job board scraping and browser automation
- `scoring` — rubric evaluation and score fusion
- `indexing` — RAG indexing and ChromaDB operations
- `adapter` — board adapter registry and implementations
- `config` — settings, rubrics, role archetypes
- `export` — results export (CSV, Markdown)
- `session` — session management and persistence
- `deps` — dependency updates

### Description (required)

- Imperative mood: "add", "fix", "remove" — not "added", "fixes", "removed"
- Lowercase first letter (after the colon)
- No period at the end
- No more than 72 characters total for the first line

### Body (optional)

Explain *what* and *why*, not *how*. Wrap at 72 characters.

### Footer (optional)

- `BREAKING CHANGE: <description>` — triggers major bump
- `Refs: #123` — links to an issue
- `Co-authored-by: Name <email>` — attribution

---

## Examples

```
feat: add get_workflow_template tool

The agent needs a way to learn the workflow format after a one-click
install. This tool returns the template with format spec, skeleton,
and a concrete example — all from a resource file that ships inside
the package.

Refs: #42
```

```
fix(scoring): handle empty rubric sections without crashing

Previously, a role archetype with an empty skills section caused a
KeyError in the score calculator. Now it returns a zero score and
logs a warning.
```

```
ci(release): add python-semantic-release workflow

Automates version bumping and GitHub Release creation on push to
master. Reads conventional commit history since the last tag and
determines the appropriate semver increment.
```

```
docs: update README with installation instructions
```

```
test(indexing): add RAG retrieval assertions
```

```
build(deps): bump playwright from 1.40.0 to 1.42.0

BREAKING CHANGE: playwright 1.42.0 removes the deprecated sync API.
All browser automation has been updated to use async.
```

---

## Rules

1. **Every commit gets a type.** No untyped commits.
2. **One logical change per commit.** Don't bundle a feature with a refactor.
3. **Squash-merge PRs** should use the PR title as the commit message, which
   must also follow this format.
4. **`feat` and `fix` are the only types that trigger version bumps.** All
   other types are recorded in the changelog but don't change the version.
5. **Use `!` or `BREAKING CHANGE:` for breaking changes.** Both are equivalent;
   `!` is shorter for simple cases.

---

## Why This Exists

`python-semantic-release` reads commit messages to determine version bumps
automatically. Without consistent conventional commits, the automation cannot
determine whether a change is a feature, fix, or breaking change — and either
bumps incorrectly or not at all.

```
