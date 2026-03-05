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

- `!` after the type/scope: `feat!: redesign workflow format`
- A `BREAKING CHANGE:` footer in the body:

```
feat: redesign workflow format

BREAKING CHANGE: Step headers now use `####` instead of `###`.
Existing workflows must be updated.
```

### Scope (optional)

Scope narrows the area of change. Use the module or subsystem name:

```
feat(auth): add OAuth2 token refresh
fix(api): handle missing query parameter gracefully
test(models): add validation assertions
ci(release): add semantic-release workflow
```

Define scopes based on your project's module structure. Good scopes are
short, stable names that map to subsystems or directories — e.g., `api`,
`auth`, `models`, `cli`, `config`, `deps`, `release`.

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
feat(auth): add OAuth2 token refresh

Access tokens now refresh automatically when they expire within
5 minutes of a request. This prevents 401 errors during long
running operations without requiring manual re-authentication.

Refs: #42
```

```
fix(api): handle missing query parameter without crashing

Previously, an empty `filter` parameter caused a TypeError in
the query builder. Now it defaults to an empty dict and logs
a warning.
```

```
ci(release): add semantic-release workflow

Automates version bumping and GitHub Release creation on push to
main. Reads conventional commit history since the last tag and
determines the appropriate semver increment.
```

```
docs: update README with installation instructions
```

```
test(models): add input validation assertions
```

```
build(deps): bump fastapi from 0.109.0 to 0.110.0

BREAKING CHANGE: 0.110.0 removes the deprecated `on_event`
lifecycle hook. All startup handlers have been migrated to
`lifespan` context managers.
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

Semantic release tools (e.g., `semantic-release`, `python-semantic-release`,
`release-please`) read commit messages to determine version bumps
automatically. Without consistent conventional commits, the automation cannot
determine whether a change is a feature, fix, or breaking change — and either
bumps incorrectly or not at all.
