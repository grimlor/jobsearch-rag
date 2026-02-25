---
name: tool-usage
description: "Development tool preferences and execution patterns. Use when choosing between VS Code tools and terminal commands, handling long scripts, deciding how to execute file operations, tests, searches, or git commands."
---

# Tool Usage Guidelines

Standard tool-vs-terminal decision framework for this repository.

## Prerequisites

The tool-first approach below depends on these VS Code extensions feeding diagnostics into the Problems panel (surfaced by `get_errors`):

| Extension | ID | Purpose |
|---|---|---|
| Pylance | `ms-python.vscode-pylance` | Type-checking (strict mode), unused imports, type-ignore validation |
| Ruff | `charliermarsh.ruff` | Lint rules from `pyproject.toml` (TC, RUF, E, W, F, I, N, UP, B, SIM) |
| Mypy Type Checker | `matangover.mypy` | mypy diagnostics via dmypy daemon |

**Required settings** (User or Workspace):

- `python.analysis.typeCheckingMode`: `"strict"`
- `python.analysis.diagnosticSeverityOverrides`: `reportUnusedImport: "error"`, `reportUnusedVariable: "error"`, `reportUnnecessaryTypeIgnoreComment: "error"`, `reportUnknownMemberType: "none"`
- `ruff.configurationPreference`: `"filesystemFirst"` (uses `pyproject.toml` rules)
- `ruff.fixAll`: `true`
- `mypy.dmypyExecutable`: absolute path to `.venv/bin/dmypy` (set in workspace `.vscode/settings.json` for portability)

Without these extensions and settings, `get_errors` will not cover the full lint/type surface and terminal fallbacks become necessary.

### Known gap — Ruff severity

The Ruff extension hardcodes diagnostic severity in `_get_severity()`: only `F821`, `E902`, and `E999` are reported as **Error**; every other rule is **Warning**. The `get_errors` tool only returns error-severity diagnostics—so most Ruff findings (TC, RUF, SIM, UP, N, B, etc.) are invisible to it.

**Impact:** `get_errors` reliably covers Pylance (strict) and mypy (all errors), but **not** the full Ruff rule set. After completing edits, run `task check` in the terminal to catch any Ruff warnings that `get_errors` missed. This is the one accepted exception to the tool-first rule above.

## Tool-First Approach

Use specialized VS Code tools instead of terminal commands. This is not a preference — it is a requirement. Tools provide structured output, integrated error reporting, and correct path resolution that raw terminal commands do not.

| Task | Use This Tool | Never This |
|------|--------------|----------|
| Read/edit files | `read_file`, `replace_string_in_file`, `create_file` | `cat`, `sed`, `echo` |
| Run tests | `runTests` tool | `pytest` in terminal |
| Check errors | `get_errors` tool (Pylance + mypy; partial Ruff) | `mypy` in terminal — but see Ruff gap below |
| Search code | `semantic_search`, `grep_search` | `grep`, `find` in terminal |
| Find files | `file_search`, `list_dir` | `ls`, `find` in terminal |
| Git status | `get_changed_files` | `git status`, `git diff` |
| Run Python snippets | `pylanceRunCodeSnippet` | `python` / `uv run python` in terminal |

**Running tests via terminal is not permitted** except for the coverage exception below. The `runTests` tool handles test environment setup, path configuration, and output formatting that raw `pytest` commands will get wrong or miss entirely. Any session step that would otherwise run `python -m pytest ...` or `pytest ...` in the terminal must use `runTests` instead — no exceptions, including quick sanity checks.

**Python snippets via `pylanceRunCodeSnippet`:** When you need to execute a Python script — bulk renames via regex, data transformations, one-off computations, or any throwaway code — use `pylanceRunCodeSnippet` instead of running `python` or `uv run python` in the terminal. It is faster, uses the workspace's configured Python environment automatically, and returns structured output. Reserve the terminal for commands that are not Python (shell utilities, package managers, build tools).

**Coverage exception:** `runTests` is a VS Code Test Explorer integration and cannot pass arbitrary flags like `--cov` or `--cov-report`. When the explicit goal is generating a coverage report (not just running tests), use the terminal:

```bash
uv run pytest --cov=<package> --cov-report=term-missing tests/
```

This exception applies only to deliberate coverage reporting steps, not to routine test runs during development.

**Linting via `get_errors`:** The VS Code Problems panel aggregates diagnostics from three sources: Pylance (strict type-checking), Ruff (lint rules from `pyproject.toml`), and mypy (via dmypy daemon). However, `get_errors` only returns error-severity diagnostics. Pylance and mypy report at error severity, so they are fully covered. Ruff only reports three rules as errors (F821, E902, E999) — all other Ruff findings are warnings and invisible to `get_errors`. After completing edits, run `task check` in the terminal to catch any Ruff warnings. Do not run `mypy` in the terminal; `get_errors` already covers it.

## When Terminal Is Appropriate

- **Package installation**: `uv pip install`, `npm install`, `dotnet restore`, etc.
- **Build/compilation**: Complex build processes requiring environment setup
- **Background processes**: Servers, long-running tasks (`isBackground=true`)
- **Environment setup**: Python venv configuration, Azure CLI auth
- **Databricks CLI**: Workspace deployment, notebook sync
- **Coverage reporting**: `pytest --cov` when generating a coverage report (see above)
- **Ruff lint sweep**: `task check` after edits are complete, to catch Ruff warnings invisible to `get_errors` (see Known gap above)
- **Commands with no tool equivalent**: When no specialized tool exists

`pytest` for general test runs is not on this list. It has a tool equivalent — `runTests` — and that tool must be used.

## Script Handling

| Script Size | Approach |
|-------------|----------|
| ≤ 10 lines | Run directly in terminal |
| > 10 lines | Create a script file, then execute it |

**For long scripts:**
1. Store scripts in `.copilot/scripts/` (git-ignored)
2. Use `create_file` to write the script
3. Use `run_in_terminal` to execute it
4. This prevents terminal buffer overflow and Pty failures

## Why This Matters

- **Faster execution**: Tools are optimized for VS Code integration
- **Better context**: Structured data instead of raw text parsing
- **Error handling**: Built-in validation catches issues early
- **Iteration speed**: Especially impactful for testing and file operations
