---
name: tool-usage
description: "Development tool preferences and execution patterns. Use when choosing between VS Code tools and terminal commands, handling long scripts, deciding how to execute file operations, tests, searches, or git commands."
---

# Tool Usage Guidelines

Standard tool-vs-terminal decision framework.

## Prerequisites

The tool-first approach below depends on VS Code extensions feeding diagnostics into the Problems panel (surfaced by `get_errors`). For Python projects, the recommended stack is:

| Extension | ID | Purpose |
|---|---|---|
| Pylance | `ms-python.vscode-pylance` | Type-checking via Pyright (strict mode), unused imports, type-ignore validation |
| Ruff | `charliermarsh.ruff` | Lint rules from project config |

Pylance uses Pyright as its type-checking engine. Pyright is also available as a standalone CLI (`pyright` or `uv run pyright`). However, Pylance does not surface all Pyright diagnostics through `get_errors` — some issues only appear when running Pyright directly. After completing edits, run `pyright` in the terminal as a final verification step alongside the lint sweep.

**Recommended settings** (User or Workspace):

- `python.analysis.typeCheckingMode`: `"strict"`
- `python.analysis.diagnosticSeverityOverrides`: `reportUnusedImport: "error"`, `reportUnusedVariable: "error"`, `reportUnnecessaryTypeIgnoreComment: "error"`, `reportUnknownMemberType: "none"`
- `ruff.configurationPreference`: `"filesystemFirst"` (uses project-level config)
- `ruff.fixAll`: `true`

Without these extensions and settings, `get_errors` will not cover the full lint/type surface and terminal fallbacks become necessary.

Adapt these recommendations to your project's language and toolchain. The principle — use extensions that feed diagnostics into `get_errors` — is universal.

### Known gap — Ruff severity

The Ruff extension hardcodes diagnostic severity in `_get_severity()`: only `F821`, `E902`, and `E999` are reported as **Error**; every other rule is **Warning**. The `get_errors` tool only returns error-severity diagnostics — so most Ruff findings are invisible to it.

**Impact:** `get_errors` does **not** surface the full diagnostic set from either tool. Pylance omits some Pyright findings, and Ruff reports most rules as warnings (invisible to `get_errors`). After completing edits, run both `pyright` and your project's lint command in the terminal to catch what `get_errors` missed. This is the one accepted exception to the tool-first rule above.

## Tool-First Approach

Use specialized VS Code tools instead of terminal commands. This is not a preference — it is a requirement. Tools provide structured output, integrated error reporting, and correct path resolution that raw terminal commands do not.

| Task | Use This Tool | Never This |
|------|--------------|----------|
| Read/edit files | `read_file`, `replace_string_in_file`, `create_file` | `cat`, `sed`, `echo` |
| Run tests | `runTests` tool | `pytest` in terminal |
| Check errors | `get_errors` tool (Pylance/Pyright; partial Ruff) | — |
| Search code | `semantic_search`, `grep_search` | `grep`, `find` in terminal |
| Find files | `file_search`, `list_dir` | `ls`, `find` in terminal |
| Git status | `get_changed_files` | `git status`, `git diff` |

**Running tests via terminal is not permitted** except for the coverage exception below. The `runTests` tool handles test environment setup, path configuration, and output formatting that raw test commands will get wrong or miss entirely. Any session step that would otherwise run `pytest`, `jest`, `dotnet test`, etc. in the terminal must use `runTests` instead — no exceptions, including quick sanity checks.

**Coverage exception:** `runTests` is a VS Code Test Explorer integration and cannot pass arbitrary flags like `--cov` or `--cov-report`. When the explicit goal is generating a coverage report (not just running tests), use the terminal:

```bash
# Python example
pytest --cov=<package> --cov-report=term-missing tests/
```

This exception applies only to deliberate coverage reporting steps, not to routine test runs during development.

**Terminal verification:** The VS Code Problems panel aggregates diagnostics from configured extensions, but `get_errors` does not surface everything. Pylance omits some Pyright diagnostics, and Ruff reports most rules as warnings (invisible to `get_errors`). After completing edits, run both `pyright` and your project's lint command in the terminal as a final verification step.

## When Terminal Is Appropriate

- **Package installation**: `pip install`, `npm install`, `dotnet restore`, etc.
- **Build/compilation**: Complex build processes requiring environment setup
- **Background processes**: Servers, long-running tasks (`isBackground=true`)
- **Environment setup**: Virtual environments, cloud CLI auth
- **Coverage reporting**: Test coverage report generation (see above)
- **Type-check + lint sweep**: Running `pyright` and linters after edits to catch diagnostics invisible to `get_errors`
- **Commands with no tool equivalent**: When no specialized tool exists

General test runs are not on this list. They have a tool equivalent — `runTests` — and that tool must be used.

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
