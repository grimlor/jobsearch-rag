---
name: tool-usage
description: "Development tool preferences and execution patterns. Use when choosing between VS Code tools and terminal commands, handling long scripts, deciding how to execute file operations, tests, searches, or git commands."
---

# Tool Usage Guidelines

Standard tool-vs-terminal decision framework for this repository.

## Tool-First Approach

Use specialized VS Code tools instead of terminal commands. This is not a preference — it is a requirement. Tools provide structured output, integrated error reporting, and correct path resolution that raw terminal commands do not.

| Task | Use This Tool | Never This |
|------|--------------|----------|
| Read/edit files | `read_file`, `replace_string_in_file`, `create_file` | `cat`, `sed`, `echo` |
| Run tests | `runTests` tool | `pytest` in terminal |
| Check errors | `get_errors` tool | Manual inspection |
| Search code | `semantic_search`, `grep_search` | `grep`, `find` in terminal |
| Find files | `file_search`, `list_dir` | `ls`, `find` in terminal |
| Git status | `get_changed_files` | `git status`, `git diff` |

**Running tests via terminal is not permitted** except for the coverage exception below. The `runTests` tool handles test environment setup, path configuration, and output formatting that raw `pytest` commands will get wrong or miss entirely. Any session step that would otherwise run `python -m pytest ...` or `pytest ...` in the terminal must use `runTests` instead — no exceptions, including quick sanity checks.

**Coverage exception:** `runTests` is a VS Code Test Explorer integration and cannot pass arbitrary flags like `--cov` or `--cov-report`. When the explicit goal is generating a coverage report (not just running tests), use the terminal:

```bash
uv run pytest --cov=src/jobsearch_rag --cov-report=term-missing tests/
```

This exception applies only to deliberate coverage reporting steps, not to routine test runs during development.

## When Terminal Is Appropriate

- **Package installation**: `uv pip install`, `npm install`, `dotnet restore`, etc.
- **Build/compilation**: Complex build processes requiring environment setup
- **Background processes**: Servers, long-running tasks (`isBackground=true`)
- **Environment setup**: Python venv configuration, Azure CLI auth
- **Databricks CLI**: Workspace deployment, notebook sync
- **Coverage reporting**: `pytest --cov` when generating a coverage report (see above)
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
