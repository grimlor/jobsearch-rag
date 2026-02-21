---
name: tool-usage
description: "Development tool preferences and execution patterns. Use when choosing between VS Code tools and terminal commands, handling long scripts, deciding how to execute file operations, tests, searches, or git commands."
---

# Tool Usage Guidelines

Standard tool-vs-terminal decision framework for this repository.

## Tool-First Approach

Prefer specialized VS Code tools over terminal commands. Tools are faster, provide structured output, and have built-in error handling.

| Task | Use This Tool | Not This |
|------|--------------|----------|
| Read/edit files | `read_file`, `replace_string_in_file`, `create_file` | `cat`, `sed`, `echo` |
| Run tests | `runTests` tool | `pytest` in terminal |
| Check errors | `get_errors` tool | Manual inspection |
| Search code | `semantic_search`, `grep_search` | `grep`, `find` in terminal |
| Find files | `file_search`, `list_dir` | `ls`, `find` in terminal |
| Git status | `get_changed_files` | `git status`, `git diff` |

## When Terminal Is Appropriate

- **Package installation**: `uv pip install`, `npm install`, `dotnet restore`, etc.
- **Build/compilation**: Complex build processes requiring environment setup
- **Background processes**: Servers, long-running tasks (`isBackground=true`)
- **Environment setup**: Python venv configuration, Azure CLI auth
- **Databricks CLI**: Workspace deployment, notebook sync
- **Commands with no tool equivalent**: When no specialized tool exists

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
