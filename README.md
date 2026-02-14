# Job Search RAG Assistant

> Local LLM + RAG pipeline that automates intelligent job filtering across
> multiple job boards. Encodes your personal hiring heuristics as semantic
> search rather than keyword matching.

## Quick Start

```bash
# Install dependencies (creates .venv automatically)
uv sync --extra dev

# If using direnv, allow the .envrc to auto-activate the venv
direnv allow

# Install Playwright browsers
uv run playwright install chromium

# Index your resume and role archetypes
uv run python -m jobsearch_rag index

# Run a search
uv run python -m jobsearch_rag search
```

## Development

```bash
# Lint
uv run task lint

# Format
uv run task format

# Type check
uv run task type

# Test
uv run task test

# All checks
uv run task check
```

## Architecture

See the project plan and BDD specifications in the docs directory for
full architectural documentation.
