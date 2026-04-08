# Job Search RAG Assistant

> Local LLM + RAG pipeline that automates intelligent job filtering across
> multiple job boards. Encodes your hiring judgment as semantic search —
> not keyword matching — and runs entirely on your machine.

[![CI](https://github.com/grimlor/jobsearch-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/grimlor/jobsearch-rag/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/grimlor/b6bda813bb4ef28ff592b6c81a09c791/raw/jobsearch-rag-coverage.json)](https://github.com/grimlor/jobsearch-rag/actions/workflows/ci.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11–3.13-blue)](https://www.python.org)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm%20Noncommercial-blue)](LICENSE)

---

## Why This Exists

Senior-level job searches on major boards are high-noise, low-precision.
Mislabeled IC roles, vendor-chain postings, keyword-match junk, and irrelevant
industries overwhelm every search page. Signal-based tools still produce
misaligned results because they match keywords, not judgment.

This project encodes your judgment once — as semantic embeddings built from
your resume, your target role archetypes, a global evaluation rubric, and your
own growing decision history — then applies it continuously across every board
you care about. Everything runs locally via
[Ollama](https://ollama.com) and [ChromaDB](https://www.trychroma.com/);
no data leaves your machine.

---

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│                     CLI Entry Point                          │
│               python -m jobsearch_rag [command]              │
└────────────────────────┬─────────────────────────────────────┘
                         │
          ┌──────────────┴───────────────┐
          │                              │
┌─────────▼───────────┐       ┌──────────▼──────────────────┐
│   Adapter Layer     │       │    RAG Pipeline             │
│  (IoC / Strategy)   │       │  (Ollama + ChromaDB)        │
│                     │       │                             │
│ AdapterRegistry     │       │ - Resume embedding          │
│ JobBoardAdapter ABC │       │ - Role archetypes           │
│  ├ ZipRecruiter     │       │ - Global positive signals   │
│  ├ Indeed           │       │ - Negative signal penalty   │
│  ├ WeWorkRemotely   │       │ - Decision history          │
│  └ LinkedIn*        │       │ - Compensation scoring      │
│                     │       │ - LLM disqualifier          │
└─────────┬───────────┘       └──────────┬──────────────────┘
          │                              │
          └──────────────┬───────────────┘
                         │
                ┌────────▼──────────┐
                │  Ranker &         │
                │  Exporter         │
                │                   │
                │ - Score fusion    │
                │ - Deduplication   │
                │ - Markdown / CSV  │
                │ - JD files        │
                │ - Browser tabs    │
                └───────────────────┘

* LinkedIn: overnight mode, stealth, throttled
```

### Scoring Model: Two Orthogonal Axes

The scoring pipeline encodes two independent dimensions of judgment:

- **What kind of role** — `archetype_score` (role type match) + `fit_score`
  (resume alignment) + `comp_score` (compensation vs. target)
- **What kind of environment** — `culture_score` (work model, ethics,
  autonomy preferences) + `negative_score` (red-flag penalty for adtech,
  chaos culture, seniority mismatch, etc.)

A role must score well on **both** axes to rank highly. The LLM disqualifier
acts as a binary gate for structural problems keywords can't catch (IC roles
disguised as architect titles, staffing-agency postings).

### Pipeline Steps

1. **Adapter layer** — Playwright-based browser automation loads job board pages
   exactly as a human would. Each board has its own adapter; all produce the
   same `JobListing` data contract.
2. **Scoring pipeline** — Job descriptions are embedded locally via
   `nomic-embed-text` and scored against six ChromaDB collections
   (resume, role archetypes, global positive signals, negative signals,
   decision history, plus inline compensation parsing).
3. **LLM disqualifier** — A `mistral:7b` pass screens for structural
   mismatches, personalized with your past rejection reasons.
   Defense-in-depth prompt injection mitigation protects this stage.
4. **Ranker** — Weighted score fusion, cross-board deduplication
   (cosine > 0.95), and minimum-score threshold filtering.
5. **Export** — Ranked results as Markdown table, CSV with compensation
   columns, individual JD files, and/or browser tabs.

### Feedback Loop

Your decisions feed back into the system:

```
search → review → decide (yes/no/maybe + reason)
                    │
                    ├──▶ decisions collection (future history_score)
                    ├──▶ rejection reasons → personalize LLM disqualifier
                    └──▶ rescore → eval (measure agreement, precision, recall)
```

Over time, the pipeline learns your preferences without retraining any models.

---

## Prerequisites

| Requirement | Install |
|---|---|
| [Python 3.11+](https://www.python.org) | `brew install python@3.13` or via [uv](https://docs.astral.sh/uv/) |
| [uv](https://docs.astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| [Ollama](https://ollama.com) | `brew install ollama` |
| [direnv](https://direnv.net/) | `brew install direnv` *(optional — auto-activates venv)* |

## Quick Start

```bash
# Clone and install
git clone https://github.com/grimlor/jobsearch-rag.git
cd jobsearch-rag
uv sync --extra dev

# Optional: auto-activate venv via direnv
direnv allow

# Install Playwright browsers
uv run playwright install chromium

# Start Ollama and pull models
ollama serve &                          # if not already running
ollama pull mistral:7b
ollama pull nomic-embed-text

# Configure your search (edit these files)
#   config/settings.toml        — board URLs, scoring weights, output prefs
#   config/role_archetypes.toml — descriptions of your target roles
#   config/global_rubric.toml   — universal evaluation dimensions
#   data/resume.md              — your resume in plain Markdown

# Index your resume, archetypes, and rubric into ChromaDB
uv run python -m jobsearch_rag index

# Run a search across enabled boards
uv run python -m jobsearch_rag search

# Review results interactively (builds decision history)
uv run python -m jobsearch_rag review
```

## CLI Reference

```bash
# Core workflow
jobsearch-rag index                     # Index resume + archetypes + rubric into ChromaDB
jobsearch-rag search                    # Search → score → rank → export (cumulative by default)
jobsearch-rag search --fresh            # Fresh search (discard prior accumulated results)
jobsearch-rag search --board linkedin --overnight  # Single board, overnight throttle
jobsearch-rag rescore                   # Re-score exported JDs without browser (tuning loop)
jobsearch-rag rescore --force-rescore   # Re-score including previously decided listings

# Review & decisions
jobsearch-rag review                    # Interactive batch review of undecided listings
jobsearch-rag decide <job_id> --verdict yes|no|maybe
jobsearch-rag decide <job_id> --verdict no --reason "Requires on-call rotation"
jobsearch-rag decisions show <job_id>   # Inspect a stored decision
jobsearch-rag decisions remove <job_id> # Remove from ChromaDB (JSONL audit log preserved)
jobsearch-rag decisions audit           # List all decisions with reasons

# Evaluation
jobsearch-rag eval                      # Pipeline vs. human decisions: agreement, precision, recall
jobsearch-rag eval --compare-models mistral:7b llama3.1:8b  # A/B test disqualifier models

# Utilities
jobsearch-rag boards                    # List registered adapters
jobsearch-rag login --board ziprecruiter --browser msedge  # Interactive session login
jobsearch-rag reset                     # Clear ChromaDB collections
jobsearch-rag reset --clear-output      # Also clear output directory

# Indexing variants
jobsearch-rag index --resume-only       # Re-index resume only
jobsearch-rag index --archetypes-only   # Re-index archetypes + rubric + negative signals
```

> **Note:** Replace `jobsearch-rag` with `uv run python -m jobsearch_rag` if
> your venv is not activated.

## Configuration

All configuration lives in three TOML files under `config/`:

| File | Purpose |
|---|---|
| **`settings.toml`** | Enabled boards, search URLs, scoring weights, Ollama models, output paths, comp bands, security prompts |
| **`role_archetypes.toml`** | Natural language descriptions of your target roles with positive/negative signals |
| **`global_rubric.toml`** | Ten universal evaluation dimensions (culture, ethics, seniority, operational burden, etc.) |

Your resume goes in `data/resume.md`. Past decisions (yes/no/maybe on roles
you've reviewed) are auto-stored in `data/decisions/` and build a personal
preference signal over time.

See [CONFIG.md](docs/CONFIG.md) for the full schema, defaults, and
validation rules.

## Development

```bash
# Individual checks
uv run task lint          # ruff check --fix
uv run task format        # ruff format
uv run task type          # pyright type checking
uv run task test          # pytest -v (unit tests only)
uv run task live          # integration + live tests (requires Ollama)

# All checks at once
uv run task check         # format → lint → type → test
```

> **Note:** The `uv run` prefix is optional if your venv is already activated —
> either manually (`source .venv/bin/activate`) or automatically via `direnv allow`.

CI runs on **3 OS × 3 Python versions** (Ubuntu, macOS, Windows × 3.11, 3.12,
3.13) with lint, type checking, and test coverage on every push.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow,
testing philosophy, and PR process.

## Project Status

**Phases 1–8 complete.** Active daily use for real job search filtering.

| Phase | Status | Highlights |
|---|---|---|
| 1 — Core Adapter Infrastructure | ✅ | Adapter registry, session manager, ZipRecruiter adapter, error hierarchy |
| 2 — RAG Foundation | ✅ | Ollama embeddings, ChromaDB vector store, resume/archetype indexing, semantic scoring |
| 3 — Scoring Pipeline | ✅ | Score fusion, cross-board dedup, LLM disqualifier, decision history |
| 4 — Export & Polish | ✅ | Markdown/CSV/JD file export, compensation scoring, context-length safety, operational resilience, interactive review, rejection reason learning, file identity refactor |
| 5 — Observability & Evaluation | ✅ | Structured session tracing, inference metrics, retrieval quality metrics, eval harness with Spearman correlation, model A/B comparison |
| 6 — Security & Data Hygiene | ✅ | Threat model, prompt injection defense (4 layers), input validation, decision audit, privacy verification test, ZipRecruiter Next.js rewrite |
| 7 — Cumulative Search | ✅ | Accumulate results across runs, parallel scoring loop, live/integration test automation |
| 8 — Config Externalization | ✅ | All persona-specific, operational, and tuning parameters moved from source to `settings.toml` |
| 9 — Portfolio Polish | 🔧 | Documentation updates (current phase) |

**868+ tests** across 35 test files. 100% statement coverage on all core
modules. CI validated on 3 OS × 3 Python versions.

## Responsible Use

This tool performs **browser automation** — it loads pages exactly as a human
would, navigates via visible links, and reads only data visible in your own
authenticated session. It does not redistribute data, bypass access controls,
or perform bulk harvesting.

That said, job boards have terms of service governing automated access.
**You are responsible for understanding and complying with the ToS of any
board you configure.** This project is provided as-is for personal,
educational, and research purposes. The authors assume no liability for
misuse.

Some practical guidelines:

- Use human-like request timing (the built-in throttle provides this)
- Run overnight passes for boards with stricter detection (e.g., LinkedIn)
- Don't redistribute scraped data
- Respect rate limits and session caps

## Documentation

| Document | Scope |
|---|---|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, adapter pattern, scoring model overview |
| [CONFIG.md](docs/CONFIG.md) | Full schema, defaults, and validation rules for all TOML files |
| [SCORING_ENGINE.md](docs/SCORING_ENGINE.md) | Fusion formula, compensation parsing, disqualification, dedup |
| [RAG_PIPELINE.md](docs/RAG_PIPELINE.md) | Embedding, indexing, retrieval, and ChromaDB collections |
| [DATA_FLOW.md](docs/DATA_FLOW.md) | End-to-end data lifecycle and persistence points |
| [FEEDBACK_LOOP.md](docs/FEEDBACK_LOOP.md) | Decide → rescore → eval cycle and tuning workflow |
| [FAILURE_MODES.md](docs/FAILURE_MODES.md) | Error catalog, recovery, bot detection, prompt injection |
| [EVOLUTION.md](docs/EVOLUTION.md) | How the system grew phase by phase |
| [TEAM_SCALING.md](docs/TEAM_SCALING.md) | What would change with multiple users |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Development setup, testing standards, PR process |
| [SECURITY.md](SECURITY.md) | Threat model, privacy guarantees, prompt injection defense |

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — free for personal use, learning, and noncommercial purposes.
