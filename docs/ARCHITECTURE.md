# Architecture

> System design guide for contributors. If you want to add a board adapter,
> fix a bug, or understand how the pieces connect, start here.
>
> This document is the high-level overview. Detail docs cover specific areas:
>
> | Doc | Scope |
> |---|---|
> | [CONFIG.md](CONFIG.md) | Full schema, defaults, and validation rules for all TOML files |
> | [SCORING_ENGINE.md](SCORING_ENGINE.md) | Fusion formula, compensation parsing, disqualification, dedup |
> | [RAG_PIPELINE.md](RAG_PIPELINE.md) | Embedding, indexing, retrieval, and ChromaDB collections |
> | [DATA_FLOW.md](DATA_FLOW.md) | End-to-end data lifecycle and persistence points |
> | [FEEDBACK_LOOP.md](FEEDBACK_LOOP.md) | Decide → rescore → eval cycle and tuning workflow |
> | [FAILURE_MODES.md](FAILURE_MODES.md) | Error catalog, recovery, bot detection, prompt injection |
> | [EVOLUTION.md](EVOLUTION.md) | How the system grew phase by phase |
> | [TEAM_SCALING.md](TEAM_SCALING.md) | What would change with multiple users |

---

## Design Principles

1. **Local-first** -- All processing (LLM inference, embeddings, vector storage)
   runs on your machine. Nothing leaves your network.
2. **Ports and adapters (hexagonal architecture)** -- Domain code depends on
   protocol interfaces (`EmbeddingPort`, `VectorStorePort`, `HealthCheckable`,
   `MetricsProvider`) defined in `ports.py`, never on concrete infrastructure.
   Production wiring and test fakes both satisfy the same contracts.
3. **Board-agnostic pipeline** -- Only the adapter layer knows about specific
   job boards. Everything downstream (scoring, ranking, export) works against
   the `JobListing` data contract.
4. **Tests are the spec** -- There is no separate requirements document. The
   test suite (1 000+ tests across 35+ test files) is the living specification.
   Each test class documents WHO needs it, WHAT it proves, and WHY. If there
   isn't **100% test coverage**, then the implementation is underspecified.
5. **Actionable errors** -- Every error carries enough context for the operator
   (or an AI assistant) to resolve it without searching logs or source code.

---

## System Overview

```
CLI (12 subcommands)
 │
 ├── index ──▶ Indexer ──▶ ChromaDB (6 collections)
 │
 ├── search ──▶ Adapter Layer ──▶ Scorer ──▶ Ranker ──▶ Export
 │               │                  │
 │               │                  ├── resume                  → fit_score
 │               │                  ├── role_archetypes         → archetype_score
 │               │                  ├── decisions               → history_score
 │               │                  ├── negative_signals        → negative_score
 │               │                  ├── global_positive_signals → culture_score
 │               │                  └── comp_parser             → comp_score
 │               │
 │               ├── ZipRecruiter (Next.js DOM extraction)
 │               ├── Indeed
 │               ├── WeWorkRemotely
 │               └── LinkedIn (overnight / CDP mode)
 │
 ├── rescore   ──▶ Rescorer (re-score JDs from disk, no browser)
 ├── decide    ──▶ DecisionRecorder
 ├── review    ──▶ ReviewSession (interactive batch)
 ├── eval      ──▶ EvalRunner (pipeline vs. human verdicts)
 └── decisions ──▶ show / remove / audit
```

### Typical Search Run

1. The CLI loads enabled boards from `config/settings.toml`
2. `create_pipeline(settings)` resolves concrete port implementations from
   the `[ports]` config section and injects them into `PipelineRunner`
3. The adapter registry resolves board names to adapter classes (IoC)
4. Ollama health check via `HealthCheckable.health_check()` -- fail fast
   if models aren't pulled
5. Auto-index if any ChromaDB collection is empty
6. Each adapter uses Playwright to navigate search results and extract listings
7. The scorer embeds each JD and queries six ChromaDB collections
8. The LLM disqualifier screens for structural red flags
9. Compensation is parsed via regex and scored against a configurable base salary
10. The ranker fuses all six component scores, deduplicates across boards, and
    applies a minimum score threshold
11. Results export as Markdown, CSV, individual JD files, and/or browser tabs

---

## Adapter Layer

### Data Contract: `JobListing`

Every adapter produces `JobListing` instances -- the single abstraction that
makes the rest of the pipeline board-agnostic:

```python
@dataclass
class JobListing:
    board: str                                      # "ziprecruiter", "linkedin", etc.
    external_id: str                                # Board's own job ID, for deduplication
    title: str                                      # Sanitized (filesystem-safe)
    company: str                                    # Sanitized
    location: str
    url: str
    full_text: str                                  # Full JD text (max ~250K chars)
    posted_at: datetime | None = None
    raw_html: str | None = None
    comp_min: float | None = None                   # Parsed compensation range
    comp_max: float | None = None
    comp_source: str | None = None                  # "employer" or "estimated"
    comp_text: str | None = None                    # Original matched snippet
    metadata: dict[str, str] = field(default_factory=dict)
```

`__post_init__` validates `full_text` length and sanitizes `title`/`company`
to strip path-traversal sequences and filesystem-unsafe characters.

### Abstract Base Class: `JobBoardAdapter`

All adapters implement this interface:

| Method / Property | Purpose |
|---|---|
| `board_name` | Unique string identifier (matches `settings.toml` key) |
| `authenticate(page)` | Establish session; detect CAPTCHAs and expired cookies |
| `search(page, query, max_pages)` | Paginate search results → `list[JobListing]` |
| `extract_detail(page, listing)` | Navigate to listing URL → populate `full_text` |
| `rate_limit_seconds` | `(min, max)` tuple for random throttle jitter (default 1.5–3.5s) |

### Adapter Registry (IoC)

Adapters self-register via a decorator. The pipeline runner never imports
concrete adapter classes:

```python
@AdapterRegistry.register
class ZipRecruiterAdapter(JobBoardAdapter):
    ...
```

The runner resolves adapters from config:

```python
enabled = settings["boards"]["enabled"]     # ["ziprecruiter", ...]
adapters = [AdapterRegistry.get(name) for name in enabled]
```

Adding a new board adapter requires **zero changes** to the pipeline, ranker,
or export code.

### Concrete Adapters

| Adapter | Strategy | Rate Limit |
|---|---|---|
| **ZipRecruiter** | Next.js DOM extraction (`article[id^="job-card-"]`) + JSON-LD URLs | configurable (default 1.5–3.5s) |
| **WeWorkRemotely** | HTML scraping | configurable (default 1.5–3.5s) |
| **LinkedIn** | CDP mode (system browser), overnight-only, bot-detection checks | configurable (default 8–20s) |
| **Indeed** | Stub (not yet implemented) | configurable |

The ZipRecruiter adapter extracts job cards from `article[id^="job-card-"]`
DOM elements in the Next.js SERP, deduplicates responsive mobile/desktop
duplicates, and resolves canonical URLs from JSON-LD `ItemList` structured
data. Salary text (e.g., `$185K - $240K/yr`) is parsed directly from card
DOM. Full JD text comes from clicking each card to populate the detail panel.
Per-board rate limits and throttle parameters are configurable in `settings.toml`.

---

## Session Management

`SessionManager` handles Playwright browser context lifecycle:

- **Two launch modes:**
  1. **Playwright-managed** (default) -- `chromium.launch()` -- simple, but
     Cloudflare detects automation flags
  2. **CDP mode** -- launches a real system browser (Edge/Chrome) as a subprocess
     with `--remote-debugging-port`, then connects via `connect_over_cdp()` --
     no automation flags, bypasses Cloudflare
- **Storage state persistence** -- Cookies are saved per board to
  `data/{board}_session.json` to avoid re-authentication on every run
- **Stealth patches** -- Optional `playwright-stealth` integration
- **Throttling** -- `throttle()` applies random jitter within the adapter's
  `rate_limit_seconds` range between every page navigation

Board-specific detection logic (e.g., LinkedIn's authwall redirects and
challenge interstitials) lives in the adapter module, not in `SessionManager`.
The session manager stays board-agnostic.

---

## Error Hierarchy

All errors extend `ActionableError` with structured remediation targeting
three audiences:

| Audience | Field | Example |
|---|---|---|
| Calling code | `error_type: ErrorType` | Route recovery logic |
| Human operator | `suggestion`, `troubleshooting` | "Re-run with --login to refresh session" |
| AI agent | `ai_guidance: AIGuidance` | `action_required`, `command`, `checks` |

**ErrorType categories:** `AUTHENTICATION`, `CONFIG`, `CONNECTION`, `EMBEDDING`,
`INDEX`, `PARSE`, `DECISION`, `VALIDATION`, `UNEXPECTED`

Factory classmethods enforce consistent construction:

```python
raise ActionableError.authentication(
    board="ziprecruiter",
    raw_error=exc,
    suggestion="Re-run with --login to refresh session",
)
```

A catch-all `from_exception()` factory auto-classifies unknown exceptions by
keyword matching (e.g., "timeout" → `CONNECTION`, "not found" → `UNEXPECTED`).

---

## Port Architecture (Hexagonal / Ports-and-Adapters)

The system uses a **ports-and-adapters** (hexagonal) architecture to
decouple domain logic from infrastructure. The domain core (scoring,
ranking, pipeline orchestration) depends only on protocol interfaces
defined in `ports.py`. Concrete implementations are injected at runtime.

### Why Ports and Adapters?

Traditional layered architectures create direct dependencies from business
logic to infrastructure (database clients, HTTP SDKs, etc.). This makes
the code hard to test -- you end up mocking implementation details. With
ports and adapters:

- **Ports** are protocol interfaces that define *what* the domain needs
  (e.g., "embed this text", "store this document")
- **Adapters** are concrete implementations that fulfill those contracts
  using real infrastructure (Ollama, ChromaDB)
- **Test fakes** are alternative adapters that satisfy the same ports with
  deterministic, in-memory behavior

The domain never imports `ollama` or `chromadb` directly. It imports
`EmbeddingPort` and `VectorStorePort`.

### Port Definitions (`ports.py`)

| Protocol | Purpose | Key Methods/Attributes |
|---|---|---|
| `EmbeddingPort` | Embedding + LLM classification | `embed()`, `classify()`, `max_embed_chars`, `llm_model` |
| `VectorStorePort` | All vector storage operations | `add_documents()`, `query()`, `get_documents()`, `count()`, `reset_collection()` |
| `HealthCheckable` | Pre-flight connectivity check | `health_check()` |
| `MetricsProvider` | Inference metrics exposure | `metrics` property → `MetricGroup` |
| `PortFactory` | Construction from settings | `from_settings(settings)` classmethod |

All protocols are `@runtime_checkable`, enabling both static type checking
(pyright) and runtime `isinstance` verification in tests.

### Production Adapters

| Port | Adapter | Infrastructure |
|---|---|---|
| `EmbeddingPort` + `HealthCheckable` + `MetricsProvider` | `OllamaEmbedder` | Ollama SDK (`AsyncClient`) |
| `VectorStorePort` | `ChromaDBStore` | ChromaDB embedded client |

### Test Fakes

| Port | Fake | Behavior |
|---|---|---|
| `EmbeddingPort` + `HealthCheckable` + `MetricsProvider` | `FakeEmbedder` | Configurable fixed vectors, no-op health check |
| `VectorStorePort` | `InMemoryVectorStore` | Dict-backed storage, cosine similarity |

Both fakes satisfy the same protocols as production adapters. No mocking
frameworks are needed -- the fakes *are* the test doubles.

### Wiring: Constructor Injection + Factory

`PipelineRunner` accepts `store: VectorStorePort` and `embedder: EmbeddingPort`
as constructor arguments.  Production wiring goes through `create_pipeline()`:

```python
# Production: factory resolves adapters from config
runner = create_pipeline(settings)

# Tests: inject fakes directly
runner = PipelineRunner(settings, store=InMemoryVectorStore(), embedder=FakeEmbedder())
```

`create_pipeline()` reads the `[ports]` section of `settings.toml` to resolve
implementation class paths at runtime, keeping the runner decoupled from any
concrete adapter.

### Observability via `@observable` Decorator

The `@observable` class decorator wraps any `EmbeddingPort` implementation
with call tracing and metric accumulation:

- Emits `embed_call` and `classify_call` structured log events
- Accumulates `MetricGroup` counters (call counts, tokens, latency)
- Makes the decorated class satisfy `MetricsProvider` automatically
- Applied to both `OllamaEmbedder` and `FakeEmbedder`

`PipelineRunner` reads metrics via `MetricsProvider.metrics` during the
session summary. `HealthCheckable.health_check()` is called unconditionally
before pipeline execution -- all embedder implementations provide it.

### Multi-Service Metrics Design

`MetricGroup` is intentionally generic -- a string-keyed accumulator, not an
embedder-specific structure. Each port implementation that satisfies
`MetricsProvider` holds its own `MetricGroup` instance with domain-appropriate
keys (e.g., `embed_calls` for the embedder, `query_calls` for a vector store).

If `ChromaDBStore` (or a future database adapter) gains observability, it
would satisfy `MetricsProvider` and publish its own `MetricGroup` with keys
like `query_calls`, `query_latency_ms_total`, `add_calls`. Because each
service owns a separate `MetricGroup` instance, key names are scoped per
service without collision. The session summary would then collect metrics
from both ports and emit them with a service discriminator:

```json
{"service": "ollama", "embed_calls": 42, "llm_calls": 12}
{"service": "chromadb", "query_calls": 252, "query_latency_ms_total": 1830}
```

This design avoids prefixed key conventions (e.g., `chromadb.query_calls`)
in favor of structural separation at the data model level.

---

## RAG Pipeline

### Six-Collection Scoring Model

Each job description is embedded and scored against six ChromaDB collections:

| Collection | Source | Score | Purpose |
|---|---|---|---|
| `resume` | `data/resume.md`, chunked by `##` heading | `fit_score` | How well your background matches |
| `role_archetypes` | `config/role_archetypes.toml` descriptions + positive signals | `archetype_score` | Does this match what you're targeting? |
| `decisions` | Past yes/no/maybe verdicts (only `yes` contributes) | `history_score` | Resembles roles you've approved |
| `negative_signals` | Global rubric + per-archetype negative signals | `negative_score` | Penalty for red-flag patterns |
| `global_positive_signals` | `config/global_rubric.toml` dimensions (one doc per dimension) | `culture_score` | Alignment with culture/work-model preferences |
| *(inline)* | Regex extraction from JD text | `comp_score` | Compensation vs. configurable base salary |

### Indexer

The `Indexer` populates ChromaDB collections from source files:

| Method | Collection(s) | Strategy |
|---|---|---|
| `index_resume()` | `resume` | Split on `##` headings → one chunk per section |
| `index_archetypes()` | `role_archetypes` | Synthesize `description + signals_positive` per archetype |
| `index_negative_signals()` | `negative_signals` | Merge global rubric + per-archetype `signals_negative` |
| `index_global_positive_signals()` | `global_positive_signals` | One document per rubric dimension |

All operations are idempotent -- the collection is reset before re-indexing.

### Scorer

For each JD:

1. **Chunk** long JDs into overlapping segments (2000-char overlap) if they
   exceed the embedding model's context window
2. **Embed** each chunk via `nomic-embed-text` through Ollama
3. **Query** each collection -- keep the best (max) score across all chunks
   per component
4. **Distance → score:** `max(0.0, min(1.0, 1.0 - cosine_distance))`
5. **Parse compensation** via regex (annual ranges, hourly→annual conversion,
   false-positive screening for employee counts and revenue figures)
6. **LLM disqualifier** (multi-layer defense):
   - Layer 1: Prompt-injection screening
   - Layer 2: Regex sanitization of JD text
   - Layer 3: Disqualifier prompt via `mistral:7b`
   - Safe default: if JSON parsing fails, listing is **not** disqualified

### Score Fusion (Ranker)

```
positive = archetype_weight × archetype_score
         + fit_weight       × fit_score
         + history_weight   × history_score
         + comp_weight      × comp_score
         + culture_weight   × culture_score

final_score = max(0.0, positive − negative_weight × negative_score)

if disqualified: final_score = 0.0
```

All weights are configurable in `config/settings.toml`. Listings below
`min_score_threshold` are excluded from output.

### Deduplication

The ranker collapses duplicates in two passes:

1. **Exact** -- same `(board, external_id)`
2. **Near** -- cosine similarity > 0.95 on `full_text` embeddings

The highest-scored instance is kept; other boards are noted in
`duplicate_boards`.

### Cross-Run Deduplication

Listings with an existing decision (yes/no/maybe) are excluded from scoring
on subsequent runs, unless `--force-rescore` is passed.

---

## Pipeline Orchestration

### Runner

`PipelineRunner` accepts port implementations via constructor injection
(`store: VectorStorePort`, `embedder: EmbeddingPort`) and coordinates
the search flow:

1. Start session logging (JSONL with `session_id`)
2. Health check via `HealthCheckable.health_check()` (fail fast before
   launching browsers)
3. Auto-index empty collections
4. Search all enabled boards concurrently
5. Score, rank, and export
6. Collect metrics via `MetricsProvider.metrics` and emit structured
   `session_summary`

`create_pipeline(settings)` is the production factory -- it reads the
`[ports]` config section to resolve concrete adapter classes and injects
them.  Tests bypass the factory and inject fakes directly.

Returns `RunResult` with ranked listings, summary statistics, failure
counts, and any errors encountered.

### Rescorer

Re-scores previously exported JD files through updated collections without
launching a browser. Useful after re-indexing with updated resume, archetypes,
or rubric.

### Interactive Review

`ReviewSession` presents undecided listings one at a time for batch
verdicts (y/n/m + optional reason). Opens JD files or URLs in the browser
on demand.

### Eval

`EvalRunner` compares pipeline scoring against human decisions:

- **Agreement rate** -- fraction where pipeline and human agree
- **Precision** -- of listings the pipeline would surface, how many did you
  approve?
- **Recall** -- of listings you approved, how many did the pipeline surface?
- **Spearman rank correlation** -- ordinal alignment between verdicts and scores
- **Model comparison** -- `--compare-models A B` runs dual evaluation with
  delta reporting

Writes a Markdown report to `output/` and appends to `data/eval_history.jsonl`.

---

## CLI

Entry point: `jobsearch-rag` (via `__main__.py → cli.py`)

| Subcommand | Purpose |
|---|---|
| `index` | Index resume and/or archetypes into ChromaDB |
| `search` | Full pipeline: search → score → rank → export |
| `rescore` | Re-score exported JDs through updated collections |
| `export` | Re-export last results in a specific format |
| `decide` | Record your verdict on a specific job (yes/no/maybe) |
| `review` | Interactive batch review of undecided listings |
| `decisions` | Manage decisions: `show`, `remove`, `audit` |
| `eval` | Evaluate scoring pipeline vs. human decisions |
| `boards` | List registered adapters |
| `login` | Open headed browser for interactive authentication |
| `reset` | Reset ChromaDB collections, optionally clear output |

---

## Configuration

Three TOML files under `config/`:

| File | Purpose |
|---|---|
| `settings.toml` | Board configs, scoring weights, Ollama connection, output settings, ChromaDB path |
| `role_archetypes.toml` | Target role descriptions with positive and negative signals |
| `global_rubric.toml` | Universal evaluation dimensions (10 dimensions, each with positive/negative signals) |

Key scoring settings:

```toml
[scoring]
archetype_weight = 0.5
fit_weight = 0.3
history_weight = 0.2
comp_weight = 0.15
negative_weight = 0.4
culture_weight = 0.2
base_salary = 220000
disqualify_on_llm_flag = true
min_score_threshold = 0.45
```

See [CONFIG.md](CONFIG.md) for the full schema, all defaults, and validation rules.

---

## Export

Four output formats, all driven from ranked results:

| Exporter | Output | Notes |
|---|---|---|
| **Markdown** | `output/results.md` | Summary table with score breakdowns |
| **CSV** | `output/results.csv` | All score components; excludes disqualified listings |
| **JD files** | `output/jds/{external_id}_company_title.md` | One file per listing with metadata header and full JD text |
| **Browser tabs** | *(opens in default browser)* | Top-N URLs from `open_top_n` setting |

---

## Observability

Structured JSONL session logs under `data/logs/`:

- One file per run: `session_{id}_{timestamp}.jsonl`
- Events: `score_computed` (per listing), `embed_call`, `classify_call`,
  `retrieval_summary` (per collection), `session_summary`
- The `@observable` decorator accumulates `MetricGroup` counters (`MetricKey`
  constants: `embed_calls`, `embed_tokens_total`, `llm_calls`,
  `llm_tokens_total`, `llm_latency_ms_total`, `slow_llm_calls`)
- `session_summary` includes `wall_clock_ms` -- end-to-end pipeline duration

---

## Project Structure

```
jobsearch-rag/
├── config/
│   ├── settings.toml               # Board config, scoring weights, Ollama settings
│   ├── role_archetypes.toml        # Target role descriptions + signals
│   └── global_rubric.toml          # Universal evaluation dimensions
├── data/
│   ├── resume.md                   # Your resume in plain Markdown
│   ├── chroma_db/                  # Vector store persistence (git-ignored)
│   ├── decisions/                  # Past verdicts as JSONL (git-ignored)
│   └── logs/                       # Session JSONL logs (git-ignored)
├── src/jobsearch_rag/
│   ├── __main__.py                 # Entry point
│   ├── cli.py                      # Argument parser + handler dispatch
│   ├── config.py                   # Settings/BoardConfig/ScoringConfig loaders
│   ├── errors.py                   # ActionableError hierarchy + factories
│   ├── logging.py                  # File + structured session logging
│   ├── observability.py            # @observable decorator, MetricGroup, MetricKey
│   ├── ports.py                    # Port protocols: EmbeddingPort, VectorStorePort, etc.
│   ├── text.py                     # slugify, text normalization
│   ├── adapters/
│   │   ├── base.py                 # JobListing dataclass + JobBoardAdapter ABC
│   │   ├── registry.py             # AdapterRegistry (decorator-based IoC)
│   │   ├── session.py              # SessionManager (Playwright / CDP lifecycle)
│   │   ├── ziprecruiter.py         # ZipRecruiter Next.js DOM extraction
│   │   ├── indeed.py               # Indeed adapter (stub)
│   │   ├── weworkremotely.py       # WeWorkRemotely HTML scraping
│   │   └── linkedin.py             # LinkedIn overnight adapter (CDP + stealth)
│   ├── rag/
│   │   ├── embedder.py             # OllamaEmbedder (implements EmbeddingPort + HealthCheckable)
│   │   ├── store.py                # ChromaDBStore (implements VectorStorePort)
│   │   ├── indexer.py              # Resume/archetype/rubric → ChromaDB
│   │   ├── scorer.py               # Semantic scoring + LLM disqualification
│   │   ├── comp_parser.py          # Regex compensation extraction + scoring
│   │   └── decisions.py            # Verdict recording + audit + removal
│   ├── pipeline/
│   │   ├── runner.py               # PipelineRunner orchestration + create_pipeline()
│   │   ├── ranker.py               # Score fusion + dedup + threshold
│   │   ├── rescorer.py             # Re-score JDs from disk
│   │   ├── review.py               # Interactive batch review session
│   │   └── eval.py                 # Pipeline evaluation + model comparison
│   └── export/
│       ├── markdown.py             # Markdown table export
│       ├── csv_export.py           # CSV export
│       ├── jd_files.py             # Individual JD file export
│       └── browser_tabs.py         # Open top-N URLs in browser
├── tests/
│   ├── conftest.py                 # Shared fixtures
│   ├── constants.py                # Test constants
│   ├── fakes.py                    # FakeEmbedder + InMemoryVectorStore (test doubles)
│   ├── fixtures/                   # HTML fixtures, sample JD JSON
    └── test_*.py                   # 1 000+ BDD-style tests across 35+ files
└── docs/
    ├── ARCHITECTURE.md             # ← you are here
    ├── CONFIG.md                   # Configuration schema + validation
    ├── SCORING_ENGINE.md           # Scoring model + fusion formula
    ├── RAG_PIPELINE.md             # Embedding, indexing, retrieval
    ├── DATA_FLOW.md                # End-to-end data lifecycle
    ├── FEEDBACK_LOOP.md            # Decide → rescore → eval cycle
    ├── FAILURE_MODES.md            # Error catalog + recovery
    ├── EVOLUTION.md                # System growth over time
    └── TEAM_SCALING.md             # Multi-user considerations
```

---

## Testing Philosophy

Tests are organized by **behavioral requirement**, not by code structure.
Each test class documents:

- **WHO** needs this behavior
- **WHAT** the behavior is (including failure modes)
- **WHY** it matters (what breaks if the contract is violated)

```python
class TestAuthenticationFailures:
    """
    REQUIREMENT: Authentication failures are detected early and reported clearly.

    WHO: The operator running the tool; the pipeline runner
    WHAT: Expired sessions are detected before search begins; CAPTCHA
          encounters halt the run gracefully
    WHY: An unauthenticated scrape returns login-page HTML silently,
         producing zero valid listings with no error
    """
```

Key principles:

- **Mock I/O boundaries, not implementation** -- Tests use HTML fixtures, not
  mocked method calls. When the adapter internals were rewritten from CSS
  selectors to JSON extraction, zero existing tests changed.
- **Failure specs are as important as happy-path specs** -- An unspecified
  failure is an unhandled failure.
- **Missing spec = missing requirement** -- When a bug is found, step one is
  always "add the missing spec."

---

## Adding a New Board Adapter

1. Create `src/jobsearch_rag/adapters/yourboard.py`
2. Subclass `JobBoardAdapter` and implement all abstract methods
3. Decorate with `@AdapterRegistry.register`
4. Add a `[boards.yourboard]` section to `config/settings.toml`
5. Add the board name to `boards.enabled`
6. Create test fixtures in `tests/fixtures/`
7. Write tests organized by behavioral requirement

The adapter must produce `JobListing` instances with all required fields
populated. The rest of the pipeline will pick it up automatically.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11–3.13 | First-class support for local AI stack; CI tested on 3 OS × 3 versions |
| Browser | Playwright | Async-native, clean cookie persistence, CDP support |
| LLM | Ollama + mistral:7b | Zero-cost, privacy-respecting, no data egress |
| Embeddings | nomic-embed-text via Ollama | Fast local embeddings, consistent with local-first philosophy |
| Vector store | ChromaDB (embedded) | Persistent, no server required, cosine similarity built-in |
| Config | TOML | Human-readable, easy to extend per board |
| Lint/format | ruff | Fast, replaces flake8 + isort + black |
| Type checking | pyright (strict) | Catches contract violations at dev time |
| Testing | pytest + pytest-asyncio | Async adapter tests, BDD-style organization |
| Error framework | actionable-errors | Structured remediation for humans and AI agents |

---

## Further Reading

- [CONFIG.md](CONFIG.md) -- Configuration schema, defaults, and validation
- [SCORING_ENGINE.md](SCORING_ENGINE.md) -- Scoring model, fusion formula, compensation, disqualification
- [RAG_PIPELINE.md](RAG_PIPELINE.md) -- Embedding, indexing, retrieval, vector store internals
- [DATA_FLOW.md](DATA_FLOW.md) -- End-to-end data lifecycle, persistence points, contracts
- [FEEDBACK_LOOP.md](FEEDBACK_LOOP.md) -- Decide → rescore → eval cycle, tuning workflow
- [FAILURE_MODES.md](FAILURE_MODES.md) -- Error catalog, recovery, bot detection, prompt injection defense
- [EVOLUTION.md](EVOLUTION.md) -- How the system grew from scraper to scoring pipeline
- [TEAM_SCALING.md](TEAM_SCALING.md) -- What would change with multiple users
