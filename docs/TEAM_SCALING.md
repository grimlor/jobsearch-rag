# What Would Change With a Team

> This system was built for a single operator. This document examines what
> would need to change to support multiple users, collaborative workflows,
> and production deployment. Organized by the boundaries that would shift.

---

## Current Single-Operator Assumptions

The following assumptions are baked into the current design:

1. **One resume** — `data/resume.md` is a single file
2. **One set of preferences** — archetypes, rubric, and weights reflect one
   person's priorities
3. **One ChromaDB instance** — embedded (SQLite), local, single-writer
4. **One decision history** — verdicts are from one reviewer's perspective
5. **One machine** — Ollama, ChromaDB, and Playwright all run locally
6. **No concurrent writes** — no locking or conflict resolution
7. **JSONL audit logs** — append-only, no indexing or querying beyond `grep`

---

## What Changes: Persistence Layer

### ChromaDB → Shared Vector Store

The embedded ChromaDB (SQLite file) cannot support concurrent writers. With
a team:

- **Upgrade to ChromaDB server mode** — run ChromaDB as a separate service
  with an HTTP API
- **Or replace** with a managed vector store (Pinecone, Weaviate, PGVector)
- **Connection string** replaces `persist_dir` in `ChromaConfig`
- **Multi-tenancy** — each user's decisions go into partitioned
  sub-collections or use metadata filtering. Shared collections (resume,
  archetypes, rubric) remain common.

### JSONL → Structured Database

The append-only JSONL files work for audit but don't support queries like
"show me all team decisions this week" or "which roles did Alice approve
that Bob rejected?"

- **Decision store** → PostgreSQL or similar with columns for `user_id`,
  `verdict`, `reason`, `timestamp`, `job_id`
- **Session logs** → time-series database or structured logging service
  (Loki, CloudWatch)
- **Eval history** → same database, enabling cross-user comparison

---

## What Changes: Configuration

### Per-User vs. Shared Config

| Config | Current | Team |
|---|---|---|
| `settings.toml` (boards, Ollama) | Shared | Shared — infrastructure config |
| `role_archetypes.toml` | Single-user | Per-user or per-team profiles |
| `global_rubric.toml` | Single-user | Base rubric shared; per-user overrides |
| Scoring weights | Single-user | Per-user profiles |
| `resume.md` | Single-user | Per-user (obviously) |
| `base_salary` | Per-user | Per-user |

The config system would need to support profiles:

```toml
[profiles.alice]
resume_path = "data/resumes/alice.md"
archetypes_path = "config/archetypes/alice.toml"
base_salary = 220000

[profiles.bob]
resume_path = "data/resumes/bob.md"
archetypes_path = "config/archetypes/bob.toml"
base_salary = 180000
```

### Collection Namespacing

ChromaDB collections would need user prefixes or metadata:

- `resume_alice`, `resume_bob` — or a single `resume` collection with
  `user_id` metadata filtering
- `decisions` collection would need `user_id` metadata on every document
- Negative signals and global rubric might stay shared (team values) with
  per-user overrides

---

## What Changes: Scoring

### Multi-Resume Fit Scoring

Each user's `fit_score` queries their own `resume` collection. The scorer
needs to accept a user context:

```python
async def score(jd_text: str, user_id: str) -> ScoreResult:
    # Query user-specific resume collection
    # Query shared archetype collection (or user-specific)
    # Query user-specific decisions
    # ...
```

### Decision History Contamination

Currently, `history_score` reflects one person's preferences. With a team:

- **Option A:** Per-user history — each person's "yes" verdicts only
  influence their own scores. Most isolated.
- **Option B:** Team history — all "yes" verdicts feed one collection.
  Creates a team consensus signal.
- **Option C:** Weighted team history — weight your own verdicts higher
  than teammates'. Most complex but most useful.

### Disqualifier Personalization

Currently, past "no" reasons are injected into the disqualifier prompt from
one person's decisions. With a team, the prompt would need to select relevant
rejection reasons from the current user's history, not everyone's.

---

## What Changes: Infrastructure

### Ollama → Shared Inference Service

Running Ollama locally per user is wasteful. With a team:

- **Shared Ollama server** on a GPU box or cloud instance
- **Or switch to API-based inference** (OpenAI, Anthropic) with appropriate
  key management and cost allocation
- The `OllamaConfig.base_url` already supports pointing to a remote server
- **Rate limiting** becomes necessary — multiple users embedding concurrently

### Browser Automation → Proxy or API

Playwright running on each user's machine doesn't scale:

- **Centralized scraping service** — one instance per board, shared results
- **Job board APIs** where available (Indeed, LinkedIn APIs have access tiers)
- **Proxy rotation** — shared IP pool to avoid per-user rate limiting
- Session cookies become per-user secrets requiring secure storage

### Observability

- **Centralized logging** — session logs aggregate across users
- **Dashboards** — scoring quality metrics per user, per board, over time
- **Alerting** — detect when a board's adapter starts failing (selector
  changes)

---

## What Changes: Pipeline Architecture

### From CLI to Service

The CLI dispatch model (run → exit) doesn't support:

- Concurrent users
- Background scheduling
- Webhook notifications

The pipeline would split into:

| Component | Current | Team |
|---|---|---|
| Search trigger | `jobsearch-rag search` (CLI) | Scheduled worker or API endpoint |
| Scoring | In-process, synchronous | Queue-based worker (Celery, Temporal) |
| Review | Interactive CLI session | Web UI |
| Decision recording | CLI → direct write | API endpoint → database |
| Export | Write to local `output/` | Serve via web UI or push to Slack/email |

### Adapter Layer

The adapter layer is already the cleanest boundary for team scaling:

- Adapters produce `JobListing` objects — this contract doesn't change
- A centralized search service would run adapters on a schedule and store
  raw listings in a database
- Users would score and review from the shared listing pool

### Progressive Deployment

A realistic migration path for a team adoption:

1. **Shared Ollama server** — lowest friction, change `base_url` only
2. **Shared ChromaDB server** — move from embedded to server mode
3. **Per-user profiles** — config system supports multiple users
4. **Centralized scraping** — one search service, shared listing pool
5. **Web UI** — replace CLI review with browser-based interface
6. **API layer** — RESTful API over the pipeline for external integrations

---

## What Stays the Same

The core abstractions hold up under team scaling:

- **`JobListing` data contract** — board-agnostic, user-agnostic
- **`ScoreResult` structure** — same six components, same fusion formula
- **Adapter ABC + Registry** — board-specific logic stays encapsulated
- **`ActionableError` hierarchy** — error design works for operators, UIs,
  and AI agents
- **Score fusion formula** — weights may differ per user but the formula
  is universal
- **Deduplication** — exact + near-dedup logic works regardless of who's
  reviewing
- **Eval metrics** — precision, recall, Spearman ρ are per-user measurements
  that compose into team dashboards

The architecture was designed around data contracts and clear boundaries.
Most team-scaling changes are at the edges (persistence, config, transport),
not in the scoring core.
