# Data Flow

> How data moves through the system — from raw HTML on a job board to a
> ranked, exported, and reviewed result. This document follows a single job
> listing through the entire pipeline and traces the persistence points.

---

## End-to-End Flow

```
  Job Board (web)
       │
       ▼
  ┌──────────────┐    Playwright
  │   Adapter    │◀── SessionManager (cookies, throttle, CDP/stealth)
  │  .search()   │
  │  .extract()  │
  └──────┬───────┘
         │ list[JobListing]
         ▼
  ┌──────────────┐
  │   Scorer     │──▶ Embedder ──▶ Ollama (nomic-embed-text)
  │              │──▶ VectorStore ──▶ ChromaDB (6 collections)
  │              │──▶ CompParser (regex)
  │              │──▶ Disqualifier ──▶ Ollama (mistral:7b)
  └──────┬───────┘
         │ list[(JobListing, ScoreResult)]
         ▼
  ┌──────────────┐
  │   Ranker     │  Score fusion → dedup → threshold filter
  └──────┬───────┘
         │ list[RankedListing]
         ▼
  ┌──────────────┐
  │   Export     │──▶ Markdown table (output/results.md)
  │              │──▶ CSV (output/results.csv)
  │              │──▶ JD files (output/jds/*.md)
  │              │──▶ Browser tabs (top-N URLs)
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   Review     │  Interactive: y/n/m per listing
  └──────┬───────┘
         │ verdict + reason
         ▼
  ┌──────────────┐
  │  Decisions   │──▶ ChromaDB (decisions collection)
  │              │──▶ JSONL (data/decisions/YYYY-MM-DD.jsonl)
  └──────────────┘
```

---

## Lifecycle of a Job Listing

### 1. Extraction

The adapter navigates a job board, extracts search results, and populates
`JobListing` instances.

**Input:** Job board HTML / JSON
**Output:** `JobListing` with all required fields (`board`, `external_id`,
`title`, `company`, `location`, `url`, `full_text`)

**Persistence:** None at this stage. Listings exist only in memory.

**Security:** `__post_init__` sanitizes `title` and `company` to strip
path-traversal sequences and filesystem-unsafe characters. `full_text` is
length-capped at ~250K characters.

### 2. Cross-Run Dedup Check

Before scoring, each listing's `external_id` is checked against the
`decisions` collection. If a prior verdict exists, the listing is skipped
(unless `--force-rescore`).

**Input:** `external_id`
**Output:** Score or skip

### 3. Scoring

The scorer embeds the JD text and queries six ChromaDB collections.

**Input:** `JobListing.full_text`
**Output:** `ScoreResult` (six component scores + disqualification status)

**Side effects:**
- Embeddings cached in memory for near-dedup
- Compensation parsed and attached to `JobListing` (`comp_min`, `comp_max`,
  `comp_source`, `comp_text`)
- `score_computed` event emitted to session log

### 4. Ranking

The ranker fuses scores, deduplicates, and filters.

**Input:** `list[(JobListing, ScoreResult)]` + cached embeddings
**Output:** `list[RankedListing]` sorted by `final_score` descending

### 5. Export

Ranked listings are written to disk in multiple formats.

**Output files:**
- `output/results.md` — summary table
- `output/results.csv` — all score components
- `output/jds/{external_id}_company_title.md` — individual JD files with
  metadata headers

### 6. Review

The operator reviews undecided listings interactively.

**Input:** `list[RankedListing]` (filtered to undecided)
**Output:** Verdict (`yes`/`no`/`maybe`) + optional reason per listing

### 7. Decision Recording

Verdicts are persisted in two forms:

1. **ChromaDB** — The JD text (optionally concatenated with the operator's
   reasoning) is embedded and stored in the `decisions` collection
2. **JSONL** — An append-only audit log: `data/decisions/YYYY-MM-DD.jsonl`

The embedded decision shifts future `history_score` calculations. Only
`yes` verdicts have `scoring_signal=true` and contribute to scoring.

---

## Persistence Points

| Data | Location | Format | Lifecycle |
|---|---|---|---|
| Session cookies | `data/{board}_session.json` | Playwright storage state | Persists across runs |
| ChromaDB collections | `data/chroma_db/` | SQLite + binary | Persists across runs, reset on `index` |
| Decision audit log | `data/decisions/YYYY-MM-DD.jsonl` | JSON-lines (append-only) | Permanent record |
| Session logs | `data/logs/session_{id}_{ts}.jsonl` | JSON-lines | One file per run |
| Export results | `output/results.md`, `results.csv` | Markdown, CSV | Overwritten each run |
| JD files | `output/jds/*.md` | Markdown | Overwritten each run |
| Eval reports | `output/eval_YYYY-MM-DD.md` | Markdown | One per eval run |
| Eval history | `data/eval_history.jsonl` | JSON-lines (append-only) | Tracks eval over time |

---

## Data Contracts

### Between Adapter and Scorer

`JobListing` — the universal contract. All required fields must be populated
after `extract_detail()`. The scorer only reads `full_text`.

### Between Scorer and Ranker

`ScoreResult` — six float scores plus disqualification status. The ranker
reads all fields but modifies none.

### Between Ranker and Export

`RankedListing` — wraps `JobListing` + `ScoreResult` + `final_score` +
`duplicate_boards`. Exporters read all fields.

### Between Review and Decisions

Verdicts are strings (`"yes"`, `"no"`, `"maybe"`) with optional free-text
reason. The `DecisionRecorder` handles embedding and persistence.

---

## Rescore Flow (Alternate Path)

The `rescore` subcommand bypasses the adapter layer entirely:

```
  output/jds/*.md
       │
       ▼
  ┌──────────────┐
  │ load_jd_files│  Parse markdown headers → reconstruct JobListing
  └──────┬───────┘
         │ list[JobListing]
         ▼
    (same as steps 3–5 above)
```

This enables fast iteration on scoring configuration (weights, archetypes,
rubric) without re-running browser searches.

---

## Session Logging Events

Each pipeline run emits structured JSONL events to `data/logs/`:

| Event | When | Key Fields |
|---|---|---|
| `embed_call` | Each embedding operation | `model`, `input_chars`, `latency_ms`, `tokens` |
| `classify_call` | Each LLM call | `model`, `input_chars`, `latency_ms`, `tokens` |
| `disqualifier_call` | Disqualification check | `model`, `input_chars`, `outcome`, `reason` |
| `prompt_injection_detected` | Injection screening | `job_id`, `pattern` |
| `score_computed` | Per-listing scoring | `job_id`, all six scores, `disqualified` |
| `retrieval_summary` | Per-collection stats | `collection`, `n_scored`, `score_min/p50/p90/max` |
| `session_summary` | End of run | `InferenceMetrics` fields, listing counts |

All events share the same `session_id` for correlation.
