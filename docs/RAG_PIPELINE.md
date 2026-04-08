# RAG Pipeline

> Embedding, indexing, retrieval, and vector storage internals. Covers how
> documents get into ChromaDB, how queries work, and the design decisions
> behind chunking, truncation, and collection layout. See
> [SCORING_ENGINE.md](SCORING_ENGINE.md) for how retrieved scores are fused
> and ranked.

---

## Overview

The RAG pipeline is the core intelligence layer. It converts unstructured
text (resumes, role descriptions, rubric signals, job descriptions) into
vector embeddings, stores them in ChromaDB collections, and retrieves
similarity scores at query time.

```
Source Files                    Indexer                       ChromaDB
─────────────         ─────────────────────            ─────────────────────
resume.md        ──▶  chunk by ## heading       ──▶    resume
archetypes.toml  ──▶  synthesize desc+signals   ──▶    role_archetypes
                 ──▶  extract signals_negative. ──▶    negative_signals
global_rubric    ──▶  synthesize per dimension  ──▶    global_positive_signals
                 ──▶  extract signals_negative. ──▶    negative_signals
decisions        ──▶  embed jd_text + reason    ──▶    decisions

Query Time:
JD text ──▶ Embedder ──▶ query each collection ──▶ ScoreResult
```

---

## Embedder

The `Embedder` class wraps Ollama's async client for both embedding and LLM
classification.

### Embedding (`embed`)

- **Model:** `nomic-embed-text` (configurable via `ollama.embed_model`)
- **Input validation:** Empty or whitespace-only text raises a `VALIDATION`
  error
- **Truncation:** Text exceeding 8,000 characters is truncated using a
  head+tail strategy before embedding

### Truncation Strategy

Real JDs front-load title, company, and overview but place key signals
(compensation, hands-on requirements, tech stack) in the final third.
Naive head-only truncation loses these signals.

```
┌──────────────────────────────────────────────────────┐
│  Head: 60% (4,800 chars)                             │
│  Title, company, overview, requirements              │
├──────────────────────────────────────────────────────┤
│  [...] marker                                        │
├──────────────────────────────────────────────────────┤
│  Tail: 40% (3,200 chars)                             │
│  Comp, tech stack, hands-on details, culture         │
└──────────────────────────────────────────────────────┘
```

- **Head ratio:** 60% of `max_embed_chars` (4,800 chars)
- **Tail ratio:** 40% of `max_embed_chars` (3,200 chars)
- **Marker:** `"\n[…]\n"` inserted between sections
- **Constant:** `max_embed_chars = 8,000`

### Classification (`classify`)

- **Model:** `mistral:7b` (configurable via `ollama.llm_model`)
- **System message:** "You are a job listing classifier..."
- Used for disqualification prompts and injection screening

### Retry Logic

Both `embed` and `classify` retry transient failures:

- **Max retries:** 3
- **Backoff:** Exponential — `1.0s × 2^(attempt−1)` → 1s, 2s, 4s
- **Retryable status codes:** 408, 429, 500, 502, 503, 504
- **Non-retryable:** Immediate failure (e.g., 404 model not found)

### Health Check

`health_check()` runs before any pipeline work:

1. Verify Ollama is reachable at `base_url`
2. Check that `embed_model` is pulled
3. Check that `llm_model` is pulled
4. Raises `CONNECTION` error if unreachable, `EMBEDDING` error if models
   are missing (with `ollama pull <model>` in the suggestion)

### Inference Metrics

```python
@dataclass
class InferenceMetrics:
    embed_calls: int = 0
    embed_tokens_total: int = 0
    llm_calls: int = 0
    llm_tokens_total: int = 0
    llm_latency_ms_total: int = 0
    slow_llm_calls: int = 0       # calls exceeding slow_llm_threshold_ms
```

Metrics are accumulated per run and emitted in the `session_summary` event.

---

## ChromaDB Collections

All collections use cosine distance. The `VectorStore` class wraps ChromaDB's
client with typed methods.

| Collection | Documents | ID Pattern | Key Metadata |
|---|---|---|---|
| `resume` | Resume section chunks | `resume-{slug}` | `source="resume"`, `section=heading` |
| `role_archetypes` | Synthesized archetype text | `archetype-{slug}` | `name=archetype_name` |
| `negative_signals` | Individual signal statements | `neg-{source-slug}-{signal-slug}` | `source="rubric:{dim}"` or `"archetype:{name}"` |
| `global_positive_signals` | Synthesized dimension text | `pos-{dimension-slug}` | `source=dimension_name`, `signal_count=N` |
| `decisions` | JD text (+ operator reasoning) | `{external_id}` | `verdict`, `scoring_signal`, `reason`, `board`, `title`, `company` |

### Collection Behaviors

- **Required:** `resume` and `role_archetypes` — an `INDEX` error is raised
  if either is empty at query time
- **Optional:** `decisions`, `negative_signals`, `global_positive_signals` —
  return a score of 0.0 if the collection is empty or missing
- **Query size:** Top 3 results per query (`n_results = min(count, 3)`)
- **Upsert semantics:** `add_documents` performs upserts — existing IDs are
  updated, not duplicated

---

## Indexing

The `Indexer` class populates collections from source files. All operations
are idempotent — the target collection is reset (dropped and recreated)
before re-indexing.

### Resume Indexing

```
resume.md ──▶ split on "## " headings ──▶ one chunk per section
```

- The `#` title heading is skipped
- `###` sub-headings remain within their parent `##` section
- Each chunk is embedded independently
- ID: `resume-{slugified_heading}`
- Metadata: `{"source": "resume", "section": "Experience"}` etc.

### Archetype Indexing

```
role_archetypes.toml ──▶ per archetype:
    description + "\n\nKey signals:\n" + signals_positive
    ──▶ one document per archetype
```

Combining the narrative description with concrete signal keywords creates
richer embeddings that capture both the intent and the vocabulary of the
target role.

### Negative Signal Indexing

Two sources feed the `negative_signals` collection:

1. **Global rubric** — Each dimension's `signals_negative` list produces
   individual documents: `neg-{dimension-slug}-{signal-slug}`
2. **Per-archetype** — Each archetype's `signals_negative` list produces
   individual documents: `neg-{archetype-slug}-{signal-slug}`

Each signal is embedded independently (not synthesized), so the similarity
query matches against the most relevant individual signal.

### Global Positive Signal Indexing

```
global_rubric.toml ──▶ per dimension:
    dimension name + "\n" + signals_positive (joined)
    ──▶ one document per dimension
```

Dimensions without `signals_positive` produce no document. The
"Compensation Red Flags" dimension typically has only negative signals and
produces no positive document.

---

## Query-Time Scoring

When the scorer processes a JD:

1. **Chunk** the JD if `len(jd_text) > max_embed_chars` (overlap: 2,000 chars)
2. For each chunk:
   a. **Embed** via `nomic-embed-text`
   b. **Query** `resume` → distances → `fit_score`
   c. **Query** `role_archetypes` → distances → `archetype_score`
   d. **Query** `decisions` (where `verdict="yes"`) → distances → `history_score`
   e. **Query** `negative_signals` → distances → `negative_score`
   f. **Query** `global_positive_signals` → distances → `culture_score`
3. **Aggregate:** Keep the maximum score per component across all chunks
4. **Comp score:** Parse compensation via regex, compute against `base_salary`
5. **Disqualifier:** Run LLM disqualification (independent of embedding queries)
6. Return `ScoreResult` with all six components + disqualification status

### Why Max-Score Aggregation?

A strong signal in any chunk should count. If chunk 3 of 5 mentions your
exact tech stack, that's a genuine fit — even if chunks 1–2 are generic
company boilerplate. Taking the maximum preserves the strongest match.

---

## Design Decisions

### Why Cosine Similarity?

Cosine distance measures directional alignment between vectors, ignoring
magnitude. This is ideal for semantic similarity: two documents about the
same topic will point in similar directions regardless of length or
verbosity.

### Why Separate Collections?

Each scoring dimension lives in its own collection so that:

1. Scores are independently interpretable (fit vs. archetype vs. culture)
2. Weights can be tuned per dimension without affecting others
3. Collections can be re-indexed independently (e.g., update resume without
   re-indexing archetypes)
4. Missing collections degrade gracefully (0.0 default) rather than blocking
   the pipeline

### Why Embed Signals Individually (Negative) vs. Synthesized (Positive)?

- **Negative signals** are embedded individually because a single red flag
  in a JD should trigger the penalty. Synthesis would dilute specific signals.
- **Positive signals** are synthesized per dimension because they represent
  a holistic quality (e.g., "good culture" = remote + async + autonomy
  together). The combined embedding captures the concept better than any
  single keyword.

### Why Head+Tail Truncation?

Standard head-only truncation loses compensation and tech stack details
that cluster at the end of JDs. The 60/40 split preserves both the role
overview and the concrete details that drive scoring decisions.
