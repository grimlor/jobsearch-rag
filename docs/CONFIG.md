# Configuration Reference

> Complete schema, defaults, and validation rules for all TOML configuration
> files. See [ARCHITECTURE.md](ARCHITECTURE.md) for how these feed into the
> pipeline.

---

## File Overview

| File | Purpose | Required |
|---|---|---|
| `config/settings.toml` | Board config, scoring weights, Ollama connection, output, ChromaDB | Yes |
| `config/role_archetypes.toml` | Target role descriptions with positive/negative signals | Yes |
| `config/global_rubric.toml` | Universal evaluation dimensions (culture, structure, comp) | Yes |
| `data/resume.md` | Your resume in plain Markdown | Yes (for indexing) |

---

## `settings.toml`

### `[boards]`

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | `list[str]` | *(required)* | Board names to search on every run |
| `overnight_boards` | `list[str]` | `[]` | Boards requiring extended throttling (e.g., LinkedIn) |
| `session_storage_dir` | `str` | `"data"` | Directory for Playwright session cookie files |

Each enabled board must have a corresponding `[boards.<name>]` section.

### `[boards.<name>]`

| Key | Type | Default | Description |
|---|---|---|---|
| `searches` | `list[str]` | *(required)* | Search URLs or query terms |
| `max_pages` | `int` | `3` | Maximum search result pages to paginate |
| `headless` | `bool` | `true` | Run browser without visible window |
| `browser_channel` | `str \| null` | `null` | System browser for CDP mode: `"msedge"`, `"chrome"`, `"chromium"` |
| `rate_limit_range` | `[float, float]` | *(required)* | `[min, max]` seconds of random jitter between search-URL navigations and between JD detail-page requests |
| `login_url` | `str \| null` | `null` | Override login URL for `login --board` command |
| `stealth` | `bool` | `false` | Enable playwright-stealth patches (fingerprint masking) |
| `throttle_max_retries` | `int` | `3` | Max retries on throttle detection (board-specific) |
| `throttle_base_delay` | `float` | `2.0` | Base delay in seconds for exponential throttle backoff; doubles each consecutive throttle (e.g., 4 → 8 → 16 s with base 4.0). Resets on success. |

When `browser_channel` is set, the session manager launches the real system
browser and connects via Chrome DevTools Protocol instead of Playwright's
bundled Chromium. This bypasses Cloudflare bot detection.

### `[scoring]`

| Key | Type | Default | Valid Range | Description |
|---|---|---|---|---|
| `archetype_weight` | `float` | `0.5` | 0.0–1.0 | Weight for role-archetype similarity |
| `fit_weight` | `float` | `0.3` | 0.0–1.0 | Weight for resume similarity |
| `history_weight` | `float` | `0.2` | 0.0–1.0 | Weight for past-decision similarity |
| `comp_weight` | `float` | `0.15` | 0.0–1.0 | Weight for compensation alignment |
| `negative_weight` | `float` | `0.4` | 0.0–1.0 | Penalty multiplier for negative signals |
| `culture_weight` | `float` | `0.2` | 0.0–1.0 | Weight for culture/work-model alignment |
| `base_salary` | `float` | `220000` | > 0 | Reference salary for compensation scoring |
| `disqualify_on_llm_flag` | `bool` | `true` | — | Enable LLM-based disqualification |
| `min_score_threshold` | `float` | `0.45` | 0.0–1.0 | Exclude listings scoring below this |
| `missing_comp_score` | `float` | `0.5` | 0.0–1.0 | Score when no salary data found (neutral) |
| `chunk_overlap` | `int` | `2000` | > 0 | Character overlap between JD chunks |
| `dedup_similarity_threshold` | `float` | `0.95` | 0.0–1.0 | Cosine threshold for near-deduplication |
| `max_parallel` | `int` | `2` | 1–8 | Concurrent scoring tasks (coordinate with `OLLAMA_NUM_PARALLEL`) |

Weights are **not** required to sum to 1.0. They are applied as raw
multipliers in the fusion formula (see [SCORING_ENGINE.md](SCORING_ENGINE.md)).

#### Weight Tuning Guidance

- **`history_weight`** — Start at 0.2. After ~50 decisions, consider raising
  to 0.3. After ~200, the history collection may be your strongest signal.
- **`culture_weight`** — Start at 0.2. Global positive signals encode
  environment preferences that JDs express inconsistently — "async-first" and
  "remote-first" appear explicitly in some JDs and implicitly (or not at all)
  in others. The history signal eventually complements this as culture-based
  yes/no decisions accumulate.
- **`negative_weight`** — Start at 0.4. A strong negative match (adtech,
  surveillance, chaos culture) should meaningfully suppress ranking even when
  positive signals are present.

### `[ollama]`

| Key | Type | Default | Description |
|---|---|---|---|
| `base_url` | `str` | `"http://localhost:11434"` | Ollama API endpoint (must start with `http://` or `https://`) |
| `llm_model` | `str` | `"mistral:7b"` | Model for disqualification and classification |
| `embed_model` | `str` | `"nomic-embed-text"` | Model for embedding generation |
| `slow_llm_threshold_ms` | `int` | `30000` | Log warning when LLM calls exceed this |
| `classify_system_prompt` | `str` | *(required)* | System message for the LLM classifier |
| `max_retries` | `int` | `3` | Maximum retry attempts for transient Ollama failures |
| `base_delay` | `float` | `1.0` | Base delay in seconds for exponential backoff |
| `max_embed_chars` | `int` | `8000` | Maximum characters sent to the embedding model |
| `head_ratio` | `float` | `0.6` | Fraction of `max_embed_chars` allocated to head (vs. tail) during truncation |
| `retryable_status_codes` | `list[int]` | `[408, 429, 500, 502, 503, 504]` | HTTP status codes that trigger retry |

### `[output]`

| Key | Type | Default | Description |
|---|---|---|---|
| `default_format` | `str` | `"markdown"` | Export format: `"markdown"`, `"csv"`, `"json"` |
| `output_dir` | `str` | `"./output"` | Directory for results and JD files |
| `open_top_n` | `int` | `5` | Number of top results to open in browser tabs |
| `jd_dir` | `str` | `"output/jds"` | Directory for individual JD markdown files |
| `decisions_dir` | `str` | `"data/decisions"` | Directory for JSONL decision audit logs |
| `log_dir` | `str` | `"data/logs"` | Directory for structured JSONL session logs |
| `eval_history_path` | `str` | `"data/eval_history.jsonl"` | Append-only file for eval run metrics |

### `[chroma]`

| Key | Type | Default | Description |
|---|---|---|---|
| `persist_dir` | `str` | `"./data/chroma_db"` | ChromaDB storage directory |

### `[security]`

| Key | Type | Default | Description |
|---|---|---|---|
| `screen_prompt` | `str` | *(required)* | Prompt for LLM injection screening pass (reviews JD text for AI-directed instructions) |

### `[adapters]`

| Key | Type | Default | Description |
|---|---|---|---|
| `cdp_timeout` | `float` | `15.0` | Seconds to wait for CDP browser to start accepting connections |

### `[adapters.browser_paths]`

Optional per-channel browser binary paths. When provided for a channel,
these paths are used exclusively (no fallback to platform defaults).

| Key | Type | Default | Description |
|---|---|---|---|
| `msedge` | `list[str]` | *(platform defaults)* | Paths to Microsoft Edge binary |
| `chrome` | `list[str]` | *(platform defaults)* | Paths to Google Chrome binary |
| `chromium` | `list[str]` | *(platform defaults)* | Paths to Chromium binary |

### `[[scoring.comp_bands]]`

Compensation scoring curve defined as `{ ratio, score }` pairs. Ratios are
relative to `base_salary` (e.g., `ratio = 0.90` means 90% of base). The
scorer linearly interpolates between adjacent bands.

| Key | Type | Valid Range | Description |
|---|---|---|---|
| `ratio` | `float` | 0.0–1.0+ | `comp_max / base_salary` threshold |
| `score` | `float` | 0.0–1.0 | Score assigned at this ratio |

**Constraints:** Must have ≥2 entries. Ratios must be strictly descending
(highest first). Above the top band → top score. Below the bottom band →
bottom score.

### Path Settings (top-level)

| Key | Type | Default | Description |
|---|---|---|---|
| `resume_path` | `str` | `"data/resume.md"` | Path to resume file |
| `archetypes_path` | `str` | `"config/role_archetypes.toml"` | Path to role archetypes |
| `global_rubric_path` | `str` | `"config/global_rubric.toml"` | Path to global rubric |

---

## Validation Rules

The config loader (`config.py`) validates on startup:

| Rule | Error Type | Triggered When |
|---|---|---|
| All weight fields in [0.0, 1.0] | `VALIDATION` | Weight outside range |
| `base_salary > 0` | `VALIDATION` | Zero or negative salary |
| `ollama.base_url` starts with `http://` or `https://` | `VALIDATION` | Invalid URL scheme |
| Each board in `enabled` has a `[boards.<name>]` section | `CONFIG` | Missing board section |
| `global_rubric_path` file exists on disk | `CONFIG` | File not found |
| TOML syntax is valid | `PARSE` | Malformed TOML |

---

## `role_archetypes.toml`

Defines the roles you're targeting. Each archetype produces one document in
the `role_archetypes` ChromaDB collection and contributes negative signals
to the `negative_signals` collection.

### Schema

```toml
[[archetypes]]
name = "AI Systems Engineer / AI Platform Architect"
description = """
Multi-paragraph narrative describing the ideal role.
The more specific, the better the embedding match quality.
"""
signals_positive = [
    "LLM systems and agentic workflows",
    "Vector database design",
    "Evaluation and observability for AI systems",
]
signals_negative = [
    "Primarily frontend or UI development",
    "Junior or entry-level role",
]
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | Yes | Unique identifier for the archetype |
| `description` | `str` | Yes | Rich narrative (embedded with positive signals) |
| `signals_positive` | `list[str]` | No | Keywords/phrases that boost archetype match |
| `signals_negative` | `list[str]` | No | Keywords/phrases indexed as negative signals |

**Embedding synthesis:** The indexer combines `description + signals_positive`
into a single embedding text. This creates richer vectors that capture both
the narrative role description and concrete skill keywords.

**Archetype ordering reflects priority:** Archetypes are listed in priority
order — the first entry is treated as the primary career target. Order them
from most desired to least desired role type.

---

## `global_rubric.toml`

Universal evaluation dimensions that apply to **all** roles regardless of
archetype. Each dimension produces documents in two collections:
`global_positive_signals` (one doc per dimension) and `negative_signals`
(one doc per negative signal).

### Schema

```toml
[[dimensions]]
name = "Culture & Work Model"
signals_positive = [
    "Remote-first or distributed team",
    "Async communication culture",
    "Psychological safety emphasis",
]
signals_negative = [
    "Mandatory return-to-office",
    "Startup burnout culture",
    "Military-style hierarchy",
]
```

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | `str` | Yes | Dimension name (stored as metadata) |
| `signals_positive` | `list[str]` | No | Positive indicators for this dimension |
| `signals_negative` | `list[str]` | No | Negative indicators (indexed as penalty signals) |

### Current Dimensions

1. **Role Scope** — cross-team influence, organizational reach
2. **Technical Depth** — infrastructure, platform, distributed systems
3. **Operational Burden** — on-call ownership vs. platform engineering
4. **Industry Alignment** — devtools, climate, health-tech vs. adtech, gambling
5. **Employment Structure** — direct hire vs. staffing agencies
6. **Seniority Mismatch** — Staff/Principal expectations vs. offered level
7. **Compensation Red Flags** — transparent range, market alignment
8. **Culture & Work Model** — remote, async, autonomy, sustainability
9. **Neurodivergence Compatibility** — clear communication, low context-switching
10. **Ethical Alignment** — mission-driven, privacy-respecting

#### Signal Authoring Guidance

- **Use full sentences.** Signals like "Cross-team architectural influence
  and organizational scope" embed with significantly better cosine
  discrimination than short keyword phrases like "architecture ownership."
  `nomic-embed-text` is a sentence-transformer — it produces richer vectors
  from complete thoughts.
- **Neurodivergence Compatibility is the weakest signal.** JDs rarely use
  language like "low-meeting culture" directly. This dimension will
  strengthen as the `decisions` history collection builds up examples of
  culture-based yes/no verdicts.

---

## `data/resume.md`

Plain Markdown file. The indexer splits on `##` headings, producing one
ChromaDB document per section. `###` sub-headings are kept within their
parent section.

The `#` title heading (if present) is skipped — only `##` sections are indexed.

---

## Session & Authentication

Playwright's `storage_state` saves cookies per board to separate JSON files
under `session_storage_dir` (default `data/`). Re-authentication is only
needed when sessions expire. Each board gets its own file
(`<board_name>_session.json`).
