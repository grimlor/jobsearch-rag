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

Each enabled board must have a corresponding `[boards.<name>]` section.

### `[boards.<name>]`

| Key | Type | Default | Description |
|---|---|---|---|
| `searches` | `list[str]` | *(required)* | Search URLs or query terms |
| `max_pages` | `int` | `3` | Maximum search result pages to paginate |
| `headless` | `bool` | `true` | Run browser without visible window |
| `browser_channel` | `str \| null` | `null` | System browser for CDP mode: `"msedge"`, `"chrome"`, `"chromium"` |

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

Weights are **not** required to sum to 1.0. They are applied as raw
multipliers in the fusion formula (see [SCORING_ENGINE.md](SCORING_ENGINE.md)).

### `[ollama]`

| Key | Type | Default | Description |
|---|---|---|---|
| `base_url` | `str` | `"http://localhost:11434"` | Ollama API endpoint (must start with `http://` or `https://`) |
| `llm_model` | `str` | `"mistral:7b"` | Model for disqualification and classification |
| `embed_model` | `str` | `"nomic-embed-text"` | Model for embedding generation |
| `slow_llm_threshold_ms` | `int` | `30000` | Log warning when LLM calls exceed this |

### `[output]`

| Key | Type | Default | Description |
|---|---|---|---|
| `default_format` | `str` | `"markdown"` | Export format: `"markdown"`, `"csv"`, `"json"` |
| `output_dir` | `str` | `"./output"` | Directory for results and JD files |
| `open_top_n` | `int` | `5` | Number of top results to open in browser tabs |

### `[chroma]`

| Key | Type | Default | Description |
|---|---|---|---|
| `persist_dir` | `str` | `"./data/chroma_db"` | ChromaDB storage directory |

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

---

## `data/resume.md`

Plain Markdown file. The indexer splits on `##` headings, producing one
ChromaDB document per section. `###` sub-headings are kept within their
parent section.

The `#` title heading (if present) is skipped — only `##` sections are indexed.
