# Scoring Engine Design

> How job listings are scored, ranked, and filtered. Covers the six-component
> scoring model, fusion formula, compensation parsing, LLM disqualification,
> and deduplication. See [RAG_PIPELINE.md](RAG_PIPELINE.md) for the embedding
> and retrieval layer beneath this.

---

## Component Scores

Each job description produces six float scores in [0.0, 1.0]:

| Score | Source | Collection | Default | Purpose |
|---|---|---|---|---|
| `fit_score` | Resume similarity | `resume` | *(required)* | How well your background matches this JD |
| `archetype_score` | Role archetype similarity | `role_archetypes` | *(required)* | How well this JD matches your target roles |
| `history_score` | Past "yes" decisions | `decisions` | 0.0 | Similarity to roles you've previously approved |
| `comp_score` | Regex extraction from JD text | *(inline)* | 0.5 | Compensation vs. your base salary |
| `culture_score` | Global rubric positive signals | `global_positive_signals` | 0.0 | Alignment with culture/work-model preferences |
| `negative_score` | Negative signal similarity | `negative_signals` | 0.0 | Penalty for red-flag patterns |

### Distance-to-Score Conversion

ChromaDB returns cosine distance (lower = more similar). The scorer converts
to a similarity score:

```
score = max(0.0, min(1.0, 1.0 − min(distances)))
```

The best (minimum) distance across the top-3 query results is used. An empty
result set produces a score of 0.0.

### Multi-Chunk Strategy

When a JD exceeds the embedding model's context window (8,000 chars), it is
split into overlapping chunks:

- **Chunk size:** 8,000 characters (matches `Embedder.MAX_EMBED_CHARS`)
- **Overlap:** 2,000 characters between consecutive chunks
- **Aggregation:** The **maximum** score per component across all chunks is kept

This ensures that a strong signal buried deep in a long JD isn't lost to
truncation. The overlap prevents signals that span a chunk boundary from
being split.

---

## Score Fusion

The ranker combines component scores into a single `final_score`:

```
positive = archetype_weight × archetype_score
         + fit_weight       × fit_score
         + history_weight   × history_score
         + comp_weight      × comp_score
         + culture_weight   × culture_score

final_score = max(0.0, positive − negative_weight × negative_score)
```

If the LLM disqualifier flags the listing: `final_score = 0.0`.

### Default Weights

| Weight | Default | Effect |
|---|---|---|
| `archetype_weight` | 0.5 | Strongest positive signal — is this the kind of role you want? |
| `fit_weight` | 0.3 | Second — does your background actually match? |
| `history_weight` | 0.2 | Third — pattern matching against past approvals |
| `comp_weight` | 0.15 | Modest boost/penalty for compensation alignment |
| `culture_weight` | 0.2 | Environment and work-model alignment |
| `negative_weight` | 0.4 | Subtractive — penalty for matched negative signals |

Weights are **not** normalized to sum to 1.0. The theoretical maximum
`final_score` depends on the weight configuration. With defaults:

```
max positive = 0.5 + 0.3 + 0.2 + 0.15 + 0.2 = 1.35
max penalty  = 0.4 × 1.0 = 0.4
```

### Threshold Filtering

Listings with `final_score < min_score_threshold` (default 0.45) are excluded
from export. The threshold is applied after fusion and deduplication.

---

## Compensation Scoring

Compensation is scored via regex extraction — no LLM involved.

### Parsing Pipeline

1. **Pattern matching** — Regex patterns detect salary formats:
   - `$180,000 - $220,000` (annual range)
   - `$180K - $220K` (K-suffix, case-insensitive)
   - `$85/hr` or `$85 per hour` (hourly, converted via × 2,080 hours/year)
   - `$200,000 per year` or `$200,000 per annum`
   - Single values: `$180,000` (sets both min and max)

2. **False-positive screening** — Context around the match is checked to
   reject:
   - Employee counts ("500-1,000 employees")
   - Revenue figures ("$50M ARR", "$1B funding")
   - Non-salary dollar amounts in surrounding text

3. **Source classification** — Employer-stated salary is preferred over
   board-estimated salary. Source is tagged as `"employer"` or `"estimated"`.

### Comp Score Bands

The `comp_max` value is compared to the configurable `base_salary`:

| Ratio (`comp_max / base_salary`) | Score Range | Interpolation |
|---|---|---|
| ≥ 100% | 1.0 | Flat — meets or exceeds target |
| 90–100% | 0.7–0.9 | Linear within band |
| 77–90% | 0.4–0.7 | Linear within band |
| 68–77% | 0.0–0.4 | Linear within band |
| < 68% | 0.0 | Flat — too far below target |
| Missing | 0.5 | Neutral — no data, don't penalize or reward |

The bands produce a continuous, piecewise-linear curve with no discontinuities
at boundaries. The `0.5` default for missing data ensures unspecified
compensation neither helps nor hurts a listing.

---

## LLM Disqualification

A multi-layer defense screens JDs for structural problems that semantic
similarity alone can't detect.

### Disqualification Criteria

The disqualifier prompt instructs the LLM to flag listings that are:

- **IC-disguised-as-architect** — titles say "architect" but responsibilities
  are individual contributor work
- **Primarily SRE on-call** — the role's primary function is incident response
  and on-call rotation
- **Staffing agency / vendor chain** — posted by a recruiting firm, not the
  actual employer
- **Primarily full-stack web development** — frontend-heavy roles mislabeled
  as platform/infrastructure

### Defense Layers

1. **Prompt-injection screening** — A separate LLM call checks whether the
   JD text contains language attempting to override system instructions
2. **Regex sanitization** — Known injection patterns are stripped from the JD
   before it reaches the disqualifier prompt
3. **Disqualifier prompt** — The sanitized JD is sent to `mistral:7b` with a
   system message requesting JSON output: `{"disqualified": bool, "reason": str}`
4. **Safe default** — If JSON parsing fails, the listing is **not**
   disqualified. False negatives (missing a bad listing) are preferred over
   false positives (rejecting a good one).

### Decision-Augmented Prompts

When the `decisions` collection contains prior "no" verdicts with reasons,
those reasons are injected into the disqualifier prompt as examples. This
personalizes disqualification to the operator's past rejection patterns.

---

## Deduplication

The ranker collapses duplicates in two passes:

### Pass 1: Exact Deduplication

Remove listings with identical `(board, external_id)` pairs. This catches
re-posts within a single board.

### Pass 2: Near-Deduplication

1. Sort scored listings by `final_score` descending
2. For each listing, compute cosine similarity against all higher-ranked
   listings using their `full_text` embeddings
3. If similarity > 0.95, the lower-ranked listing is absorbed:
   - Its `board` is appended to the survivor's `duplicate_boards` list
   - The duplicate is removed from results

The highest-scored instance always survives. Export output notes which other
boards carried the same listing.

### Cross-Run Deduplication

On subsequent runs, the pipeline checks the `decisions` collection for each
listing's `external_id`. If a decision (yes/no/maybe) exists, the listing is
excluded from scoring entirely — it has already been reviewed.

The `--force-rescore` flag overrides this behavior, re-scoring all listings
regardless of decision status. Useful after re-indexing or weight changes.

---

## Data Structures

### `ScoreResult`

```python
@dataclass
class ScoreResult:
    fit_score: float
    archetype_score: float
    history_score: float
    disqualified: bool
    disqualifier_reason: str | None = None
    comp_score: float = 0.5
    negative_score: float = 0.0
    culture_score: float = 0.0
```

`is_valid` property verifies all component scores are in [0.0, 1.0].

### `RankedListing`

```python
@dataclass
class RankedListing:
    listing: JobListing
    scores: ScoreResult
    final_score: float
    duplicate_boards: list[str] = field(default_factory=list)
```

### `RankSummary`

```python
@dataclass
class RankSummary:
    total_found: int = 0
    total_scored: int = 0
    total_excluded: int = 0
    total_deduplicated: int = 0
```

### `CompResult`

```python
@dataclass
class CompResult:
    comp_min: float
    comp_max: float
    comp_source: str       # "employer" or "estimated"
    comp_text: str          # original matched snippet
```
