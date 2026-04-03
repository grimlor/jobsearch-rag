# Evolution Over Time

> How the system grew from a simple scraper to a six-dimensional scoring
> pipeline with a feedback loop. Documents the design decisions, inflection
> points, and what drove each addition.

---

## Phase 1 — Extraction

**Problem:** Manually browsing job boards is slow and repetitive.

**Solution:** Playwright-based adapters that extract `JobListing` objects
from board HTML/JSON.

**Key decisions:**
- `JobListing` as a dataclass contract — decouples extraction from everything
  downstream
- Adapter ABC + decorator-based registry — adding a new board requires zero
  changes to the pipeline
- Session cookie persistence — avoids re-authentication on every run

**What existed:** CLI → adapters → raw listing output. No scoring.

---

## Phase 2 — RAG Scoring (Three Collections)

**Problem:** Extracted listings need to be ranked by relevance, not just listed.

**Solution:** Embed JDs and compare against three ChromaDB collections.

**Initial collections:**
1. `resume` → `fit_score` — does your background match?
2. `role_archetypes` → `archetype_score` — is this the kind of role you want?
3. `decisions` → `history_score` — does this resemble roles you've approved?

**Key decisions:**
- Local-only inference via Ollama — no API keys, no data egress
- ChromaDB embedded mode — no server, no docker, just a SQLite file
- Cosine similarity — directional alignment, invariant to text length
- Separate collections per score dimension — independently tunable

**What changed:** Listings are now ranked. The operator sees the best matches
first instead of a flat list.

---

## Phase 3 — Score Fusion and Deduplication

**Problem:** Three separate scores need a single ranking. Same job appears on
multiple boards.

**Solution:** Weighted fusion formula and two-pass deduplication.

**Key decisions:**
- Weights are not normalized to 1.0 — each component is a raw multiplier,
  allowing the operator to boost specific dimensions independently
- Near-dedup threshold at 0.95 cosine similarity — tight enough to avoid
  false collapses, loose enough to catch reformatted reposts
- Exact dedup on `(board, external_id)` runs first as a cheap pre-filter

---

## Phase 4 — Actionable Errors

**Problem:** When things go wrong (expired cookies, Ollama not running, model
not pulled), the operator gets opaque Python tracebacks.

**Solution:** `ActionableError` hierarchy with three-audience error design.

**Key decisions:**
- Errors classified by **recovery path**, not by origin module
- Factory classmethods enforce consistent construction
- `AIGuidance` field for automated recovery by AI assistants
- `from_exception()` auto-classifies unknown exceptions by keyword matching
- Health check runs before browser work — fail fast, not fail late

---

## Phase 5 — LLM Disqualification

**Problem:** Semantic similarity can't catch structural problems. An
"Architect" title on a junior IC role scores high on archetype match.

**Solution:** An LLM pass (`mistral:7b`) that screens for disqualification
criteria the embedding model can't detect.

**Key decisions:**
- Safe default: if the LLM produces unparseable output, the listing is **not**
  disqualified. False negatives preferred over false positives.
- Multi-layer injection defense — JDs are untrusted input going into an LLM
  prompt. Screening, regex sanitization, and output validation defend against
  prompt injection.
- Disqualification zeroes out `final_score` entirely — it's not a penalty,
  it's a gate.

---

## Phase 6 — Compensation Scoring

**Problem:** Two otherwise identical roles at $120K vs. $220K should rank
differently.

**Solution:** Regex-based compensation extraction and a piecewise-linear
scoring curve.

**Key decisions:**
- Regex, not LLM — salary patterns are predictable. No inference cost for
  a reliably solvable problem.
- Head+tail truncation — real JDs put comp info in the tail third, so
  naive head-only truncation loses it before embedding. The 60/40 split
  preserves both overview and details.
- `0.5` default for missing comp — neutral, not punitive. Many JDs don't
  list compensation.
- Piecewise-linear curve with continuous boundaries — no score jumps at
  band transitions.
- `base_salary` is configurable — the curve scales with your expectations.

---

## Phase 7 — Negative Signals and Culture Scoring

**Problem:** Fit and archetype similarity are necessary but not sufficient.
A high-fit role at an adtech company with mandatory RTO should rank lower
than the same role at a remote-first devtools company.

**Solution:** Two new collections — `negative_signals` (subtractive penalty)
and `global_positive_signals` (additive culture match).

**Key decisions:**
- Negative signals are embedded individually — a single red flag should
  trigger the penalty. Synthesizing them would dilute the signal.
- Positive signals are synthesized per dimension — they represent holistic
  qualities ("good culture" = remote + async + autonomy together).
- `negative_weight` is subtractive, not multiplicative — it reduces the
  final score directly: `max(0.0, positive - penalty)`.
- Global rubric is domain-universal; archetype signals are role-specific.
  Both feed `negative_signals`, but only the rubric feeds
  `global_positive_signals`.

---

## Phase 8 — Feedback Loop (Decide, Review, Eval)

**Problem:** The scoring pipeline is only as good as its initial
configuration. There's no way to measure or improve it.

**Solution:** Decision recording, interactive review, and quantitative
evaluation.

**Key decisions:**
- Only "yes" verdicts are scoring signals — rejections are too diverse
- Operator reasoning augments embedding vectors — shifts decisions toward
  the conceptual framing of *why*
- Past "no" reasons personalize the disqualifier prompt
- Eval metrics (precision, recall, Spearman ρ) give quantitative feedback
  on scoring quality
- Model comparison (`--compare-models`) enables A/B testing without
  affecting production data

---

## Phase 9 — Rescore Pipeline

**Problem:** After tuning weights, archetypes, or rubric signals, the
operator has to run a full browser search to see the effect.

**Solution:** `rescore` subcommand that re-scores previously exported JDs
from disk without browser interaction.

**Key decisions:**
- JD files contain enough metadata (title, company, board, URL) to
  reconstruct `JobListing` objects without the original extraction
- Enables tight tuning loops: edit config → rescore → eval → repeat
- Zero browser overhead for configuration iteration

---

## Phase 10 — Observability

**Problem:** When scores seem wrong, there's no way to diagnose whether the
issue is embedding quality, collection contents, or weight miscalibration.

**Solution:** Structured JSONL session logs with per-call and per-collection
metrics.

**Key decisions:**
- One JSONL file per run with a session ID for correlation
- Per-listing `score_computed` events show all six components
- Per-collection `retrieval_summary` shows score distribution statistics
- `InferenceMetrics` tracks call counts, tokens, and latency
- Slow LLM threshold is configurable — flags calls that may indicate
  resource pressure

---

## What's Next

Areas identified but not yet implemented:

- **Glassdoor/salary API integration** — supplement regex comp extraction
  with external salary data
- **Structured archetype matching** — beyond cosine similarity, use
  skill-graph or taxonomy matching
- **Notification pipeline** — push new high-scoring listings to Slack/email
  instead of requiring the operator to run searches manually
- **Historical score tracking** — track how a specific listing's score
  changes as the scoring configuration evolves
