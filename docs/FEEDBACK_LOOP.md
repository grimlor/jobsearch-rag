# Feedback Loop

> How human decisions improve the system over time. Covers the
> decide → index → rescore → eval cycle and the mechanisms that make the
> scoring pipeline learn from your past choices.

---

## The Loop

```
        ┌───────────────────────────────────────────┐
        │                                           │
        ▼                                           │
  ┌───────────┐     ┌──────────┐     ┌──────────┐   │
  │  search   │────▶│  review  │────▶│  decide  │───┘
  └───────────┘     └──────────┘     └──────────┘
        │                                 │
        │                                 ▼
        │                           ┌────────────┐
        │                           │ decisions  │──▶ ChromaDB
        │                           │ collection │──▶ JSONL audit
        │                           └────────────┘
        │                                 │
        │          ┌───────────┐          │
        ├─────────▶│  rescore  │◀─────────┘
        │          └───────────┘    (re-score with updated history)
        │                │
        │                ▼
        │          ┌──────────┐
        └─────────▶│   eval   │  measure: agreement, precision, recall
                   └──────────┘
```

---

## How Decisions Influence Scoring

### History Score

When you record a verdict, it enters the `decisions` ChromaDB collection:

- **"yes" verdicts** have `scoring_signal=true` — they contribute to
  `history_score` on future runs
- **"no" and "maybe" verdicts** have `scoring_signal=false` — they are
  stored for audit and cross-run dedup but do not influence scoring

The scorer queries the `decisions` collection with
`where={"verdict": "yes"}`, so future JDs similar to ones you've approved
score higher on `history_score`.

**Design rationale:** Only positive decisions are scoring signals because
rejections are too diverse in their reasons. A "no" could mean wrong
industry, wrong level, wrong comp, or wrong work model — these are
already covered by dedicated collections (negative_signals, comp_score,
culture_score). Mixing rejection reasons into a single history signal
would add noise.

### Operator Reasoning as Embedding Signal

When you provide a reason with your verdict, the decision recorder appends
it to the JD text before embedding:

```
{jd_text}\n\nOperator reasoning: {reason}
```

This shifts the embedding vector toward the operator's conceptual framing.
If you reject a listing reasoning "too much frontend work", future JDs with
similar frontend emphasis will have higher cosine similarity to that
decision — but since only "yes" decisions contribute to `history_score`,
this reasoning augmentation primarily helps the disqualifier prompt (which
receives past "no" reasons as examples).

### Disqualifier Personalization

Past "no" verdicts with reasons are injected into the disqualifier prompt
as examples of what the operator has previously rejected. This teaches the
LLM your specific rejection patterns beyond the hardcoded disqualification
criteria.

---

## The Rescore Cycle

After recording several decisions, you can re-score previous results without
re-running browser searches:

```bash
jobsearch-rag rescore
```

This:

1. Loads JD files from `output/jds/*.md`
2. Reconstructs `JobListing` objects from Markdown metadata headers
3. Re-scores each through the current scorer (with updated collections)
4. Re-ranks with current weights
5. Re-exports results

The rescore cycle is the primary mechanism for iterating on scoring quality:

1. Run a search → get initial results
2. Review and record verdicts on some listings
3. Update archetypes, rubric, or weights based on what you see
4. **Rescore** → see how rankings change without waiting for new searches
5. **Eval** → measure agreement between pipeline and your decisions
6. Repeat steps 3–5 until scoring aligns with your judgment

---

## Evaluation

The `eval` subcommand quantifies how well the pipeline matches your judgment:

### Metrics

| Metric | Definition | What It Tells You |
|---|---|---|
| **Agreement rate** | `agreed / total` | Overall alignment between you and the pipeline |
| **Precision** | `true_positive / pipeline_positive` | Of listings the pipeline would surface, how many would you approve? |
| **Recall** | `true_positive / human_yes` | Of listings you approved, how many did the pipeline surface? |
| **Spearman ρ** | Rank correlation between scores and verdicts | Do higher scores correspond to more positive verdicts? |

### Verdict Classification

- **Human positive:** "yes" or "maybe"
- **Human negative:** "no"
- **Pipeline positive:** `final_score ≥ min_score_threshold`

### Model Comparison

`eval --compare-models mistral:7b llama3:8b` runs dual evaluation:

1. Score all decisions with model A
2. Score all decisions with model B
3. Compute deltas: `agreement_δ`, `precision_δ`, `recall_δ`, `spearman_δ`

This enables A/B testing of LLM models without affecting production data.

### Eval Artifacts

- **Report:** `output/eval_YYYY-MM-DD.md` — summary metrics + disagreement
  list (which listings you and the pipeline disagree on)
- **History:** `data/eval_history.jsonl` — one line per eval run for tracking
  improvement over time

---

## Tuning Workflow

### Improving Precision (too many bad listings surfaced)

1. Run `eval` — check which "no" verdicts the pipeline scored above threshold
2. Look at the disagreement list — what pattern do they share?
3. Options:
   - **Add negative signals** to `global_rubric.toml` if it's a universal pattern
   - **Add archetype `signals_negative`** if it's role-specific
   - **Increase `negative_weight`** to amplify the penalty
   - **Increase `min_score_threshold`** to be more selective
4. Re-index: `jobsearch-rag index --archetypes-only`
5. Rescore and re-eval

### Improving Recall (too many good listings missed)

1. Run `eval` — check which "yes" verdicts the pipeline scored below threshold
2. Look at the disagreement list — what's missing from the archetype?
3. Options:
   - **Expand `signals_positive`** in the archetype that should match
   - **Add a new archetype** if none of the existing ones capture the pattern
   - **Decrease `min_score_threshold`** to let more through
   - **Increase `archetype_weight`** or `fit_weight`
4. Re-index and rescore

### Improving Spearman Correlation

Low Spearman ρ means the ranking order doesn't match your preferences even
if the threshold decisions are correct. This usually indicates weight
miscalibration:

- **High archetype but low agreement:** `fit_weight` may need to increase
- **Good matches get low scores:** Check if `negative_weight` is too
  aggressive
- **Comp-appropriate roles rank low:** Increase `comp_weight`

---

## Cross-Run Deduplication

Decisions also serve as a deduplication mechanism across runs:

1. First run: you see listing X, decide "no"
2. Second run: listing X appears again on the same or different board
3. Pipeline checks `decisions` collection for `external_id` — finds your
   prior "no"
4. Listing X is **skipped** — you never see it again

This prevents decision fatigue from seeing the same listings repeatedly.
The `--force-rescore` flag overrides this when you want to re-evaluate
everything (e.g., after a significant scoring configuration change).
