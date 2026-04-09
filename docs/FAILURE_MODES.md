# Failure Modes

> Catalog of what can go wrong, how the system detects it, and what recovery
> looks like. Every failure maps to an `ErrorType` in the error hierarchy.
> See [ARCHITECTURE.md](ARCHITECTURE.md) for the `ActionableError` design.

---

## Error Philosophy

Every error in this system is designed to be **immediately actionable** by
three audiences:

1. **The operator** — `suggestion` and `troubleshooting` steps in plain English
2. **The calling code** — `error_type` for routing recovery logic
3. **An AI assistant** — `ai_guidance` with concrete next actions, commands,
   and diagnostic checks

The principle: **no error should require searching source code to resolve.**

---

## Failure Categories

### Authentication Failures

| Failure | Detection | Error Type | Recovery |
|---|---|---|---|
| Expired session cookies | Adapter detects redirect to login page | `AUTHENTICATION` | `jobsearch-rag login --board <name>` |
| CAPTCHA encountered | Adapter detects challenge interstitial | `AUTHENTICATION` | Wait, then `login --board <name>` manually |
| Missing storage state file | `SessionManager` finds no cookies file | `AUTHENTICATION` | First-run `login` (not a crash) |
| LinkedIn authwall redirect | URL check in `check_linkedin_detection()` | `AUTHENTICATION` | Wait 24h+ before retrying |
| LinkedIn challenge interstitial | Page content check | `AUTHENTICATION` | Wait, avoid rapid retries |

**Design decision:** CAPTCHA detection halts the entire board run immediately.
No retries are attempted because retrying increases ban risk. Partial results
collected before the CAPTCHA are preserved and exported.

### Connection Failures

| Failure | Detection | Error Type | Recovery |
|---|---|---|---|
| Ollama unreachable | `health_check()` HTTP error | `CONNECTION` | Start Ollama: `ollama serve` |
| Ollama timeout on embedding | HTTP 408/504 during `embed()` | `CONNECTION` | Check system resources, restart Ollama |
| Job board unreachable | Playwright navigation timeout | `CONNECTION` | Check network, retry later |
| ChromaDB corruption | SQLite errors on collection access | `CONNECTION` | `jobsearch-rag reset` to rebuild |

**Retry policy for Ollama:** Transient failures (HTTP 408, 429, 500, 502, 503,
504) are retried 3 times with exponential backoff (1s, 2s, 4s). Non-transient
failures (404, 401) fail immediately.

### Model Failures

| Failure | Detection | Error Type | Recovery |
|---|---|---|---|
| Embed model not pulled | `health_check()` model list check | `EMBEDDING` | `ollama pull nomic-embed-text` |
| LLM model not pulled | `health_check()` model list check | `EMBEDDING` | `ollama pull mistral:7b` |
| Embedding call fails after retries | 3 retries exhausted | `EMBEDDING` | Check Ollama logs, system memory |

**Design decision:** Health check runs before any browser work. This avoids
the scenario where the operator waits for search extraction only to discover
Ollama isn't running when scoring begins.

### Parse Failures

| Failure | Detection | Error Type | Recovery |
|---|---|---|---|
| Board HTML structure changed | Adapter returns no results from known-good query | `PARSE` | Update selectors in adapter |
| DOM cards missing or changed | ZipRecruiter `article[id^="job-card-"]` not found or JSON-LD structure changed | `PARSE` | Board may have redesigned; update adapter |
| Malformed TOML config | `tomllib.loads()` raises | `PARSE` | Fix TOML syntax (error includes file path) |
| Malformed global rubric | Missing expected keys | `PARSE` | Fix rubric structure |

### Index Failures

| Failure | Detection | Error Type | Recovery |
|---|---|---|---|
| Empty `resume` collection | `collection_count("resume") == 0` at query time | `INDEX` | `jobsearch-rag index --resume-only` |
| Empty `role_archetypes` collection | `collection_count("role_archetypes") == 0` | `INDEX` | `jobsearch-rag index --archetypes-only` |
| Missing collection entirely | ChromaDB collection not found | `INDEX` | `jobsearch-rag index` |

**Auto-recovery:** The pipeline runner auto-indexes if collections are empty
at startup. This handles the first-run case transparently.

### Config Failures

| Failure | Detection | Error Type | Recovery |
|---|---|---|---|
| Missing `settings.toml` | File not found on load | `CONFIG` | Create from template |
| Enabled board has no config section | Validation check | `CONFIG` | Add `[boards.<name>]` section |
| `global_rubric_path` file missing | Validation check | `CONFIG` | Create the file or fix the path |

### Validation Failures

| Failure | Detection | Error Type | Recovery |
|---|---|---|---|
| Weight outside [0.0, 1.0] | Config validation | `VALIDATION` | Fix value in `settings.toml` |
| `base_salary ≤ 0` | Config validation | `VALIDATION` | Set positive value |
| Invalid Ollama URL scheme | Config validation | `VALIDATION` | Use `http://` or `https://` |
| `full_text` exceeds 250K chars | `JobListing.__post_init__` | `VALIDATION` | Adapter extraction bug — truncate |
| Empty text sent to embedder | `embed()` whitespace check | `VALIDATION` | Adapter returned empty `full_text` |

### Decision Failures

| Failure | Detection | Error Type | Recovery |
|---|---|---|---|
| Decision for unknown job ID | `get_decision()` returns None | `DECISION` | Check job ID against results |

### Extraction Failures (Graceful Degradation)

These are not fatal errors — the pipeline continues with partial results:

| Failure | Behavior | Logging |
|---|---|---|
| 404 on detail page | Listing skipped, pipeline continues | Warning with job URL |
| Empty extracted text | Listing excluded from scoring | Warning with job URL |
| Network timeout on single listing | One retry, then skip | Warning with retry count |
| Comp parsing finds nothing | `comp_score = 0.5` (neutral) | No warning (normal) |
| Disqualifier JSON malformed | Listing **not** disqualified (safe default) | Warning with raw response |
| Optional collection empty | Score = 0.0 for that component | No warning (expected on first run) |

---

## Bot Detection and Rate Limiting

### LinkedIn-Specific

LinkedIn has aggressive bot detection. The system handles this with:

1. **CDP mode** — Uses real system browser (Edge/Chrome) instead of
   Playwright's bundled Chromium, avoiding automation flags
2. **Stealth patches** — `playwright-stealth` removes WebDriver indicators
3. **Extended throttling** — 8–20 second random jitter between requests
   (vs. 1.5–3.5s for other boards)
4. **Detection checks** — `check_linkedin_detection()` inspects page content
   for authwall redirects and challenge interstitials
5. **Halt on detection** — No retries after detection. The run stops
   immediately for that board. Partial results are preserved.

### ZipRecruiter Throttling

After approximately 1.5 pages of results, ZipRecruiter starts returning
"We encountered an error while loading this job" in the detail panel during
card click-through — this is server-side rate limiting, not an adapter bug.
The adapter detects this via `is_throttle_response()` and retries with
exponential backoff (`throttle_base_delay × 2^(n−1)`, e.g., 4 → 8 → 16 s
with base 4.0) up to `throttle_max_retries` (both
configurable per board in `settings.toml`). The backoff counter resets
after each successful panel load, so only consecutive throttle responses
escalate the delay.

**Estimated vs. employer-stated salary:** ZipRecruiter shows its own salary
estimate on cards, which can differ from the employer-stated range in the JD
body. The comp parser tracks `comp_source` to distinguish these;
employer-stated is preferred when both are present.

### General Throttling

All boards are throttled via `SessionManager.throttle(adapter)`:

- Random sleep within the board's `rate_limit_range` from `settings.toml`
- Applied between search-URL navigations and between individual JD
  detail-page requests
- Boards that enrich `full_text` during search (e.g., ZR card
  click-through) use their own internal pacing; the runner's per-detail
  throttle is skipped for already-enriched listings
- LinkedIn's range (8–20s) is much wider than other boards (1.5–3.5s)

---

## Prompt Injection Defense

Job descriptions are untrusted input that gets passed to the LLM
disqualifier. The system defends against prompt injection in four layers:

| Layer           | Mechanism                                              | What It Catches                                   |
| --------------- | ------------------------------------------------------ | ------------------------------------------------- |
| 1. Screening    | Separate LLM call checks for injection language        | "Ignore previous instructions", "Disregard above" |
| 2. Regex        | Strip known injection patterns from JD text            | JSON override attempts, role-play instructions    |
| 3. Validation   | Parse disqualifier output as JSON with expected schema | Injected freeform text instead of JSON            |
| 4. Safe default | Malformed JSON → `disqualified=False`                  | Any bypass that produces unparseable output       |

**Philosophy:** False negatives (missing a bad listing) are preferred over
false positives (rejecting a good listing due to a failed injection attempt).

---

## Observability for Failures

Session logs capture failure context:

- `embed_call` events with non-200 status codes and retry counts
- `disqualifier_call` events with malformed response markers
- `prompt_injection_detected` events with the triggering pattern
- `session_summary` includes `failed_listings` count and `slow_llm_calls`

The `retrieval_summary` event per collection includes:
- `below_threshold` count — listings that scored below `min_score_threshold`
  for that specific collection
- Score distribution (`min`, `p50`, `p90`, `max`) — helps identify if a
  collection's embeddings are well-calibrated
