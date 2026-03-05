# BDD Mock Audit — jobsearch-rag/tests/

**Principle:** Mock at I/O boundaries only. Never mock our own code.

I/O boundaries (OK to mock): `ollama.AsyncClient` methods, `subprocess.Popen`,
Playwright `async_playwright` / page objects, `urllib.request.urlopen`,
`webbrowser.open`, `builtins.input`, `asyncio.sleep`, `shutil.which/rmtree`,
`tempfile.mkdtemp`, `random.uniform`, `builtins.__import__`, `sys.modules`.

Our code (never mock): `Embedder`, `Scorer`, `Indexer`, `VectorStore`,
`PipelineRunner`, `AdapterRegistry`, `SessionManager`, `DecisionRecorder`,
`Ranker`, `Rescorer`, `load_settings`, `ReviewSession`, `card_to_listing`.

---

## File-by-File Audit

### conftest.py

| Mock target | Verdict | Rationale |
|---|---|---|
| `Embedder.__new__` + `embed`/`classify`/`health_check = AsyncMock()` | ✅ CORRECT | Bypasses `__init__` (which creates `ollama.AsyncClient`); stubs only the methods that make Ollama HTTP calls |
| `VectorStore(persist_dir=tmp_path)` — real instance | ✅ CORRECT | No mock; real ChromaDB backed by temp dir |
| `DecisionRecorder(store, embedder)` — real instance | ✅ CORRECT | No mock; wired with real store + I/O-stubbed embedder |

---

### test_embedder.py (404 lines)

| Mock target | Verdict | Rationale |
|---|---|---|
| `patch.object(embedder, "_client")` → stubs `_client.embed`, `_client.chat`, `_client.list` | ✅ CORRECT | `_client` is the `ollama.AsyncClient` instance — the exact I/O boundary. All Ollama HTTP calls are mocked; all Embedder logic runs real. |

**Rating: EXEMPLARY** — every test exercises real Embedder logic; only the
Ollama wire protocol is stubbed.

---

### test_indexer.py (720 lines)

| Mock target | Verdict | Rationale |
|---|---|---|
| `mock_embedder` fixture (same Embedder.__new__ pattern) | ✅ CORRECT | I/O boundary |
| Real `VectorStore` with temp dir | ✅ CORRECT | No mock |
| Real `Indexer` with real store + mocked embedder | ✅ CORRECT | No mock |
| `store._get_existing_collection(...)` in assertion | ⚠️ BORDERLINE | Accesses private member for verification — not a mock, but couples test to internal API |

**Rating: EXEMPLARY** — real indexing pipeline end-to-end; only Ollama I/O stubbed.

---

### test_cross_run_dedup.py (317 lines)

| Mock target | Verdict | Rationale |
|---|---|---|
| `MagicMock()` for `DecisionRecorder` | ❌ WRONG | Our own class — should use real DecisionRecorder with real store |
| `runner._decision_recorder = _mock_recorder(...)` | ❌ WRONG | Replaces our own component via private attribute |
| `patch.object(runner, "_search_board", ...)` | ❌ WRONG | Mocks PipelineRunner private method |
| `patch.object(runner, "_ensure_indexed", ...)` | ❌ WRONG | Mocks PipelineRunner private method |
| `patch.object(runner._scorer, "score", ...)` | ❌ WRONG | Mocks our Scorer — should use real Scorer with mocked embedder |
| `patch.object(runner._embedder, "health_check", ...)` | ✅ CORRECT | Ollama health endpoint = I/O boundary |
| `patch.object(runner._embedder, "embed", ...)` | ✅ CORRECT | Ollama embed endpoint = I/O boundary |

**Rating: POOR** — 5 mocks target our own code. The dedup behaviour should be
tested with a real `PipelineRunner` + real `Scorer` + real `DecisionRecorder`,
mocking only the Ollama client and the board adapter (I/O).

---

### test_registry.py

| Mock target | Verdict | Rationale |
|---|---|---|
| `page = MagicMock()` (Playwright page) | ✅ CORRECT | Browser page = I/O boundary |
| Real adapter classes, real `AdapterRegistry` | ✅ CORRECT | No mocks |

**Rating: EXEMPLARY** — zero anti-pattern mocks.

---

### test_runner.py (945 lines)

| Mock target | Verdict | Rationale |
|---|---|---|
| `patch("...runner.Embedder", ...)` | ❌ WRONG | Mocks our Embedder class at construction time |
| `patch("...runner.Scorer", ...)` | ❌ WRONG | Mocks our Scorer class |
| `patch("...runner.AdapterRegistry", mock_registry)` | ❌ WRONG | Mocks our AdapterRegistry |
| `patch("...runner.Indexer", ...)` | ❌ WRONG | Mocks our Indexer class |
| `patch("...runner.SessionManager", ...)` | ⚠️ BORDERLINE | SessionManager wraps browser I/O but is our code |
| `patch("...runner.throttle", ...)` | ⚠️ BORDERLINE | `throttle()` wraps `asyncio.sleep` — I/O-adjacent |
| Mock adapter with `_auth`/`_search`/`_extract` as AsyncMock | ⚠️ BORDERLINE | Adapters represent the board-I/O boundary; MagicMock is acceptable here |

**Rating: POOR** — the test factory `_make_runner_with_mocks()` replaces 4+ of
our own classes. The runner should be constructed with real Scorer + real Indexer +
I/O-stubbed Embedder + mock Playwright adapter.

---

### test_scorer.py (1273 lines)

| Mock target | Verdict | Rationale |
|---|---|---|
| `mock_embedder` fixture | ✅ CORRECT | I/O boundary (Ollama) |
| Real `VectorStore` (populated with test data) | ✅ CORRECT | No mock |
| Real `Scorer` | ✅ CORRECT | No mock |
| `populated_store.query = MagicMock(...)` (1 test) | ⚠️ BORDERLINE | Monkey-patches query to simulate empty distances edge case |
| All `Ranker` / `TestCrossBoardDeduplication` tests | ✅ CORRECT | Pure function — zero mocks |

**Rating: GOOD** — only one borderline case; the vast majority exercises the
real scoring pipeline end-to-end.

---

### test_cli.py (1772 lines)

| Mock target | Verdict | Rationale |
|---|---|---|
| `patch("...cli.load_settings", ...)` | ❌ WRONG | Our own config loader |
| `patch("...cli.Embedder", ...)` | ❌ WRONG | Our Embedder class |
| `patch("...cli.VectorStore")` | ❌ WRONG | Our VectorStore class |
| `patch("...cli.Indexer", ...)` | ❌ WRONG | Our Indexer class |
| `patch("...cli.PipelineRunner", ...)` | ❌ WRONG | Our PipelineRunner class |
| `patch("...cli.AdapterRegistry")` | ❌ WRONG | Our AdapterRegistry |
| `patch("...cli.DecisionRecorder", ...)` | ❌ WRONG | Our DecisionRecorder |
| `patch("...cli.Scorer")` | ❌ WRONG | Our Scorer |
| `patch("...cli.Rescorer", ...)` | ❌ WRONG | Our Rescorer |
| `patch("...cli.SessionManager", ...)` | ⚠️ BORDERLINE | Wraps browser I/O |
| `patch("webbrowser.open")` | ✅ CORRECT | I/O boundary |
| `patch("builtins.input", ...)` | ✅ CORRECT | I/O boundary |
| `patch("...review.webbrowser.open")` | ✅ CORRECT | I/O boundary |
| `TestParserConstruction` — no mocks, pure function tests | ✅ CORRECT | No mocks |

**Rating: POOR** — 9 distinct mocks target our own code. The CLI handler
tests mock every dependency instead of wiring the real dependency graph.
This is the single worst file in the suite.

**Refactoring note:** `handle_search`, `handle_index`, `handle_reset`,
`handle_rescore` and `handle_review` each independently patch 3-6 of our
own classes. The correct approach is to construct the real object graph
(real VectorStore + temp ChromaDB, real Scorer + I/O-stubbed Embedder,
real Indexer, real PipelineRunner), mocking **only** Ollama and Playwright.

---

### test_browser_failures.py (419 lines)

| Mock target | Verdict | Rationale |
|---|---|---|
| Real `ActionableError` construction — no mocks | ✅ CORRECT | Pure value-object tests |
| Real `_FakeAdapter` + real `throttle()` | ✅ CORRECT | No mocks |
| `page = MagicMock()` (Playwright page) | ✅ CORRECT | Browser I/O boundary |

**Rating: EXEMPLARY**

---

### test_export.py (753 lines)

| Mock target | Verdict | Rationale |
|---|---|---|
| Real `MarkdownExporter` + `tmp_path` | ✅ CORRECT | No mocks |
| Real `CSVExporter` | ✅ CORRECT | No mocks |
| `patch("webbrowser.open")` | ✅ CORRECT | I/O boundary |
| Real `JDFileExporter` | ✅ CORRECT | No mocks |

**Rating: EXEMPLARY**

---

### test_session_cdp.py (673 lines)

| Mock target | Verdict | Rationale |
|---|---|---|
| `patch("...session.async_playwright", ...)` | ✅ CORRECT | Playwright launch = I/O |
| `patch("...session.subprocess.Popen")` | ✅ CORRECT | Subprocess = I/O |
| `patch("urllib.request.urlopen")` | ✅ CORRECT | HTTP I/O |
| `patch("...session.tempfile.mkdtemp", ...)` | ✅ CORRECT | Filesystem I/O |
| `patch("shutil.which", ...)` | ✅ CORRECT | Filesystem lookup I/O |
| `patch("...session.shutil.rmtree")` | ✅ CORRECT | Filesystem I/O |
| `patch("...session.asyncio.sleep", ...)` | ✅ CORRECT | I/O boundary |
| `patch("builtins.__import__", ...)` | ✅ CORRECT | Import system I/O |
| `patch.dict("sys.modules", ...)` | ✅ CORRECT | Simulating optional dependency |
| `patch("...session._BROWSER_PATHS", ...)` | ⚠️ BORDERLINE | Module constant, not a function |
| `patch("...session._wait_for_cdp", ...)` | ⚠️ BORDERLINE | Our function wrapping network poll I/O |
| `patch("...session._STORAGE_DIR", tmp_path)` | ⚠️ BORDERLINE | Module constant |

**Rating: GOOD** — nearly all mocks target true I/O. The borderline items
are defensible: `_BROWSER_PATHS` and `_STORAGE_DIR` are environment-dependent
constants; `_wait_for_cdp` is a thin wrapper over network polling.

---

### test_rag_pipeline.py (988 lines)

| Mock target | Verdict | Rationale |
|---|---|---|
| Real `Embedder` with bad URL (no mocks!) | ✅ CORRECT | Tests real connection-error path |
| `patch("...embedder.ollama_sdk.AsyncClient", ...)` | ✅ CORRECT | I/O boundary |
| `mock_embedder` from conftest | ✅ CORRECT | I/O boundary |
| Real `VectorStore`, real `Indexer`, real `Scorer` | ✅ CORRECT | No mocks |

**Rating: EXEMPLARY** — full integration pipeline with real ChromaDB; only
the Ollama SDK is stubbed.

---

### test_ziprecruiter_extraction.py (1095 lines)

| Mock target | Verdict | Rationale |
|---|---|---|
| Pure functions (`extract_js_variables`, `parse_job_cards`, `card_to_listing`, `html_to_text`) | ✅ CORRECT | No mocks needed |
| `_make_mock_page()` — MagicMock for Playwright page | ✅ CORRECT | Browser I/O boundary |
| `patch("...ziprecruiter.asyncio.sleep", ...)` | ✅ CORRECT | I/O boundary |
| `patch("...ziprecruiter.random.uniform", ...)` | ✅ CORRECT | Randomness source I/O |
| `patch("...ziprecruiter._CF_WAIT_TIMEOUT", 1)` | ⚠️ BORDERLINE | Module constant override |
| `patch("...ziprecruiter.card_to_listing", ...)` (1 test: `test_search_skips_unparseable_card`) | ❌ WRONG | Mocks our own pure function to simulate per-card failure |

**Rating: GOOD** — one mock of our code in a single test; everything else
is I/O-boundary or pure function testing.

---

### test_interactive_review.py (393 lines)

| Mock target | Verdict | Rationale |
|---|---|---|
| Real `decision_recorder`, real `vector_store` (conftest) | ✅ CORRECT | No mocks |
| `patch("...review.webbrowser.open")` | ✅ CORRECT | I/O boundary |

**Rating: EXEMPLARY**

---

### test_config.py (564 lines)

No mocks at all. Pure function tests on `load_settings()` with real TOML parsing.

**Rating: EXEMPLARY**

---

### test_decisions.py (334 lines)

| Mock target | Verdict | Rationale |
|---|---|---|
| `mock_embedder` fixture | ✅ CORRECT | I/O boundary |
| Real `VectorStore`, real `DecisionRecorder` | ✅ CORRECT | No mocks |
| `mock_store = MagicMock()` (1 test: `test_get_decision_returns_none_when_collection_missing`) | ❌ WRONG | Mocks our VectorStore to simulate a missing-collection error |

**Rating: GOOD** — one anti-pattern in a single test (should use a real store
and query a nonexistent collection).

---

### test_rescore.py (300 lines)

| Mock target | Verdict | Rationale |
|---|---|---|
| `MagicMock()` for `Scorer` (`mock.score = AsyncMock(...)`) | ❌ WRONG | Mocks our Scorer |
| Real `Ranker` | ✅ CORRECT | No mock |
| Real `load_jd_files()` | ✅ CORRECT | No mock |

**Rating: POOR** — the Rescorer tests should use a real Scorer wired with
I/O-stubbed Embedder + real VectorStore.

---

### test_throttle_detection.py (401 lines)

| Mock target | Verdict | Rationale |
|---|---|---|
| `MagicMock()`/`AsyncMock()` for Playwright page elements | ✅ CORRECT | Browser I/O boundary |
| `patch("...ziprecruiter.asyncio.sleep", ...)` | ✅ CORRECT | I/O boundary |
| `patch("...ziprecruiter._wait_for_cloudflare", ...)` | ⚠️ BORDERLINE | Our function wrapping browser I/O |

**Rating: GOOD** — the helper `_patch_search_to_click_through()` explicitly
runs real parsing functions (`extract_js_variables`, `parse_job_cards`, etc.)
and only stubs I/O.

---

### test_vector_store.py (383 lines)

No mocks at all. Real ChromaDB with `tmp_path`.

**Rating: EXEMPLARY**

---

### test_errors.py (272 lines)

No mocks at all. Pure `ActionableError` construction tests.

**Rating: EXEMPLARY**

---

### test_file_logging.py (150 lines)

No mocks at all. Real file I/O with `tmp_path`.

**Rating: EXEMPLARY**

---

### test_comp_parser.py (217 lines)

No mocks at all. Pure function tests on `parse_compensation()`.

**Rating: EXEMPLARY**

---

### test_comp_scoring.py (176 lines)

No mocks at all. Pure function tests on `compute_comp_score()`.

**Rating: EXEMPLARY**

---

### test_integration.py

No mocks. End-to-end integration.

**Rating: EXEMPLARY**

---

### test_text.py

No mocks. Pure text-processing tests.

**Rating: EXEMPLARY**

---

## Final Tally

| Category | Count | Meaning |
|---|---|---|
| ✅ CORRECT | ~54 distinct patterns | I/O boundary mock or no mock needed |
| ❌ WRONG | 21 | Mocks our own classes/functions |
| ⚠️ BORDERLINE | 11 | Defensible but worth reviewing |

### Breakdown of ❌ WRONG by file

| File | ❌ Count | Targets mocked |
|---|---|---|
| test_cli.py | **9** | load_settings, Embedder, VectorStore, Indexer, PipelineRunner, AdapterRegistry, DecisionRecorder, Scorer, Rescorer |
| test_cross_run_dedup.py | **5** | DecisionRecorder, _search_board, _ensure_indexed, _scorer.score, _decision_recorder assignment |
| test_runner.py | **4** | Embedder, Scorer, AdapterRegistry, Indexer |
| test_rescore.py | **1** | Scorer |
| test_decisions.py | **1** | VectorStore (1 test) |
| test_ziprecruiter_extraction.py | **1** | card_to_listing (1 test) |

### Worst offenders (priority refactoring targets)

1. **test_cli.py** — 9 distinct our-code mocks. Every handler test mocks the entire dependency graph.
2. **test_cross_run_dedup.py** — 5 mocks. Mocks PipelineRunner internals and Scorer.
3. **test_runner.py** — 4 mocks. The `_make_runner_with_mocks()` factory replaces all domain classes.

### Best examples (reference implementations)

1. **test_embedder.py** — mocks only `ollama.AsyncClient._client`; all Embedder logic runs real.
2. **test_rag_pipeline.py** — full integration with real ChromaDB; only Ollama SDK stubbed.
3. **test_config.py** / **test_vector_store.py** / **test_export.py** — zero mocks.
4. **test_session_cdp.py** — mocks only `subprocess`, `async_playwright`, `urllib`, `shutil`.
5. **test_throttle_detection.py** — real parsing functions; only Playwright page I/O stubbed.

### Recommended remediation pattern

For files rated POOR, the fix follows one template:

```python
# Instead of:
with patch("...cli.Scorer"), patch("...cli.Embedder"), ...

# Construct the real graph:
embedder = Embedder.__new__(Embedder)
embedder.embed = AsyncMock(return_value=[0.1, 0.2, ...])
embedder.classify = AsyncMock(return_value="yes")
embedder.health_check = AsyncMock()

store = VectorStore(persist_dir=str(tmp_path / "chroma"))
scorer = Scorer(store=store, embedder=embedder)
indexer = Indexer(store=store, embedder=embedder)
# ... wire the real object graph, mock only Ollama + Playwright
```

This keeps the "System Specification, Not Unit Testing" principle intact:
tests validate real behaviour through real code paths, with seams only at
true I/O boundaries.
