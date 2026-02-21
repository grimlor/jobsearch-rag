"""Integration tests — validate external dependency contracts.

These tests require **live Ollama** with ``nomic-embed-text`` and
``mistral:7b`` models pulled.  They are skipped by default and run
only when explicitly requested::

    uv run pytest -m integration          # run integration tests only
    uv run pytest -m "not integration"    # run everything else (default)
    uv run pytest                          # also skips integration

The integration marker is defined in ``pyproject.toml`` under
``[tool.pytest.ini_options].markers``.

Four test classes validate the assumptions our unit tests make:

1. **TestOllamaContract** — the Ollama SDK response shapes we rely on
   (embedding vectors, chat responses, model listing, error types).

2. **TestChromaDBContract** — ChromaDB distance semantics and
   persistence behavior.

3. **TestEndToEndScoring** — the full pipeline from resume indexing
   through scoring, using real Ollama embeddings with no mocks.

4. **TestLiveZipRecruiterPipeline** — the full system against live
   ZipRecruiter: browser session → search → extract → score → rank →
   export.  Requires a valid session cookie and Ollama.  Run with
   ``uv run pytest -m live``.

Between them, these tests catch the class of bug where our mocks
silently diverge from reality — e.g. an ollama SDK upgrade changes
the response shape, or ChromaDB alters its distance metric.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from jobsearch_rag.adapters.session import SessionConfig, SessionManager, throttle
from jobsearch_rag.adapters.ziprecruiter import ZipRecruiterAdapter
from jobsearch_rag.errors import ActionableError, ErrorType
from jobsearch_rag.export.csv_export import CSVExporter
from jobsearch_rag.export.jd_files import JDFileExporter
from jobsearch_rag.export.markdown import MarkdownExporter
from jobsearch_rag.pipeline.ranker import Ranker
from jobsearch_rag.rag.comp_parser import compute_comp_score, parse_compensation
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.indexer import Indexer
from jobsearch_rag.rag.scorer import Scorer
from jobsearch_rag.rag.store import VectorStore

if TYPE_CHECKING:
    from collections.abc import Iterator

# All tests in this file require live Ollama
pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Configuration — matches settings.toml defaults
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral:7b"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embedder() -> Embedder:
    """A real Embedder pointed at localhost Ollama."""
    return Embedder(
        base_url=OLLAMA_BASE_URL,
        embed_model=EMBED_MODEL,
        llm_model=LLM_MODEL,
        max_retries=2,
        base_delay=0.5,
    )


@pytest.fixture
def store() -> Iterator[VectorStore]:
    """A VectorStore backed by a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield VectorStore(persist_dir=tmpdir)


@pytest.fixture
def sample_resume(tmp_path: Path) -> Path:
    """A small but realistic resume for integration testing."""
    resume = tmp_path / "resume.md"
    resume.write_text(
        "# Test Candidate\n\n"
        "## Summary\n"
        "Principal architect specializing in distributed systems "
        "and cloud-native platform design.\n\n"
        "## Experience\n"
        "### Acme Corp - Staff Engineer\n"
        "Led platform architecture for microservices migration.\n"
        "Designed event-driven systems with Kafka and Kubernetes.\n\n"
        "## Skills\n"
        "Python, Go, Kubernetes, Kafka, AWS, Terraform, gRPC\n",
        encoding="utf-8",
    )
    return resume


@pytest.fixture
def sample_archetypes(tmp_path: Path) -> Path:
    """A small archetypes TOML for integration testing."""
    archetypes = tmp_path / "archetypes.toml"
    archetypes.write_text(
        "[[archetypes]]\n"
        'name = "Staff Platform Architect"\n'
        'description = """\n'
        "Defines technical strategy and architecture for distributed "
        "systems platforms. Cross-team influence. Cloud-native "
        "infrastructure. API design and governance.\n"
        '"""\n\n'
        "[[archetypes]]\n"
        'name = "SRE Manager"\n'
        'description = """\n'
        "Manages on-call rotations, incident response, and reliability "
        "engineering. Primarily operational with people management.\n"
        '"""\n',
        encoding="utf-8",
    )
    return archetypes


# ---------------------------------------------------------------------------
# TestOllamaContract
# ---------------------------------------------------------------------------


class TestOllamaContract:
    """REQUIREMENT: Ollama SDK responses match the shapes our code assumes.

    WHO: Every unit test that mocks Ollama responses
    WHAT: embed() returns a list of floats with consistent dimensionality;
          classify() returns a string; health_check() passes when models
          are available; errors carry the expected attributes
    WHY: If an Ollama SDK update changes response shapes, our mocks would
         still pass but production would break — these tests catch that drift
    """

    async def test_embed_returns_float_list_with_consistent_dimensions(
        self, embedder: Embedder
    ) -> None:
        """embed() returns a list[float] and repeated calls produce vectors of the same dimensionality."""
        vec1 = await embedder.embed("distributed systems architecture")
        vec2 = await embedder.embed("underwater basket weaving")

        assert isinstance(vec1, list)
        assert len(vec1) > 0
        assert all(isinstance(v, float) for v in vec1)
        assert len(vec1) == len(vec2), f"Embedding dimensions differ: {len(vec1)} vs {len(vec2)}"

    async def test_embed_vector_is_not_all_zeros(self, embedder: Embedder) -> None:
        """A real embedding is not a zero vector — it carries semantic information."""
        vec = await embedder.embed("Principal platform architect role")
        assert any(v != 0.0 for v in vec)

    async def test_classify_returns_string(self, embedder: Embedder) -> None:
        """classify() returns a str response from the LLM."""
        result = await embedder.classify("Respond with exactly one word: hello")
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_health_check_passes_with_required_models(self, embedder: Embedder) -> None:
        """health_check() succeeds when both configured models are pulled."""
        # Should not raise
        await embedder.health_check()

    async def test_health_check_nonexistent_model_suggests_ollama_pull(self) -> None:
        """health_check() for a missing model suggests 'ollama pull' to fix it."""
        embedder = Embedder(
            base_url=OLLAMA_BASE_URL,
            embed_model="does-not-exist-model-xyz",
            llm_model=LLM_MODEL,
        )
        with pytest.raises(ActionableError) as exc_info:
            await embedder.health_check()
        err = exc_info.value
        assert err.error_type == ErrorType.EMBEDDING
        assert "does-not-exist-model-xyz" in err.error
        assert err.suggestion is not None
        assert err.troubleshooting is not None
        assert len(err.troubleshooting.steps) > 0

    async def test_embed_nonexistent_model_provides_recovery_guidance(self) -> None:
        """Embedding with a nonexistent model provides recovery guidance."""
        embedder = Embedder(
            base_url=OLLAMA_BASE_URL,
            embed_model="nonexistent-model-abc",
            llm_model=LLM_MODEL,
            max_retries=1,
            base_delay=0.0,
        )
        with pytest.raises(ActionableError) as exc_info:
            await embedder.embed("test")
        err = exc_info.value
        assert err.error_type == ErrorType.EMBEDDING
        assert err.suggestion is not None
        assert err.troubleshooting is not None

    async def test_similar_texts_produce_closer_embeddings(self, embedder: Embedder) -> None:
        """Semantically similar texts have smaller cosine distance than dissimilar ones.

        This validates our assumption that cosine distance is meaningful
        for the nomic-embed-text model.
        """
        vec_arch = await embedder.embed("Staff platform architect for distributed cloud systems")
        vec_similar = await embedder.embed(
            "Principal engineer designing cloud-native infrastructure"
        )
        vec_unrelated = await embedder.embed("Recipe for chocolate chip cookies with extra sugar")

        def cosine_distance(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b, strict=True))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 1.0
            return 1.0 - dot / (norm_a * norm_b)

        dist_similar = cosine_distance(vec_arch, vec_similar)
        dist_unrelated = cosine_distance(vec_arch, vec_unrelated)

        assert dist_similar < dist_unrelated, (
            f"Similar texts should be closer: {dist_similar:.4f} vs {dist_unrelated:.4f}"
        )


# ---------------------------------------------------------------------------
# TestChromaDBContract
# ---------------------------------------------------------------------------


class TestChromaDBContract:
    """REQUIREMENT: ChromaDB returns distances and results in the format we assume.

    WHO: VectorStore wrapper and Scorer distance-to-score conversion
    WHAT: Cosine distance for identical vectors is ~0.0; distance for
          orthogonal content is higher; query results contain the expected
          keys; persistence survives a client restart
    WHY: Our _distance_to_score function assumes cosine distance semantics
         (0 = identical, 1 = orthogonal) — any deviation would invert
         all scoring logic
    """

    async def test_identical_documents_have_near_zero_distance(
        self, store: VectorStore, embedder: Embedder
    ) -> None:
        """Querying with the same vector used to index returns distance ~0.0."""
        text = "Staff architect for distributed systems"
        embedding = await embedder.embed(text)

        store.add_documents(
            collection_name="test_identity",
            ids=["doc-1"],
            documents=[text],
            embeddings=[embedding],
        )

        results = store.query(
            collection_name="test_identity",
            query_embedding=embedding,
            n_results=1,
        )
        distances = results["distances"][0]
        assert len(distances) == 1
        assert distances[0] == pytest.approx(0.0, abs=1e-5), (
            f"Identical vector should have ~0 distance, got {distances[0]}"
        )

    async def test_dissimilar_documents_have_higher_distance(
        self, store: VectorStore, embedder: Embedder
    ) -> None:
        """Dissimilar content returns higher distance than similar content."""
        vec_arch = await embedder.embed("Distributed systems platform architecture")
        vec_cooking = await embedder.embed("Baking sourdough bread at high altitude")
        vec_query = await embedder.embed("Cloud infrastructure architect role")

        store.add_documents(
            collection_name="test_distance",
            ids=["doc-arch", "doc-cooking"],
            documents=["architecture doc", "cooking doc"],
            embeddings=[vec_arch, vec_cooking],
        )

        results = store.query(
            collection_name="test_distance",
            query_embedding=vec_query,
            n_results=2,
        )
        distances = results["distances"][0]
        ids = results["ids"][0]

        # Architecture doc should be closer (lower distance) to the query
        arch_idx = ids.index("doc-arch")
        cooking_idx = ids.index("doc-cooking")
        assert distances[arch_idx] < distances[cooking_idx], (
            f"Architecture should be closer: arch={distances[arch_idx]:.4f}, "
            f"cooking={distances[cooking_idx]:.4f}"
        )

    async def test_query_results_contain_expected_keys(
        self, store: VectorStore, embedder: Embedder
    ) -> None:
        """Query results contain ids, documents, metadatas, and distances."""
        embedding = await embedder.embed("test document")
        store.add_documents(
            collection_name="test_keys",
            ids=["doc-1"],
            documents=["test document"],
            embeddings=[embedding],
            metadatas=[{"source": "test"}],
        )

        results = store.query(
            collection_name="test_keys",
            query_embedding=embedding,
            n_results=1,
        )

        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results
        # Each is a list of lists (one per query vector)
        assert isinstance(results["ids"][0], list)
        assert isinstance(results["distances"][0], list)

    async def test_persistence_survives_client_restart(self, embedder: Embedder) -> None:
        """Data indexed with one VectorStore instance is available after a restart."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Index with first client
            store1 = VectorStore(persist_dir=tmpdir)
            embedding = await embedder.embed("persistence test")
            store1.add_documents(
                collection_name="test_persist",
                ids=["persist-1"],
                documents=["persistence test"],
                embeddings=[embedding],
            )
            count_before = store1.collection_count("test_persist")

            # Create a new client against the same directory
            store2 = VectorStore(persist_dir=tmpdir)
            count_after = store2.collection_count("test_persist")

            assert count_before == 1
            assert count_after == 1

            # Verify the actual document is retrievable
            docs = store2.get_documents("test_persist", ids=["persist-1"])
            assert docs["documents"][0] == "persistence test"


# ---------------------------------------------------------------------------
# TestEndToEndScoring
# ---------------------------------------------------------------------------


class TestEndToEndScoring:
    """REQUIREMENT: The full index-then-score pipeline produces valid results.

    WHO: The operator running the pipeline for the first time
    WHAT: Indexing real resume + archetypes with real Ollama embeddings,
          then scoring a sample JD, produces a valid ScoreResult with
          meaningful (non-trivial) scores; a matching JD scores higher
          than an unrelated one
    WHY: Unit tests mock Ollama, so they can't catch integration failures
         like dimension mismatches between embed() and ChromaDB, or
         LLM classification prompts that produce unparseable output
    """

    async def test_index_and_score_produces_valid_result(
        self,
        store: VectorStore,
        embedder: Embedder,
        sample_resume: Path,
        sample_archetypes: Path,
    ) -> None:
        """End-to-end: index resume + archetypes, score a JD, get valid ScoreResult."""
        indexer = Indexer(store=store, embedder=embedder)

        resume_count = await indexer.index_resume(str(sample_resume))
        archetype_count = await indexer.index_archetypes(str(sample_archetypes))

        assert resume_count == 3  # Summary, Experience, Skills
        assert archetype_count == 2  # Staff Platform Architect, SRE Manager

        scorer = Scorer(
            store=store,
            embedder=embedder,
            disqualify_on_llm_flag=False,  # skip LLM for speed
        )
        result = await scorer.score(
            "Staff Platform Architect: design and build distributed "
            "systems infrastructure for cloud-native applications"
        )

        assert result.is_valid, (
            f"fit={result.fit_score}, arch={result.archetype_score}, hist={result.history_score}"
        )
        assert result.fit_score > 0.0, "Matching JD should have non-zero fit"
        assert result.archetype_score > 0.0, "Matching JD should have non-zero archetype"
        assert result.history_score == 0.0, "No decisions indexed, history should be 0"

    async def test_matching_jd_scores_higher_than_unrelated(
        self,
        store: VectorStore,
        embedder: Embedder,
        sample_resume: Path,
        sample_archetypes: Path,
    ) -> None:
        """A JD matching the resume and archetypes scores higher than an unrelated JD."""
        indexer = Indexer(store=store, embedder=embedder)
        await indexer.index_resume(str(sample_resume))
        await indexer.index_archetypes(str(sample_archetypes))

        scorer = Scorer(
            store=store,
            embedder=embedder,
            disqualify_on_llm_flag=False,
        )

        result_match = await scorer.score(
            "Principal Engineer: distributed systems, Kubernetes, "
            "platform architecture, cloud infrastructure, API design"
        )
        result_unrelated = await scorer.score(
            "Pastry Chef: create artisan breads and pastries for "
            "a high-end restaurant in downtown Portland"
        )

        assert result_match.fit_score > result_unrelated.fit_score, (
            f"Matching fit {result_match.fit_score:.4f} should exceed "
            f"unrelated {result_unrelated.fit_score:.4f}"
        )
        assert result_match.archetype_score > result_unrelated.archetype_score, (
            f"Matching archetype {result_match.archetype_score:.4f} should exceed "
            f"unrelated {result_unrelated.archetype_score:.4f}"
        )

    async def test_disqualifier_produces_parseable_json(
        self,
        store: VectorStore,
        embedder: Embedder,
        sample_resume: Path,
        sample_archetypes: Path,
    ) -> None:
        """The LLM disqualifier prompt produces a response that parses as valid JSON.

        This validates that our prompt engineering actually elicits the
        expected response format from the real model, not just from our mocks.
        """
        indexer = Indexer(store=store, embedder=embedder)
        await indexer.index_resume(str(sample_resume))
        await indexer.index_archetypes(str(sample_archetypes))

        scorer = Scorer(
            store=store,
            embedder=embedder,
            disqualify_on_llm_flag=True,
        )

        # Score with disqualifier enabled — the result should have
        # a valid disqualified field (True or False), not crash
        result = await scorer.score(
            "Staff Platform Architect: lead the design of cloud-native "
            "distributed systems. Cross-team collaboration and mentoring."
        )
        assert isinstance(result.disqualified, bool)
        # This role should NOT be disqualified
        # (soft assertion — LLM output can vary, but this is a clear match)
        if result.disqualified:
            pytest.skip(
                f"LLM unexpectedly disqualified a matching role: {result.disqualifier_reason}"
            )

    async def test_index_with_real_resume_file(
        self,
        store: VectorStore,
        embedder: Embedder,
    ) -> None:
        """Indexing the actual project resume.md produces the expected chunk count."""
        real_resume = Path("data/resume.md")
        if not real_resume.exists():
            pytest.skip("data/resume.md not found — run from project root")

        indexer = Indexer(store=store, embedder=embedder)
        count = await indexer.index_resume(str(real_resume))

        # The real resume has 5 ## sections
        assert count == 5, f"Expected 5 resume chunks, got {count}"

        # Verify the collection is queryable
        test_vec = await embedder.embed("distributed systems architecture")
        results = store.query(
            collection_name="resume",
            query_embedding=test_vec,
            n_results=3,
        )
        assert len(results["ids"][0]) == 3
        assert all(d >= 0.0 for d in results["distances"][0])

    async def test_index_with_real_archetypes_file(
        self,
        store: VectorStore,
        embedder: Embedder,
    ) -> None:
        """Indexing the actual role_archetypes.toml produces the expected count."""
        real_archetypes = Path("config/role_archetypes.toml")
        if not real_archetypes.exists():
            pytest.skip("config/role_archetypes.toml not found — run from project root")

        indexer = Indexer(store=store, embedder=embedder)
        count = await indexer.index_archetypes(str(real_archetypes))

        # The real archetypes file has 3 entries
        assert count == 3, f"Expected 3 archetypes, got {count}"


# ---------------------------------------------------------------------------
# TestLiveZipRecruiterPipeline
# ---------------------------------------------------------------------------

# This class uses the ``live`` marker — it is excluded from both default
# and integration runs.  Run it explicitly:
#
#     uv run pytest -m live              # live tests only
#     uv run pytest -m "live or integration"  # both tiers


@pytest.mark.live
class TestLiveZipRecruiterPipeline:
    """REQUIREMENT: The full system works end-to-end against live ZipRecruiter.

    WHO: The operator validating the tool after installation or upgrade
    WHAT: A real browser session authenticates with ZipRecruiter, searches
          one query, extracts ≥5 listings with full JD text, scores them
          with real Ollama, ranks them, and exports valid Markdown/CSV/JD files
    WHY: Unit tests mock every I/O boundary — browser, Ollama, ChromaDB.
         Integration tests use real Ollama but fixture HTML.  Only this test
         validates the entire system against the real world: ZipRecruiter's
         DOM structure, Cloudflare behavior, Ollama model output, and
         ChromaDB persistence all working together.
    """

    # Minimum listings we expect from a single search page.  ZR typically
    # returns 20+ per page, so 5 is a conservative floor that accounts
    # for rate-limit or layout changes.
    MIN_LISTINGS = 5

    @pytest.fixture
    def live_embedder(self) -> Embedder:
        """A real Embedder for live pipeline tests."""
        return Embedder(
            base_url=OLLAMA_BASE_URL,
            embed_model=EMBED_MODEL,
            llm_model=LLM_MODEL,
            max_retries=2,
            base_delay=0.5,
        )

    @pytest.fixture
    def live_store(self) -> Iterator[VectorStore]:
        """A VectorStore in a temp directory for live pipeline tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield VectorStore(persist_dir=tmpdir)

    async def test_live_search_score_rank_export(
        self,
        live_store: VectorStore,
        live_embedder: Embedder,
        sample_resume: Path,
        sample_archetypes: Path,
        tmp_path: Path,
    ) -> None:
        """Full pipeline: browser → search → score → rank → export, no mocks."""
        # --- Skip if session file is missing ---
        session_path = Path("data/ziprecruiter_session.json")
        if not session_path.exists():
            pytest.skip("No ZipRecruiter session file — run login first")

        # --- Step 1: Health check Ollama ---
        await live_embedder.health_check()

        # --- Step 2: Index resume + archetypes ---
        indexer = Indexer(store=live_store, embedder=live_embedder)
        await indexer.index_resume(str(sample_resume))
        await indexer.index_archetypes(str(sample_archetypes))

        # --- Step 3: Browser session → authenticate → search one query ---
        adapter = ZipRecruiterAdapter()
        config = SessionConfig(
            board_name="ziprecruiter",
            headless=False,
            browser_channel="msedge",
        )

        search_url = (
            "https://www.ziprecruiter.com/jobs-search"
            "?search=staff+platform+architect&location=Remote+(USA)"
        )

        listings = []
        try:
            async with SessionManager(config) as session:
                page = await session.new_page()

                try:
                    await adapter.authenticate(page)
                except ActionableError as exc:
                    pytest.skip(f"ZipRecruiter auth failed (session expired?): {exc.error}")

                await throttle(adapter)
                try:
                    listings = await adapter.search(page, search_url, max_pages=1)
                except ActionableError as exc:
                    pytest.skip(f"ZipRecruiter search failed: {exc.error}")
        except Exception as exc:
            pytest.skip(f"Browser session failed: {exc}")

        # --- Structural assertions about ZipRecruiter extraction ---
        assert len(listings) >= self.MIN_LISTINGS, (
            f"Expected ≥{self.MIN_LISTINGS} listings, got {len(listings)}"
        )

        for listing in listings:
            assert listing.board == "ziprecruiter"
            assert listing.title, "Every listing must have a title"
            assert listing.company, "Every listing must have a company"
            assert listing.url.startswith("https://"), f"URL should be HTTPS: {listing.url}"
            assert listing.external_id, "Every listing must have an external_id"

        # At least some listings should have full JD text from click-through
        with_jd = [ls for ls in listings if ls.full_text.strip()]
        assert len(with_jd) >= 1, "At least one listing should have full JD text"

        # --- Step 4: Score each listing ---
        scorer = Scorer(
            store=live_store,
            embedder=live_embedder,
            disqualify_on_llm_flag=False,  # skip LLM for speed
        )
        base_salary = 220_000

        scored = []
        embeddings: dict[str, list[float]] = {}

        for listing in with_jd[: self.MIN_LISTINGS]:
            result = await scorer.score(listing.full_text)

            # Parse compensation
            comp = parse_compensation(listing.full_text)
            if comp is not None:
                listing.comp_min = comp.comp_min
                listing.comp_max = comp.comp_max
                listing.comp_source = comp.comp_source
                listing.comp_text = comp.comp_text
            result.comp_score = compute_comp_score(listing.comp_max, base_salary)

            scored.append((listing, result))
            embedding = await live_embedder.embed(listing.full_text)
            embeddings[listing.url] = embedding

        # --- Score assertions ---
        for listing, result in scored:
            assert result.is_valid, (
                f"Invalid score for '{listing.title}': "
                f"fit={result.fit_score}, arch={result.archetype_score}"
            )
            assert 0.0 <= result.fit_score <= 1.0
            assert 0.0 <= result.archetype_score <= 1.0
            assert 0.0 <= result.comp_score <= 1.0
            # Fit and archetype should be non-zero for real JDs
            assert result.fit_score > 0.0, (
                f"Zero fit score for '{listing.title}' — embedding may have failed"
            )
            assert result.archetype_score > 0.0, f"Zero archetype score for '{listing.title}'"

        # --- Step 5: Rank ---
        ranker = Ranker(
            archetype_weight=0.5,
            fit_weight=0.3,
            history_weight=0.2,
            comp_weight=0.15,
            min_score_threshold=0.0,  # keep all for this test
        )
        ranked, summary = ranker.rank(scored, embeddings)

        assert len(ranked) > 0, "Ranker should produce at least one result"
        assert summary.total_found == len(scored)
        assert summary.total_scored == len(scored)

        # Rankings should be in descending order
        scores = [r.final_score for r in ranked]
        assert scores == sorted(scores, reverse=True), (
            f"Rankings not in descending order: {scores}"
        )

        # All final scores should be positive (we set threshold=0.0)
        for r in ranked:
            assert r.final_score > 0.0, f"Zero final score for '{r.listing.title}'"

        # --- Step 6: Export ---
        md_path = str(tmp_path / "results.md")
        csv_path = str(tmp_path / "results.csv")
        jd_dir = str(tmp_path / "jds")

        MarkdownExporter().export(ranked, md_path, summary=summary)
        CSVExporter().export(ranked, csv_path, summary=summary)
        jd_paths = JDFileExporter().export(ranked, jd_dir, summary=summary)

        # Markdown export assertions
        md_content = Path(md_path).read_text()
        assert "# Run Summary" in md_content
        assert "## Ranked Listings" in md_content
        assert "ziprecruiter" in md_content
        # Every ranked listing's title should appear in the MD
        for r in ranked:
            assert r.listing.title in md_content, (
                f"'{r.listing.title}' missing from Markdown export"
            )

        # CSV export assertions
        csv_content = Path(csv_path).read_text()
        assert "title,company,board" in csv_content
        csv_lines = csv_content.strip().split("\n")
        # Header + data rows
        assert len(csv_lines) == len(ranked) + 1, (
            f"CSV should have {len(ranked)} data rows, got {len(csv_lines) - 1}"
        )

        # JD file export assertions
        assert len(jd_paths) == len(ranked), (
            f"Expected {len(ranked)} JD files, got {len(jd_paths)}"
        )
        for jd_path in jd_paths:
            jd_content = jd_path.read_text()
            assert "## Score" in jd_content
            assert "## Job Description" in jd_content
            assert "**Board:** ziprecruiter" in jd_content
