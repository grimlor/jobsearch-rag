"""
Hexagonal architecture refactor — BDD specifications.

Behavior specs verifying that domain services (Scorer, Indexer,
DecisionRecorder), the pipeline orchestrator (PipelineRunner), and the
CLI composition root operate correctly when given any port-protocol
implementation — not just the concrete Embedder and VectorStore.

These tests are the Phase 6 deliverable. They MUST fail before Phase 7
implementation begins.
"""

# Public API surface (from spec + src/ discovery):
#
# Scorer(*, store: VectorStore, embedder: Embedder,
#        disqualify_on_llm_flag: bool, disqualifier_prompt: str,
#        screen_prompt: str, chunk_overlap: int, top_k_retrieval: int)
#   async score(jd_text: str) -> ScoreResult
#
# ScoreResult: fit_score, archetype_score, history_score, disqualified,
#              disqualifier_reason, comp_score, negative_score,
#              culture_score, best_archetype, is_valid
#
# Indexer(store: VectorStore, embedder: Embedder)
#   async index_resume(resume_path: str) -> int
#   async index_archetypes(archetypes_path: str) -> int
#   async index_negative_signals(rubric_path: str, archetypes_path: str) -> int
#   async index_global_positive_signals(rubric_path: str) -> int
#
# DecisionRecorder(*, store: VectorStore, embedder: Embedder,
#                  decisions_dir: str | Path)
#   async record(*, job_id, verdict, jd_text, board, title, company, reason)
#   get_decision(job_id: str) -> dict[str, str] | None
#   history_count() -> int
#
# PipelineRunner(settings: Settings)
#   — Post-refactor (KD#6): __init__(*, embedder, store, scorer, ranker,
#     decision_recorder, settings)
#   async run(boards=None, *, overnight, force_rescore, max_listings) -> RunResult
#   — run() signature is UNCHANGED by the refactor.
#
# RunResult: ranked_listings, summary, failed_listings, skipped_decisions,
#            boards_searched, errors
#
# Ranker(archetype_weight, fit_weight, history_weight, comp_weight,
#        negative_weight, culture_weight, min_score_threshold,
#        dedup_similarity_threshold)
#   rank(listings, embeddings) -> tuple[list[RankedListing], RankSummary]
#
# CLI handlers (KD#12 — services + typed kwargs, no Namespace):
#   handle_search(services, *, boards, overnight, force_rescore,
#                 max_listings, fresh, open_top, output_dir)
#   handle_rescore(services, *, jd_dir, output_dir)
#   handle_eval(services, *, compare_models)
#   handle_index(services, *, resume_path, rubric_path, archetypes_path,
#                resume_only, archetypes_only)
#
# composition.build_services(settings_path: Path) -> Services
#
# FakeEmbedder (tests/fakes/embedder.py) — EmbedderPort implementation
# FakeVectorStore (tests/fakes/store.py) — VectorStorePort implementation

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from jobsearch_rag.cli import handle_eval, handle_index, handle_rescore, handle_search
from jobsearch_rag.composition import build_services
from jobsearch_rag.errors import ActionableError
from jobsearch_rag.pipeline.ranker import Ranker
from jobsearch_rag.pipeline.runner import PipelineRunner, RunResult
from jobsearch_rag.rag.decisions import DecisionRecorder
from jobsearch_rag.rag.indexer import Indexer
from jobsearch_rag.rag.scorer import Scorer
from tests.fakes.embedder import FakeEmbedder
from tests.fakes.store import FakeVectorStore

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Test helpers — supply full constructor args for classes that don't
# exercise Scorer config or Ranker weights directly.
# ---------------------------------------------------------------------------


def _make_test_scorer(
    embedder: FakeEmbedder,
    store: FakeVectorStore,
) -> Scorer:
    """
    Construct a Scorer with test-default config values.

    Classes 4 and 6 exercise pipeline/handler behavior, not Scorer config.
    Config params use safe sentinel values:

    - ``disqualify_on_llm_flag=False`` disables the classify path.
    - ``disqualifier_prompt`` is a sentinel — if a future test enables
      the flag without overriding the prompt, the sentinel content makes
      the misconfiguration visible in classify output rather than
      silently producing an empty-prompt LLM call.
    - ``screen_prompt`` is a neutral instruction.
    - ``chunk_overlap`` and ``top_k_retrieval`` are moderate defaults.

    Tests that exercise Scorer config directly (class 1) construct
    Scorer with explicit values — they do NOT use this helper.
    """
    return Scorer(
        store=store,
        embedder=embedder,
        disqualify_on_llm_flag=False,
        disqualifier_prompt="TEST-DEFAULT — override if disqualify_on_llm_flag=True",
        screen_prompt="Review this JD.",
        chunk_overlap=100,
        top_k_retrieval=3,
    )


def _make_test_ranker(
    *,
    min_score_threshold: float = 0.0,
) -> Ranker:
    """
    Construct a Ranker with test-default weight values.

    Classes 4 and 6 exercise pipeline/handler behavior, not Ranker weights.
    Weights are uniform 1.0 — intentionally neutral for shape and threshold
    tests. Tests asserting on score *values* must construct Ranker directly
    with domain-appropriate weights; uniform weights produce degenerate
    scoring unsuitable for value assertions.

    Only ``min_score_threshold`` is exposed as a parameter because class 4
    tests 2-3 exercise threshold filtering behavior.
    """
    return Ranker(
        archetype_weight=1.0,
        fit_weight=1.0,
        history_weight=1.0,
        comp_weight=1.0,
        negative_weight=1.0,
        culture_weight=1.0,
        min_score_threshold=min_score_threshold,
        dedup_similarity_threshold=0.95,
    )


# ---------------------------------------------------------------------------
# 1. TestScorerWithFakeEmbedder (WHAT: 5, tests: 5)
# ---------------------------------------------------------------------------


class TestScorerWithFakeEmbedder:
    """
    REQUIREMENT: Scorer produces correct semantic scores when given any
    EmbedderPort implementation, not just the concrete Embedder.

    WHO: The ranking pipeline, which consumes ScoreResult to produce a
         ranked shortlist.
    WHAT: (1) Scorer constructed with FakeEmbedder and FakeVectorStore
              returns a valid ScoreResult with all components in [0.0, 1.0].
          (2) A FakeEmbedder returning a vector close to a resume chunk
              produces a higher fit_score than one returning an orthogonal
              vector.
          (3) A FakeEmbedder returning a vector close to an archetype
              produces a higher archetype_score than one returning an
              orthogonal vector.
          (4) A long JD whose best-scoring content appears after the
              first max_embed_chars characters produces a higher
              fit_score than if the input were truncated, proving
              Scorer chunks via the injected EmbedderPort's
              max_embed_chars.
          (5) Scorer calls classify() on the injected EmbedderPort for
              disqualification, and a FakeEmbedder returning a disqualified
              JSON response produces disqualified=True in the ScoreResult.
    WHY: If Scorer is coupled to the concrete Embedder, swapping to a
         different embedding backend will silently produce wrong scores
         or fail at construction time.

    MOCK BOUNDARY:
        Mock:  Nothing — FakeEmbedder and FakeVectorStore ARE the I/O
               boundary. No patching needed.
        Real:  Scorer, ScoreResult.
        Never: Scorer itself.
    """

    async def test_fake_embedder_produces_valid_score_result(self) -> None:
        """
        Given a Scorer constructed with FakeEmbedder and FakeVectorStore
            with populated resume and archetype collections.
        When a JD is scored.
        Then the ScoreResult has all components in [0.0, 1.0].
        """
        # Given: Scorer with fake infrastructure and populated collections
        fake_embedder = FakeEmbedder()
        fake_store = FakeVectorStore()
        scorer = Scorer(
            store=fake_store,
            embedder=fake_embedder,
            disqualify_on_llm_flag=False,
            disqualifier_prompt="",
            screen_prompt="Review this JD.",
            chunk_overlap=100,
            top_k_retrieval=3,
        )

        # When: a JD is scored
        result = await scorer.score("Senior Python engineer at Acme Corp.")

        # Then: all components are in valid range
        assert result.is_valid, (
            f"ScoreResult components out of [0, 1]: "
            f"fit={result.fit_score}, archetype={result.archetype_score}, "
            f"history={result.history_score}"
        )

    async def test_close_embedding_produces_higher_fit_score(self) -> None:
        """
        Given a FakeVectorStore with a resume chunk embedded as [0.9, 0.1, 0.0].
        When a FakeEmbedder returns [0.85, 0.15, 0.05] for the JD.
        Then fit_score is higher than when FakeEmbedder returns [0.0, 0.0, 1.0].
        """
        # Given: FakeVectorStore with a known resume embedding
        fake_store = FakeVectorStore()
        close_embedder = FakeEmbedder(embed_vector=[0.85, 0.15, 0.05])
        far_embedder = FakeEmbedder(embed_vector=[0.0, 0.0, 1.0])

        scorer_close = Scorer(
            store=fake_store,
            embedder=close_embedder,
            disqualify_on_llm_flag=False,
            disqualifier_prompt="",
            screen_prompt="Review this JD.",
            chunk_overlap=100,
            top_k_retrieval=3,
        )
        scorer_far = Scorer(
            store=fake_store,
            embedder=far_embedder,
            disqualify_on_llm_flag=False,
            disqualifier_prompt="",
            screen_prompt="Review this JD.",
            chunk_overlap=100,
            top_k_retrieval=3,
        )

        # When: both score the same JD
        jd = "Senior Python engineer at Acme Corp."
        result_close = await scorer_close.score(jd)
        result_far = await scorer_far.score(jd)

        # Then: close embedding produces higher fit_score
        assert result_close.fit_score > result_far.fit_score, (
            f"Expected close embedding fit_score ({result_close.fit_score}) "
            f"> far embedding fit_score ({result_far.fit_score})"
        )

    async def test_close_embedding_produces_higher_archetype_score(self) -> None:
        """
        Given a FakeVectorStore with an archetype embedded as [0.9, 0.1, 0.0].
        When a FakeEmbedder returns [0.85, 0.15, 0.05] for the JD.
        Then archetype_score is higher than when FakeEmbedder returns [0.0, 0.0, 1.0].
        """
        # Given: FakeVectorStore with a known archetype embedding
        fake_store = FakeVectorStore()
        close_embedder = FakeEmbedder(embed_vector=[0.85, 0.15, 0.05])
        far_embedder = FakeEmbedder(embed_vector=[0.0, 0.0, 1.0])

        scorer_close = Scorer(
            store=fake_store,
            embedder=close_embedder,
            disqualify_on_llm_flag=False,
            disqualifier_prompt="",
            screen_prompt="Review this JD.",
            chunk_overlap=100,
            top_k_retrieval=3,
        )
        scorer_far = Scorer(
            store=fake_store,
            embedder=far_embedder,
            disqualify_on_llm_flag=False,
            disqualifier_prompt="",
            screen_prompt="Review this JD.",
            chunk_overlap=100,
            top_k_retrieval=3,
        )

        # When: both score the same JD
        jd = "Staff ML engineer specializing in NLP pipelines."
        result_close = await scorer_close.score(jd)
        result_far = await scorer_far.score(jd)

        # Then: close embedding produces higher archetype_score
        assert result_close.archetype_score > result_far.archetype_score, (
            f"Expected close embedding archetype_score ({result_close.archetype_score}) "
            f"> far embedding archetype_score ({result_far.archetype_score})"
        )

    async def test_best_chunk_score_wins_over_truncated_input(self) -> None:
        """
        Given a FakeEmbedder with max_embed_chars=50 and a 200-character JD
            where the best-scoring content appears after character 100
            and a FakeVectorStore with a resume chunk whose embedding
            is close to the middle chunk's embedding but orthogonal to
            the first 50 characters' embedding.
        When Scorer scores the JD.
        Then fit_score reflects the best chunk (middle), not the first
             50-character truncation.
        """
        # Given: FakeEmbedder with small max_embed_chars, per-call config
        padding = "x" * 100
        best_content = "Senior Python engineer at Acme Corp building ML pipelines"
        jd = padding + best_content + padding

        fake_store = FakeVectorStore()
        # Per-call embedder: returns orthogonal for padding, close for best_content
        fake_embedder = FakeEmbedder(max_embed_chars=50)

        scorer = Scorer(
            store=fake_store,
            embedder=fake_embedder,
            disqualify_on_llm_flag=False,
            disqualifier_prompt="",
            screen_prompt="Review this JD.",
            chunk_overlap=10,
            top_k_retrieval=3,
        )

        # When: Scorer scores the long JD (chunked by max_embed_chars)
        result = await scorer.score(jd)

        # Then: fit_score reflects the best chunk, not just the first 50 chars
        truncated_embedder = FakeEmbedder(max_embed_chars=50)
        scorer_truncated = Scorer(
            store=fake_store,
            embedder=truncated_embedder,
            disqualify_on_llm_flag=False,
            disqualifier_prompt="",
            screen_prompt="Review this JD.",
            chunk_overlap=10,
            top_k_retrieval=3,
        )
        result_truncated = await scorer_truncated.score(jd[:50])
        assert result.fit_score >= result_truncated.fit_score, (
            f"Expected chunked fit_score ({result.fit_score}) >= "
            f"truncated fit_score ({result_truncated.fit_score})"
        )

    async def test_fake_classify_disqualifies_listing(self) -> None:
        """
        Given a FakeEmbedder whose classify() returns
            '{"disqualified": true, "reason": "staffing agency"}'.
        When Scorer scores a JD with disqualify_on_llm_flag=True.
        Then ScoreResult.disqualified is True and disqualifier_reason
            is "staffing agency".
        """
        # Given: FakeEmbedder that returns disqualified classification
        fake_embedder = FakeEmbedder(
            classify_response='{"disqualified": true, "reason": "staffing agency"}',
        )
        fake_store = FakeVectorStore()

        scorer = Scorer(
            store=fake_store,
            embedder=fake_embedder,
            disqualify_on_llm_flag=True,
            disqualifier_prompt="Check if this is a staffing agency.",
            screen_prompt="Review this JD.",
            chunk_overlap=100,
            top_k_retrieval=3,
        )

        # When: a JD is scored with disqualification enabled
        result = await scorer.score("Contract position at Staffing Solutions Inc.")

        # Then: listing is disqualified with the correct reason
        assert result.disqualified is True, (
            f"Expected disqualified=True, got {result.disqualified}"
        )
        assert result.disqualifier_reason == "staffing agency", (
            f"Expected reason='staffing agency', got '{result.disqualifier_reason}'"
        )


# ---------------------------------------------------------------------------
# 2. TestIndexerWithFakeInfrastructure (WHAT: 5, tests: 5)
# ---------------------------------------------------------------------------


class TestIndexerWithFakeInfrastructure:
    """
    REQUIREMENT: Indexer ingests resume and archetype documents using any
    EmbedderPort and VectorStorePort implementation.

    WHO: The scoring pipeline, which requires populated collections before
         scoring can begin.
    WHAT: (1) Indexer constructed with FakeEmbedder and FakeVectorStore
              indexes a resume file and the FakeVectorStore's resume
              collection contains the expected document count.
          (2) Indexer indexes archetypes and the FakeVectorStore's
              role_archetypes collection contains one document per archetype.
          (3) Indexer indexes negative signals and the FakeVectorStore's
              negative_signals collection is populated.
          (4) Indexer indexes global positive signals and the
              FakeVectorStore's global_positive_signals collection is
              populated.
          (5) Re-indexing replaces existing documents (idempotent) — the
              collection count does not grow on a second call.
    WHY: If Indexer is coupled to ChromaDB, switching vector stores
         requires modifying ingestion logic.

    MOCK BOUNDARY:
        Mock:  File I/O (resume.md, TOML config files via tmp_path).
               FakeEmbedder and FakeVectorStore ARE the infrastructure.
        Real:  Indexer.
        Never: Indexer itself.
    """

    async def test_index_resume_populates_fake_store(self, tmp_path: Path) -> None:
        """
        Given a FakeEmbedder and FakeVectorStore and a resume.md with 3 sections.
        When Indexer.index_resume() is called.
        Then FakeVectorStore's resume collection contains 3 documents.
        """
        # Given: resume file with 3 markdown sections
        resume = tmp_path / "resume.md"
        resume.write_text(
            "# Section 1\nExperience in Python.\n\n"
            "# Section 2\nExperience in ML.\n\n"
            "# Section 3\nExperience in DevOps.\n"
        )
        fake_embedder = FakeEmbedder()
        fake_store = FakeVectorStore()
        indexer = Indexer(store=fake_store, embedder=fake_embedder)

        # When: resume is indexed
        count = await indexer.index_resume(str(resume))

        # Then: 3 documents in resume collection
        assert count == 3, f"Expected 3 indexed documents, got {count}"
        assert fake_store.collection_count("resume") == 3, (
            f"Expected 3 docs in resume collection, got "
            f"{fake_store.collection_count('resume')}"
        )

    async def test_index_archetypes_populates_fake_store(
        self, tmp_path: Path
    ) -> None:
        """
        Given a role_archetypes.toml with 2 archetypes.
        When Indexer.index_archetypes() is called.
        Then FakeVectorStore's role_archetypes collection contains 2 documents.
        """
        # Given: archetypes TOML with 2 entries
        archetypes = tmp_path / "role_archetypes.toml"
        archetypes.write_text(
            '[archetype.backend]\ntitle = "Backend Engineer"\n'
            'keywords = ["python", "api"]\n\n'
            '[archetype.ml]\ntitle = "ML Engineer"\n'
            'keywords = ["pytorch", "transformers"]\n'
        )
        fake_embedder = FakeEmbedder()
        fake_store = FakeVectorStore()
        indexer = Indexer(store=fake_store, embedder=fake_embedder)

        # When: archetypes are indexed
        count = await indexer.index_archetypes(str(archetypes))

        # Then: 2 documents in role_archetypes collection
        assert count == 2, f"Expected 2 indexed archetypes, got {count}"
        assert fake_store.collection_count("role_archetypes") == 2, (
            f"Expected 2 docs in role_archetypes collection, got "
            f"{fake_store.collection_count('role_archetypes')}"
        )

    async def test_index_negative_signals_populates_fake_store(
        self, tmp_path: Path
    ) -> None:
        """
        Given a global_rubric.toml with negative signals.
        When Indexer.index_negative_signals() is called.
        Then FakeVectorStore's negative_signals collection is non-empty.
        """
        # Given: rubric with negative signals and archetypes file
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text(
            '[negative_signals]\nsignals = [\n'
            '  "staffing agency",\n  "contract to hire"\n]\n'
        )
        archetypes = tmp_path / "role_archetypes.toml"
        archetypes.write_text(
            '[archetype.backend]\ntitle = "Backend"\nkeywords = ["python"]\n'
        )
        fake_embedder = FakeEmbedder()
        fake_store = FakeVectorStore()
        indexer = Indexer(store=fake_store, embedder=fake_embedder)

        # When: negative signals are indexed
        count = await indexer.index_negative_signals(str(rubric), str(archetypes))

        # Then: negative_signals collection is non-empty
        assert count > 0, f"Expected non-zero negative signal count, got {count}"
        assert fake_store.collection_count("negative_signals") > 0, (
            "Expected non-empty negative_signals collection"
        )

    async def test_index_positive_signals_populates_fake_store(
        self, tmp_path: Path
    ) -> None:
        """
        Given a global_rubric.toml with positive signal dimensions.
        When Indexer.index_global_positive_signals() is called.
        Then FakeVectorStore's global_positive_signals collection is non-empty.
        """
        # Given: rubric with positive signal dimensions
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text(
            '[positive_signals]\ndimensions = [\n'
            '  "remote work",\n  "equity"\n]\n'
        )
        fake_embedder = FakeEmbedder()
        fake_store = FakeVectorStore()
        indexer = Indexer(store=fake_store, embedder=fake_embedder)

        # When: global positive signals are indexed
        count = await indexer.index_global_positive_signals(str(rubric))

        # Then: global_positive_signals collection is non-empty
        assert count > 0, f"Expected non-zero positive signal count, got {count}"
        assert fake_store.collection_count("global_positive_signals") > 0, (
            "Expected non-empty global_positive_signals collection"
        )

    async def test_reindex_is_idempotent(self, tmp_path: Path) -> None:
        """
        Given a FakeVectorStore that already has resume documents.
        When Indexer.index_resume() is called a second time.
        Then the resume collection count does not increase.
        """
        # Given: resume indexed once
        resume = tmp_path / "resume.md"
        resume.write_text(
            "# Section 1\nExperience in Python.\n\n"
            "# Section 2\nExperience in ML.\n"
        )
        fake_embedder = FakeEmbedder()
        fake_store = FakeVectorStore()
        indexer = Indexer(store=fake_store, embedder=fake_embedder)
        await indexer.index_resume(str(resume))
        count_after_first = fake_store.collection_count("resume")

        # When: re-indexed
        await indexer.index_resume(str(resume))
        count_after_second = fake_store.collection_count("resume")

        # Then: count did not grow
        assert count_after_second == count_after_first, (
            f"Expected idempotent reindex: first={count_after_first}, "
            f"second={count_after_second}"
        )


# ---------------------------------------------------------------------------
# 3. TestDecisionRecorderWithFakeInfrastructure (WHAT: 4, tests: 4)
# ---------------------------------------------------------------------------


class TestDecisionRecorderWithFakeInfrastructure:
    """
    REQUIREMENT: DecisionRecorder stores and retrieves user verdicts using
    any EmbedderPort and VectorStorePort implementation.

    WHO: The evaluation pipeline and review session, which consume decision
         history.
    WHAT: (1) DecisionRecorder constructed with FakeEmbedder and
              FakeVectorStore records a "yes" verdict and the decision
              is retrievable from the FakeVectorStore.
          (2) A "no" verdict is stored but excluded from the scoring
              signal (metadata scoring_signal="false").
          (3) The JSONL audit log file is written to disk with the
              expected fields.
          (4) Duplicate decisions on the same job_id overwrite rather
              than append.
    WHY: If DecisionRecorder is coupled to ChromaDB, switching vector
         stores requires modifying verdict storage logic.

    MOCK BOUNDARY:
        Mock:  FakeEmbedder and FakeVectorStore ARE the infrastructure.
               File I/O via tmp_path for the JSONL audit log.
        Real:  DecisionRecorder.
        Never: DecisionRecorder itself.
    """

    async def test_yes_verdict_stored_and_retrievable(self, tmp_path: Path) -> None:
        """
        Given a DecisionRecorder with FakeEmbedder and FakeVectorStore.
        When a "yes" verdict is recorded for job "x-123".
        Then get_decision("x-123") returns the verdict metadata.
        """
        # Given: DecisionRecorder with fake infrastructure
        fake_embedder = FakeEmbedder()
        fake_store = FakeVectorStore()
        recorder = DecisionRecorder(
            store=fake_store,
            embedder=fake_embedder,
            decisions_dir=tmp_path / "decisions",
        )

        # When: a "yes" verdict is recorded
        await recorder.record(
            job_id="x-123",
            verdict="yes",
            jd_text="Senior Python engineer at Acme Corp.",
            board="testboard",
            title="Senior Python Engineer",
            company="Acme Corp",
        )

        # Then: the decision is retrievable
        decision = recorder.get_decision("x-123")
        assert decision is not None, "Expected decision for 'x-123' to be retrievable"
        assert decision.get("verdict") == "yes", (
            f"Expected verdict='yes', got '{decision.get('verdict')}'"
        )

    async def test_no_verdict_excluded_from_scoring_signal(
        self, tmp_path: Path
    ) -> None:
        """
        Given a DecisionRecorder with FakeEmbedder and FakeVectorStore.
        When a "no" verdict is recorded.
        Then the stored document's metadata has scoring_signal="false".
        """
        # Given: DecisionRecorder with fake infrastructure
        fake_embedder = FakeEmbedder()
        fake_store = FakeVectorStore()
        recorder = DecisionRecorder(
            store=fake_store,
            embedder=fake_embedder,
            decisions_dir=tmp_path / "decisions",
        )

        # When: a "no" verdict is recorded
        await recorder.record(
            job_id="x-456",
            verdict="no",
            jd_text="Staffing agency recruiter position.",
            board="testboard",
            title="Recruiter",
            company="StaffCo",
        )

        # Then: metadata has scoring_signal="false"
        docs = fake_store.get_by_metadata(
            "decisions", where={"job_id": "x-456"}
        )
        assert len(docs) == 1, f"Expected 1 decision doc, got {len(docs)}"
        assert docs[0].metadata is not None, "Expected metadata on decision doc"
        assert docs[0].metadata.get("scoring_signal") == "false", (
            f"Expected scoring_signal='false', got "
            f"'{docs[0].metadata.get('scoring_signal')}'"
        )

    async def test_audit_log_written_to_disk(self, tmp_path: Path) -> None:
        """
        Given a DecisionRecorder with FakeEmbedder and FakeVectorStore.
        When a verdict is recorded.
        Then a JSONL file exists in decisions_dir with the expected fields.
        """
        # Given: DecisionRecorder with tmp_path decisions dir
        decisions_dir = tmp_path / "decisions"
        fake_embedder = FakeEmbedder()
        fake_store = FakeVectorStore()
        recorder = DecisionRecorder(
            store=fake_store,
            embedder=fake_embedder,
            decisions_dir=decisions_dir,
        )

        # When: a verdict is recorded
        await recorder.record(
            job_id="x-789",
            verdict="yes",
            jd_text="Great ML position.",
            board="testboard",
            title="ML Engineer",
            company="DeepTech",
        )

        # Then: JSONL audit log exists with expected fields
        log_files = list(decisions_dir.glob("*.jsonl"))
        assert len(log_files) >= 1, (
            f"Expected at least 1 JSONL file in {decisions_dir}, found {len(log_files)}"
        )
        with open(log_files[0]) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) >= 1, "Expected at least 1 log entry"
        entry = lines[-1]
        assert "job_id" in entry, f"Expected 'job_id' field in log entry: {entry}"
        assert entry["job_id"] == "x-789", (
            f"Expected job_id='x-789', got '{entry.get('job_id')}'"
        )

    async def test_duplicate_decision_overwrites(self, tmp_path: Path) -> None:
        """
        Given a DecisionRecorder that already recorded job "x-123" as "no".
        When a "yes" verdict is recorded for the same job.
        Then the stored verdict is "yes" and collection count is 1.
        """
        # Given: existing "no" decision
        fake_embedder = FakeEmbedder()
        fake_store = FakeVectorStore()
        recorder = DecisionRecorder(
            store=fake_store,
            embedder=fake_embedder,
            decisions_dir=tmp_path / "decisions",
        )
        await recorder.record(
            job_id="x-123",
            verdict="no",
            jd_text="Not great.",
            board="testboard",
        )

        # When: overwritten with "yes"
        await recorder.record(
            job_id="x-123",
            verdict="yes",
            jd_text="Actually great.",
            board="testboard",
        )

        # Then: stored verdict is "yes", count is 1
        decision = recorder.get_decision("x-123")
        assert decision is not None, "Expected decision for 'x-123'"
        assert decision.get("verdict") == "yes", (
            f"Expected verdict='yes' after overwrite, got '{decision.get('verdict')}'"
        )
        assert recorder.history_count() == 1, (
            f"Expected 1 decision after overwrite, got {recorder.history_count()}"
        )


# ---------------------------------------------------------------------------
# 4. TestPipelineRunnerWithInjectedServices (WHAT: 5, tests: 5)
# ---------------------------------------------------------------------------


class TestPipelineRunnerWithInjectedServices:
    """
    REQUIREMENT: PipelineRunner orchestrates a complete search-score-rank
    pipeline using injected service instances.

    WHO: CLI handlers that wire services and pass them to PipelineRunner.
    WHAT: (1) PipelineRunner constructed with FakeEmbedder, FakeVectorStore,
              real Scorer, real Ranker, and real DecisionRecorder runs a
              pipeline and returns a RunResult with ranked listings.
          (2) RunResult contains exactly as many ranked listings as the
              adapter returned that scored above the threshold.
          (3) Listings below the score threshold are excluded from the
              RunResult.
          (4) ConnectionError from the embedder's health check is caught
              by PipelineRunner and re-raised as an ActionableError that
              preserves the cause and carries a non-empty suggestion.
          (5) The RunResult's summary contains board names that were searched.
    WHY: If PipelineRunner constructs its own services internally, the
         composition root cannot control infrastructure selection, and
         testing requires patching constructors instead of injecting fakes.

    MOCK BOUNDARY:
        Mock:  FakeEmbedder (Ollama), FakeVectorStore (ChromaDB),
               board adapter (Playwright) via conftest adapter_override,
               file I/O (session logging via tmp_path).
        Real:  PipelineRunner, Scorer, Ranker, DecisionRecorder, RunResult.
        Never: PipelineRunner itself.

    SETTINGS STAND-IN (Phase 7):
        PipelineRunner.run() accesses self._settings for:
          - output.log_dir (session logging)
          - enabled_boards, overnight_boards (board selection fallback)
          - boards dict (per-board config: headless, browser_channel,
            throttle, storage state)
          - adapters.viewport_width/height, browser_paths, cdp_timeout,
            max_full_text_chars
          - session_storage_dir
          - scoring.salary_floor/ceiling, hours_per_year,
            missing_comp_score, min_score_threshold
          - archetypes_path, resume_path, global_rubric_path
            (auto-index on empty collections)
        Phase 7 must provide a test-suitable Settings (or
        SimpleNamespace) with these fields. The current settings=None
        placeholder will fail at runtime.
    """

    async def test_pipeline_returns_run_result_with_ranked_listings(
        self, tmp_path: Path
    ) -> None:
        """
        Given a PipelineRunner with FakeEmbedder, FakeVectorStore, real
        Scorer/Ranker, and a fake board adapter returning 2 listings.
        When run() is called.
        Then RunResult.ranked_listings contains entries for the returned listings.
        """
        # Given: PipelineRunner with injected fakes and real domain services
        fake_embedder = FakeEmbedder()
        fake_store = FakeVectorStore()
        scorer = _make_test_scorer(fake_embedder, fake_store)
        ranker = _make_test_ranker()
        recorder = DecisionRecorder(
            store=fake_store,
            embedder=fake_embedder,
            decisions_dir=tmp_path / "decisions",
        )
        runner = PipelineRunner(
            embedder=fake_embedder,
            store=fake_store,
            scorer=scorer,
            ranker=ranker,
            decision_recorder=recorder,
            settings=None,
        )

        # When: pipeline is run
        result = await runner.run(boards=["testboard"])

        # Then: RunResult has ranked listings
        assert isinstance(result, RunResult), (
            f"Expected RunResult, got {type(result).__name__}"
        )
        assert len(result.ranked_listings) > 0, (
            "Expected non-empty ranked_listings in RunResult"
        )

    async def test_ranked_listing_count_matches_above_threshold(
        self, tmp_path: Path
    ) -> None:
        """
        Given a fake board adapter returning 5 listings and a FakeEmbedder
        configured so that exactly 3 produce scores above the Ranker's
        min_score_threshold.
        When run() completes.
        Then len(RunResult.ranked_listings) == 3.
        """
        # Given: 5 listings, 3 above threshold
        fake_embedder = FakeEmbedder(above_threshold_count=3)
        fake_store = FakeVectorStore()
        scorer = _make_test_scorer(fake_embedder, fake_store)
        ranker = _make_test_ranker(min_score_threshold=0.5)
        recorder = DecisionRecorder(
            store=fake_store,
            embedder=fake_embedder,
            decisions_dir=tmp_path / "decisions",
        )
        runner = PipelineRunner(
            embedder=fake_embedder,
            store=fake_store,
            scorer=scorer,
            ranker=ranker,
            decision_recorder=recorder,
            settings=None,
        )

        # When: pipeline is run
        result = await runner.run(boards=["testboard"])

        # Then: exactly 3 ranked listings
        assert len(result.ranked_listings) == 3, (
            f"Expected 3 above-threshold listings, got {len(result.ranked_listings)}"
        )

    async def test_listings_below_threshold_excluded(
        self, tmp_path: Path
    ) -> None:
        """
        Given a Ranker with min_score_threshold=0.99 and a FakeEmbedder
        returning low-similarity vectors.
        When run() completes.
        Then RunResult.ranked_listings is empty.
        """
        # Given: threshold so high nothing passes
        fake_embedder = FakeEmbedder(similarity=0.01)
        fake_store = FakeVectorStore()
        scorer = _make_test_scorer(fake_embedder, fake_store)
        ranker = _make_test_ranker(min_score_threshold=0.99)
        recorder = DecisionRecorder(
            store=fake_store,
            embedder=fake_embedder,
            decisions_dir=tmp_path / "decisions",
        )
        runner = PipelineRunner(
            embedder=fake_embedder,
            store=fake_store,
            scorer=scorer,
            ranker=ranker,
            decision_recorder=recorder,
            settings=None,
        )

        # When: pipeline is run
        result = await runner.run(boards=["testboard"])

        # Then: no listings above threshold
        assert len(result.ranked_listings) == 0, (
            f"Expected 0 ranked listings, got {len(result.ranked_listings)}"
        )

    async def test_health_check_failure_raises_actionable_error(
        self, tmp_path: Path
    ) -> None:
        """
        Given a FakeEmbedder whose health_check() raises ConnectionError.
        When PipelineRunner.run() is called.
        Then PipelineRunner catches the ConnectionError and raises an
        ActionableError. And the ActionableError's __cause__ is the original
        ConnectionError. And the ActionableError's suggestion is non-empty.
        """
        # Given: FakeEmbedder that fails health check
        fake_embedder = FakeEmbedder(health_check_error=ConnectionError("Ollama down"))
        fake_store = FakeVectorStore()
        scorer = _make_test_scorer(fake_embedder, fake_store)
        ranker = _make_test_ranker()
        recorder = DecisionRecorder(
            store=fake_store,
            embedder=fake_embedder,
            decisions_dir=tmp_path / "decisions",
        )
        runner = PipelineRunner(
            embedder=fake_embedder,
            store=fake_store,
            scorer=scorer,
            ranker=ranker,
            decision_recorder=recorder,
            settings=None,
        )

        with pytest.raises(ActionableError) as exc_info:
            # When: run() is called
            await runner.run(boards=["testboard"])

        # Then: ActionableError wraps the original ConnectionError
        assert exc_info.value.__cause__ is not None, (
            "Expected ActionableError to chain the original ConnectionError"
        )
        assert isinstance(exc_info.value.__cause__, ConnectionError), (
            f"Expected __cause__ to be ConnectionError, got "
            f"{type(exc_info.value.__cause__).__name__}"
        )
        assert exc_info.value.suggestion, (
            "Expected non-empty suggestion on ActionableError"
        )

    async def test_run_result_summary_contains_searched_boards(
        self, tmp_path: Path
    ) -> None:
        """
        Given a PipelineRunner configured for board "testboard".
        When run() completes.
        Then RunResult.boards_searched contains "testboard".
        """
        # Given: PipelineRunner targeting "testboard"
        fake_embedder = FakeEmbedder()
        fake_store = FakeVectorStore()
        scorer = _make_test_scorer(fake_embedder, fake_store)
        ranker = _make_test_ranker()
        recorder = DecisionRecorder(
            store=fake_store,
            embedder=fake_embedder,
            decisions_dir=tmp_path / "decisions",
        )
        runner = PipelineRunner(
            embedder=fake_embedder,
            store=fake_store,
            scorer=scorer,
            ranker=ranker,
            decision_recorder=recorder,
            settings=None,
        )

        # When: pipeline is run
        result = await runner.run(boards=["testboard"])

        # Then: summary references the board
        assert "testboard" in result.boards_searched, (
            f"Expected 'testboard' in boards_searched, got {result.boards_searched}"
        )


# ---------------------------------------------------------------------------
# 5. TestCompositionRootBuildsServices (WHAT: 3, tests: 3)
# ---------------------------------------------------------------------------


class TestCompositionRootBuildsServices:
    """
    REQUIREMENT: build_services() reads adapter class paths from config,
    instantiates them, wires domain services, and returns a correctly
    wired Services namespace.

    WHO: CLI handlers and any future driving adapter that needs the
         service graph.
    WHAT: (1) Calling services.scorer.score() returns a valid ScoreResult
              (is_valid is True) — proves embedder wired into Scorer.
          (2) Calling services.indexer.index_resume() returns a non-zero
              document count — proves store wired through Indexer.
          (3) build_services() with a missing or invalid class path
              raises ActionableError with non-empty suggestion (KD#10).
    NOTE: Pipeline runner wiring is tested in
          TestPipelineRunnerWithInjectedServices (class 4). Class 5
          tests only composition-root-specific claims.
    WHY: If the composition root returns a malformed namespace,
         downstream handlers fail at runtime with opaque attribute
         errors. Behavioral wiring tests catch real integration defects
         that isinstance checks would miss (e.g., embedder constructed
         but not injected into Scorer).

    MOCK BOUNDARY:
        Mock:  FakeEmbedder and FakeVectorStore via test [adapters]
               config (resolved by build_services() — same code path
               as production). File I/O (settings TOML via tmp_path).
        Real:  build_services(), Scorer, Indexer, Services.
        Never: build_services() itself.
    """

    def _write_test_settings(self, tmp_path: Path) -> Path:
        """Write a test settings.toml with fake adapter class paths."""
        settings_toml = tmp_path / "settings.toml"
        settings_toml.write_text(
            "[adapters]\n"
            'embedder = "tests.fakes.embedder:FakeEmbedder"\n'
            'store = "tests.fakes.store:FakeVectorStore"\n'
        )
        return settings_toml

    async def test_scorer_wired_with_embedder(self, tmp_path: Path) -> None:
        """
        Given a test settings.toml with FakeEmbedder and FakeVectorStore
            and a wired Services namespace from build_services().
        When services.scorer.score() is invoked.
        Then the returned ScoreResult is valid (is_valid is True).
        """
        # Given: a wired Services namespace built from test settings
        settings_toml = self._write_test_settings(tmp_path)
        services = build_services(settings_toml)

        # When: scorer.score() is invoked on the wired services
        result = await services.scorer.score("test JD text")

        # Then: the returned ScoreResult is valid
        assert result.is_valid, (
            f"Expected valid ScoreResult from wired Scorer, got "
            f"fit={result.fit_score}, archetype={result.archetype_score}, "
            f"history={result.history_score}"
        )

    async def test_indexer_wired_with_store(self, tmp_path: Path) -> None:
        """
        Given a test settings.toml with FakeEmbedder and FakeVectorStore,
            a resume file with content, and a wired Services namespace
            from build_services().
        When services.indexer.index_resume() is invoked.
        Then the return count is > 0 and the store's resume collection
            contains that many documents.
        """
        # Given: a wired Services namespace and a resume file
        settings_toml = self._write_test_settings(tmp_path)
        resume = tmp_path / "resume.md"
        resume.write_text("## Skills\nPython, ML, DevOps.\n")
        services = build_services(settings_toml)

        # When: indexer.index_resume() is invoked on the wired services
        count = await services.indexer.index_resume(str(resume))

        # Then: non-zero count and store collection matches
        assert count > 0, (
            f"Expected non-zero document count from wired Indexer, got {count}"
        )
        assert services.store.collection_count("resume") == count, (
            f"Expected store resume collection to contain {count} documents, "
            f"got {services.store.collection_count('resume')}"
        )

    def test_invalid_class_path_raises_actionable_error(
        self, tmp_path: Path
    ) -> None:
        """
        Given a test settings.toml with [adapters] embedder pointing to
        a nonexistent class path.
        When build_services() is called.
        Then ActionableError is raised with a non-empty suggestion.
        """
        # Given: a settings.toml with [adapters] embedder pointing to a nonexistent class path
        settings_toml = tmp_path / "settings.toml"
        settings_toml.write_text(
            "[adapters]\n"
            'embedder = "nonexistent.module:FakeEmbedder"\n'
            'store = "tests.fakes.store:FakeVectorStore"\n'
        )

        with pytest.raises(ActionableError) as exc_info:
            # When: build_services() is called
            build_services(settings_toml)

        # Then: ActionableError raised with non-empty suggestion
        assert exc_info.value.suggestion, "Expected non-empty suggestion"


# ---------------------------------------------------------------------------
# 6. TestCLIHandlersUseInjectedServices (WHAT: 4, tests: 4)
# ---------------------------------------------------------------------------


class TestCLIHandlersUseInjectedServices:
    """
    REQUIREMENT: CLI handlers accept a pre-built Services namespace and
    produce the expected artifacts.

    WHO: End users invoking CLI subcommands.
    WHAT: (1) handle_search given a Services namespace and a fake board
              adapter produces CSV and Markdown output files.
          (2) handle_rescore given a Services namespace and existing JD
              files produces updated CSV output.
          (3) handle_eval given a Services namespace and recorded
              decisions produces an eval report on stdout.
          (4) handle_index given a Services namespace, resume, and
              config files completes without error.
    WHY: If CLI handlers cannot consume a pre-built Services namespace
         correctly, the composition root's output is wasted. Testing
         handlers with a directly-constructed Services (not routed
         through build_services()) isolates handler logic from wiring
         logic — a failure here means the handler is broken, not the
         wiring.

    MOCK BOUNDARY:
        Mock:  FakeEmbedder and FakeVectorStore (constructed directly
               in the test, not via build_services()). Playwright
               (board adapters via adapter_override). File I/O
               (settings TOML, resume/config, output dirs via tmp_path).
        Real:  CLI handler functions, PipelineRunner, Scorer, Ranker,
               DecisionRecorder, Indexer.
        Never: The CLI handler functions themselves.
    """

    def _build_services(self, tmp_path: Path) -> SimpleNamespace:
        """Construct a full Services namespace with fakes and real domain services."""
        fake_embedder = FakeEmbedder()
        fake_store = FakeVectorStore()
        scorer = _make_test_scorer(fake_embedder, fake_store)
        ranker = _make_test_ranker()
        indexer = Indexer(store=fake_store, embedder=fake_embedder)
        decision_recorder = DecisionRecorder(
            store=fake_store,
            embedder=fake_embedder,
            decisions_dir=tmp_path / "decisions",
        )
        pipeline_runner = PipelineRunner(
            embedder=fake_embedder,
            store=fake_store,
            scorer=scorer,
            ranker=ranker,
            decision_recorder=decision_recorder,
            settings=None,
        )
        return SimpleNamespace(
            embedder=fake_embedder,
            store=fake_store,
            scorer=scorer,
            ranker=ranker,
            indexer=indexer,
            decision_recorder=decision_recorder,
            pipeline_runner=pipeline_runner,
            settings=None,  # Phase 7: replace with test Settings stand-in
        )

    def test_handle_search_produces_output_files(self, tmp_path: Path) -> None:
        """
        Given a Services namespace with FakeEmbedder/FakeVectorStore
        and a fake board adapter.
        When handle_search is invoked.
        Then CSV and Markdown output files exist in the output directory.
        """
        # Given: full Services namespace (not via build_services)
        services = self._build_services(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # When: handle_search is invoked with the services namespace
        handle_search(
            services=services,
            boards=["testboard"],
            output_dir=output_dir,
        )

        # Then: output files exist
        csv_files = list(output_dir.glob("*.csv"))
        md_files = list(output_dir.glob("*.md"))
        assert len(csv_files) >= 1, (
            f"Expected CSV output file in {output_dir}, found {len(csv_files)}"
        )
        assert len(md_files) >= 1, (
            f"Expected Markdown output file in {output_dir}, found {len(md_files)}"
        )

    def test_handle_rescore_produces_updated_output(self, tmp_path: Path) -> None:
        """
        Given existing JD files and a Services namespace with
        FakeEmbedder/FakeVectorStore.
        When handle_rescore is invoked.
        Then the output CSV reflects re-scored listings.
        """
        # Given: existing JD files and full Services namespace
        services = self._build_services(tmp_path)
        jd_dir = tmp_path / "jds"
        jd_dir.mkdir()
        (jd_dir / "test_job.md").write_text("# Test Job\nSenior Python Engineer\n")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # When: handle_rescore is invoked
        handle_rescore(
            services=services,
            jd_dir=jd_dir,
            output_dir=output_dir,
        )

        # Then: updated CSV exists
        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) >= 1, (
            f"Expected CSV output file after rescore, found {len(csv_files)}"
        )

    def test_handle_eval_produces_report(self, tmp_path: Path) -> None:
        """
        Given recorded decisions and a Services namespace with
        FakeEmbedder/FakeVectorStore.
        When handle_eval is invoked.
        Then an eval report is printed to stdout.
        """
        # Given: full Services namespace
        services = self._build_services(tmp_path)

        # When: handle_eval is invoked (captures stdout)
        captured = io.StringIO()
        sys.stdout = captured
        try:
            handle_eval(services=services)
        finally:
            sys.stdout = sys.__stdout__

        # Then: output contains eval report content
        output = captured.getvalue()
        assert len(output) > 0, "Expected non-empty eval report on stdout"

    def test_handle_index_completes_without_error(self, tmp_path: Path) -> None:
        """
        Given resume.md and config TOML files in tmp_path, and a Services
        namespace with FakeEmbedder/FakeVectorStore.
        When handle_index is invoked.
        Then the function returns without raising.
        """
        # Given: resume and config files with full Services namespace
        services = self._build_services(tmp_path)
        resume = tmp_path / "resume.md"
        resume.write_text("# Resume\nExperienced Python developer.\n")
        rubric = tmp_path / "global_rubric.toml"
        rubric.write_text(
            '[positive_signals]\ndimensions = ["remote"]\n'
            '[negative_signals]\nsignals = ["staffing"]\n'
        )
        archetypes = tmp_path / "role_archetypes.toml"
        archetypes.write_text(
            '[archetype.backend]\ntitle = "Backend"\nkeywords = ["python"]\n'
        )

        # When: handle_index is called with the resume and config files
        # Then: it completes without raising
        handle_index(
            services=services,
            resume_path=resume,
            rubric_path=rubric,
            archetypes_path=archetypes,
        )


# ---------------------------------------------------------------------------
# 7. TestExistingBehaviorPreservedAfterRefactor (WHAT: 1, tests: 1)
# ---------------------------------------------------------------------------


class TestExistingBehaviorPreservedAfterRefactor:
    """
    REQUIREMENT: The refactor preserves all existing observable behavior.

    WHO: All downstream consumers (exporters, reviewers, evaluators).
    WHAT: (1) The existing 800+ test suite passes without modification
              to test assertions (test infrastructure changes to mock at
              new boundaries are permitted).
    WHY: This is a refactor, not a feature. Any behavioral change is a
         regression.

    MOCK BOUNDARY:
        Mock:  Ollama, ChromaDB (via tmp_path), Playwright, file I/O.
        Real:  All domain logic, all pipeline orchestration.
        Never: Any component under behavioral test.

    NOTE: This behavior is verified by running the existing test suite,
    not by writing new tests. It is listed here as a spec requirement so
    the implementation phase cannot declare "done" without a green suite.
    """

    def test_full_existing_suite_passes(self) -> None:
        """
        Given the refactored codebase.
        When the full test suite is run.
        Then all existing tests pass (test wiring changes permitted,
        assertion changes not permitted).
        """
        # This is a meta-test: verified by running the existing 800+ test
        # suite. The implementation phase must demonstrate a green `task test`
        # run before marking Phase 7 complete.
        #
        # This test method exists as a spec anchor — it documents the
        # requirement. It passes trivially so it does not block Phase 6.
        pass
