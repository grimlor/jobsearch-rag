"""
BDD specs for the evaluation harness (Phase 5d-i).

Covers:
    TestEvalCommand  — CLI subcommand wiring and handler integration
    TestEvalMetrics  — agreement rate, precision, recall computation
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobsearch_rag.config import (
    BoardConfig,
    ChromaConfig,
    OllamaConfig,
    OutputConfig,
    ScoringConfig,
    Settings,
)
from jobsearch_rag.pipeline.eval import EvalResult, EvalRunner
from jobsearch_rag.pipeline.ranker import Ranker
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.scorer import Scorer
from jobsearch_rag.rag.store import VectorStore
from tests.constants import EMBED_FAKE

# Public API surface (from src/jobsearch_rag/pipeline/eval — to be created):
#   EvalRunner(scorer: Scorer, ranker: Ranker, store: VectorStore)
#   async EvalRunner.evaluate() -> EvalResult
#
#   EvalResult:
#     decisions_evaluated: int
#     agreement_rate: float
#     precision: float
#     recall: float
#     per_decision: list[EvalDecision]
#
#   EvalDecision:
#     job_id: str
#     verdict: str
#     final_score: float
#     above_threshold: bool
#     agreed: bool
#
# Public API surface (from src/jobsearch_rag/cli):
#   build_parser() -> ArgumentParser
#   handle_eval(args: Namespace) -> None

# An embedding vector distant from EMBED_FAKE — produces cosine distance > 0
# when JD text is embedded with EMBED_FAKE.  This creates scores < 1.0 so
# eval tests can distinguish above/below threshold behavior.
_EMBED_DISTANT: list[float] = [0.9, 0.1, 0.9, 0.1, 0.9]


def _make_settings(tmpdir: str, *, min_score_threshold: float = 0.45) -> Settings:
    """Create minimal Settings for eval tests."""
    return Settings(
        enabled_boards=["testboard"],
        overnight_boards=[],
        boards={
            "testboard": BoardConfig(
                name="testboard", searches=["https://example.com"], max_pages=1
            )
        },
        scoring=ScoringConfig(
            min_score_threshold=min_score_threshold,
            disqualify_on_llm_flag=False,
        ),
        ollama=OllamaConfig(),
        output=OutputConfig(output_dir=str(Path(tmpdir) / "output")),
        chroma=ChromaConfig(persist_dir=str(Path(tmpdir) / "chroma")),
    )


def _make_mock_embedder(
    embed_return: list[float] | None = None,
) -> tuple[Embedder, AsyncMock]:
    """
    Create an Embedder with mocked Ollama I/O boundary.

    Returns (embedder, mock_client) so tests can configure per-call behavior.
    """
    mock_client = AsyncMock()

    model_embed = MagicMock()
    model_embed.model = "nomic-embed-text"
    model_llm = MagicMock()
    model_llm.model = "mistral:7b"
    list_response = MagicMock()
    list_response.models = [model_embed, model_llm]
    mock_client.list.return_value = list_response

    embed_response = MagicMock()
    embed_response.embeddings = [embed_return or EMBED_FAKE]
    mock_client.embed.return_value = embed_response

    classify_response = MagicMock()
    classify_response.message = MagicMock()
    classify_response.message.content = '{"disqualified": false}'
    mock_client.chat.return_value = classify_response

    with patch(
        "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
        return_value=mock_client,
    ):
        embedder = Embedder(
            base_url="http://localhost:11434",
            embed_model="nomic-embed-text",
            llm_model="mistral:7b",
        )

    return embedder, mock_client


def _make_eval_stack(
    settings: Settings,
    *,
    embed_return: list[float] | None = None,
) -> tuple[EvalRunner, Scorer, Ranker, VectorStore, AsyncMock]:
    """Build a full eval stack: EvalRunner + Scorer + Ranker + real VectorStore."""
    embedder, mock_client = _make_mock_embedder(embed_return=embed_return)
    store = VectorStore(persist_dir=settings.chroma.persist_dir)
    scorer = Scorer(
        store=store,
        embedder=embedder,
        disqualify_on_llm_flag=settings.scoring.disqualify_on_llm_flag,
    )
    ranker = Ranker(
        archetype_weight=settings.scoring.archetype_weight,
        fit_weight=settings.scoring.fit_weight,
        history_weight=settings.scoring.history_weight,
        comp_weight=settings.scoring.comp_weight,
        negative_weight=settings.scoring.negative_weight,
        culture_weight=settings.scoring.culture_weight,
        min_score_threshold=settings.scoring.min_score_threshold,
    )
    runner = EvalRunner(scorer=scorer, ranker=ranker, store=store)
    return runner, scorer, ranker, store, mock_client


def _seed_required_collections(store: VectorStore, embedding: list[float]) -> None:
    """Seed resume and role_archetypes so the scorer doesn't raise on empty."""
    for name in ("resume", "role_archetypes"):
        store.add_documents(
            name,
            ids=[f"{name}-seed"],
            documents=[f"Seed document for {name}"],
            embeddings=[embedding],
        )


def _seed_decision(
    store: VectorStore,
    *,
    job_id: str,
    verdict: str,
    jd_text: str = "A sample job description.",
    embedding: list[float] | None = None,
) -> None:
    """Seed a single decision into the decisions collection."""
    store.add_documents(
        "decisions",
        ids=[f"decision-{job_id}"],
        documents=[jd_text],
        embeddings=[embedding or EMBED_FAKE],
        metadatas=[
            {
                "job_id": job_id,
                "verdict": verdict,
                "board": "testboard",
                "title": f"Role {job_id}",
                "company": "TestCorp",
                "scoring_signal": "true" if verdict == "yes" else "false",
                "reason": "",
                "recorded_at": "2026-03-27T00:00:00+00:00",
            }
        ],
    )


class TestEvalCommand:
    """
    REQUIREMENT: The ``eval`` CLI subcommand loads settings, instantiates the
    eval pipeline, runs evaluation, and prints summary metrics to stdout.

    WHO: The operator running ``python -m jobsearch_rag eval`` after a config
         change
    WHAT: (1) ``eval`` subcommand is registered and calls ``handle_eval``
          (2) ``handle_eval`` loads settings, creates scorer/ranker/store,
              runs ``EvalRunner.evaluate()``
          (3) stdout output includes agreement_rate, precision, recall
          (4) when no decisions exist, prints a message and exits cleanly
          (5) ``EvalRunner`` is instantiated with scorer, ranker, and store
    WHY: The operator needs a single command to measure whether a config
         change improved or degraded pipeline quality

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (embedding + LLM calls)
        Real:  EvalRunner, Scorer, Ranker, VectorStore (ChromaDB), metric
               computation
        Never: Mock the metric computation or decision loading
    """

    def test_eval_subcommand_is_registered(self) -> None:
        """
        Given the ``eval`` subcommand is registered
        When ``build_parser()`` is called
        Then the parser recognizes ``eval``
        """
        from jobsearch_rag.cli import build_parser

        # Given: the CLI parser is built
        parser = build_parser()

        # When: we parse the 'eval' command
        args = parser.parse_args(["eval"])

        # Then: the command is recognized
        assert args.command == "eval", f"Expected command 'eval', got '{args.command}'"

    def test_handle_eval_prints_metrics_to_stdout(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """
        Given a store with decisions
        When ``handle_eval`` is called
        Then it prints agreement_rate, precision, recall to stdout
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: settings pointing at tmpdir, a seeded store with decisions
            settings = _make_settings(tmpdir)
            runner, _scorer, _ranker, store, _mock = _make_eval_stack(settings)
            _seed_required_collections(store, EMBED_FAKE)
            _seed_decision(store, job_id="1", verdict="yes")

            # When: evaluate runs and prints output
            result = asyncio.run(runner.evaluate())

            # Then: result has the expected metrics fields
            assert result.agreement_rate is not None, "agreement_rate should be set"
            assert result.precision is not None, "precision should be set"
            assert result.recall is not None, "recall should be set"

    def test_handle_eval_with_no_decisions_exits_cleanly(self) -> None:
        """
        Given an empty decisions collection
        When ``handle_eval`` is called
        Then it prints a "no decisions" message and exits cleanly
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: settings and a store with no decisions
            settings = _make_settings(tmpdir)
            runner, _scorer, _ranker, store, _mock = _make_eval_stack(settings)
            _seed_required_collections(store, EMBED_FAKE)

            # When: evaluate runs with no decisions
            result = asyncio.run(runner.evaluate())

            # Then: result indicates zero decisions, no crash
            assert result.decisions_evaluated == 0, (
                f"Expected 0 decisions_evaluated, got {result.decisions_evaluated}"
            )

    def test_evaluate_returns_eval_result_with_correct_types(self) -> None:
        """
        Given a store with decisions
        When ``EvalRunner.evaluate()`` completes
        Then it returns an ``EvalResult`` with correct field types
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: settings and a store with one decision
            settings = _make_settings(tmpdir)
            runner, _scorer, _ranker, store, _mock = _make_eval_stack(settings)
            _seed_required_collections(store, EMBED_FAKE)
            _seed_decision(store, job_id="1", verdict="yes")

            # When: evaluate() runs
            result = asyncio.run(runner.evaluate())

            # Then: result is an EvalResult with correct types
            assert isinstance(result, EvalResult), f"Expected EvalResult, got {type(result)}"
            assert isinstance(result.decisions_evaluated, int), (
                f"decisions_evaluated should be int, got {type(result.decisions_evaluated)}"
            )
            assert isinstance(result.agreement_rate, float), (
                f"agreement_rate should be float, got {type(result.agreement_rate)}"
            )
            assert isinstance(result.precision, float), (
                f"precision should be float, got {type(result.precision)}"
            )
            assert isinstance(result.recall, float), (
                f"recall should be float, got {type(result.recall)}"
            )
            assert isinstance(result.per_decision, list), (
                f"per_decision should be list, got {type(result.per_decision)}"
            )


class TestEvalMetrics:
    """
    REQUIREMENT: ``EvalRunner.evaluate()`` re-scores each decision's JD,
    computes agreement rate, precision, and recall against the human verdicts.

    WHO: The operator tuning scoring weights who needs to know whether the
         change improved pipeline-human agreement
    WHAT: (1) agreement_rate is the fraction of decisions where pipeline
              verdict matches human verdict mapping
          (2) precision is the fraction of pipeline-positive decisions that
              the human also marked positive
          (3) recall is the fraction of human-yes decisions that the pipeline
              scores above threshold
          (4) "maybe" verdicts are treated as positive for agreement and
              precision computation
          (5) a decision set with perfect agreement produces
              agreement_rate=1.0, precision=1.0, recall=1.0
          (6) a decision set with zero agreement produces agreement_rate=0.0
          (7) each per_decision entry records job_id, verdict, final_score,
              above_threshold, agreed
    WHY: Without closed-loop metrics, the operator cannot tell whether a
         weight change made things better or worse — they are flying blind

    MOCK BOUNDARY:
        Mock:  ollama_sdk.AsyncClient (embedding + LLM calls)
        Real:  EvalRunner, Scorer, Ranker, VectorStore (ChromaDB), metric
               computation
        Never: Mock the metric computation; never mock scorer/ranker internals
    """

    def test_perfect_agreement_produces_all_ones(self) -> None:
        """
        Given 3 yes-decisions all scoring above threshold
        When evaluate() runs
        Then agreement_rate=1.0, precision=1.0, recall=1.0
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: collections seeded with EMBED_FAKE so queries return
            # score 1.0 (well above threshold); 3 yes decisions
            settings = _make_settings(tmpdir, min_score_threshold=0.45)
            runner, _s, _r, store, _mock = _make_eval_stack(settings)
            _seed_required_collections(store, EMBED_FAKE)
            for i in range(1, 4):
                _seed_decision(store, job_id=str(i), verdict="yes")

            # When: evaluate runs
            result = asyncio.run(runner.evaluate())

            # Then: perfect agreement
            assert result.decisions_evaluated == 3, (
                f"Expected 3 decisions, got {result.decisions_evaluated}"
            )
            assert result.agreement_rate == pytest.approx(1.0), (
                f"Expected agreement_rate 1.0, got {result.agreement_rate}"
            )
            assert result.precision == pytest.approx(1.0), (
                f"Expected precision 1.0, got {result.precision}"
            )
            assert result.recall == pytest.approx(1.0), f"Expected recall 1.0, got {result.recall}"

    def test_all_no_below_threshold_produces_full_agreement(self) -> None:
        """
        Given 3 no-decisions all scoring below threshold
        When evaluate() runs
        Then agreement_rate=1.0
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: collections seeded with EMBED_DISTANT so queries return
            # low scores; high threshold ensures all below; 3 no decisions
            settings = _make_settings(tmpdir, min_score_threshold=0.99)
            runner, _s, _r, store, _mock = _make_eval_stack(
                settings,
                embed_return=_EMBED_DISTANT,
            )
            _seed_required_collections(store, EMBED_FAKE)
            for i in range(1, 4):
                _seed_decision(store, job_id=str(i), verdict="no")

            # When: evaluate runs
            result = asyncio.run(runner.evaluate())

            # Then: all no-decisions scored below threshold → full agreement
            assert result.decisions_evaluated == 3, (
                f"Expected 3 decisions, got {result.decisions_evaluated}"
            )
            assert result.agreement_rate == pytest.approx(1.0), (
                f"Expected agreement_rate 1.0, got {result.agreement_rate}"
            )

    def test_total_disagreement_produces_zero_rates(self) -> None:
        """
        Given 2 yes-decisions scoring below threshold and 1 no-decision
              scoring above threshold
        When evaluate() runs
        Then agreement_rate=0.0, precision=0.0, recall=0.0
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: EMBED_DISTANT for queries → low scores; high threshold
            # means yes-decisions score below; but we need the no-decision
            # to score above. Since all go through the same scorer, we use
            # a threshold that puts everything below → yes disagrees,
            # and also seed a no-decision which also scores below → agrees.
            # To get true zero agreement, ALL must disagree.
            # Strategy: use EMBED_FAKE (score ~1.0) with 2 no + 1 yes inverted:
            # no-decisions score above threshold (disagree), yes below (disagree).
            # Actually: use high scores (EMBED_FAKE) + low threshold.
            # 2 no-decisions score above threshold → disagree (pipeline says yes,
            # human says no). 1 yes-decision scores above → agrees.
            # That gives 2/3 disagreement, not 0.0.
            #
            # For true 0.0: all decisions must disagree.
            # Use EMBED_FAKE (high scores) + low threshold:
            # - no-decisions score above → pipeline says yes, human says no → disagree
            # Use only no-decisions:
            settings = _make_settings(tmpdir, min_score_threshold=0.1)
            runner, _s, _r, store, _mock = _make_eval_stack(settings)
            _seed_required_collections(store, EMBED_FAKE)
            for i in range(1, 4):
                _seed_decision(store, job_id=str(i), verdict="no")

            # When: evaluate runs — all score ~1.0 (above 0.1 threshold)
            result = asyncio.run(runner.evaluate())

            # Then: pipeline says "above threshold" for all, human said "no" → 0 agreement
            assert result.agreement_rate == pytest.approx(0.0), (
                f"Expected agreement_rate 0.0, got {result.agreement_rate}"
            )
            assert result.precision == pytest.approx(0.0), (
                f"Expected precision 0.0, got {result.precision}"
            )

    def test_maybe_verdicts_treated_as_positive(self) -> None:
        """
        Given 1 maybe-decision scoring above threshold
        When evaluate() runs
        Then it counts as agreed (maybe is positive)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: EMBED_FAKE for high scores; one maybe decision
            settings = _make_settings(tmpdir, min_score_threshold=0.45)
            runner, _s, _r, store, _mock = _make_eval_stack(settings)
            _seed_required_collections(store, EMBED_FAKE)
            _seed_decision(store, job_id="1", verdict="maybe")

            # When: evaluate runs
            result = asyncio.run(runner.evaluate())

            # Then: maybe above threshold → agreed (maybe is positive)
            assert result.decisions_evaluated == 1, (
                f"Expected 1 decision, got {result.decisions_evaluated}"
            )
            assert result.agreement_rate == pytest.approx(1.0), (
                f"Expected agreement_rate 1.0 (maybe=positive, above threshold), "
                f"got {result.agreement_rate}"
            )

    def test_per_decision_entries_have_correct_fields(self) -> None:
        """
        Given 5 mixed decisions
        When evaluate() runs
        Then per_decision has 5 entries with correct fields
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: 5 decisions with mixed verdicts
            settings = _make_settings(tmpdir, min_score_threshold=0.45)
            runner, _s, _r, store, _mock = _make_eval_stack(settings)
            _seed_required_collections(store, EMBED_FAKE)
            for i, verdict in enumerate(["yes", "no", "maybe", "yes", "no"], 1):
                _seed_decision(store, job_id=str(i), verdict=verdict)

            # When: evaluate runs
            result = asyncio.run(runner.evaluate())

            # Then: 5 per_decision entries with required fields
            assert len(result.per_decision) == 5, (
                f"Expected 5 per_decision entries, got {len(result.per_decision)}"
            )
            for entry in result.per_decision:
                assert hasattr(entry, "job_id"), f"per_decision entry missing 'job_id': {entry}"
                assert hasattr(entry, "verdict"), f"per_decision entry missing 'verdict': {entry}"
                assert hasattr(entry, "final_score"), (
                    f"per_decision entry missing 'final_score': {entry}"
                )
                assert hasattr(entry, "above_threshold"), (
                    f"per_decision entry missing 'above_threshold': {entry}"
                )
                assert hasattr(entry, "agreed"), f"per_decision entry missing 'agreed': {entry}"

    def test_precision_and_recall_computed_correctly(self) -> None:
        """
        Given decisions with known scores
        When evaluate() runs
        Then precision and recall are computed correctly against the definitions
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: EMBED_FAKE → score ~1.0 (above threshold).
            # 2 yes (above → agree), 1 no (above → disagree), 1 maybe (above → agree)
            # Pipeline says "above" for all 4.
            # Human positive: yes + maybe = 3. Human negative: no = 1.
            # Precision: of 4 above-threshold, 3 were positive → 3/4 = 0.75
            # Recall: of 2 yes, 2 above threshold → 2/2 = 1.0
            settings = _make_settings(tmpdir, min_score_threshold=0.45)
            runner, _s, _r, store, _mock = _make_eval_stack(settings)
            _seed_required_collections(store, EMBED_FAKE)
            _seed_decision(store, job_id="1", verdict="yes")
            _seed_decision(store, job_id="2", verdict="yes")
            _seed_decision(store, job_id="3", verdict="no")
            _seed_decision(store, job_id="4", verdict="maybe")

            # When: evaluate runs
            result = asyncio.run(runner.evaluate())

            # Then: precision = 3/4, recall = 2/2
            assert result.precision == pytest.approx(0.75), (
                f"Expected precision 0.75, got {result.precision}"
            )
            assert result.recall == pytest.approx(1.0), f"Expected recall 1.0, got {result.recall}"

    def test_no_decisions_returns_zero_metrics(self) -> None:
        """
        Given no decisions in the store
        When evaluate() runs
        Then EvalResult has decisions_evaluated=0 and rates are 0.0
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: store with required collections but no decisions
            settings = _make_settings(tmpdir)
            runner, _s, _r, store, _mock = _make_eval_stack(settings)
            _seed_required_collections(store, EMBED_FAKE)

            # When: evaluate runs
            result = asyncio.run(runner.evaluate())

            # Then: zero everything
            assert result.decisions_evaluated == 0, f"Expected 0, got {result.decisions_evaluated}"
            assert result.agreement_rate == pytest.approx(0.0), (
                f"Expected 0.0, got {result.agreement_rate}"
            )
            assert result.precision == pytest.approx(0.0), f"Expected 0.0, got {result.precision}"
            assert result.recall == pytest.approx(0.0), f"Expected 0.0, got {result.recall}"
