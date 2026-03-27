"""
Evaluation harness — measures pipeline-vs-human agreement.

Re-scores every stored decision's JD through the current scorer/ranker
configuration and computes agreement rate, precision, and recall against
the human verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobsearch_rag.logging import logger

if TYPE_CHECKING:
    from jobsearch_rag.pipeline.ranker import Ranker
    from jobsearch_rag.rag.scorer import Scorer
    from jobsearch_rag.rag.store import VectorStore


@dataclass
class EvalDecision:
    """Per-decision evaluation result."""

    job_id: str
    verdict: str
    final_score: float
    above_threshold: bool
    agreed: bool


@dataclass
class EvalResult:
    """Aggregate evaluation metrics."""

    decisions_evaluated: int
    agreement_rate: float
    precision: float
    recall: float
    per_decision: list[EvalDecision] = field(default_factory=lambda: list[EvalDecision]())


class EvalRunner:
    """Re-scores stored decisions and computes agreement metrics."""

    def __init__(
        self,
        scorer: Scorer,
        ranker: Ranker,
        store: VectorStore,
    ) -> None:
        """Initialize with a scorer, ranker, and vector store."""
        self._scorer = scorer
        self._ranker = ranker
        self._store = store

    async def evaluate(self) -> EvalResult:
        """
        Re-score every decision and compute agreement metrics.

        Verdict mapping:
            Human positive: yes, maybe
            Human negative: no
            Pipeline positive: final_score >= ranker.min_score_threshold

        Metrics:
            agreement_rate = agreed / total
            precision = true_positive / pipeline_positive  (0.0 if none)
            recall = true_positive / human_yes_count  (0.0 if none)
        """
        decisions = self._load_decisions()
        if not decisions:
            logger.info("No decisions found — nothing to evaluate.")
            return EvalResult(
                decisions_evaluated=0,
                agreement_rate=0.0,
                precision=0.0,
                recall=0.0,
            )

        per_decision: list[EvalDecision] = []
        agreed_count = 0
        true_positive = 0
        pipeline_positive_count = 0
        human_yes_count = 0
        human_yes_above = 0

        for job_id, verdict, jd_text in decisions:
            score_result = await self._scorer.score(jd_text)
            final_score = self._ranker.compute_final_score(score_result)
            above_threshold = final_score >= self._ranker.min_score_threshold

            human_positive = verdict in ("yes", "maybe")
            pipeline_positive = above_threshold

            agreed = human_positive == pipeline_positive

            if agreed:
                agreed_count += 1
            if pipeline_positive:
                pipeline_positive_count += 1
                if human_positive:
                    true_positive += 1
            if verdict == "yes":
                human_yes_count += 1
                if pipeline_positive:
                    human_yes_above += 1

            per_decision.append(
                EvalDecision(
                    job_id=job_id,
                    verdict=verdict,
                    final_score=final_score,
                    above_threshold=above_threshold,
                    agreed=agreed,
                )
            )

        total = len(decisions)
        agreement_rate = agreed_count / total
        precision = true_positive / pipeline_positive_count if pipeline_positive_count > 0 else 0.0
        recall = human_yes_above / human_yes_count if human_yes_count > 0 else 0.0

        return EvalResult(
            decisions_evaluated=total,
            agreement_rate=agreement_rate,
            precision=precision,
            recall=recall,
            per_decision=per_decision,
        )

    def _load_decisions(self) -> list[tuple[str, str, str]]:
        """Load all decisions from the store as (job_id, verdict, jd_text) tuples."""
        try:
            collection = self._store.get_or_create_collection("decisions")
        except Exception:
            logger.debug("Decisions collection not accessible.")
            return []

        result = collection.get(include=["documents", "metadatas"])

        ids = result["ids"]
        documents = result["documents"] or []
        metadatas = result["metadatas"] or []

        decisions: list[tuple[str, str, str]] = []
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if metadatas and i < len(metadatas) else None
            doc = documents[i] if documents and i < len(documents) else None
            if meta is None or doc is None:
                logger.warning("Skipping decision %s — missing metadata or document.", doc_id)
                continue
            job_id = str(meta.get("job_id", doc_id))
            verdict = str(meta.get("verdict", ""))
            if not verdict:
                logger.warning("Skipping decision %s — no verdict.", doc_id)
                continue
            decisions.append((job_id, verdict, doc))

        return decisions
