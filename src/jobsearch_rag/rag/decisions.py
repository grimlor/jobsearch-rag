"""Decision history recording and retrieval.

Records user verdicts (yes / no / maybe) on scored job listings and
stores them in the ``decisions`` ChromaDB collection.  Only ``yes``
verdicts contribute to ``history_score`` — rejected roles have too
many confounding reasons to be a useful negative signal.

Decisions are persisted in two forms:

1. **ChromaDB** — the ``decisions`` collection stores the JD embedding
   alongside metadata (verdict, job_id, board) so it can be queried
   for ``history_score`` on future runs.

2. **JSONL on disk** — a daily append-only ``data/decisions/YYYY-MM-DD.jsonl``
   file for audit and debugging.  The JSONL file contains the full JD text
   (which is too large for ChromaDB metadata) plus all scoring data.

The ChromaDB ``decisions`` collection filters on ``verdict == "yes"``
during scoring queries.  No/maybe verdicts are stored in ChromaDB too
(for auditability) but carry metadata ``{"scoring_signal": "false"}``
so the scorer can exclude them without deleting data.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from jobsearch_rag.errors import ActionableError, ErrorType

if TYPE_CHECKING:
    from jobsearch_rag.rag.embedder import Embedder
    from jobsearch_rag.rag.store import VectorStore

logger = logging.getLogger(__name__)

# Valid verdicts
VALID_VERDICTS = frozenset({"yes", "no", "maybe"})

# Default directory for JSONL decision logs
_DECISIONS_DIR = Path("data/decisions")


class DecisionRecorder:
    """Records and retrieves user verdicts on job listings.

    Usage::

        recorder = DecisionRecorder(store=vector_store, embedder=embedder)
        await recorder.record(
            job_id="ziprecruiter-12345",
            verdict="yes",
            jd_text="Full job description text...",
            board="ziprecruiter",
            title="Staff Platform Architect",
            company="Acme Corp",
        )
    """

    def __init__(
        self,
        *,
        store: VectorStore,
        embedder: Embedder,
        decisions_dir: str | Path = _DECISIONS_DIR,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._decisions_dir = Path(decisions_dir)

    async def record(
        self,
        *,
        job_id: str,
        verdict: str,
        jd_text: str,
        board: str,
        title: str = "",
        company: str = "",
    ) -> None:
        """Record a user verdict on a job listing.

        Stores the decision in both ChromaDB (for scoring) and as JSONL
        on disk (for audit).  Re-recording a verdict on the same ``job_id``
        **overwrites** the previous decision (upsert semantics).

        Args:
            job_id: Unique job identifier (``board-external_id`` format).
            verdict: One of ``"yes"``, ``"no"``, ``"maybe"``.
            jd_text: Full job description text for embedding.
            board: Source board name.
            title: Job title (for audit log).
            company: Company name (for audit log).

        Raises:
            ActionableError (DECISION): if verdict is invalid.
            ActionableError (VALIDATION): if jd_text is empty.
        """
        if verdict not in VALID_VERDICTS:
            from jobsearch_rag.errors import AIGuidance, Troubleshooting

            raise ActionableError(
                error=f"Invalid verdict '{verdict}' for job '{job_id}'",
                error_type=ErrorType.DECISION,
                service="decisions",
                suggestion=f"Use one of: {', '.join(sorted(VALID_VERDICTS))}",
                ai_guidance=AIGuidance(
                    action_required=f"Replace '{verdict}' with a valid verdict",
                    checks=[f"Valid verdicts are: {', '.join(sorted(VALID_VERDICTS))}"],
                ),
                troubleshooting=Troubleshooting(
                    steps=[
                        f"1. You used verdict '{verdict}' which is not recognized",
                        f"2. Valid verdicts are: {', '.join(sorted(VALID_VERDICTS))}",
                        "3. Re-run the decide command with a valid verdict",
                    ]
                ),
            )

        if not jd_text.strip():
            raise ActionableError.validation(
                field_name="jd_text",
                reason=f"empty JD text for job '{job_id}'",
                suggestion="Ensure the job listing has full_text before recording a decision",
            )

        # Embed the JD text
        embedding = await self._embedder.embed(jd_text)

        # Determine whether this verdict participates in scoring
        scoring_signal = "true" if verdict == "yes" else "false"

        # Upsert into ChromaDB decisions collection
        self._store.add_documents(
            collection_name="decisions",
            ids=[f"decision-{job_id}"],
            documents=[jd_text],
            embeddings=[embedding],
            metadatas=[{
                "job_id": job_id,
                "verdict": verdict,
                "board": board,
                "title": title,
                "company": company,
                "scoring_signal": scoring_signal,
                "recorded_at": datetime.now(UTC).isoformat(),
            }],
        )

        # Append to daily JSONL log
        self._append_jsonl(
            job_id=job_id,
            verdict=verdict,
            board=board,
            title=title,
            company=company,
            jd_text=jd_text,
        )

        logger.info(
            "Recorded '%s' verdict for %s (%s @ %s)",
            verdict,
            job_id,
            title,
            company,
        )

    def get_decision(self, job_id: str) -> dict[str, str] | None:
        """Retrieve the stored decision for a job_id, or None if not found.

        Returns metadata dict with keys: job_id, verdict, board, title,
        company, scoring_signal, recorded_at.
        """
        doc_id = f"decision-{job_id}"
        try:
            results = self._store.get_documents(
                collection_name="decisions",
                ids=[doc_id],
            )
        except ActionableError:
            return None

        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        if not ids or not metadatas:
            return None

        return dict(metadatas[0]) if metadatas[0] else None

    def history_count(self) -> int:
        """Return the number of decisions in the history collection."""
        try:
            return self._store.collection_count("decisions")
        except ActionableError:
            return 0

    def _append_jsonl(
        self,
        *,
        job_id: str,
        verdict: str,
        board: str,
        title: str,
        company: str,
        jd_text: str,
    ) -> None:
        """Append a decision record to the daily JSONL file."""
        self._decisions_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        filepath = self._decisions_dir / f"{today}.jsonl"

        record = {
            "job_id": job_id,
            "verdict": verdict,
            "board": board,
            "title": title,
            "company": company,
            "jd_text": jd_text,
            "recorded_at": datetime.now(UTC).isoformat(),
        }

        with filepath.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        logger.debug("Appended decision to %s", filepath)
