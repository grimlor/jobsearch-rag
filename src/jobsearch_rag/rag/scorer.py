"""Semantic scoring and LLM disqualifier classification.

The Scorer bridges two RAG concerns:

1. **Semantic similarity** — embed the JD text, query VectorStore collections
   (resume, role_archetypes, decisions) and convert cosine distances to
   similarity scores in [0.0, 1.0].

2. **LLM disqualification** — send a structured prompt to the Embedder's
   ``classify()`` method asking the LLM whether the role is structurally
   unsuitable (e.g. IC-disguised-as-architect, SRE on-call ownership,
   staffing agency chain).  The response is expected as JSON; malformed
   responses fall back to *not disqualified* (safe default).

ChromaDB returns *distances* (cosine distance = 1 - cosine_similarity).
The ``_distance_to_score`` helper converts the closest distance to a
similarity score clamped to [0.0, 1.0].
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from jobsearch_rag.errors import ActionableError

if TYPE_CHECKING:
    from jobsearch_rag.rag.embedder import Embedder
    from jobsearch_rag.rag.store import VectorStore

logger = logging.getLogger(__name__)

# The disqualifier prompt sent as a *system* message.  The JD text is sent
# as the *user* message.
_DISQUALIFIER_SYSTEM_PROMPT = """\
You are a role-fit classifier for a senior/staff-level platform architect.
Analyse the following job description and decide whether it is structurally
unsuitable for a Principal/Staff Platform Architect candidate.

Disqualify if the role is:
- An individual-contributor coding role disguised with an "Architect" title
- Primarily SRE/on-call operations ownership
- A staffing-agency or vendor-chain posting
- Primarily full-stack web development

Respond ONLY with a JSON object (no markdown fences):
{"disqualified": true/false, "reason": "short explanation or null"}
"""


@dataclass
class ScoreResult:
    """Component scores for a single job listing."""

    fit_score: float
    archetype_score: float
    history_score: float
    disqualified: bool
    disqualifier_reason: str | None = None

    @property
    def is_valid(self) -> bool:
        """All component scores are in [0.0, 1.0]."""
        return all(
            0.0 <= s <= 1.0 for s in (self.fit_score, self.archetype_score, self.history_score)
        )


def _distance_to_score(distances: list[float]) -> float:
    """Convert a list of cosine distances to a single similarity score.

    ChromaDB returns *cosine distance* (1 - cosine_similarity).  We take
    the *minimum* distance (closest match) and clamp to [0.0, 1.0]:

        score = max(0.0, min(1.0, 1.0 - distance))
    """
    if not distances:
        return 0.0
    best = min(distances)
    return max(0.0, min(1.0, 1.0 - best))


class Scorer:
    """Computes semantic similarity scores and runs the LLM disqualifier.

    Parameters
    ----------
    store:
        A VectorStore with (at minimum) ``resume`` and ``role_archetypes``
        collections.  A ``decisions`` collection is optional — if absent or
        empty, the history score defaults to 0.0.
    embedder:
        An Embedder instance used to embed the JD text and to call the
        LLM disqualifier.
    disqualify_on_llm_flag:
        Whether ``score()`` should run the disqualifier prompt.  When False
        (e.g. for benchmarking), the disqualification step is skipped.
    """

    def __init__(
        self,
        *,
        store: VectorStore,
        embedder: Embedder,
        disqualify_on_llm_flag: bool = True,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._disqualify_on_llm_flag = disqualify_on_llm_flag

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def score(self, jd_text: str) -> ScoreResult:
        """Score a job description against resume, archetypes, and decision history.

        1. Embed ``jd_text``.
        2. Query the ``resume`` collection — best-match distance → ``fit_score``.
        3. Query the ``role_archetypes`` collection — → ``archetype_score``.
        4. Query the ``decisions`` collection (if present) — → ``history_score``.
        5. Optionally run the LLM disqualifier.

        Raises ``ActionableError`` (INDEX) if the ``resume`` collection is
        empty or missing — the pipeline *must* index a resume before scoring.
        """
        embedding = await self._embedder.embed(jd_text)

        fit_score = self._query_collection("resume", embedding)
        archetype_score = self._query_collection("role_archetypes", embedding)
        history_score = self._query_collection_optional("decisions", embedding)

        disqualified = False
        reason: str | None = None
        if self._disqualify_on_llm_flag:
            disqualified, reason = await self.disqualify(jd_text)

        return ScoreResult(
            fit_score=fit_score,
            archetype_score=archetype_score,
            history_score=history_score,
            disqualified=disqualified,
            disqualifier_reason=reason,
        )

    async def disqualify(self, jd_text: str) -> tuple[bool, str | None]:
        """Run the LLM disqualifier prompt. Returns ``(disqualified, reason)``.

        If the LLM response is not valid JSON, the role is kept (safe
        default) and a warning is logged with the raw response so the
        prompt engineer can diagnose.
        """
        raw = await self._embedder.classify(_DISQUALIFIER_SYSTEM_PROMPT + "\n\n" + jd_text)
        return self._parse_disqualifier_response(raw)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _query_collection(self, collection_name: str, embedding: list[float]) -> float:
        """Query a collection and return the best-match similarity score.

        Raises ``ActionableError.index`` if the collection is empty or
        does not exist.
        """
        count = self._store.collection_count(collection_name)
        if count == 0:
            msg = f"Collection '{collection_name}' is empty. Run the indexer before scoring."
            raise ActionableError.index(msg)

        n_results = min(count, 3)
        results = self._store.query(
            collection_name=collection_name,
            query_embedding=embedding,
            n_results=n_results,
        )
        distances: list[float] = results.get("distances", [[]])[0]
        return _distance_to_score(distances)

    def _query_collection_optional(self, collection_name: str, embedding: list[float]) -> float:
        """Query a collection, returning 0.0 if it's empty or missing."""
        try:
            count = self._store.collection_count(collection_name)
        except ActionableError:
            return 0.0
        if count == 0:
            return 0.0
        n_results = min(count, 3)
        results = self._store.query(
            collection_name=collection_name,
            query_embedding=embedding,
            n_results=n_results,
        )
        distances: list[float] = results.get("distances", [[]])[0]
        return _distance_to_score(distances)

    @staticmethod
    def _parse_disqualifier_response(raw: str) -> tuple[bool, str | None]:
        """Parse the LLM disqualifier JSON response.

        Expected format::

            {"disqualified": true, "reason": "short explanation"}

        Falls back to ``(False, None)`` on any parse error — keeping the
        role is the safe default.
        """
        try:
            data = json.loads(raw)
            disqualified = bool(data.get("disqualified", False))
            reason = data.get("reason")
            if reason is not None:
                reason = str(reason)
                if reason.lower() == "null":
                    reason = None
            return disqualified, reason
        except (json.JSONDecodeError, AttributeError, TypeError):
            logger.warning("Malformed disqualifier response (keeping role): %s", raw)
            return False, None
