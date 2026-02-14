"""Decision history tests.

Maps to BDD spec: TestDecisionRecording
"""

from __future__ import annotations


class TestDecisionRecording:
    """REQUIREMENT: User decisions are recorded and build the history signal over time.

    WHO: The scorer computing history_score on future runs
    WHAT: A verdict (yes/no/maybe) is stored with the JD embedding and job_id;
          recording a decision for an unknown job_id raises a clear error;
          the history collection grows with each decision;
          only 'yes' decisions contribute to history_score (maybes and nos do not)
    WHY: If 'no' decisions contributed to scoring, roles similar to rejected ones
         would score lower — but the signal we want is 'what did I like',
         not 'what did I reject' (rejections have too many confounding reasons)
    """

    def test_yes_verdict_is_stored_in_history_collection(self) -> None: ...
    def test_no_verdict_is_stored_but_excluded_from_scoring_signal(self) -> None: ...
    def test_maybe_verdict_is_stored_but_excluded_from_scoring_signal(self) -> None: ...
    def test_unknown_job_id_raises_decision_error_with_job_id(self) -> None: ...
    def test_history_collection_count_increases_after_each_decision(self) -> None: ...
    def test_duplicate_decision_on_same_job_id_overwrites_not_appends(self) -> None: ...
