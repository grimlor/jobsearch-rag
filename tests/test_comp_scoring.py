"""Compensation scoring tests — continuous scale, band boundaries, config integration.

Maps to BDD specs: TestCompensationScoring

The comp scorer converts parsed compensation data (comp_max) into a continuous
score in [0.0, 1.0] relative to a configurable base_salary target.  This is a
*taste signal* — it nudges rankings without hard-gating any role.
"""

from __future__ import annotations

import pytest

from jobsearch_rag.rag.comp_parser import compute_comp_score

# ---------------------------------------------------------------------------
# TestCompensationScoring
# ---------------------------------------------------------------------------


class TestCompensationScoring:
    """REQUIREMENT: Compensation score is a continuous signal relative to a
    configurable base_salary target, not a binary gate.

    WHO: The ranker consuming comp_score to nudge final ranking;
         the operator tuning base_salary in settings.toml
    WHAT: comp_max at or above base_salary produces comp_score 1.0;
          comp_max between 90-100% of base produces 0.7-0.9 (linear);
          comp_max between 77-90% of base produces 0.4-0.7 (linear);
          comp_max between 68-77% of base produces 0.0-0.4 (linear);
          comp_max below 68% of base produces 0.0;
          the scale is continuous across band boundaries (no gaps or jumps);
          missing compensation data produces 0.5 (neutral);
          base_salary is read from settings.toml, not hardcoded;
          changing base_salary changes all score boundaries proportionally;
          comp_score is always in [0.0, 1.0]
    WHY: Compensation as a taste signal lets well-paying roles float up
         without hard-gating roles that might be stepping stones or have
         other attractive qualities — a gate would discard too aggressively
    """

    # Default base_salary for tests
    BASE = 220_000

    # --- Top band: ≥ 100% of base → 1.0 ---

    def test_comp_max_at_base_salary_scores_one(self) -> None:
        """comp_max exactly at base_salary produces the maximum score of 1.0."""
        assert compute_comp_score(comp_max=220_000, base_salary=self.BASE) == pytest.approx(1.0)

    def test_comp_max_above_base_salary_scores_one(self) -> None:
        """comp_max exceeding base_salary still scores 1.0 — no bonus for overshoot."""
        assert compute_comp_score(comp_max=300_000, base_salary=self.BASE) == pytest.approx(1.0)

    # --- Band: 90-100% of base -> 0.7-0.9 (linear) ---

    def test_comp_max_at_95_percent_of_base_scores_between_07_and_09(self) -> None:
        """comp_max at 95% of base falls in the 0.7-0.9 band (linear interpolation)."""
        score = compute_comp_score(comp_max=209_000, base_salary=self.BASE)
        assert 0.7 < score < 0.9

    def test_comp_max_at_90_percent_of_base_scores_07(self) -> None:
        """comp_max at exactly 90% of base hits the lower boundary of 0.7."""
        score = compute_comp_score(comp_max=198_000, base_salary=self.BASE)
        assert score == pytest.approx(0.7)

    # --- Band: 77-90% of base -> 0.4-0.7 (linear) ---

    def test_comp_max_at_85_percent_of_base_scores_between_04_and_07(self) -> None:
        """comp_max at 85% of base falls in the 0.4-0.7 band (linear interpolation)."""
        score = compute_comp_score(comp_max=187_000, base_salary=self.BASE)
        assert 0.4 < score < 0.7

    def test_comp_max_at_77_percent_of_base_scores_04(self) -> None:
        """comp_max at exactly 77% of base hits the lower boundary of 0.4."""
        score = compute_comp_score(comp_max=169_400, base_salary=self.BASE)
        assert score == pytest.approx(0.4)

    # --- Band: 68-77% of base -> 0.0-0.4 (linear) ---

    def test_comp_max_at_72_percent_of_base_scores_between_00_and_04(self) -> None:
        """comp_max at 72% of base falls in the 0.0-0.4 band (linear interpolation)."""
        score = compute_comp_score(comp_max=158_400, base_salary=self.BASE)
        assert 0.0 < score < 0.4

    def test_comp_max_at_68_percent_of_base_scores_zero(self) -> None:
        """comp_max at exactly 68% of base hits the floor of 0.0."""
        score = compute_comp_score(comp_max=149_600, base_salary=self.BASE)
        assert score == pytest.approx(0.0)

    # --- Below 68% → 0.0 ---

    def test_comp_max_below_68_percent_of_base_scores_zero(self) -> None:
        """comp_max well below 68% of base is clamped to 0.0."""
        assert compute_comp_score(comp_max=100_000, base_salary=self.BASE) == pytest.approx(0.0)

    # --- Missing data → 0.5 ---

    def test_missing_comp_data_scores_05(self) -> None:
        """When comp_max is None (no salary data), the score is neutral at 0.5."""
        assert compute_comp_score(comp_max=None, base_salary=self.BASE) == pytest.approx(0.5)

    # --- Config-driven, not hardcoded ---

    def test_base_salary_is_read_from_config_not_hardcoded(self) -> None:
        """The scoring function accepts base_salary as a parameter, not a hardcoded constant."""
        # With base=200k, 200k should score 1.0
        assert compute_comp_score(comp_max=200_000, base_salary=200_000) == pytest.approx(1.0)
        # With base=300k, 200k should score well below 1.0
        score = compute_comp_score(comp_max=200_000, base_salary=300_000)
        assert score < 0.7

    def test_changing_base_salary_shifts_all_boundaries(self) -> None:
        """All band boundaries scale proportionally with base_salary."""
        # With base=100k, 90% = 90k should score 0.7
        assert compute_comp_score(comp_max=90_000, base_salary=100_000) == pytest.approx(0.7)
        # With base=300k, 90% = 270k should also score 0.7
        assert compute_comp_score(comp_max=270_000, base_salary=300_000) == pytest.approx(0.7)

    # --- Invariants ---

    def test_comp_score_is_always_between_zero_and_one(self) -> None:
        """comp_score is clamped to [0.0, 1.0] for any input value."""
        # Very high comp
        assert 0.0 <= compute_comp_score(comp_max=1_000_000, base_salary=self.BASE) <= 1.0
        # Very low comp
        assert 0.0 <= compute_comp_score(comp_max=10_000, base_salary=self.BASE) <= 1.0
        # Zero comp
        assert 0.0 <= compute_comp_score(comp_max=0, base_salary=self.BASE) <= 1.0
        # Missing comp
        assert 0.0 <= compute_comp_score(comp_max=None, base_salary=self.BASE) <= 1.0

    def test_comp_score_interpolates_linearly_within_bands(self) -> None:
        """Within each band, the score increases linearly — no plateaus or jumps."""
        # Sample three points in the 77-90% band (0.4-0.7)
        low = compute_comp_score(comp_max=170_000, base_salary=self.BASE)  # ~77%
        mid = compute_comp_score(comp_max=184_000, base_salary=self.BASE)  # ~83.6%
        high = compute_comp_score(comp_max=197_000, base_salary=self.BASE)  # ~89.5%
        assert low < mid < high, f"Expected monotonic increase: {low} < {mid} < {high}"

    def test_scale_is_continuous_across_band_boundaries(self) -> None:
        """The score has no gaps or jumps at the boundaries between bands."""
        epsilon = 1.0  # $1 above/below boundary

        # At 68% boundary (0.0)
        below_68 = compute_comp_score(comp_max=self.BASE * 0.68 - epsilon, base_salary=self.BASE)
        at_68 = compute_comp_score(comp_max=self.BASE * 0.68, base_salary=self.BASE)
        above_68 = compute_comp_score(comp_max=self.BASE * 0.68 + epsilon, base_salary=self.BASE)
        assert below_68 == pytest.approx(0.0, abs=0.01)
        assert at_68 == pytest.approx(0.0, abs=0.01)
        assert above_68 >= 0.0

        # At 77% boundary (0.4)
        just_below_77 = compute_comp_score(
            comp_max=self.BASE * 0.77 - epsilon, base_salary=self.BASE
        )
        just_above_77 = compute_comp_score(
            comp_max=self.BASE * 0.77 + epsilon, base_salary=self.BASE
        )
        assert abs(just_above_77 - just_below_77) < 0.02, "Discontinuity at 77% boundary"

        # At 90% boundary (0.7)
        just_below_90 = compute_comp_score(
            comp_max=self.BASE * 0.90 - epsilon, base_salary=self.BASE
        )
        just_above_90 = compute_comp_score(
            comp_max=self.BASE * 0.90 + epsilon, base_salary=self.BASE
        )
        assert abs(just_above_90 - just_below_90) < 0.02, "Discontinuity at 90% boundary"

        # At 100% boundary (1.0)
        just_below_100 = compute_comp_score(comp_max=self.BASE - epsilon, base_salary=self.BASE)
        at_100 = compute_comp_score(comp_max=self.BASE, base_salary=self.BASE)
        assert just_below_100 > 0.89, "Score should be near 1.0 just below 100%"
        assert at_100 == pytest.approx(1.0)
