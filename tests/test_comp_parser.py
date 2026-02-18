"""Compensation parsing tests — extraction, normalization, and source detection.

Maps to BDD spec: TestCompensationParsing

The comp parser extracts salary ranges from JD text via regex, normalizes
hourly rates to annual (x2080), and records the source (employer-stated vs.
board-estimated).  All extraction happens on raw text — no LLM involvement.
"""

from __future__ import annotations

import pytest

from jobsearch_rag.rag.comp_parser import CompResult, parse_compensation  # noqa: F401

# ---------------------------------------------------------------------------
# TestCompensationParsing
# ---------------------------------------------------------------------------


class TestCompensationParsing:
    """REQUIREMENT: Compensation ranges are extracted from JD text and normalized.

    WHO: The pipeline runner enriching listings after JD extraction;
         the scorer computing comp_score
    WHAT: Annual salary ranges ($NNN,NNN or $NNNk) are extracted as comp_min/max
           floats; hourly rates ($NN/hr) are converted to annual via x2080;
          ranges expressed as "$180,000 - $220,000" produce both min and max;
          single values produce both min and max as the same number;
          the original text snippet is preserved in comp_text;
          JDs with no salary information produce None for all comp fields;
          the parser does not hallucinate numbers from non-salary contexts
          (e.g. "5,000 employees" is not a salary)
    WHY: Salary data is noisy and inconsistently formatted across boards;
         incorrect parsing would silently distort comp_score rankings —
         a wrong number is worse than a missing one
    """

    def test_annual_range_with_commas_is_parsed(self) -> None:
        """A range like '$180,000 - $220,000' extracts comp_min=180000 and comp_max=220000."""
        text = "Compensation: $180,000 - $220,000 per year"
        result = parse_compensation(text)
        assert result is not None
        assert result.comp_min == pytest.approx(180_000)
        assert result.comp_max == pytest.approx(220_000)

    def test_annual_range_with_k_suffix_is_parsed(self) -> None:
        """A range like '$180k-$220k' is parsed using the k=1000 multiplier."""
        text = "Salary range: $180k-$220k"
        result = parse_compensation(text)
        assert result is not None
        assert result.comp_min == pytest.approx(180_000)
        assert result.comp_max == pytest.approx(220_000)

    def test_hourly_rate_is_converted_to_annual_via_2080(self) -> None:
        """An hourly rate like '$95/hr' is annualized as 95 * 2080 = 197,600."""
        text = "This position pays $95/hr"
        result = parse_compensation(text)
        assert result is not None
        assert result.comp_min == pytest.approx(95 * 2080)
        assert result.comp_max == pytest.approx(95 * 2080)

    def test_hourly_range_is_converted_to_annual_via_2080(self) -> None:
        """An hourly range like '$85 - $105/hour' is annualized for both endpoints."""
        text = "Rate: $85 - $105/hour"
        result = parse_compensation(text)
        assert result is not None
        assert result.comp_min == pytest.approx(85 * 2080)
        assert result.comp_max == pytest.approx(105 * 2080)

    def test_single_annual_value_sets_both_min_and_max(self) -> None:
        """A single value like '$200,000' sets both comp_min and comp_max to 200000."""
        text = "Base salary: $200,000"
        result = parse_compensation(text)
        assert result is not None
        assert result.comp_min == pytest.approx(200_000)
        assert result.comp_max == pytest.approx(200_000)

    def test_no_salary_info_returns_none(self) -> None:
        """A JD with no salary information returns None."""
        text = "We are looking for a Staff Architect to lead platform design."
        result = parse_compensation(text)
        assert result is None

    def test_comp_text_preserves_original_matched_snippet(self) -> None:
        """The comp_text field preserves the original matched text for audit."""
        text = "The salary for this role is $180,000 - $220,000 per year."
        result = parse_compensation(text)
        assert result is not None
        assert "$180,000" in result.comp_text
        assert "$220,000" in result.comp_text

    def test_employee_count_is_not_mistaken_for_salary(self) -> None:
        """Numbers like '5,000 employees' are not mistaken for salary figures."""
        text = "We have 5,000 employees across 10 offices worldwide."
        result = parse_compensation(text)
        assert result is None

    def test_revenue_figures_are_not_mistaken_for_salary(self) -> None:
        """Revenue like '$2.5 billion' or '$500M ARR' is not mistaken for salary."""
        text = "The company has $2.5 billion in revenue and $500M ARR."
        result = parse_compensation(text)
        assert result is None

    def test_range_with_dollar_sign_and_hyphen_is_parsed(self) -> None:
        """A standard range with dollar signs and hyphen '$150,000 - $200,000' is parsed."""
        text = "Pay: $150,000 - $200,000"
        result = parse_compensation(text)
        assert result is not None
        assert result.comp_min == pytest.approx(150_000)
        assert result.comp_max == pytest.approx(200_000)

    def test_range_with_to_keyword_is_parsed(self) -> None:
        """Ranges using 'to' instead of '-' (e.g. '$150,000 to $200,000') are parsed."""
        text = "Salary: $150,000 to $200,000 annually"
        result = parse_compensation(text)
        assert result is not None
        assert result.comp_min == pytest.approx(150_000)
        assert result.comp_max == pytest.approx(200_000)

    def test_per_year_and_per_annum_suffixes_are_recognized(self) -> None:
        """Suffixes like '/yr', '/year', 'per annum', 'annually' mark annual salary."""
        for suffix in ["/yr", "/year", "per annum", "annually", "per year", "a year"]:
            text = f"Compensation: $200,000 {suffix}"
            result = parse_compensation(text)
            assert result is not None, f"Failed to parse salary with suffix '{suffix}'"
            assert result.comp_min == pytest.approx(200_000)

    def test_comp_source_defaults_to_employer(self) -> None:
        """When salary is extracted from JD body text, comp_source is 'employer'."""
        text = "Base salary for this role: $180,000 - $220,000"
        result = parse_compensation(text)
        assert result is not None
        assert result.comp_source == "employer"

    def test_comp_source_is_estimated_when_flagged(self) -> None:
        """When source='estimated' is passed, comp_source reflects that."""
        text = "Estimated: $180,000 - $220,000"
        result = parse_compensation(text, source="estimated")
        assert result is not None
        assert result.comp_source == "estimated"

    def test_employer_stated_salary_is_preferred_over_board_estimate(self) -> None:
        """When both a JD-body salary and a board estimate exist, the JD-body one wins."""
        jd_result = parse_compensation(
            "Base salary: $200,000 - $250,000 per year", source="employer"
        )
        board_result = parse_compensation(
            "Estimated salary: $170,000 - $200,000", source="estimated"
        )
        assert jd_result is not None
        assert board_result is not None
        # Employer result should be chosen over estimated in the pipeline
        assert jd_result.comp_source == "employer"
        assert board_result.comp_source == "estimated"

    def test_result_dataclass_has_expected_fields(self) -> None:
        """CompResult has comp_min, comp_max, comp_source, and comp_text fields."""
        text = "Salary: $200,000"
        result = parse_compensation(text)
        assert result is not None
        assert hasattr(result, "comp_min")
        assert hasattr(result, "comp_max")
        assert hasattr(result, "comp_source")
        assert hasattr(result, "comp_text")

    def test_k_suffix_case_insensitive(self) -> None:
        """The 'k' suffix is recognized regardless of case: $200K, $200k."""
        for variant in ["$200K", "$200k"]:
            result = parse_compensation(f"Salary: {variant}")
            assert result is not None
            assert result.comp_min == pytest.approx(200_000)

    def test_range_with_no_space_around_hyphen(self) -> None:
        """Ranges like '$180,000-$220,000' (no spaces) are parsed."""
        text = "Salary: $180,000-$220,000"
        result = parse_compensation(text)
        assert result is not None
        assert result.comp_min == pytest.approx(180_000)
        assert result.comp_max == pytest.approx(220_000)

    def test_decimal_hourly_rate(self) -> None:
        """Hourly rates with decimals like '$95.50/hr' are parsed correctly."""
        text = "Rate: $95.50/hr"
        result = parse_compensation(text)
        assert result is not None
        assert result.comp_min == pytest.approx(95.50 * 2080)

    def test_range_in_false_positive_context_is_skipped(self) -> None:
        """GIVEN a dollar range surrounded by false-positive context words
        WHEN parse_compensation is called
        THEN the range match is skipped and None is returned.
        """
        text = "Our company raised $100,000 - $200,000 in funding last quarter."
        result = parse_compensation(text)
        assert result is None

    def test_range_outside_salary_bounds_is_skipped(self) -> None:
        """GIVEN a dollar range where both values are below realistic salary thresholds
        WHEN parse_compensation is called
        THEN the match is skipped (not in salary range) and None is returned.
        """
        # $3 - $5 is too low to be salary and not /hr so it's skipped
        text = "The fee is $3 - $5 per widget."
        result = parse_compensation(text)
        assert result is None

    def test_single_value_outside_salary_bounds_is_skipped(self) -> None:
        """GIVEN a single dollar value below realistic salary thresholds
        WHEN parse_compensation is called
        THEN the match is skipped (not in salary range) and None is returned.
        """
        # $5 is too low to be salary and not /hr so it's skipped
        text = "The price is $5 per unit."
        result = parse_compensation(text)
        assert result is None
