"""Export tests — Markdown and CSV.

Maps to BDD specs: TestMarkdownExport, TestCSVExport
(Specs to be completed — placeholder structure)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# TestMarkdownExport
# ---------------------------------------------------------------------------


class TestMarkdownExport:
    """REQUIREMENT: Markdown output is human-readable and complete.

    WHO: The operator reviewing results in Obsidian or a text editor
    WHAT: Output includes title, company, board, final score, component scores,
          disqualifier status, and a clickable URL; disqualified roles are
          omitted from output (score 0.0 = excluded); results are sorted
          descending by final score; run summary appears at top
    WHY: The ranked list is the primary product of every run —
         missing fields or wrong sort order defeats the purpose
    """

    def test_output_includes_all_required_fields_per_listing(self) -> None: ...
    def test_listings_are_sorted_descending_by_final_score(self) -> None: ...
    def test_disqualified_roles_are_not_present_in_output(self) -> None: ...
    def test_score_explanation_shows_all_three_component_scores(self) -> None: ...
    def test_run_summary_appears_at_top_of_output(self) -> None: ...
    def test_run_summary_includes_total_found_scored_excluded_deduplicated(self) -> None: ...
    def test_url_is_present_and_non_empty_for_every_listing(self) -> None: ...
    def test_empty_result_set_produces_output_with_summary_and_no_table(self) -> None: ...


# ---------------------------------------------------------------------------
# TestCSVExport
# ---------------------------------------------------------------------------


class TestCSVExport:
    """REQUIREMENT: CSV export is valid and importable by standard tools.

    WHO: The operator importing results into a spreadsheet or ATS tracker
    WHAT: Output is valid CSV with a header row; all required columns are present;
          JD text is excluded from CSV (too large); special characters in
          company names or titles do not break CSV formatting
    WHY: A CSV with unescaped commas or missing headers silently corrupts
         on import — the operator may not notice
    """

    def test_csv_has_header_row_with_all_required_columns(self) -> None: ...
    def test_full_jd_text_is_not_included_in_csv_output(self) -> None: ...
    def test_company_names_with_commas_are_properly_quoted(self) -> None: ...
    def test_empty_result_set_produces_header_only_csv(self) -> None: ...
    def test_csv_row_count_matches_scored_listing_count(self) -> None: ...
