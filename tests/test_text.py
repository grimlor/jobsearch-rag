"""Tests for :mod:`jobsearch_rag.text` — shared text-processing utilities.

Covers :class:`TestSlugify`.
"""

from __future__ import annotations

from jobsearch_rag.text import MAX_SLUG_LEN, slugify


class TestSlugify:
    """
    REQUIREMENT: Text is converted to filesystem-safe slugs consistently
    across CLI, export, and pipeline layers.

    WHO: Any module that builds file paths from human-readable text
         (JD exporter, review session, CLI read_jd_text)
    WHAT: Slugify lowercases, strips unsafe characters, collapses
          whitespace/underscores to hyphens, trims leading/trailing
          hyphens, and truncates to a configurable max length
    WHY: Inconsistent slugification across modules would cause file-not-found
         errors when one module writes a JD file and another tries to open it
    """

    def test_basic_title_is_lowercased_and_hyphenated(self) -> None:
        """
        When a mixed-case title with spaces is slugified
        Then the result is lowercase with hyphens replacing spaces
        """
        result = slugify("Senior Staff Engineer")

        assert result == "senior-staff-engineer", (
            f"Expected lowercase hyphenated slug, got {result!r}"
        )

    def test_special_characters_are_stripped(self) -> None:
        """
        When text contains parentheses, em-dashes, and other punctuation
        Then non-alphanumeric characters (except hyphens) are removed
        """
        result = slugify("Senior Staff Engineer — Platform (Remote)")

        assert result == "senior-staff-engineer-platform-remote", (
            f"Expected special chars stripped, got {result!r}"
        )

    def test_underscores_collapse_to_single_hyphen(self) -> None:
        """
        When text contains underscores
        Then they are replaced with hyphens like whitespace
        """
        result = slugify("some_company_name")

        assert result == "some-company-name", f"Expected underscores→hyphens, got {result!r}"

    def test_consecutive_whitespace_collapses_to_single_hyphen(self) -> None:
        """
        When text contains multiple consecutive spaces or mixed whitespace
        Then the run collapses to a single hyphen
        """
        result = slugify("too   many   spaces")

        assert result == "too-many-spaces", f"Expected collapsed whitespace, got {result!r}"

    def test_leading_and_trailing_hyphens_are_stripped(self) -> None:
        """
        When text would produce leading or trailing hyphens
        Then those hyphens are removed
        """
        result = slugify("--leading and trailing--")

        assert result == "leading-and-trailing", f"Expected stripped edges, got {result!r}"

    def test_truncation_at_default_max_length(self) -> None:
        """
        When text exceeds MAX_SLUG_LEN characters after slugification
        Then the result is truncated to MAX_SLUG_LEN
        """
        long_text = "a " * (MAX_SLUG_LEN + 10)

        result = slugify(long_text)

        assert len(result) <= MAX_SLUG_LEN, f"Expected len ≤ {MAX_SLUG_LEN}, got {len(result)}"

    def test_custom_max_len_overrides_default(self) -> None:
        """
        When a custom max_len is provided
        Then the result is truncated to that length instead of the default
        """
        result = slugify("this-is-a-long-slug-that-should-be-truncated", max_len=10)

        assert len(result) <= 10, f"Expected len ≤ 10, got {len(result)}"
        assert result == "this-is-a-", f"Expected first 10 chars, got {result!r}"

    def test_empty_string_returns_empty(self) -> None:
        """
        When an empty string is slugified
        Then the result is an empty string
        """
        result = slugify("")

        assert result == "", f"Expected empty string, got {result!r}"

    def test_already_clean_slug_is_unchanged(self) -> None:
        """
        When text is already a valid slug
        Then the result is identical to the input (lowercased)
        """
        result = slugify("already-clean")

        assert result == "already-clean", f"Expected no change, got {result!r}"

    def test_company_and_title_round_trip_matches_jd_filename_convention(self) -> None:
        """
        When company and title are slugified and assembled into the JD filename pattern
        Then the result matches the NNN_company_title.md convention used by JDFileExporter
        """
        # Given — realistic values from a job listing
        company = "Acme Corp."
        title = "Staff Engineer — Platform"
        rank = 7

        # When
        filename = f"{rank:03d}_{slugify(company)}_{slugify(title)}.md"

        # Then
        assert filename == "007_acme-corp_staff-engineer-platform.md", (
            f"Expected JD filename convention, got {filename!r}"
        )
