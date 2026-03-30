"""
Input validation and security boundary tests.

Spec classes:
    TestJobListingValidation — construction rejects malformed input at the
                               adapter boundary where untrusted web content
                               enters the system
"""

from __future__ import annotations

import pytest

from jobsearch_rag.adapters.base import JobListing

# Public API surface (from src/jobsearch_rag/adapters/base):
#   JobListing(board, external_id, title, company, location, url, full_text,
#              posted_at=None, raw_html=None, comp_min=None, comp_max=None,
#              comp_source=None, comp_text=None, metadata={})
#   _sanitize_filename_field(value: str) -> str   (module-level helper)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED = {
    "board": "test-board",
    "external_id": "sec-001",
    "title": "Staff Platform Architect",
    "company": "Acme Corp",
    "location": "Remote (USA)",
    "url": "https://example.org/job/sec-001",
    "full_text": "A normal job description.",
}


def _make(**overrides: object) -> JobListing:
    """Build a JobListing with sensible defaults; override any field."""
    fields = {**_REQUIRED, **overrides}
    return JobListing(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestJobListingValidation
# ---------------------------------------------------------------------------


class TestJobListingValidation:
    """
    REQUIREMENT: JobListing construction rejects malformed input at the
    boundary where untrusted web content enters the system.

    WHO: The scoring pipeline; the exporter constructing output filenames
    WHAT: (1) a listing whose full_text exceeds 500,000 characters is rejected with a ValueError
          (2) full_text at exactly 500,000 characters constructs without error
          (3) path traversal sequences in title are replaced, leaving no path separators
          (4) path traversal sequences in company are replaced, leaving no path separators
          (5) filesystem-unsafe characters (< > : " | ? *) are stripped from title
          (6) well-formed content constructs without error and all fields are accessible
          (7) sanitisation does not affect optional fields, which remain at their defaults
    WHY: An oversized JD causes an Ollama out-of-memory crash with no
         actionable error. A title containing '../' can write files outside
         the output directory — a path traversal vulnerability even in a
         local single-user context

    MOCK BOUNDARY:
        Mock:  nothing — JobListing is a data class with validation in __init__
        Real:  JobListing constructor called with explicit field values
        Never: Bypass the constructor; test all validation paths through
               JobListing(...) with adversarial input strings
    """

    def test_full_text_exceeding_500k_chars_raises_value_error(self) -> None:
        """
        GIVEN a JobListing constructed with full_text of 500,001 characters
        WHEN the constructor executes
        THEN a ValueError is raised.
        """
        # Given: oversized full_text
        oversized = "x" * 500_001

        # Then: construction raises ValueError
        with pytest.raises(ValueError, match="full_text exceeds maximum length"):
            _make(full_text=oversized)

    def test_full_text_at_500k_chars_constructs_without_error(self) -> None:
        """
        GIVEN a JobListing constructed with full_text of exactly 500,000 characters
        WHEN the constructor executes
        THEN no error is raised.
        """
        # Given: exactly at the boundary
        at_limit = "x" * 500_000

        # When: construction succeeds
        listing = _make(full_text=at_limit)

        # Then: full_text is preserved
        assert len(listing.full_text) == 500_000

    def test_path_traversal_in_title_is_removed(self) -> None:
        r"""
        GIVEN a JobListing constructed with title containing '../'
        WHEN the listing is inspected
        THEN the title field contains no '/' or '\' characters.
        """
        # Given: path traversal in title
        listing = _make(title="../../etc/passwd Engineer")

        # Then: no path separators remain
        assert "/" not in listing.title, "title must not contain '/'"
        assert "\\" not in listing.title, "title must not contain '\\'"

    def test_path_traversal_in_company_name_is_removed(self) -> None:
        """
        GIVEN a JobListing constructed with company containing '../../etc'
        WHEN the listing is inspected
        THEN the company field contains no path separator characters.
        """
        # Given: path traversal in company
        listing = _make(company="../../etc/shadow Corp")

        # Then: no path separators remain
        assert "/" not in listing.company, "company must not contain '/'"
        assert "\\" not in listing.company, "company must not contain '\\'"

    def test_filesystem_unsafe_characters_are_stripped_from_title(self) -> None:
        """
        GIVEN a title containing characters from the set < > : " | ? *
        WHEN the listing is constructed
        THEN the title field contains none of those characters.
        """
        # Given: filesystem-unsafe characters in title
        listing = _make(title='Staff <Eng> "Platform" | Arch? *Senior*')

        # Then: none of the unsafe chars remain
        unsafe = set('<>:"|?*')
        remaining = unsafe & set(listing.title)
        assert not remaining, f"unsafe characters remain: {remaining}"

    def test_well_formed_listing_constructs_without_error(self) -> None:
        """
        GIVEN all required fields with normal content
        WHEN a JobListing is constructed
        THEN no error is raised and all fields are accessible.
        """
        # When: normal construction
        listing = _make()

        # Then: all required fields accessible
        assert listing.board == "test-board"
        assert listing.external_id == "sec-001"
        assert listing.title == "Staff Platform Architect"
        assert listing.company == "Acme Corp"
        assert listing.location == "Remote (USA)"
        assert listing.url == "https://example.org/job/sec-001"
        assert listing.full_text == "A normal job description."

    def test_sanitisation_does_not_affect_optional_fields(self) -> None:
        """
        GIVEN a listing with None for optional fields
        WHEN the listing is constructed
        THEN optional fields remain None without error.
        """
        # When: listing with defaults for optional fields
        listing = _make()

        # Then: optional fields are their defaults
        assert listing.posted_at is None
        assert listing.raw_html is None
        assert listing.comp_min is None
        assert listing.comp_max is None
        assert listing.comp_source is None
        assert listing.comp_text is None
        assert listing.metadata == {}
