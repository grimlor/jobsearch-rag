"""ZipRecruiter adapter extraction tests.

Spec classes:
    TestZipRecruiterJsonExtraction — JSON blob extraction from React SPA
    TestHtmlToText — HTML-to-plain-text conversion for embedding
    TestRealWorldExtraction — Regression guard against production HTML
    TestAuthenticate — Session verification and Cloudflare/CAPTCHA detection
    TestSearch — SERP navigation, card extraction, and click-through enrichment
    TestExtractDetailPassthrough — extract_detail passthrough when full_text populated

Validates the JSON-based extraction strategy against ZipRecruiter's
React SPA structure where all job data is embedded in a
``<script id="js_variables">`` JSON blob, and the search() method that
enriches listings via SERP click-through.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobsearch_rag.adapters.base import JobListing
from jobsearch_rag.adapters.ziprecruiter import (
    ZipRecruiterAdapter,
    card_to_listing,
    extract_jd_text,
    extract_js_variables,
    html_to_text,
    parse_job_cards,
)
from jobsearch_rag.errors import ActionableError

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent / "fixtures"
_SERP_FIXTURE = _FIXTURES / "ziprecruiter_results.html"
_JD_FIXTURE = _FIXTURES / "ziprecruiter_jd.html"
_REAL_FIXTURE = _FIXTURES / "ziprecruiter_serp_real.html"


# ---------------------------------------------------------------------------
# TestZipRecruiterJsonExtraction
# ---------------------------------------------------------------------------


class TestZipRecruiterJsonExtraction:
    """REQUIREMENT: Job data is extracted from ZipRecruiter's js_variables JSON blob.

    WHO: The ZipRecruiter adapter extraction pipeline
    WHAT: (1) The ``<script id="js_variables">`` tag is located and parsed as JSON
          (2) job cards are extracted from ``hydrateJobCardsResponse.jobCards``
          (3) each card maps correctly to a ``JobListing``
          (4) full JD HTML is converted to plain text from ``htmlFullDescription``
          (5) a missing js_variables script tag raises ActionableError naming the selector
          (6) malformed JSON in js_variables raises ActionableError identifying the parse problem
          (7) missing hydrateJobCardsResponse key returns an empty card list
          (8) salary metadata is included when pay data is present, omitted when absent
          (9) URL construction falls back to applyButtonConfig when canonical path is missing
          (10) SERP card full_text is empty after extraction (populated during search)
          (11) extract_jd_text returns empty string when description or details response is absent
          (12) card_to_listing maps the card title to the listing title
          (13) card_to_listing extracts the company name from the nested company object
          (14) card_to_listing maps location.displayName to the listing location
          (15) card_to_listing uses listingKey as the listing external_id
          (16) card_to_listing builds the full URL from the canonical path
          (17) card_to_listing sets the board field to 'ziprecruiter'
          (18) real fixture parsing produces the expected count of job cards
          (19) extract_jd_text returns clean text from real production fixture data
          (20) extract_jd_text returns empty string when getJobDetailsResponse is absent
          (21) real SERP fixture parses with expected session authentication flags
    WHY: ZipRecruiter is a React SPA — the HTML body contains only empty
         hydration roots.  CSS selectors against rendered DOM will never match.
         All data lives in the embedded JSON blob.

    MOCK BOUNDARY:
        Mock:  (none — pure functions operating on fixture HTML)
        Real:  extract_js_variables, parse_job_cards, card_to_listing,
               extract_jd_text, html_to_text
        Never: Patch extraction functions or fixture file contents
    """

    def test_extract_js_variables_from_serp_fixture(self) -> None:
        """
        GIVEN a synthetic SERP HTML fixture
        WHEN extract_js_variables is called
        THEN the js_variables JSON blob is parsed successfully.
        """
        # Given: synthetic fixture HTML
        html = _SERP_FIXTURE.read_text()

        # When: extract
        js_vars = extract_js_variables(html)

        # Then: valid dict with expected key
        assert isinstance(js_vars, dict), f"Expected dict, got {type(js_vars)}"
        assert "hydrateJobCardsResponse" in js_vars, "Missing hydrateJobCardsResponse key"

    def test_extract_js_variables_from_real_fixture(self) -> None:
        """
        GIVEN a real ZipRecruiter SERP HTML fixture
        WHEN extract_js_variables is called
        THEN the js_variables JSON blob is parsed with expected keys.
        """
        # Given: real fixture HTML
        html = _REAL_FIXTURE.read_text()

        # When: extract
        js_vars = extract_js_variables(html)

        # Then: valid dict with expected keys
        assert isinstance(js_vars, dict), f"Expected dict, got {type(js_vars)}"
        assert "hydrateJobCardsResponse" in js_vars, "Missing hydrateJobCardsResponse key"
        assert js_vars.get("isLoggedIn") is False, (
            f"Expected isLoggedIn=False, got {js_vars.get('isLoggedIn')!r}"
        )

    def test_extract_js_variables_missing_script_tag_names_the_selector(self) -> None:
        """
        GIVEN HTML without a js_variables script tag
        WHEN extract_js_variables is called
        THEN an ActionableError naming the missing selector is raised.
        """
        # Given: HTML with no JSON blob
        html = "<html><body><p>No JSON here</p></body></html>"

        # When/Then: raises with actionable guidance
        with pytest.raises(ActionableError) as exc_info:
            extract_js_variables(html)

        # Then: error references selector
        err = exc_info.value
        assert "js_variables" in err.error, f"Error should name selector: {err.error!r}"
        assert err.suggestion is not None, "Should include suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"

    def test_extract_js_variables_malformed_json_identifies_parse_problem(self) -> None:
        """
        GIVEN HTML with a js_variables script tag containing invalid JSON
        WHEN extract_js_variables is called
        THEN an ActionableError identifying the parse problem is raised.
        """
        # Given: malformed JSON
        html = '<script id="js_variables" type="application/json">{not valid json</script>'

        # When/Then: raises with parse guidance
        with pytest.raises(ActionableError) as exc_info:
            extract_js_variables(html)

        # Then: error identifies parse failure
        err = exc_info.value
        assert "Failed to parse" in err.error, f"Error should mention parse: {err.error!r}"
        assert err.suggestion is not None, "Should include suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"

    def test_parse_job_cards_returns_correct_count_from_synthetic(self) -> None:
        """
        GIVEN a synthetic SERP fixture with 3 job cards
        WHEN parse_job_cards is called
        THEN exactly 3 cards are returned.
        """
        # Given: synthetic fixture
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)

        # When: parse
        cards = parse_job_cards(js_vars)

        # Then: 3 cards
        assert len(cards) == 3, f"Expected 3 cards, got {len(cards)}"

    def test_parse_job_cards_returns_20_from_real_fixture(self) -> None:
        """
        GIVEN a real ZipRecruiter SERP fixture
        WHEN parse_job_cards is called
        THEN 20 job cards are returned (page 1 default).
        """
        # Given: real fixture
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)

        # When: parse
        cards = parse_job_cards(js_vars)

        # Then: 20 cards
        assert len(cards) == 20, f"Expected 20 cards, got {len(cards)}"

    def test_parse_job_cards_handles_missing_key_gracefully(self) -> None:
        """
        GIVEN an empty js_vars dict without hydrateJobCardsResponse
        WHEN parse_job_cards is called
        THEN an empty list is returned.
        """
        # When: parse empty dict
        cards = parse_job_cards({})

        # Then: empty list
        assert cards == [], f"Expected empty list, got {cards}"

    def test_card_to_listing_maps_title(self) -> None:
        """
        GIVEN a parsed job card from the synthetic fixture
        WHEN card_to_listing is called
        THEN the title maps to 'Staff Platform Architect'.
        """
        # Given: first card from fixture
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)

        # When: convert
        listing = card_to_listing(cards[0])

        # Then: title matches
        assert listing.title == "Staff Platform Architect", (
            f"Expected 'Staff Platform Architect', got {listing.title!r}"
        )

    def test_card_to_listing_maps_company(self) -> None:
        """
        GIVEN a parsed job card with a nested company object
        WHEN card_to_listing is called
        THEN the company name is extracted correctly.
        """
        # Given: first card from fixture
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)

        # When: convert
        listing = card_to_listing(cards[0])

        # Then: company matches
        assert listing.company == "Acme Corp", f"Expected 'Acme Corp', got {listing.company!r}"

    def test_card_to_listing_maps_location(self) -> None:
        """
        GIVEN a parsed job card with location.displayName
        WHEN card_to_listing is called
        THEN the location is mapped correctly.
        """
        # Given: first card from fixture
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)

        # When: convert
        listing = card_to_listing(cards[0])

        # Then: location matches
        assert listing.location == "Remote (USA)", (
            f"Expected 'Remote (USA)', got {listing.location!r}"
        )

    def test_card_to_listing_maps_external_id_from_listing_key(self) -> None:
        """
        GIVEN a parsed job card with a listingKey
        WHEN card_to_listing is called
        THEN the listingKey becomes the external_id.
        """
        # Given: first card from fixture
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)

        # When: convert
        listing = card_to_listing(cards[0])

        # Then: external_id matches listingKey
        assert listing.external_id == "abc123key", (
            f"Expected 'abc123key', got {listing.external_id!r}"
        )

    def test_card_to_listing_builds_full_url_from_canonical_path(self) -> None:
        """
        GIVEN a parsed job card with a canonical URL path
        WHEN card_to_listing is called
        THEN the full URL is built from the ZipRecruiter base URL.
        """
        # Given: first card from fixture
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)

        # When: convert
        listing = card_to_listing(cards[0])

        # Then: URL starts with expected base
        assert listing.url.startswith("https://www.ziprecruiter.com/c/Acme-Corp/"), (
            f"Unexpected URL prefix: {listing.url!r}"
        )

    def test_card_to_listing_includes_salary_metadata(self) -> None:
        """
        GIVEN a parsed job card with pay.minAnnual/maxAnnual
        WHEN card_to_listing is called
        THEN salary_range is included in metadata.
        """
        # Given: first card from fixture (has salary)
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)

        # When: convert
        listing = card_to_listing(cards[0])

        # Then: salary_range present with correct values
        assert "salary_range" in listing.metadata, "Missing salary_range metadata"
        assert "$210,000" in listing.metadata["salary_range"], (
            f"Missing $210,000 in salary_range: {listing.metadata['salary_range']!r}"
        )
        assert "$250,000" in listing.metadata["salary_range"], (
            f"Missing $250,000 in salary_range: {listing.metadata['salary_range']!r}"
        )

    def test_card_to_listing_omits_salary_when_missing(self) -> None:
        """
        GIVEN a parsed job card without pay data
        WHEN card_to_listing is called
        THEN salary_range is absent from metadata.
        """
        # Given: third card (no pay info)
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)

        # When: convert card without pay
        listing = card_to_listing(cards[2])

        # Then: no salary_range
        assert "salary_range" not in listing.metadata, (
            f"salary_range should be absent, got {listing.metadata.get('salary_range')!r}"
        )

    def test_card_to_listing_falls_back_to_apply_button_url(self) -> None:
        """
        GIVEN a card missing rawCanonicalZipJobPageUrl at top level
        WHEN card_to_listing is called
        THEN the URL is built from applyButtonConfig fallback.
        """
        # Given: card with URL only in applyButtonConfig
        card = {
            "listingKey": "fallback-key",
            "title": "Test Role",
            "company": {"name": "TestCo"},
            "location": {"displayName": "Remote"},
            "applyButtonConfig": {
                "rawCanonicalZipJobPageUrl": "/c/TestCo/Job/Test-Role/-in-Remote?jid=fallback",
            },
        }

        # When: convert
        listing = card_to_listing(card)

        # Then: URL from applyButtonConfig
        assert (
            listing.url
            == "https://www.ziprecruiter.com/c/TestCo/Job/Test-Role/-in-Remote?jid=fallback"
        ), f"Unexpected URL: {listing.url!r}"

    def test_card_to_listing_sets_board_name(self) -> None:
        """
        GIVEN any parsed job card
        WHEN card_to_listing is called
        THEN board is set to 'ziprecruiter'.
        """
        # Given: first card from fixture
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)

        # When: convert
        listing = card_to_listing(cards[0])

        # Then: board is ziprecruiter
        assert listing.board == "ziprecruiter", f"Expected 'ziprecruiter', got {listing.board!r}"

    def test_card_to_listing_full_text_is_empty(self) -> None:
        """
        GIVEN a job card from search results
        WHEN card_to_listing is called
        THEN full_text is empty (detail extraction is a separate step).
        """
        # Given: first card from fixture
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)

        # When: convert
        listing = card_to_listing(cards[0])

        # Then: empty full_text
        assert listing.full_text == "", f"Expected empty full_text, got {listing.full_text!r}"

    def test_extract_jd_text_from_synthetic_fixture(self) -> None:
        """
        GIVEN a synthetic JD HTML fixture with htmlFullDescription
        WHEN extract_jd_text is called
        THEN clean plain text is returned with HTML tags stripped.
        """
        # Given: JD fixture
        html = _JD_FIXTURE.read_text()
        js_vars = extract_js_variables(html)

        # When: extract
        text = extract_jd_text(js_vars)

        # Then: content present, HTML stripped
        assert "Staff Platform Architect" in text, "Missing title in JD text"
        assert "distributed systems" in text, "Missing content in JD text"
        assert "<div>" not in text, "HTML tags should be stripped"
        assert "<strong>" not in text, "HTML tags should be stripped"

    def test_extract_jd_text_from_real_fixture(self) -> None:
        """
        GIVEN a real SERP fixture with getJobDetailsResponse
        WHEN extract_jd_text is called
        THEN clean text is returned with Spotnana content.
        """
        # Given: real fixture
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)

        # When: extract
        text = extract_jd_text(js_vars)

        # Then: Spotnana content present, HTML stripped
        assert "Spotnana" in text, "Missing Spotnana in JD text"
        assert "travel industry" in text, "Missing content in JD text"
        assert "<div>" not in text, "HTML tags should be stripped"

    def test_extract_jd_text_returns_empty_when_no_description(self) -> None:
        """
        GIVEN js_vars with getJobDetailsResponse but no htmlFullDescription
        WHEN extract_jd_text is called
        THEN an empty string is returned.
        """
        # Given: no description in job details
        js_vars: dict[str, Any] = {"getJobDetailsResponse": {"jobDetails": {}}}

        # When: extract
        text = extract_jd_text(js_vars)

        # Then: empty
        assert text == "", f"Expected empty string, got {text!r}"

    def test_extract_jd_text_returns_empty_when_no_details_response(self) -> None:
        """
        GIVEN js_vars without getJobDetailsResponse entirely
        WHEN extract_jd_text is called
        THEN an empty string is returned.
        """
        # When: extract from empty dict
        text = extract_jd_text({})

        # Then: empty
        assert text == "", f"Expected empty string, got {text!r}"


# ---------------------------------------------------------------------------
# TestHtmlToText
# ---------------------------------------------------------------------------


class TestHtmlToText:
    """REQUIREMENT: HTML job descriptions are converted to clean plain text.

    WHO: The JD extraction pipeline before RAG embedding
    WHAT: (1) The system removes all HTML tags and leaves only the text content.
          (2) The system collapses consecutive whitespace into single spaces.
          (3) The system preserves the text content of all list items in nested list HTML.
          (4) The system returns an empty string when given an empty HTML string.
          (5) The system returns plain text unchanged when no HTML tags are present.
    WHY: RAG embeddings work best on clean text — HTML artifacts degrade
         retrieval quality

    MOCK BOUNDARY:
        Mock:  (none — pure function tests)
        Real:  html_to_text
        Never: Patch html_to_text internals
    """

    def test_strips_html_tags(self) -> None:
        """
        GIVEN HTML with strong and em tags
        WHEN html_to_text is called
        THEN all tags are removed, leaving only text.
        """
        # Given: HTML with tags
        html = "<div><strong>Hello</strong> <em>world</em></div>"

        # When: convert
        result = html_to_text(html)

        # Then: no HTML tags remain
        assert "<" not in result, f"HTML tags should be stripped: {result!r}"
        assert "Hello" in result, "Missing 'Hello' in result"
        assert "world" in result, "Missing 'world' in result"

    def test_normalizes_whitespace(self) -> None:
        """
        GIVEN HTML with multiple consecutive whitespace characters
        WHEN html_to_text is called
        THEN whitespace is collapsed to single spaces.
        """
        # Given: HTML with excess whitespace
        html = "<p>Hello    world</p>"

        # When: convert
        result = html_to_text(html)

        # Then: normalized
        assert "  " not in result, f"Double spaces remain: {result!r}"
        assert "Hello world" in result, f"Expected 'Hello world' in: {result!r}"

    def test_handles_nested_lists(self) -> None:
        """
        GIVEN HTML with ul/li list items
        WHEN html_to_text is called
        THEN all list item text is preserved.
        """
        # Given: HTML list
        html = "<ul><li>Item one</li><li>Item two</li></ul>"

        # When: convert
        result = html_to_text(html)

        # Then: text preserved
        assert "Item one" in result, f"Missing 'Item one' in: {result!r}"
        assert "Item two" in result, f"Missing 'Item two' in: {result!r}"

    def test_handles_empty_string(self) -> None:
        """
        GIVEN an empty HTML string
        WHEN html_to_text is called
        THEN an empty string is returned.
        """
        # Then: empty in, empty out
        assert html_to_text("") == "", "Empty HTML should produce empty text"

    def test_handles_plain_text_passthrough(self) -> None:
        """
        GIVEN plain text without any HTML tags
        WHEN html_to_text is called
        THEN the text passes through unchanged.
        """
        # Given: plain text
        text = "Just plain text, nothing fancy."

        # Then: passthrough
        assert html_to_text(text) == text, "Plain text should pass through unchanged"


# ---------------------------------------------------------------------------
# TestRealWorldExtraction
# ---------------------------------------------------------------------------


class TestRealWorldExtraction:
    """REQUIREMENT: Extraction works correctly against real ZipRecruiter data.

    WHO: The adapter maintainer validating against production HTML
    WHAT: (1) The system produces exactly 20 listings from the real ZipRecruiter SERP fixture.
          (2) The system converts the first real SERP card into a Spotnana listing whose title contains 'Senior Staff Software Engineer'.
          (3) The system assigns every real listing a non-empty unique external_id.
          (4) The system assigns every real listing a URL that starts with the ZipRecruiter base URL.
          (5) The system captures a salary_range for the Spotnana listing that includes $210,000-$240,000.
          (6) The system extracts a maxPages value of 2 from js_vars in the real SERP fixture.
          (7) The system extracts a totalListings value of 30 from listJobKeysResponse in the real SERP fixture.
          (8) The system extracts substantial Spotnana job description content from the real SERP fixture.
          (9) The system detects an unauthenticated session by reporting isLoggedIn as False and isLoggedOut as True.
    WHY: Synthetic fixtures can drift from production reality — these tests
         serve as a regression guard against ZipRecruiter structure changes

    MOCK BOUNDARY:
        Mock:  (none — pure functions operating on real fixture HTML)
        Real:  extract_js_variables, parse_job_cards, card_to_listing,
               extract_jd_text
        Never: Patch extraction functions or modify fixture data
    """

    def test_real_serp_produces_20_listings(self) -> None:
        """
        GIVEN the real ZipRecruiter SERP fixture
        WHEN all cards are converted to listings
        THEN exactly 20 listings are produced.
        """
        # Given: real fixture
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)

        # When: convert all
        listings = [card_to_listing(c) for c in cards]

        # Then: 20 listings
        assert len(listings) == 20, f"Expected 20 listings, got {len(listings)}"

    def test_real_first_listing_is_spotnana(self) -> None:
        """
        GIVEN the real SERP fixture
        WHEN the first card is converted
        THEN the company is Spotnana and title contains 'Senior Staff Software Engineer'.
        """
        # Given: real fixture, first card
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)

        # When: convert first
        listing = card_to_listing(cards[0])

        # Then: Spotnana
        assert listing.company == "Spotnana", f"Expected 'Spotnana', got {listing.company!r}"
        assert "Senior Staff Software Engineer" in listing.title, (
            f"Expected title with 'Senior Staff Software Engineer', got {listing.title!r}"
        )

    def test_real_listings_have_valid_external_ids(self) -> None:
        """
        GIVEN all 20 real listings
        WHEN external_ids are checked
        THEN every listing has a non-empty unique external_id.
        """
        # Given: all real listings
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        listings = [card_to_listing(c) for c in cards]

        # Then: all non-empty and unique
        assert all(item.external_id for item in listings), "All listings must have external_id"
        assert len(set(item.external_id for item in listings)) == 20, (
            "All 20 external_ids should be unique"
        )

    def test_real_listings_have_valid_urls(self) -> None:
        """
        GIVEN all 20 real listings
        WHEN URLs are checked
        THEN every URL starts with the ZipRecruiter base URL.
        """
        # Given: all real listings
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        listings = [card_to_listing(c) for c in cards]

        # Then: all URLs valid
        assert all(item.url.startswith("https://www.ziprecruiter.com/c/") for item in listings), (
            "All URLs should start with ZipRecruiter base"
        )

    def test_real_spotnana_salary_range(self) -> None:
        """
        GIVEN the Spotnana listing from the real fixture
        WHEN salary metadata is checked
        THEN salary_range includes $210,000-$240,000.
        """
        # Given: Spotnana listing
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        listing = card_to_listing(cards[0])

        # Then: salary range present
        assert "salary_range" in listing.metadata, "Missing salary_range metadata"
        assert "$210,000" in listing.metadata["salary_range"], (
            f"Missing $210,000: {listing.metadata['salary_range']!r}"
        )
        assert "$240,000" in listing.metadata["salary_range"], (
            f"Missing $240,000: {listing.metadata['salary_range']!r}"
        )

    def test_real_max_pages_is_extracted(self) -> None:
        """
        GIVEN the real SERP fixture
        WHEN maxPages is read from js_vars
        THEN the value is 2.
        """
        # Given: real fixture
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)

        # Then: maxPages is 2
        assert js_vars.get("maxPages") == 2, (
            f"Expected maxPages=2, got {js_vars.get('maxPages')!r}"
        )

    def test_real_total_listings_count(self) -> None:
        """
        GIVEN the real SERP fixture
        WHEN totalListings is read from listJobKeysResponse
        THEN the value is 30.
        """
        # Given: real fixture
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        list_response = js_vars.get("listJobKeysResponse", {})

        # Then: 30 total
        assert list_response.get("totalListings") == 30, (
            f"Expected 30 total listings, got {list_response.get('totalListings')!r}"
        )

    def test_real_jd_html_is_extractable(self) -> None:
        """
        GIVEN the real SERP fixture
        WHEN extract_jd_text is called
        THEN substantial Spotnana content is returned.
        """
        # Given: real fixture
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)

        # When: extract
        text = extract_jd_text(js_vars)

        # Then: substantial content
        assert "Spotnana" in text, "Missing Spotnana in JD text"
        assert len(text) > 500, f"JD should be substantial, got {len(text)} chars"

    def test_real_unauthenticated_session_is_detected(self) -> None:
        """
        GIVEN the real fixture captured without login
        WHEN session flags are checked
        THEN isLoggedIn is False and isLoggedOut is True.
        """
        # Given: real fixture
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)

        # Then: unauthenticated
        assert js_vars.get("isLoggedIn") is False, (
            f"Expected isLoggedIn=False, got {js_vars.get('isLoggedIn')!r}"
        )
        assert js_vars.get("isLoggedOut") is True, (
            f"Expected isLoggedOut=True, got {js_vars.get('isLoggedOut')!r}"
        )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_listing(
    external_id: str = "test-1",
    title: str = "Staff Architect",
    company: str = "Acme Corp",
    full_text: str = "",
    short_description: str = "",
) -> JobListing:
    """Create a test JobListing."""
    metadata: dict[str, str] = {}
    if short_description:
        metadata["short_description"] = short_description
    return JobListing(
        board="ziprecruiter",
        external_id=external_id,
        title=title,
        company=company,
        location="Remote",
        url=f"https://www.ziprecruiter.com/c/{external_id}",
        full_text=full_text,
        metadata=metadata,
    )


def _make_mock_page(
    *,
    content_html: str = "",
    title: str = "Jobs - ZipRecruiter",
    url: str = "https://www.ziprecruiter.com/jobs-search",
    card_count: int = 0,
    panel_texts: list[str] | None = None,
    captcha: bool = False,
) -> MagicMock:
    """Create a mock Playwright page with configurable I/O responses.

    This is the I/O boundary for all ZipRecruiter adapter tests.
    ``page`` represents the browser automation interface.
    """
    mock_page = MagicMock()
    mock_page.goto = AsyncMock()
    mock_page.title = AsyncMock(return_value=title)
    mock_page.content = AsyncMock(return_value=content_html)
    mock_page.url = url
    mock_page.query_selector = AsyncMock(return_value=MagicMock() if captcha else None)

    # Card locator
    mock_card_locator = MagicMock()
    mock_card_locator.count = AsyncMock(return_value=card_count)
    cards: list[MagicMock] = []
    for _ in range(card_count):
        card = MagicMock()
        card.click = AsyncMock()
        cards.append(card)
    mock_card_locator.nth.side_effect = lambda i: cards[i] if i < len(cards) else MagicMock()  # pyright: ignore[reportUnknownLambdaType]

    # Panel locator
    mock_panel_locator = MagicMock()
    mock_panel_locator.wait_for = AsyncMock()
    if panel_texts:
        mock_panel_locator.inner_text = AsyncMock(side_effect=panel_texts)
    else:
        mock_panel_locator.inner_text = AsyncMock(return_value="")

    mock_page.locator.side_effect = lambda sel: {  # pyright: ignore[reportUnknownLambdaType]
        "[class*='job_result'] article": mock_card_locator,
        "[data-testid='job-details-scroll-container']": mock_panel_locator,
    }[sel]

    return mock_page


# ---------------------------------------------------------------------------
# TestAuthenticate
# ---------------------------------------------------------------------------


class TestAuthenticate:
    """REQUIREMENT: Session verification detects expired sessions and Cloudflare blocks.

    WHO: The pipeline runner during the authenticate step
    WHAT: (1) The system completes authentication without error when the session is valid.
          (2) The system raises an ActionableError that mentions CAPTCHA and suggests manual solve when it detects a CAPTCHA.
          (3) The system raises an ActionableError that states the session expired when authentication is redirected to `/login`.
          (4) The system raises an ActionableError that states the session expired when authentication is redirected to `/sign-in`.
          (5) The system raises an ActionableError that mentions Cloudflare and suggests headed mode when the Cloudflare challenge persists until timeout.
    WHY: Starting a search against an expired or blocked session wastes
         time and produces zero results — fail fast with actionable advice

    MOCK BOUNDARY:
        Mock:  Playwright page (browser I/O via _make_mock_page helper),
               asyncio.sleep + _CF_WAIT_TIMEOUT (timing control in
               cloudflare timeout test)
        Real:  ZipRecruiterAdapter.authenticate
        Never: Patch internal request/response parsing logic
    """

    @pytest.mark.asyncio
    async def test_authenticate_succeeds_with_valid_session(self) -> None:
        """
        GIVEN a mock page that loads without Cloudflare/CAPTCHA/redirect
        WHEN authenticate is called
        THEN it completes without error.
        """
        # Given: valid mock page
        adapter = ZipRecruiterAdapter()
        page = _make_mock_page()

        # When: authenticate
        await adapter.authenticate(page)

        # Then: page.goto was called
        page.goto.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_authenticate_captcha_suggests_manual_solve(self) -> None:
        """
        GIVEN a mock page with a CAPTCHA element detected
        WHEN authenticate is called
        THEN an ActionableError suggesting manual-solve is raised.
        """
        # Given: page with CAPTCHA
        adapter = ZipRecruiterAdapter()
        page = _make_mock_page(captcha=True)

        # When/Then: raises with CAPTCHA guidance
        with pytest.raises(ActionableError) as exc_info:
            await adapter.authenticate(page)

        # Then: actionable error
        err = exc_info.value
        assert "CAPTCHA" in err.error, f"Error should mention CAPTCHA: {err.error!r}"
        assert err.suggestion is not None, "Should include suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"

    @pytest.mark.asyncio
    async def test_authenticate_login_redirect_tells_operator_to_reauthenticate(self) -> None:
        """
        GIVEN a mock page redirected to /login
        WHEN authenticate is called
        THEN an ActionableError about session expiration is raised.
        """
        # Given: page redirected to login
        adapter = ZipRecruiterAdapter()
        page = _make_mock_page(url="https://www.ziprecruiter.com/login")

        # When/Then: raises session expired
        with pytest.raises(ActionableError) as exc_info:
            await adapter.authenticate(page)

        # Then: actionable error
        err = exc_info.value
        assert "Session expired" in err.error, f"Error should mention expiration: {err.error!r}"
        assert err.suggestion is not None, "Should include suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"

    @pytest.mark.asyncio
    async def test_authenticate_sign_in_redirect_tells_operator_to_reauthenticate(self) -> None:
        """
        GIVEN a mock page redirected to /sign-in
        WHEN authenticate is called
        THEN an ActionableError about session expiration is raised.
        """
        # Given: page redirected to sign-in
        adapter = ZipRecruiterAdapter()
        page = _make_mock_page(url="https://www.ziprecruiter.com/sign-in")

        # When/Then: raises session expired
        with pytest.raises(ActionableError) as exc_info:
            await adapter.authenticate(page)

        # Then: actionable error
        err = exc_info.value
        assert "Session expired" in err.error, f"Error should mention expiration: {err.error!r}"
        assert err.suggestion is not None, "Should include suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"

    @pytest.mark.asyncio
    async def test_authenticate_cloudflare_timeout_suggests_headed_mode(self) -> None:
        """
        GIVEN a mock page showing a persistent Cloudflare challenge
        WHEN authenticate times out waiting for resolution
        THEN an ActionableError suggesting headed mode is raised.
        """
        # Given: Cloudflare challenge page
        adapter = ZipRecruiterAdapter()
        page = _make_mock_page(title="Just a moment...")

        # When/Then: raises after timeout (asyncio.sleep mocked so 15 iterations are instant)
        with (
            patch(
                "jobsearch_rag.adapters.ziprecruiter.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            pytest.raises(ActionableError) as exc_info,
        ):
            await adapter.authenticate(page)

        # Then: actionable error about Cloudflare
        err = exc_info.value
        assert "Cloudflare" in err.error, f"Error should mention Cloudflare: {err.error!r}"
        assert err.suggestion is not None, "Should include suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"


# ---------------------------------------------------------------------------
# TestSearch
# ---------------------------------------------------------------------------


class TestSearch:
    """REQUIREMENT: search() navigates SERP pages, extracts cards, and enriches via click-through.

    WHO: The pipeline runner collecting listings from ZipRecruiter
    WHAT: (1) The system returns three listings from the fixture and includes 'Staff Platform Architect' as the first title.
          (2) The system populates the first card's full text from `htmlFullDescription` in `js_variables`.
          (3) The system enriches cards after the first with click-through panel text.
          (4) The system falls back to `shortDescription` when panel text is too short.
          (5) The system falls back to `shortDescription` for a card when click-through fails.
          (6) The system stops pagination at the fixture's maximum page limit even when a higher `max_pages` value is requested.
          (7) The system stops pagination when a later page has no cards and returns listings only from earlier populated pages.
          (8) The system returns an empty listing list when it cannot load the `js_variables` data.
          (9) The system skips an unparseable card and returns the remaining valid listings.
          (10) The system returns listings from JSON without attempting click-through when no cards exist in the DOM.
          (11) The system appends `&page=2` when paginating a search URL that already contains a query string.
          (12) The system stops click-through at the DOM card count while still returning all JSON listings.
          (13) The system appends `?page=2` when paginating a search URL that has no query string.
          (14) The system enriches the first card through click-through instead of pre-populating it when no job details exist in `js_variables`.
          (15) The system respects the `max_pages` argument by navigating only the requested number of pages.
    WHY: ZipRecruiter is a React SPA — all data lives in an embedded
         JSON blob.  Click-through on the SERP avoids Cloudflare
         challenges that would block per-URL navigation.

    MOCK BOUNDARY:
        Mock:  Playwright page (browser I/O via _make_mock_page helper),
               random.uniform (delay control)
        Real:  ZipRecruiterAdapter.search, extract_js_variables,
               parse_job_cards, card_to_listing
        Never: Patch internal extraction logic (except card_to_listing in
               one resilience test that verifies per-card error isolation)
    """

    @pytest.mark.asyncio
    async def test_search_returns_listings_from_fixture(self) -> None:
        """
        GIVEN a mock page with the synthetic SERP fixture and 3 card articles
        WHEN search is called with max_pages=1
        THEN 3 listings are returned with the first titled 'Staff Platform Architect'.
        """
        # Given: mock page with fixture HTML
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()
        panel_text = "A detailed job description for testing " * 10  # > 100 chars

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=3,
            panel_texts=[panel_text, panel_text],  # cards 1 & 2 (card 0 from js_vars)
        )

        # When: search
        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        # Then: 3 listings with correct first title
        assert len(listings) == 3, f"Expected 3 listings, got {len(listings)}"
        assert listings[0].title == "Staff Platform Architect", (
            f"Expected 'Staff Platform Architect', got {listings[0].title!r}"
        )

    @pytest.mark.asyncio
    async def test_search_first_card_gets_jd_from_js_variables(self) -> None:
        """
        GIVEN a mock page with the synthetic SERP fixture
        WHEN search is called
        THEN the first card's full_text comes from htmlFullDescription in js_variables.
        """
        # Given: mock page with fixture HTML
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()
        panel_text = "Panel text for remaining cards with enough content " * 5

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=3,
            panel_texts=[panel_text, panel_text],
        )

        # When: search
        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        # Then: first card from js_variables, not panel
        assert "distributed systems" in listings[0].full_text, (
            f"First card should have js_variables content: {listings[0].full_text[:100]!r}"
        )
        assert "Staff Platform Architect" in listings[0].full_text, (
            "First card should include title from js_variables"
        )

    @pytest.mark.asyncio
    async def test_search_remaining_cards_enriched_by_click_through(self) -> None:
        """
        GIVEN a mock page with 3 cards and panel texts for cards 1 and 2
        WHEN search is called
        THEN cards after the first are enriched via click-through panel text.
        """
        # Given: mock page with distinct panel text per card
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()
        panel_b_text = "Role B detailed job description for panel " * 10
        panel_c_text = "Role C detailed job description for panel " * 10

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=3,
            panel_texts=[panel_b_text, panel_c_text],
        )

        # When: search
        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        # Then: panel text enriched
        assert "Role B" in listings[1].full_text, "Card 1 should have panel B text"
        assert "Role C" in listings[2].full_text, "Card 2 should have panel C text"

    @pytest.mark.asyncio
    async def test_search_falls_back_to_short_desc_when_panel_too_short(self) -> None:
        """
        GIVEN a mock page where panel text is under 100 chars
        WHEN search is called
        THEN shortDescription fallback is used for those cards.
        """
        # Given: mock page with short panel text
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=3,
            panel_texts=["Too short", "Also short"],
        )

        # When: search
        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        # Then: card 0 from js_variables, cards 1&2 from shortDescription fallback
        assert "distributed systems" in listings[0].full_text, (
            "Card 0 should have js_variables content"
        )
        assert "Senior Staff Engineer at Globex Corporation" in listings[1].full_text, (
            "Card 1 should have shortDescription fallback"
        )
        assert "Principal Software Architect at Initech" in listings[2].full_text, (
            "Card 2 should have shortDescription fallback"
        )

    @pytest.mark.asyncio
    async def test_search_falls_back_on_click_failure(self) -> None:
        """
        GIVEN a mock page where clicking the second card raises TimeoutError
        WHEN search is called
        THEN shortDescription fallback is used for that card.
        """
        # Given: mock page with failing card click
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=3,
        )

        # Make the second card's click raise
        card_locator = page.locator("[class*='job_result'] article")
        failing_card = MagicMock()
        failing_card.click = AsyncMock(side_effect=TimeoutError("click timeout"))
        original_nth = card_locator.nth.side_effect
        card_locator.nth.side_effect = lambda i: failing_card if i == 1 else original_nth(i)  # pyright: ignore[reportUnknownLambdaType]

        panel_text = "Panel text for card C with enough detail " * 10
        panel_locator = page.locator("[data-testid='job-details-scroll-container']")
        panel_locator.inner_text = AsyncMock(return_value=panel_text)

        # When: search
        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        # Then: card 1 used shortDescription fallback
        assert "Senior Staff Engineer at Globex Corporation" in listings[1].full_text, (
            "Card 1 should have shortDescription fallback after click failure"
        )

    @pytest.mark.asyncio
    async def test_search_stops_at_max_pages(self) -> None:
        """
        GIVEN a fixture with maxPages=1
        WHEN search is called with max_pages=5
        THEN only 1 page is navigated.
        """
        # Given: fixture with maxPages=1
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()  # maxPages=1

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=3,
            panel_texts=["Long panel text " * 20, "Long panel text " * 20],
        )

        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=5)

        # maxPages=1 in fixture, so only 1 page navigated despite max_pages=5
        assert page.goto.await_count == 1, (
            f"Expected 1 page navigated, got {page.goto.await_count}"
        )
        assert len(listings) == 3, f"Expected 3 listings, got {len(listings)}"

    @pytest.mark.asyncio
    async def test_search_stops_when_no_cards_on_page(self) -> None:
        """
        GIVEN a multi-page SERP where page 2 has zero cards
        WHEN search is called with max_pages=3
        THEN pagination stops after 2 pages with only page 1 listings.
        """
        # Given: page 1 has cards, page 2 has zero cards
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()
        no_cards_fixture = (
            "<html><head><title>Jobs</title></head><body>"
            '<script id="js_variables" type="application/json">'
            '{"hydrateJobCardsResponse":{"jobCards":[]},"maxPages":3}'
            "</script></body></html>"
        )

        call_count = {"goto": 0}
        original_goto = AsyncMock()

        async def _tracked_goto(url: str, **kwargs: object) -> None:
            call_count["goto"] += 1
            await original_goto(url, **kwargs)

        page = _make_mock_page(
            content_html=fixture_html.replace('"maxPages": 1', '"maxPages": 3'),
            card_count=3,
            panel_texts=["Long panel text " * 20, "Long panel text " * 20],
        )
        page.goto = _tracked_goto
        # After first page, return no-cards HTML
        page.content = AsyncMock(
            side_effect=[
                fixture_html.replace('"maxPages": 1', '"maxPages": 3'),
                no_cards_fixture,
            ]
        )

        # When: search is called with max_pages=3
        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=3)

        # Then: pagination stops after page 1 (2 goto calls: page 1 + page 2 with no cards)
        assert call_count["goto"] == 2, f"Expected 2 page navigations, got {call_count['goto']}"
        assert len(listings) == 3, f"Expected 3 listings from page 1 only, got {len(listings)}"

    @pytest.mark.asyncio
    async def test_search_stops_on_js_variables_failure(self) -> None:
        """
        GIVEN a page with no js_variables JSON blob
        WHEN search is called
        THEN an empty listing list is returned.
        """
        # Given: broken HTML with no JSON
        adapter = ZipRecruiterAdapter()
        broken_html = "<html><head><title>Jobs</title></head><body>No JSON here</body></html>"

        page = _make_mock_page(content_html=broken_html)

        # When: search
        listings = await adapter.search(page, "https://zr.com/search", max_pages=3)

        # Then: empty
        assert listings == [], f"Expected empty list, got {len(listings)} listings"

    @pytest.mark.asyncio
    async def test_search_skips_unparseable_card(self) -> None:
        """
        GIVEN a mock page where card_to_listing raises on the second card
        WHEN search is called
        THEN the unparseable card is skipped and 2 listings are returned.
        """
        # Given: mock page with 3 cards, one will fail
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()
        panel_text = "Long panel text for testing with enough chars " * 10

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=3,
            panel_texts=[panel_text, panel_text, panel_text],
        )

        # Make card_to_listing raise on the second card
        original_card_to_listing = card_to_listing

        call_count = {"calls": 0}

        def _failing_card_to_listing(card: dict[str, Any]) -> JobListing:
            call_count["calls"] += 1
            if call_count["calls"] == 2:
                raise ValueError("Unparseable card")
            return original_card_to_listing(card)

        # When: search with failing card_to_listing
        with (
            patch(
                "jobsearch_rag.adapters.ziprecruiter.card_to_listing",
                side_effect=_failing_card_to_listing,
            ),
            patch(
                "jobsearch_rag.adapters.ziprecruiter.random.uniform",
                return_value=0.0,
            ),
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        # Then: 3 cards, 1 failed → 2 listings
        assert len(listings) == 2, f"Expected 2 listings (1 skipped), got {len(listings)}"

    @pytest.mark.asyncio
    async def test_search_no_cards_in_dom_returns_listings_without_click(self) -> None:
        """
        GIVEN a mock page with 0 card articles in the DOM
        WHEN search is called
        THEN JSON listings are returned without panel click-through.
        """
        # Given: no card articles in DOM
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=0,  # No card articles in DOM
        )

        # When: search
        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        # Then: 3 listings from JSON, first has full_text from js_vars
        assert len(listings) == 3, f"Expected 3 listings from JSON, got {len(listings)}"
        assert listings[0].full_text.strip(), "First card should have text from js_variables"

    @pytest.mark.asyncio
    async def test_search_paginates_url_correctly(self) -> None:
        """
        GIVEN a fixture with maxPages=2 and a query URL containing '?'
        WHEN search is called with max_pages=2
        THEN page 2 URL appends '&page=2'.
        """
        # Given: a fixture with maxPages=2 and a query URL containing '?'
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text().replace('"maxPages": 1', '"maxPages": 2')
        empty_fixture = (
            "<html><head><title>Jobs</title></head><body>"
            '<script id="js_variables" type="application/json">'
            '{"hydrateJobCardsResponse":{"jobCards":[]},"maxPages":2}'
            "</script></body></html>"
        )

        goto_urls: list[str] = []

        async def _track_goto(url: str, **kwargs: object) -> None:
            goto_urls.append(url)

        page = _make_mock_page(content_html=fixture_html, card_count=3)
        page.goto = _track_goto
        page.content = AsyncMock(side_effect=[fixture_html, empty_fixture])

        panel_text = "Enough panel text for the test " * 10

        # Rebuild locators for panel
        mock_panel = MagicMock()
        mock_panel.wait_for = AsyncMock()
        mock_panel.inner_text = AsyncMock(return_value=panel_text)
        mock_card = MagicMock()
        mock_card.count = AsyncMock(return_value=3)
        for_cards: list[MagicMock] = []
        for _ in range(3):
            c = MagicMock()
            c.click = AsyncMock()
            for_cards.append(c)
        mock_card.nth.side_effect = lambda i: for_cards[i] if i < len(for_cards) else MagicMock()  # pyright: ignore[reportUnknownLambdaType]
        page.locator.side_effect = lambda sel: {  # pyright: ignore[reportUnknownLambdaType]
            "[class*='job_result'] article": mock_card,
            "[data-testid='job-details-scroll-container']": mock_panel,
        }[sel]

        # When: search is called with max_pages=2
        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            await adapter.search(page, "https://zr.com/search?q=architect", max_pages=2)

        # Then: page 2 URL appends '&page=2' since URL already has '?'
        assert goto_urls[0] == "https://zr.com/search?q=architect", (
            f"Expected page 1 URL unchanged, got {goto_urls[0]}"
        )
        assert goto_urls[1] == "https://zr.com/search?q=architect&page=2", (
            f"Expected '&page=2' appended, got {goto_urls[1]}"
        )

    @pytest.mark.asyncio
    async def test_search_card_index_exceeds_dom_count(self) -> None:
        """
        GIVEN more JSON listings than DOM card articles
        WHEN search is called
        THEN click-through stops at DOM count but all JSON listings are returned.
        """
        # Given: 3 cards in JSON but only 1 in DOM
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()  # 3 cards in JSON

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=1,  # Only 1 card article in DOM
        )

        # When: search
        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        # Then: all 3 returned, first has full text
        assert len(listings) == 3, f"Expected 3 listings, got {len(listings)}"
        assert listings[0].full_text.strip(), "First card should have text from js_variables"

    @pytest.mark.asyncio
    async def test_search_pagination_uses_question_mark_separator(self) -> None:
        """
        GIVEN a query URL without '?' (no query string)
        WHEN search paginates to page 2
        THEN '?page=2' is used as the separator.
        """
        # Given: a query URL without '?' (no query string)
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text().replace('"maxPages": 1', '"maxPages": 2')
        empty_fixture = (
            "<html><head><title>Jobs</title></head><body>"
            '<script id="js_variables" type="application/json">'
            '{"hydrateJobCardsResponse":{"jobCards":[]},"maxPages":2}'
            "</script></body></html>"
        )

        goto_urls: list[str] = []

        async def _track_goto(url: str, **kwargs: object) -> None:
            goto_urls.append(url)

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=3,
            panel_texts=["Long panel text " * 20, "Long panel text " * 20],
        )
        page.goto = _track_goto
        page.content = AsyncMock(side_effect=[fixture_html, empty_fixture])

        # When: search paginates to page 2
        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            await adapter.search(page, "https://zr.com/search", max_pages=2)

        # Then: '?page=2' is used since URL has no existing query string
        assert goto_urls[1] == "https://zr.com/search?page=2", (
            f"Expected '?page=2' separator, got {goto_urls[1]}"
        )

    @pytest.mark.asyncio
    async def test_search_first_card_not_prepopulated_when_no_jd_in_js_vars(self) -> None:
        """
        GIVEN js_variables without getJobDetailsResponse
        WHEN search is called
        THEN the first card is enriched via click-through, not pre-populated.
        """
        # Given: minimal fixture with cards but no getJobDetailsResponse
        adapter = ZipRecruiterAdapter()
        # Build a minimal fixture with cards but no getJobDetailsResponse
        fixture_html = (
            "<html><head><title>Jobs</title></head><body>"
            '<script id="js_variables" type="application/json">'
            '{"hydrateJobCardsResponse":{"jobCards":['
            '{"listingKey":"k1","title":"Role A","company":{"name":"Co"},'
            '"location":{"displayName":"Remote"},'
            '"shortDescription":"Short desc A",'
            '"applyButtonConfig":{"rawCanonicalZipJobPageUrl":"/c/Co/Job/A?jid=k1"}}'
            ']},"maxPages":1}'
            "</script></body></html>"
        )
        panel_text = "Full detail text for role A with plenty of content " * 5

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=1,
            panel_texts=[panel_text],
        )

        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        assert len(listings) == 1, f"Expected 1 listing, got {len(listings)}"
        # Full text came from click-through panel, not js_variables
        assert "Full detail text" in listings[0].full_text, (
            "Full text should come from panel click-through"
        )

    @pytest.mark.asyncio
    async def test_search_respects_max_pages_argument(self) -> None:
        """
        GIVEN a fixture reporting maxPages=3
        WHEN search is called with max_pages=1
        THEN only 1 page is navigated.
        """
        # Given: fixture with maxPages=3 but max_pages=1
        adapter = ZipRecruiterAdapter()
        # Fixture says maxPages=3 but we pass max_pages=1
        fixture_html = (
            "<html><head><title>Jobs</title></head><body>"
            '<script id="js_variables" type="application/json">'
            '{"hydrateJobCardsResponse":{"jobCards":['
            '{"listingKey":"k1","title":"Role A","company":{"name":"Co"},'
            '"location":{"displayName":"Remote"},'
            '"shortDescription":"Short desc A",'
            '"applyButtonConfig":{"rawCanonicalZipJobPageUrl":"/c/Co/Job/A?jid=k1"}}'
            ']},"maxPages":3}'
            "</script></body></html>"
        )
        panel_text = "Full detail text with enough content for tests " * 5

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=1,
            panel_texts=[panel_text],
        )

        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        # Only 1 page navigated despite site saying maxPages=3
        page.goto.assert_awaited_once()
        assert len(listings) == 1, f"Expected 1 listing, got {len(listings)}"


# ---------------------------------------------------------------------------
# TestExtractDetailPassthrough
# ---------------------------------------------------------------------------


class TestExtractDetailPassthrough:
    """REQUIREMENT: extract_detail is a passthrough when full_text is populated.

    WHO: The pipeline runner calling extract_detail after search
    WHAT: (1) The system returns the listing unchanged when full_text is already populated.
          (2) The system populates full_text from the short_description fallback when full_text is empty.
          (3) The system leaves full_text empty when no short_description fallback is available.
    WHY: SERP click-through makes per-URL extraction unnecessary — the
         runner calls extract_detail out of protocol compliance, but
         ZipRecruiter does all extraction during search()

    MOCK BOUNDARY:
        Mock:  Playwright page (unused — passthrough behaviour)
        Real:  ZipRecruiterAdapter.extract_detail
        Never: Patch extract_detail internals or full_text assignment
    """

    @pytest.mark.asyncio
    async def test_passthrough_when_full_text_present(self) -> None:
        """
        GIVEN a listing with full_text already populated
        WHEN extract_detail is called
        THEN the listing is returned unchanged.
        """
        # Given: listing with full_text
        adapter = ZipRecruiterAdapter()
        listing = _make_listing(full_text="Full JD already populated")
        mock_page = MagicMock()

        # When: extract_detail
        result = await adapter.extract_detail(mock_page, listing)

        # Then: unchanged
        assert result is listing, "Should return same object"
        assert result.full_text == "Full JD already populated", (
            f"full_text should be unchanged: {result.full_text!r}"
        )

    @pytest.mark.asyncio
    async def test_fallback_when_full_text_empty(self) -> None:
        """
        GIVEN a listing with empty full_text and a short_description in metadata
        WHEN extract_detail is called
        THEN the shortDescription fallback populates full_text.
        """
        # Given: listing with empty full_text, short_description in metadata
        adapter = ZipRecruiterAdapter()
        listing = _make_listing(full_text="", short_description="Python role at company")
        mock_page = MagicMock()

        # When: extract_detail
        result = await adapter.extract_detail(mock_page, listing)

        # Then: fallback applied
        assert "Python role at company" in result.full_text, (
            f"short_description fallback missing: {result.full_text!r}"
        )
        assert "Staff Architect at Acme Corp" in result.full_text, (
            f"Title/company context missing: {result.full_text!r}"
        )

    @pytest.mark.asyncio
    async def test_empty_when_no_fallback_available(self) -> None:
        """
        GIVEN a listing with empty full_text and no short_description
        WHEN extract_detail is called
        THEN full_text remains empty.
        """
        # Given: listing with empty full_text, no fallback
        adapter = ZipRecruiterAdapter()
        listing = _make_listing(full_text="")
        mock_page = MagicMock()

        # When: extract_detail
        result = await adapter.extract_detail(mock_page, listing)

        # Then: still empty
        assert result.full_text == "", f"Expected empty full_text, got {result.full_text!r}"
