"""ZipRecruiter adapter extraction tests.

Maps to BDD specs: TestZipRecruiterJsonExtraction
Validates the JSON-based extraction strategy against ZipRecruiter's
React SPA structure where all job data is embedded in a
``<script id="js_variables">`` JSON blob.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jobsearch_rag.adapters.ziprecruiter import (
    _html_to_text,
    card_to_listing,
    extract_jd_text,
    extract_js_variables,
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
    WHAT: The ``<script id="js_variables">`` tag is located and parsed as JSON;
          job cards are extracted from ``hydrateJobCardsResponse.jobCards``;
          each card maps correctly to a ``JobListing``; full JD HTML is
          converted to plain text from ``htmlFullDescription``
    WHY: ZipRecruiter is a React SPA — the HTML body contains only empty
         hydration roots.  CSS selectors against rendered DOM will never match.
         All data lives in the embedded JSON blob.
    """

    def test_extract_js_variables_from_serp_fixture(self) -> None:
        """The js_variables JSON blob is successfully parsed from a synthetic SERP page."""
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)

        assert isinstance(js_vars, dict)
        assert "hydrateJobCardsResponse" in js_vars

    def test_extract_js_variables_from_real_fixture(self) -> None:
        """The js_variables JSON blob is successfully parsed from a real ZipRecruiter SERP page."""
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)

        assert isinstance(js_vars, dict)
        assert "hydrateJobCardsResponse" in js_vars
        assert js_vars.get("isLoggedIn") is False

    def test_extract_js_variables_raises_on_missing_script_tag(self) -> None:
        """A page without the js_variables script tag raises a descriptive ParseError."""
        html = "<html><body><p>No JSON here</p></body></html>"

        with pytest.raises(ActionableError) as exc_info:
            extract_js_variables(html)

        assert "js_variables" in exc_info.value.error

    def test_extract_js_variables_raises_on_malformed_json(self) -> None:
        """Malformed JSON inside the script tag raises a ParseError with decode details."""
        html = '<script id="js_variables" type="application/json">{not valid json</script>'

        with pytest.raises(ActionableError) as exc_info:
            extract_js_variables(html)

        assert "Failed to parse" in exc_info.value.error

    def test_parse_job_cards_returns_correct_count_from_synthetic(self) -> None:
        """The synthetic fixture contains exactly 3 job cards."""
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)

        assert len(cards) == 3

    def test_parse_job_cards_returns_20_from_real_fixture(self) -> None:
        """The real ZipRecruiter SERP contains 20 job cards on page 1."""
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)

        assert len(cards) == 20

    def test_parse_job_cards_handles_missing_key_gracefully(self) -> None:
        """An empty js_vars dict returns an empty card list rather than crashing."""
        cards = parse_job_cards({})

        assert cards == []

    def test_card_to_listing_maps_title(self) -> None:
        """The job title is correctly mapped from the card dict to JobListing.title."""
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        listing = card_to_listing(cards[0])

        assert listing.title == "Staff Platform Architect"

    def test_card_to_listing_maps_company(self) -> None:
        """The company name is extracted from the nested company object."""
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        listing = card_to_listing(cards[0])

        assert listing.company == "Acme Corp"

    def test_card_to_listing_maps_location(self) -> None:
        """The location displayName is mapped to JobListing.location."""
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        listing = card_to_listing(cards[0])

        assert listing.location == "Remote (USA)"

    def test_card_to_listing_maps_external_id_from_listing_key(self) -> None:
        """The listingKey serves as the external_id — not a URL-parsed slug."""
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        listing = card_to_listing(cards[0])

        assert listing.external_id == "abc123key"

    def test_card_to_listing_builds_full_url_from_canonical_path(self) -> None:
        """The canonical URL path is prepended with the base URL to form a full URL."""
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        listing = card_to_listing(cards[0])

        assert listing.url.startswith("https://www.ziprecruiter.com/c/Acme-Corp/")

    def test_card_to_listing_includes_salary_metadata(self) -> None:
        """Salary range from pay.minAnnual/maxAnnual is included in metadata."""
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        listing = card_to_listing(cards[0])

        assert "salary_range" in listing.metadata
        assert "$210,000" in listing.metadata["salary_range"]
        assert "$250,000" in listing.metadata["salary_range"]

    def test_card_to_listing_omits_salary_when_missing(self) -> None:
        """Cards without pay data do not include salary_range in metadata."""
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        # Third card has no pay info
        listing = card_to_listing(cards[2])

        assert "salary_range" not in listing.metadata

    def test_card_to_listing_sets_board_name(self) -> None:
        """Every listing is tagged with board='ziprecruiter'."""
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        listing = card_to_listing(cards[0])

        assert listing.board == "ziprecruiter"

    def test_card_to_listing_full_text_is_empty(self) -> None:
        """Search results produce listings with empty full_text — detail extraction is a separate step."""
        html = _SERP_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        listing = card_to_listing(cards[0])

        assert listing.full_text == ""

    def test_extract_jd_text_from_synthetic_fixture(self) -> None:
        """Full JD text is extracted and cleaned from htmlFullDescription in the synthetic fixture."""
        html = _JD_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        text = extract_jd_text(js_vars)

        assert "Staff Platform Architect" in text
        assert "distributed systems" in text
        # HTML tags should be stripped
        assert "<div>" not in text
        assert "<strong>" not in text

    def test_extract_jd_text_from_real_fixture(self) -> None:
        """Full JD text is extracted from the real SERP fixture's getJobDetailsResponse."""
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        text = extract_jd_text(js_vars)

        assert "Spotnana" in text
        assert "travel industry" in text
        # HTML tags should be stripped
        assert "<div>" not in text

    def test_extract_jd_text_returns_empty_when_no_description(self) -> None:
        """When htmlFullDescription is absent, extract_jd_text returns empty string."""
        js_vars = {"getJobDetailsResponse": {"jobDetails": {}}}
        text = extract_jd_text(js_vars)

        assert text == ""

    def test_extract_jd_text_returns_empty_when_no_details_response(self) -> None:
        """When getJobDetailsResponse is missing entirely, returns empty string."""
        text = extract_jd_text({})

        assert text == ""


# ---------------------------------------------------------------------------
# TestHtmlToText
# ---------------------------------------------------------------------------


class TestHtmlToText:
    """REQUIREMENT: HTML job descriptions are converted to clean plain text.

    WHO: The JD extraction pipeline before RAG embedding
    WHAT: HTML tags are stripped; whitespace is normalized; nested lists
          are flattened; the output is clean plain text suitable for embedding
    WHY: RAG embeddings work best on clean text — HTML artifacts degrade
         retrieval quality
    """

    def test_strips_html_tags(self) -> None:
        """All HTML tags are removed, leaving only text content."""
        html = "<div><strong>Hello</strong> <em>world</em></div>"
        result = _html_to_text(html)

        assert "<" not in result
        assert "Hello" in result
        assert "world" in result

    def test_normalizes_whitespace(self) -> None:
        """Multiple whitespace characters are collapsed to single spaces."""
        html = "<p>Hello    world</p>"
        result = _html_to_text(html)

        assert "  " not in result
        assert "Hello world" in result

    def test_handles_nested_lists(self) -> None:
        """List items in ul/li structures are preserved as text."""
        html = "<ul><li>Item one</li><li>Item two</li></ul>"
        result = _html_to_text(html)

        assert "Item one" in result
        assert "Item two" in result

    def test_handles_empty_string(self) -> None:
        """An empty HTML string returns an empty string."""
        assert _html_to_text("") == ""

    def test_handles_plain_text_passthrough(self) -> None:
        """Plain text without any HTML tags passes through unchanged."""
        text = "Just plain text, nothing fancy."
        assert _html_to_text(text) == text


# ---------------------------------------------------------------------------
# TestRealWorldExtraction
# ---------------------------------------------------------------------------


class TestRealWorldExtraction:
    """REQUIREMENT: Extraction works correctly against real ZipRecruiter data.

    WHO: The adapter maintainer validating against production HTML
    WHAT: Real SERP data from an unauthenticated session parses correctly;
          all 20 job cards produce valid JobListing objects; salary ranges,
          company names, and canonical URLs are correctly extracted
    WHY: Synthetic fixtures can drift from production reality — these tests
         serve as a regression guard against ZipRecruiter structure changes
    """

    def test_real_serp_produces_20_listings(self) -> None:
        """The real SERP fixture yields exactly 20 job listings from page 1."""
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        listings = [card_to_listing(c) for c in cards]

        assert len(listings) == 20

    def test_real_first_listing_is_spotnana(self) -> None:
        """The first listing in the real fixture is from Spotnana."""
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        listing = card_to_listing(cards[0])

        assert listing.company == "Spotnana"
        assert "Senior Staff Software Engineer" in listing.title

    def test_real_listings_have_valid_external_ids(self) -> None:
        """Every real listing has a non-empty listingKey as its external_id."""
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        listings = [card_to_listing(c) for c in cards]

        assert all(item.external_id for item in listings)
        assert len(set(item.external_id for item in listings)) == 20  # All unique

    def test_real_listings_have_valid_urls(self) -> None:
        """Every real listing has a URL starting with the ZipRecruiter base URL."""
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        listings = [card_to_listing(c) for c in cards]

        assert all(item.url.startswith("https://www.ziprecruiter.com/c/") for item in listings)

    def test_real_spotnana_salary_range(self) -> None:
        """The Spotnana listing includes salary info: $210K-$240K."""
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        cards = parse_job_cards(js_vars)
        listing = card_to_listing(cards[0])

        assert "salary_range" in listing.metadata
        assert "$210,000" in listing.metadata["salary_range"]
        assert "$240,000" in listing.metadata["salary_range"]

    def test_real_max_pages_is_extracted(self) -> None:
        """The real fixture reports maxPages=2 for pagination control."""
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)

        assert js_vars.get("maxPages") == 2

    def test_real_total_listings_count(self) -> None:
        """The real fixture reports 30 total listings across all pages."""
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        list_response = js_vars.get("listJobKeysResponse", {})

        assert list_response.get("totalListings") == 30

    def test_real_jd_html_is_extractable(self) -> None:
        """The full JD HTML for the selected card is available and contains Spotnana content."""
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)
        text = extract_jd_text(js_vars)

        assert "Spotnana" in text
        assert len(text) > 500  # JD should be substantial

    def test_real_unauthenticated_session_is_detected(self) -> None:
        """The real fixture was captured without login — isLoggedIn should be False."""
        html = _REAL_FIXTURE.read_text()
        js_vars = extract_js_variables(html)

        assert js_vars.get("isLoggedIn") is False
        assert js_vars.get("isLoggedOut") is True
