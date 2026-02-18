"""ZipRecruiter adapter extraction tests.

Maps to BDD specs: TestZipRecruiterJsonExtraction, TestHtmlToText,
TestRealWorldExtraction, TestAuthenticate, TestSearch, TestExtractDetailPassthrough

Validates the JSON-based extraction strategy against ZipRecruiter's
React SPA structure where all job data is embedded in a
``<script id="js_variables">`` JSON blob, and the search() method that
enriches listings via SERP click-through.
"""

from __future__ import annotations

from pathlib import Path
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

    def test_card_to_listing_falls_back_to_apply_button_url(self) -> None:
        """When rawCanonicalZipJobPageUrl is missing at top level, falls back to applyButtonConfig."""
        card = {
            "listingKey": "fallback-key",
            "title": "Test Role",
            "company": {"name": "TestCo"},
            "location": {"displayName": "Remote"},
            "applyButtonConfig": {
                "rawCanonicalZipJobPageUrl": "/c/TestCo/Job/Test-Role/-in-Remote?jid=fallback",
            },
        }
        listing = card_to_listing(card)

        assert listing.url == "https://www.ziprecruiter.com/c/TestCo/Job/Test-Role/-in-Remote?jid=fallback"

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
        result = html_to_text(html)

        assert "<" not in result
        assert "Hello" in result
        assert "world" in result

    def test_normalizes_whitespace(self) -> None:
        """Multiple whitespace characters are collapsed to single spaces."""
        html = "<p>Hello    world</p>"
        result = html_to_text(html)

        assert "  " not in result
        assert "Hello world" in result

    def test_handles_nested_lists(self) -> None:
        """List items in ul/li structures are preserved as text."""
        html = "<ul><li>Item one</li><li>Item two</li></ul>"
        result = html_to_text(html)

        assert "Item one" in result
        assert "Item two" in result

    def test_handles_empty_string(self) -> None:
        """An empty HTML string returns an empty string."""
        assert html_to_text("") == ""

    def test_handles_plain_text_passthrough(self) -> None:
        """Plain text without any HTML tags passes through unchanged."""
        text = "Just plain text, nothing fancy."
        assert html_to_text(text) == text


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
    mock_page.query_selector = AsyncMock(
        return_value=MagicMock() if captcha else None
    )

    # Card locator
    mock_card_locator = MagicMock()
    mock_card_locator.count = AsyncMock(return_value=card_count)
    cards = []
    for _ in range(card_count):
        card = MagicMock()
        card.click = AsyncMock()
        cards.append(card)
    mock_card_locator.nth.side_effect = lambda i: cards[i] if i < len(cards) else MagicMock()

    # Panel locator
    mock_panel_locator = MagicMock()
    mock_panel_locator.wait_for = AsyncMock()
    if panel_texts:
        mock_panel_locator.inner_text = AsyncMock(side_effect=panel_texts)
    else:
        mock_panel_locator.inner_text = AsyncMock(return_value="")

    mock_page.locator.side_effect = lambda sel: {
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
    WHAT: A valid session passes silently; a Cloudflare challenge that
          resolves within timeout succeeds; a persistent Cloudflare
          challenge raises with headless suggestion; a CAPTCHA raises
          with manual-solve suggestion; a login redirect raises with
          session-expired suggestion
    WHY: Starting a search against an expired or blocked session wastes
         time and produces zero results — fail fast with actionable advice
    """

    @pytest.mark.asyncio
    async def test_authenticate_succeeds_with_valid_session(self) -> None:
        """A page that loads without Cloudflare/CAPTCHA/login-redirect passes."""
        adapter = ZipRecruiterAdapter()
        page = _make_mock_page()

        await adapter.authenticate(page)

        page.goto.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_authenticate_raises_on_captcha(self) -> None:
        """A CAPTCHA element on the page raises with manual-solve suggestion."""
        adapter = ZipRecruiterAdapter()
        page = _make_mock_page(captcha=True)

        with pytest.raises(ActionableError) as exc_info:
            await adapter.authenticate(page)

        assert "CAPTCHA" in exc_info.value.error

    @pytest.mark.asyncio
    async def test_authenticate_raises_on_login_redirect(self) -> None:
        """Being redirected to /login raises a session-expired error."""
        adapter = ZipRecruiterAdapter()
        page = _make_mock_page(url="https://www.ziprecruiter.com/login")

        with pytest.raises(ActionableError) as exc_info:
            await adapter.authenticate(page)

        assert "Session expired" in exc_info.value.error

    @pytest.mark.asyncio
    async def test_authenticate_raises_on_sign_in_redirect(self) -> None:
        """Being redirected to /sign-in also raises a session-expired error."""
        adapter = ZipRecruiterAdapter()
        page = _make_mock_page(url="https://www.ziprecruiter.com/sign-in")

        with pytest.raises(ActionableError) as exc_info:
            await adapter.authenticate(page)

        assert "Session expired" in exc_info.value.error

    @pytest.mark.asyncio
    async def test_authenticate_raises_on_cloudflare_timeout(self) -> None:
        """A Cloudflare challenge that never resolves raises with headless suggestion."""
        adapter = ZipRecruiterAdapter()
        page = _make_mock_page(title="Just a moment...")

        with (
            patch(
                "jobsearch_rag.adapters.ziprecruiter.asyncio.sleep",
                new_callable=AsyncMock,
            ),
            patch(
                "jobsearch_rag.adapters.ziprecruiter._CF_WAIT_TIMEOUT",
                1,
            ),
            pytest.raises(ActionableError) as exc_info,
        ):
            await adapter.authenticate(page)

        assert "Cloudflare" in exc_info.value.error


# ---------------------------------------------------------------------------
# TestSearch
# ---------------------------------------------------------------------------


class TestSearch:
    """REQUIREMENT: search() navigates SERP pages, extracts cards, and enriches via click-through.

    WHO: The pipeline runner collecting listings from ZipRecruiter
    WHAT: search() navigates to the search URL; extracts job cards from
          the js_variables JSON blob; populates the first card's JD from
          the embedded detail response; clicks remaining cards to read
          full JD from the detail panel; paginates until maxPages or no
          cards; handles extraction failures per-card without aborting;
          falls back to shortDescription when panel text is too short
    WHY: ZipRecruiter is a React SPA — all data lives in an embedded
         JSON blob.  Click-through on the SERP avoids Cloudflare
         challenges that would block per-URL navigation.
    """

    @pytest.mark.asyncio
    async def test_search_returns_listings_from_fixture(self) -> None:
        """search() extracts 3 listings from the synthetic fixture."""
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()
        panel_text = "A detailed job description for testing " * 10  # > 100 chars

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=3,
            panel_texts=[panel_text, panel_text],  # cards 1 & 2 (card 0 from js_vars)
        )

        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        assert len(listings) == 3
        assert listings[0].title == "Staff Platform Architect"

    @pytest.mark.asyncio
    async def test_search_first_card_gets_jd_from_js_variables(self) -> None:
        """The first card's full_text comes from htmlFullDescription in js_variables."""
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()
        panel_text = "Panel text for remaining cards with enough content " * 5

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=3,
            panel_texts=[panel_text, panel_text],
        )

        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        # First card's text comes from js_variables, not the panel
        assert "distributed systems" in listings[0].full_text
        assert "Staff Platform Architect" in listings[0].full_text

    @pytest.mark.asyncio
    async def test_search_remaining_cards_enriched_by_click_through(self) -> None:
        """Cards after the first are enriched by clicking and reading the detail panel."""
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()
        panel_b_text = "Role B detailed job description for panel " * 10
        panel_c_text = "Role C detailed job description for panel " * 10

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=3,
            panel_texts=[panel_b_text, panel_c_text],
        )

        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        assert "Role B" in listings[1].full_text
        assert "Role C" in listings[2].full_text

    @pytest.mark.asyncio
    async def test_search_falls_back_to_short_desc_when_panel_too_short(self) -> None:
        """When panel text is under 100 chars, shortDescription fallback is used."""
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=3,
            panel_texts=["Too short", "Also short"],
        )

        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        # Card 0: from js_variables (has full JD)
        assert "distributed systems" in listings[0].full_text
        # Cards 1 & 2: panel was too short → fallback to shortDescription
        assert "Senior Staff Engineer at Globex Corporation" in listings[1].full_text
        assert "Principal Software Architect at Initech" in listings[2].full_text

    @pytest.mark.asyncio
    async def test_search_falls_back_on_click_failure(self) -> None:
        """When a card click raises an exception, shortDescription fallback is used."""
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=3,
        )

        # Make the second card's click raise (first card is from js_vars, so index 1 clicks)
        card_locator = page.locator("[class*='job_result'] article")
        failing_card = MagicMock()
        failing_card.click = AsyncMock(side_effect=TimeoutError("click timeout"))
        original_nth = card_locator.nth.side_effect
        card_locator.nth.side_effect = lambda i: failing_card if i == 1 else original_nth(i)

        panel_text = "Panel text for card C with enough detail " * 10
        panel_locator = page.locator("[data-testid='job-details-scroll-container']")
        panel_locator.inner_text = AsyncMock(return_value=panel_text)

        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        # Card 1: click failed → fallback
        assert "Senior Staff Engineer at Globex Corporation" in listings[1].full_text

    @pytest.mark.asyncio
    async def test_search_stops_at_max_pages(self) -> None:
        """Pagination stops when page_num reaches maxPages from js_variables."""
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
        assert page.goto.await_count == 1
        assert len(listings) == 3

    @pytest.mark.asyncio
    async def test_search_stops_when_no_cards_on_page(self) -> None:
        """When a page yields zero job cards, pagination stops."""
        adapter = ZipRecruiterAdapter()
        # First page returns cards, second page returns empty
        fixture_html = _SERP_FIXTURE.read_text()
        # Override maxPages to allow multi-page
        empty_html = fixture_html.replace('"maxPages": 1', '"maxPages": 3')
        # Second page has no cards
        empty_html.replace(
            '"jobCards": [', '"jobCards": ['
        ).replace(
            '"hydrateJobCardsResponse"',
            '"hydrateJobCardsResponse"',
        )
        # Simpler: just return fixture for page 1, then empty for page 2
        no_cards_fixture = (
            '<html><head><title>Jobs</title></head><body>'
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

        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=3)

        assert call_count["goto"] == 2  # page 1 + page 2 (stops on empty)
        assert len(listings) == 3  # only from page 1

    @pytest.mark.asyncio
    async def test_search_stops_on_js_variables_failure(self) -> None:
        """An ActionableError from extract_js_variables stops pagination."""
        adapter = ZipRecruiterAdapter()
        broken_html = "<html><head><title>Jobs</title></head><body>No JSON here</body></html>"

        page = _make_mock_page(content_html=broken_html)

        listings = await adapter.search(page, "https://zr.com/search", max_pages=3)

        assert listings == []

    @pytest.mark.asyncio
    async def test_search_skips_unparseable_card(self) -> None:
        """A card_to_listing exception on one card doesn't abort the rest."""
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

        def _failing_card_to_listing(card: dict) -> JobListing:
            call_count["calls"] += 1
            if call_count["calls"] == 2:
                raise ValueError("Unparseable card")
            return original_card_to_listing(card)

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

        # 3 cards, 1 failed → 2 listings
        assert len(listings) == 2

    @pytest.mark.asyncio
    async def test_search_no_cards_in_dom_returns_listings_without_click(self) -> None:
        """When DOM has no card articles, listings are returned without panel enrichment."""
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=0,  # No card articles in DOM
        )

        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        # 3 listings from JSON, but only first has full_text (from js_vars)
        assert len(listings) == 3
        assert listings[0].full_text.strip()  # From js_variables
        # Cards 1 & 2 have no panel text and no click-through happened
        # They get short_description fallback since panel was never read

    @pytest.mark.asyncio
    async def test_search_paginates_url_correctly(self) -> None:
        """Page 2+ URLs append &page=N or ?page=N correctly."""
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text().replace(
            '"maxPages": 1', '"maxPages": 2'
        )
        empty_fixture = (
            '<html><head><title>Jobs</title></head><body>'
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
        for_cards = []
        for _ in range(3):
            c = MagicMock()
            c.click = AsyncMock()
            for_cards.append(c)
        mock_card.nth.side_effect = lambda i: for_cards[i] if i < len(for_cards) else MagicMock()
        page.locator.side_effect = lambda sel: {
            "[class*='job_result'] article": mock_card,
            "[data-testid='job-details-scroll-container']": mock_panel,
        }[sel]

        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            await adapter.search(page, "https://zr.com/search?q=architect", max_pages=2)

        assert goto_urls[0] == "https://zr.com/search?q=architect"
        assert goto_urls[1] == "https://zr.com/search?q=architect&page=2"

    @pytest.mark.asyncio
    async def test_search_card_index_exceeds_dom_count(self) -> None:
        """When more listings than DOM cards, click-through stops at DOM count."""
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text()  # 3 cards in JSON

        page = _make_mock_page(
            content_html=fixture_html,
            card_count=1,  # Only 1 card article in DOM
        )

        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            listings = await adapter.search(page, "https://zr.com/search", max_pages=1)

        # 3 listings parsed, but click-through stopped at DOM card 1
        assert len(listings) == 3
        # First card from js_variables
        assert listings[0].full_text.strip()

    @pytest.mark.asyncio
    async def test_search_pagination_uses_question_mark_separator(self) -> None:
        """When query URL has no '?', page=N is appended with '?' separator."""
        adapter = ZipRecruiterAdapter()
        fixture_html = _SERP_FIXTURE.read_text().replace(
            '"maxPages": 1', '"maxPages": 2'
        )
        empty_fixture = (
            '<html><head><title>Jobs</title></head><body>'
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

        with patch(
            "jobsearch_rag.adapters.ziprecruiter.random.uniform",
            return_value=0.0,
        ):
            await adapter.search(page, "https://zr.com/search", max_pages=2)

        # No '?' in query → uses '?' separator
        assert goto_urls[1] == "https://zr.com/search?page=2"

    @pytest.mark.asyncio
    async def test_search_first_card_not_prepopulated_when_no_jd_in_js_vars(self) -> None:
        """When js_variables has no getJobDetailsResponse, first card is not pre-populated."""
        adapter = ZipRecruiterAdapter()
        # Build a minimal fixture with cards but no getJobDetailsResponse
        fixture_html = (
            '<html><head><title>Jobs</title></head><body>'
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

        assert len(listings) == 1
        # Full text came from click-through panel, not js_variables
        assert "Full detail text" in listings[0].full_text

    @pytest.mark.asyncio
    async def test_search_respects_max_pages_argument(self) -> None:
        """When max_pages < site maxPages, the loop stops at max_pages without breaking."""
        adapter = ZipRecruiterAdapter()
        # Fixture says maxPages=3 but we pass max_pages=1
        fixture_html = (
            '<html><head><title>Jobs</title></head><body>'
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
        assert len(listings) == 1


# ---------------------------------------------------------------------------
# TestExtractDetailPassthrough
# ---------------------------------------------------------------------------


class TestExtractDetailPassthrough:
    """REQUIREMENT: extract_detail is a passthrough when full_text is populated.

    WHO: The pipeline runner calling extract_detail after search
    WHAT: If full_text was already populated during search click-through,
          extract_detail returns the listing unchanged; if still empty,
          it applies the shortDescription fallback; if no fallback exists,
          full_text remains empty
    WHY: SERP click-through makes per-URL extraction unnecessary — the
         runner calls extract_detail out of protocol compliance, but
         ZipRecruiter does all extraction during search()
    """

    @pytest.mark.asyncio
    async def test_passthrough_when_full_text_present(self) -> None:
        """extract_detail returns listing unchanged if full_text is populated."""
        adapter = ZipRecruiterAdapter()
        listing = _make_listing(full_text="Full JD already populated")
        mock_page = MagicMock()

        result = await adapter.extract_detail(mock_page, listing)

        assert result is listing
        assert result.full_text == "Full JD already populated"

    @pytest.mark.asyncio
    async def test_fallback_when_full_text_empty(self) -> None:
        """extract_detail applies shortDescription fallback when full_text is empty."""
        adapter = ZipRecruiterAdapter()
        listing = _make_listing(full_text="", short_description="Python role at company")
        mock_page = MagicMock()

        result = await adapter.extract_detail(mock_page, listing)

        assert "Python role at company" in result.full_text
        assert "Staff Architect at Acme Corp" in result.full_text

    @pytest.mark.asyncio
    async def test_empty_when_no_fallback_available(self) -> None:
        """extract_detail returns empty full_text when no short description exists."""
        adapter = ZipRecruiterAdapter()
        listing = _make_listing(full_text="")
        mock_page = MagicMock()

        result = await adapter.extract_detail(mock_page, listing)

        assert result.full_text == ""
