"""ZipRecruiter adapter — auth, search pagination, JD extraction.

ZipRecruiter renders its SERP as a React SPA.  The HTML body contains only
empty hydration roots (``#react-serp-root``); all job data is delivered as
JSON inside a ``<script id="js_variables" type="application/json">`` tag.

Extraction strategy:
  1. Navigate to the search URL and wait for DOM content.
  2. Read the ``js_variables`` script tag and ``JSON.parse`` it.
  3. Job cards live at ``hydrateJobCardsResponse.jobCards[]``.
  4. Full JD HTML for the initially-selected card is at
     ``getJobDetailsResponse.jobDetails.htmlFullDescription``.
  5. For remaining cards, navigate to the canonical detail URL
     (``rawCanonicalZipJobPageUrl``) and extract from that page's
     ``js_variables`` blob.

The adapter does **not** use CSS selectors against rendered DOM for job data
because ZipRecruiter hydrates from JSON → React; the DOM is empty in the
initial HTML response used for parse-mode extraction.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any

from jobsearch_rag.adapters.base import JobBoardAdapter, JobListing
from jobsearch_rag.adapters.registry import AdapterRegistry
from jobsearch_rag.errors import ActionableError
from jobsearch_rag.logging import logger

if TYPE_CHECKING:
    from playwright.async_api import Page

# ---------------------------------------------------------------------------
# Selectors — only the handful we still need against the actual DOM
# ---------------------------------------------------------------------------

_SELECTORS = {
    "js_variables": "script#js_variables",
    "captcha_indicator": "iframe[src*='captcha'], div.g-recaptcha",
}

# Base URL for relative canonical paths
_BASE_URL = "https://www.ziprecruiter.com"


# ---------------------------------------------------------------------------
# HTML → plain-text helper
# ---------------------------------------------------------------------------


class _HTMLTextExtractor(HTMLParser):
    """Lightweight HTML-to-text converter for JD descriptions."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _html_to_text(html: str) -> str:
    """Strip HTML tags and return plain text."""
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    return re.sub(r"\s+", " ", extractor.get_text()).strip()


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------


def extract_js_variables(html: str) -> dict[str, Any]:
    """Parse the ``js_variables`` JSON blob from a ZipRecruiter page.

    Args:
        html: Full page HTML source.

    Returns:
        Parsed dict of the ``js_variables`` content.

    Raises:
        ActionableError: If the script tag is missing or JSON is malformed.
    """
    pattern = r'<script\s+id="js_variables"\s+type="application/json"[^>]*>(.*?)</script>'
    match = re.search(pattern, html, re.DOTALL)
    if not match:
        raise ActionableError.parse(
            "ziprecruiter",
            "script#js_variables",
            "js_variables script tag not found — page structure may have changed",
        )
    try:
        return json.loads(match.group(1))  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        raise ActionableError.parse(
            "ziprecruiter",
            "script#js_variables",
            f"Failed to parse js_variables JSON: {exc}",
        ) from exc


def parse_job_cards(js_vars: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the job card list from parsed js_variables."""
    hydrate = js_vars.get("hydrateJobCardsResponse", {})
    cards: list[dict[str, Any]] = hydrate.get("jobCards", [])
    return cards


def card_to_listing(card: dict[str, Any]) -> JobListing:
    """Convert a single job card dict to a ``JobListing``."""
    listing_key = card.get("listingKey", "")
    title = card.get("title", "")
    company = card.get("company", {}).get("name", "Unknown")

    # Location: use displayName from the nested location object
    loc_obj = card.get("location", {})
    location = loc_obj.get("displayName", "Unknown")

    # Canonical URL — top-level field in card, falls back to applyButtonConfig
    raw_url = card.get("rawCanonicalZipJobPageUrl", "")
    if not raw_url:
        apply_cfg = card.get("applyButtonConfig", {})
        raw_url = apply_cfg.get("rawCanonicalZipJobPageUrl", "")
    url = f"{_BASE_URL}{raw_url}" if raw_url else ""

    # Salary metadata
    pay = card.get("pay", {})
    salary_min = pay.get("minAnnual")
    salary_max = pay.get("maxAnnual")
    metadata: dict[str, str] = {}
    if salary_min is not None and salary_max is not None:
        metadata["salary_range"] = f"${salary_min:,.0f} - ${salary_max:,.0f}"

    return JobListing(
        board="ziprecruiter",
        external_id=listing_key,
        title=title,
        company=company,
        location=location,
        url=url,
        full_text="",  # Populated by extract_detail
        metadata=metadata,
    )


def extract_jd_text(js_vars: dict[str, Any]) -> str:
    """Extract full JD plain text from the detail response in js_variables."""
    details = js_vars.get("getJobDetailsResponse", {}).get("jobDetails", {})
    html_desc = details.get("htmlFullDescription", "")
    if not html_desc:
        return ""
    return _html_to_text(html_desc)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@AdapterRegistry.register
class ZipRecruiterAdapter(JobBoardAdapter):
    """Browser automation adapter for ZipRecruiter.

    Extracts job data from the ``js_variables`` JSON blob embedded in
    ZipRecruiter's React SPA pages rather than from rendered DOM elements.
    """

    @property
    def board_name(self) -> str:
        return "ziprecruiter"

    async def authenticate(self, page: Page) -> None:
        """Verify an active session by checking for login/CAPTCHA indicators.

        ZipRecruiter auth is cookie-based via ``storage_state``.  This method
        validates the session is still alive rather than performing login.

        Raises:
            ActionableError: If the session is expired or a CAPTCHA is present.
        """
        await page.goto(
            "https://www.ziprecruiter.com/jobs-search",
            wait_until="domcontentloaded",
        )

        # Check for CAPTCHA first — it takes priority
        captcha = await page.query_selector(_SELECTORS["captcha_indicator"])
        if captcha:
            raise ActionableError.authentication(
                self.board_name,
                "CAPTCHA encountered — cannot proceed automatically",
                suggestion="Solve CAPTCHA manually in headed mode, then re-run",
            )

        # Check if we landed on a login page instead of search
        url = page.url
        if "/login" in url or "/sign-in" in url:
            raise ActionableError.authentication(
                self.board_name,
                "Session expired — redirected to login page",
                suggestion=(
                    "Run in headed mode (headless=false) to authenticate manually. "
                    "Session cookies will be saved automatically."
                ),
            )

        logger.info("ZipRecruiter session verified")

    async def search(
        self,
        page: Page,
        query: str,
        max_pages: int = 3,
    ) -> list[JobListing]:
        """Navigate search results and return shallow listings.

        Extracts job cards from the ``js_variables`` JSON blob on each
        search results page.  Pagination appends ``&page=N`` to the URL
        rather than clicking a "next" button, since the page data is
        in the initial HTML, not injected by client-side React navigation.

        Args:
            page: Playwright page with an active session.
            query: Full search URL (from settings.toml).
            max_pages: Maximum number of result pages to paginate through.

        Returns:
            List of ``JobListing`` with title/company/url populated but
            ``full_text`` empty — call :meth:`extract_detail` next.
        """
        listings: list[JobListing] = []

        logger.info("Searching ZipRecruiter: %s (max %d pages)", query, max_pages)

        for page_num in range(1, max_pages + 1):
            # Build paginated URL
            sep = "&" if "?" in query else "?"
            page_url = f"{query}{sep}page={page_num}" if page_num > 1 else query
            await page.goto(page_url, wait_until="domcontentloaded")

            logger.info("Processing search results page %d", page_num)

            html = await page.content()
            try:
                js_vars = extract_js_variables(html)
            except ActionableError:
                logger.warning(
                    "Could not extract js_variables on page %d — stopping",
                    page_num,
                )
                break

            cards = parse_job_cards(js_vars)
            if not cards:
                logger.info("No job cards on page %d — stopping", page_num)
                break

            for card in cards:
                try:
                    listing = card_to_listing(card)
                    listings.append(listing)
                except Exception as exc:
                    logger.warning("Failed to parse a job card: %s", exc)

            # Check if there are more pages
            site_max_pages = js_vars.get("maxPages", 1)
            if page_num >= site_max_pages:
                logger.info("Reached last page (%d)", page_num)
                break

        logger.info("Found %d listings across %d page(s)", len(listings), page_num)
        return listings

    async def extract_detail(
        self,
        page: Page,
        listing: JobListing,
    ) -> JobListing:
        """Navigate to the listing URL and extract full JD text.

        Reads the ``js_variables`` JSON on the detail page and extracts
        ``htmlFullDescription``, converting it to plain text.

        Returns the same listing object with ``full_text`` populated.
        Empty text is not silently accepted — it logs a warning.
        """
        logger.debug("Extracting detail for %s: %s", listing.external_id, listing.url)

        try:
            await page.goto(listing.url, wait_until="domcontentloaded")
        except Exception as exc:
            raise ActionableError.parse(
                self.board_name,
                _SELECTORS["js_variables"],
                f"Failed to navigate to {listing.url}: {exc}",
            ) from exc

        html = await page.content()
        try:
            js_vars = extract_js_variables(html)
        except ActionableError:
            logger.warning(
                "No js_variables on detail page %s — selector may have changed",
                listing.url,
            )
            listing.full_text = ""
            return listing

        text = extract_jd_text(js_vars)
        listing.full_text = text

        if not listing.full_text:
            logger.warning("Empty job description extracted from %s", listing.url)

        return listing
