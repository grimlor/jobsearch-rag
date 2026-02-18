"""ZipRecruiter adapter — auth, search pagination, JD extraction.

ZipRecruiter renders its SERP as a React SPA.  The HTML body contains only
empty hydration roots (``#react-serp-root``); all job data is delivered as
JSON inside a ``<script id="js_variables" type="application/json">`` tag.

Extraction strategy:
  1. Navigate to the search URL and wait for DOM content.
  2. Wait for any Cloudflare challenge to resolve (important: Cloudflare
     presents a "Just a moment..." interstitial on first visit).
  3. Read the ``js_variables`` script tag and ``JSON.parse`` it.
  4. Job cards live at ``hydrateJobCardsResponse.jobCards[]``.
  5. Full JD HTML for the initially-selected card is at
     ``getJobDetailsResponse.jobDetails.htmlFullDescription``.
  6. For remaining cards, **click each card article** on the SERP to
     update the detail panel (``[data-testid='job-details-scroll-container']``)
     and read the full JD text via ``inner_text()``.

The adapter stays on the SERP page and uses click-through extraction
rather than navigating to individual detail URLs, which avoids triggering
fresh Cloudflare challenges per listing.

.. important::

   ZipRecruiter uses Cloudflare bot protection.  Headless Chromium is
   typically blocked.  Set ``headless = false`` in ``settings.toml``
   for reliable operation.  Use ``jobsearch-rag login --board ziprecruiter``
   to establish a session interactively before running headless searches.
"""

from __future__ import annotations

import asyncio
import json
import random
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
    "card_articles": "[class*='job_result'] article",
    "detail_panel": "[data-testid='job-details-scroll-container']",
}

# Base URL for relative canonical paths
_BASE_URL = "https://www.ziprecruiter.com"

# Maximum seconds to wait for Cloudflare challenge to resolve
_CF_WAIT_TIMEOUT = 15

# Click-through timing (seconds) — human-like pause between card clicks
_CLICK_DELAY = (0.5, 1.5)


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


def html_to_text(html: str) -> str:
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
    """Convert a single job card dict to a ``JobListing``.

    The ``shortDescription`` snippet from the card is stored in
    ``metadata["short_description"]`` for fallback use when full
    JD extraction from the detail page is blocked.
    """
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

    # Capture short description for fallback scoring
    short_desc = card.get("shortDescription", "")
    if short_desc:
        metadata["short_description"] = short_desc

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
    return html_to_text(html_desc)


# ---------------------------------------------------------------------------
# Cloudflare challenge helper
# ---------------------------------------------------------------------------


async def _wait_for_cloudflare(page: Page, *, timeout: int = _CF_WAIT_TIMEOUT) -> None:
    """Wait for a Cloudflare "Just a moment..." challenge to resolve.

    Cloudflare inserts an interstitial page that runs a JS challenge.
    In headed mode it resolves in < 1 s.  In headless mode it usually
    blocks indefinitely (headless = false is recommended).

    This function polls the page title and returns as soon as the
    challenge clears, or raises :class:`ActionableError` on timeout.
    """
    for _ in range(timeout):
        title = await page.title()
        if "just a moment" not in title.lower():
            return
        await asyncio.sleep(1)

    raise ActionableError.authentication(
        "ziprecruiter",
        f"Cloudflare challenge did not resolve within {timeout}s",
        suggestion=(
            "Set headless = false in settings.toml for ziprecruiter. "
            "Cloudflare blocks headless browsers."
        ),
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@AdapterRegistry.register
class ZipRecruiterAdapter(JobBoardAdapter):
    """Browser automation adapter for ZipRecruiter.

    Extracts job data from the ``js_variables`` JSON blob embedded in
    ZipRecruiter's React SPA pages.  Full JD text is obtained via
    SERP click-through: clicking each card article updates the detail
    panel, avoiding per-URL navigation and Cloudflare challenges.
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

        # Wait for Cloudflare challenge if present
        await _wait_for_cloudflare(page)

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
        """Navigate search results, click through cards, and return enriched listings.

        Extracts job card metadata from the ``js_variables`` JSON blob, then
        clicks through each card article on the SERP to read full JD text
        from the detail panel.  This avoids navigating to individual detail
        URLs, which would trigger fresh Cloudflare challenges per listing.

        Args:
            page: Playwright page with an active session.
            query: Full search URL (from settings.toml).
            max_pages: Maximum number of result pages to paginate through.

        Returns:
            List of ``JobListing`` with ``full_text`` already populated
            via click-through.  ``extract_detail`` will be a no-op.
        """
        listings: list[JobListing] = []

        logger.info("Searching ZipRecruiter: %s (max %d pages)", query, max_pages)

        for page_num in range(1, max_pages + 1):
            # Build paginated URL
            sep = "&" if "?" in query else "?"
            page_url = f"{query}{sep}page={page_num}" if page_num > 1 else query
            await page.goto(page_url, wait_until="domcontentloaded")
            await _wait_for_cloudflare(page)

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

            # Parse card metadata into listings
            page_listings: list[JobListing] = []
            for card in cards:
                try:
                    listing = card_to_listing(card)
                    page_listings.append(listing)
                except Exception as exc:
                    logger.warning("Failed to parse a job card: %s", exc)

            # --- Click-through extraction ---
            # The first card's JD may already be in js_variables
            first_jd = extract_jd_text(js_vars)
            if first_jd and page_listings:
                page_listings[0].full_text = first_jd
                logger.debug(
                    "First card JD from js_variables (%d chars)",
                    len(first_jd),
                )

            # Click remaining cards to extract full JD from detail panel
            await self._click_through_cards(page, page_listings)

            listings.extend(page_listings)

            # Check if there are more pages
            site_max_pages = js_vars.get("maxPages", 1)
            if page_num >= site_max_pages:
                logger.info("Reached last page (%d)", page_num)
                break

        enriched = sum(1 for ls in listings if ls.full_text.strip())
        logger.info(
            "Found %d listings across %d page(s) (%d with full JD)",
            len(listings),
            page_num,
            enriched,
        )
        return listings

    async def _click_through_cards(
        self,
        page: Page,
        listings: list[JobListing],
    ) -> None:
        """Click each card article on the SERP and read JD from the detail panel.

        Skips listings that already have ``full_text`` populated (e.g. the
        first card extracted from ``js_variables``).  Falls back to
        ``shortDescription`` from metadata when the panel text is too short.
        """
        card_locator = page.locator(_SELECTORS["card_articles"])
        panel_locator = page.locator(_SELECTORS["detail_panel"])

        card_count = await card_locator.count()
        if card_count == 0:
            logger.warning("No card articles found on SERP — cannot click-through")
            return

        for i, listing in enumerate(listings):
            # Skip if already populated (first card from js_variables)
            if listing.full_text.strip():
                continue

            if i >= card_count:
                logger.debug("Card index %d exceeds DOM card count %d", i, card_count)
                break

            try:
                # Click the card article
                card = card_locator.nth(i)
                await card.click()

                # Human-like delay
                delay = random.uniform(*_CLICK_DELAY)
                await asyncio.sleep(delay)

                # Wait for panel to be visible and read text
                await panel_locator.wait_for(state="visible", timeout=5000)
                panel_text = await panel_locator.inner_text()

                if panel_text and len(panel_text.strip()) > 100:
                    listing.full_text = panel_text.strip()
                    logger.debug(
                        "Card %d/%d: %d chars from detail panel",
                        i + 1,
                        len(listings),
                        len(listing.full_text),
                    )
                else:
                    # Fall back to short description
                    self._apply_short_description_fallback(listing)
                    logger.debug(
                        "Card %d/%d: panel text too short, used fallback",
                        i + 1,
                        len(listings),
                    )
            except Exception as exc:
                logger.warning(
                    "Click-through failed for card %d (%s): %s",
                    i + 1,
                    listing.external_id,
                    exc,
                )
                self._apply_short_description_fallback(listing)

    @staticmethod
    def _apply_short_description_fallback(listing: JobListing) -> None:
        """Populate full_text from the SERP card's shortDescription snippet."""
        short_desc = listing.metadata.get("short_description", "")
        if short_desc:
            listing.full_text = f"{listing.title} at {listing.company}. {short_desc}"
        else:
            listing.full_text = ""

    async def extract_detail(
        self,
        page: Page,
        listing: JobListing,
    ) -> JobListing:
        """Return the listing — full JD is already populated by click-through.

        The ``search()`` method populates ``full_text`` during SERP
        click-through, so this method is a passthrough.  If ``full_text``
        is somehow still empty (e.g. click-through was skipped), falls
        back to the ``shortDescription`` captured from the SERP card.

        Returns the same listing object, unchanged or with fallback text.
        """
        if listing.full_text.strip():
            return listing

        # Fallback: use the short description from the SERP card
        logger.debug(
            "extract_detail called with empty full_text for %s — applying fallback",
            listing.external_id,
        )
        self._apply_short_description_fallback(listing)
        return listing
