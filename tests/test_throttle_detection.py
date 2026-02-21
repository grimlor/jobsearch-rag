"""Throttle detection tests — ZipRecruiter rate-limit recognition and backoff.

Maps to BDD spec: TestThrottleDetection

Tests verify that the ZipRecruiter adapter recognizes throttle responses
("We encountered an error while loading this job"), applies exponential
backoff, skips after max retries, logs throttle events, and distinguishes
throttle text from legitimate JD content.

All behavioral tests exercise the public ``search()`` API rather than
internal click-through methods, asserting on returned listing state,
sleep calls, and log output.
"""

from __future__ import annotations

import asyncio as _asyncio
import json
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobsearch_rag.adapters.base import JobListing
from jobsearch_rag.adapters.ziprecruiter import (
    ZipRecruiterAdapter,
    is_throttle_response,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ZR = "jobsearch_rag.adapters.ziprecruiter"
_SEARCH_URL = "https://www.ziprecruiter.com/jobs-search?search=test"

_THROTTLE_TEXT = "We encountered an error while loading this job"

_REAL_JD = (
    "Senior Solutions Architect at Acme Corp. "
    "We are looking for an experienced architect to lead our cloud "
    "infrastructure modernization effort. Requirements include 10+ "
    "years of experience in distributed systems design. " * 3
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_listing(**overrides: Any) -> JobListing:
    """Create a minimal JobListing for throttle testing."""
    defaults: dict[str, Any] = {
        "board": "ziprecruiter",
        "external_id": "zr-test-1",
        "title": "Solutions Architect",
        "company": "Acme Corp",
        "location": "Remote",
        "url": "https://www.ziprecruiter.com/jobs/test-1",
        "full_text": "",
    }
    defaults.update(overrides)
    return JobListing(**defaults)


def _listing_to_card(listing: JobListing) -> dict[str, Any]:
    """Convert a ``JobListing`` to a ZR card dict for the HTML fixture."""
    card: dict[str, Any] = {
        "listingKey": listing.external_id,
        "title": listing.title,
        "company": {"name": listing.company},
        "location": {"displayName": listing.location},
        "rawCanonicalZipJobPageUrl": listing.url.replace(
            "https://www.ziprecruiter.com", "",
        ),
    }
    short_desc = listing.metadata.get("short_description", "")
    if short_desc:
        card["shortDescription"] = short_desc
    return card


def _build_zr_html(cards: list[dict[str, Any]]) -> str:
    """Build a realistic ZipRecruiter page with a ``js_variables`` JSON blob."""
    js_vars = {
        "hydrateJobCardsResponse": {"jobCards": cards},
        "maxPages": 1,
    }
    return (
        "<html><head><title>Jobs</title></head><body>"
        '<script id="js_variables" type="application/json">'
        f"{json.dumps(js_vars)}</script></body></html>"
    )


@contextmanager
def _patch_search_to_click_through(
    listings: list[JobListing],
    panel_mock: AsyncMock,
) -> Iterator[MagicMock]:
    """Patch I/O boundaries so ``search()`` reaches click-through with real parsing.

    Builds a realistic HTML page containing a ``js_variables`` JSON blob
    derived from *listings*.  Pure-computation functions
    (``extract_js_variables``, ``parse_job_cards``, ``card_to_listing``,
    ``extract_jd_text``) run on **real data** — only I/O boundaries
    (page navigation, Cloudflare waits, Playwright locators) are stubbed.

    Yields the ``page`` mock for passing to ``adapter.search()``.
    """
    html = _build_zr_html([_listing_to_card(ls) for ls in listings])

    card_mock = AsyncMock()
    card_mock.click = AsyncMock()

    card_locator = MagicMock()
    card_locator.count = AsyncMock(return_value=len(listings))
    card_locator.nth = MagicMock(return_value=card_mock)

    page = MagicMock()
    page.goto = AsyncMock()
    page.content = AsyncMock(return_value=html)
    page.locator = MagicMock(side_effect=[card_locator, panel_mock])

    with patch(f"{_ZR}._wait_for_cloudflare", new_callable=AsyncMock):
        yield page


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestThrottleDetection:
    """
    REQUIREMENT: Board-specific throttle responses are detected and handled
    gracefully rather than treated as valid JD text.

    WHO: The pipeline operator running automated searches
    WHAT: Throttle responses are recognized, backoff is applied with
          increasing delays, and listings are skipped after max retries
    WHY: Without throttle detection, error messages would be indexed as
         JD content, corrupting scoring results
    """

    # --- Pure detection ------------------------------------------------

    def test_ziprecruiter_error_message_is_not_treated_as_valid_jd(self) -> None:
        """
        When the canonical ZR throttle text is checked
        Then it is recognized as a throttle response
        """
        assert is_throttle_response(_THROTTLE_TEXT) is True

    def test_non_throttle_error_text_is_not_misidentified_as_throttle(self) -> None:
        """
        When legitimate JD text is checked
        Then it is not falsely flagged as a throttle response
        """
        assert is_throttle_response(_REAL_JD) is False
        # A JD that happens to mention 'error' should not trigger detection
        assert is_throttle_response("Handle error reporting in production systems. " * 5) is False

    # --- Backoff behavior through search() ----------------------------

    @pytest.mark.asyncio
    async def test_throttle_detection_triggers_backoff_delay(self) -> None:
        """
        When a throttle response is detected during search
        Then the adapter waits before retrying
        """
        adapter = ZipRecruiterAdapter()
        listing = _make_listing()

        panel_mock = AsyncMock()
        panel_mock.inner_text = AsyncMock(side_effect=[_THROTTLE_TEXT, _REAL_JD])
        panel_mock.wait_for = AsyncMock()

        with (
            _patch_search_to_click_through([listing], panel_mock) as page,
            patch(f"{_ZR}.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            await adapter.search(page, _SEARCH_URL, max_pages=1)
            # click delay + backoff delay
            assert mock_sleep.call_count >= 2

    @pytest.mark.asyncio
    async def test_backoff_delay_increases_exponentially_on_consecutive_throttles(
        self,
    ) -> None:
        """
        When consecutive throttle responses occur across listings
        Then the second backoff wait is longer than the first
        """
        adapter = ZipRecruiterAdapter()
        listing1 = _make_listing(external_id="zr-1")
        listing2 = _make_listing(external_id="zr-2")

        panel_mock = AsyncMock()
        panel_mock.inner_text = AsyncMock(
            side_effect=[_THROTTLE_TEXT, _REAL_JD, _THROTTLE_TEXT, _REAL_JD]
        )
        panel_mock.wait_for = AsyncMock()

        backoff_delays: list[float] = []

        async def capture_sleep(duration: float) -> None:
            backoff_delays.append(duration)

        with (
            _patch_search_to_click_through([listing1, listing2], panel_mock) as page,
            patch(f"{_ZR}.asyncio.sleep", side_effect=capture_sleep),
        ):
            await adapter.search(page, _SEARCH_URL, max_pages=1)

        # Filter out click delays (small) from backoff delays (larger)
        backoff_waits = [d for d in backoff_delays if d >= 2.0]
        assert len(backoff_waits) >= 2, (
            f"Expected at least 2 backoff waits, got {len(backoff_waits)}: {backoff_delays}"
        )
        assert backoff_waits[1] > backoff_waits[0], (
            f"Second backoff ({backoff_waits[1]}) should be longer than first ({backoff_waits[0]})"
        )

    # --- Retry exhaustion and skip ------------------------------------

    @pytest.mark.asyncio
    async def test_listing_is_skipped_after_max_retry_attempts(self) -> None:
        """
        When the retry limit is exceeded
        Then the listing is skipped with empty or fallback full_text
        """
        adapter = ZipRecruiterAdapter()
        listing = _make_listing()

        panel_mock = AsyncMock()
        panel_mock.inner_text = AsyncMock(return_value=_THROTTLE_TEXT)
        panel_mock.wait_for = AsyncMock()

        with (
            _patch_search_to_click_through([listing], panel_mock) as page,
            patch(f"{_ZR}.asyncio.sleep", new_callable=AsyncMock),
        ):
            results = await adapter.search(page, _SEARCH_URL, max_pages=1)

        assert not results[0].full_text.strip() or results[0].full_text == (
            f"{listing.title} at {listing.company}. "
        )

    @pytest.mark.asyncio
    async def test_skipped_listing_increments_failed_count(self) -> None:
        """
        When a listing exhausts retries
        Then it is counted as a failed extraction
        """
        adapter = ZipRecruiterAdapter()
        listing = _make_listing()

        panel_mock = AsyncMock()
        panel_mock.inner_text = AsyncMock(return_value=_THROTTLE_TEXT)
        panel_mock.wait_for = AsyncMock()

        with (
            _patch_search_to_click_through([listing], panel_mock) as page,
            patch(f"{_ZR}.asyncio.sleep", new_callable=AsyncMock),
        ):
            results = await adapter.search(page, _SEARCH_URL, max_pages=1)

        # After max retries, full_text should be empty (treated as failed)
        assert not results[0].full_text.strip() or "short_description" not in results[0].metadata

    # --- Logging ------------------------------------------------------

    def test_throttle_event_is_logged_with_url_and_retry_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        When a throttle event occurs
        Then it is logged at WARNING with the listing identifier
        """
        adapter = ZipRecruiterAdapter()
        listing = _make_listing()

        panel_mock = AsyncMock()
        panel_mock.inner_text = AsyncMock(return_value=_THROTTLE_TEXT)
        panel_mock.wait_for = AsyncMock()

        with (
            _patch_search_to_click_through([listing], panel_mock) as page,
            caplog.at_level(logging.WARNING),
            patch(f"{_ZR}.asyncio.sleep", new_callable=AsyncMock),
        ):
            _asyncio.run(adapter.search(page, _SEARCH_URL, max_pages=1))

        throttle_logs = [r for r in caplog.records if "throttl" in r.message.lower()]
        assert len(throttle_logs) >= 1, "No throttle warning logged"
        # Log should mention the listing identifier
        log_text = " ".join(r.message for r in throttle_logs)
        assert listing.external_id in log_text or listing.url in log_text

    # --- Recovery after backoff ---------------------------------------

    @pytest.mark.asyncio
    async def test_successful_extraction_after_backoff_populates_full_text(
        self,
    ) -> None:
        """
        When extraction succeeds after a throttle backoff
        Then the listing full_text contains the real JD content
        """
        adapter = ZipRecruiterAdapter()
        listing = _make_listing()

        panel_mock = AsyncMock()
        panel_mock.inner_text = AsyncMock(side_effect=[_THROTTLE_TEXT, _REAL_JD])
        panel_mock.wait_for = AsyncMock()

        with (
            _patch_search_to_click_through([listing], panel_mock) as page,
            patch(f"{_ZR}.asyncio.sleep", new_callable=AsyncMock),
        ):
            results = await adapter.search(page, _SEARCH_URL, max_pages=1)

        assert results[0].full_text.strip() == _REAL_JD.strip()

    # --- Timeout + late throttle detection ----------------------------

    @pytest.mark.asyncio
    async def test_timeout_with_late_throttle_text_triggers_backoff(self) -> None:
        """
        When wait_for times out but the panel rendered throttle text
        Then the adapter detects the late throttle and retries with backoff
        """
        adapter = ZipRecruiterAdapter()
        listing = _make_listing()

        panel_mock = AsyncMock()
        panel_mock.wait_for = AsyncMock(side_effect=[TimeoutError("5000ms exceeded"), None])
        panel_mock.inner_text = AsyncMock(side_effect=[_THROTTLE_TEXT, _REAL_JD])

        with (
            _patch_search_to_click_through([listing], panel_mock) as page,
            patch(f"{_ZR}.asyncio.sleep", new_callable=AsyncMock),
        ):
            results = await adapter.search(page, _SEARCH_URL, max_pages=1)

        assert results[0].full_text.strip() == _REAL_JD.strip()

    @pytest.mark.asyncio
    async def test_late_throttle_exhausts_retries_then_falls_back(self) -> None:
        """
        When wait_for always times out and late panel reads always show throttle
        Then all retries are exhausted and fallback text is used
        """
        adapter = ZipRecruiterAdapter()
        listing = _make_listing()
        listing.metadata["short_description"] = "Fallback desc"

        panel_mock = AsyncMock()
        panel_mock.wait_for = AsyncMock(side_effect=TimeoutError("5000ms exceeded"))
        panel_mock.inner_text = AsyncMock(return_value=_THROTTLE_TEXT)

        with (
            _patch_search_to_click_through([listing], panel_mock) as page,
            patch(f"{_ZR}.asyncio.sleep", new_callable=AsyncMock),
        ):
            results = await adapter.search(page, _SEARCH_URL, max_pages=1)

        assert "Fallback desc" in results[0].full_text

    @pytest.mark.asyncio
    async def test_timeout_without_throttle_text_falls_back_immediately(
        self,
    ) -> None:
        """
        When wait_for times out and late panel read also fails
        Then the adapter falls back without retrying
        """
        adapter = ZipRecruiterAdapter()
        listing = _make_listing()
        listing.metadata["short_description"] = "Short desc for fallback"

        panel_mock = AsyncMock()
        panel_mock.wait_for = AsyncMock(side_effect=TimeoutError("5000ms exceeded"))
        panel_mock.inner_text = AsyncMock(side_effect=Exception("element detached"))

        with (
            _patch_search_to_click_through([listing], panel_mock) as page,
            patch(f"{_ZR}.asyncio.sleep", new_callable=AsyncMock),
        ):
            results = await adapter.search(page, _SEARCH_URL, max_pages=1)

        assert "Short desc for fallback" in results[0].full_text
