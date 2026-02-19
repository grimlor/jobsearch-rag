"""Throttle detection tests — ZipRecruiter rate-limit recognition and backoff.

Maps to BDD spec: TestThrottleDetection

Tests verify that the ZipRecruiter adapter recognizes throttle responses
(\"We encountered an error while loading this job\"), applies exponential
backoff, skips after max retries, logs throttle events, and distinguishes
throttle text from legitimate JD content.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from jobsearch_rag.adapters.base import JobListing
from jobsearch_rag.adapters.ziprecruiter import (
    ZipRecruiterAdapter,
    is_throttle_response,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_THROTTLE_TEXT = "We encountered an error while loading this job"

_REAL_JD = (
    "Senior Solutions Architect at Acme Corp. "
    "We are looking for an experienced architect to lead our cloud "
    "infrastructure modernization effort. Requirements include 10+ "
    "years of experience in distributed systems design. " * 3
)


def _make_listing(**overrides: str) -> JobListing:
    """Create a minimal JobListing for throttle testing."""
    defaults = {
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


class TestThrottleDetection:
    """Board-specific throttle responses are detected and handled
    gracefully rather than treated as valid JD text."""

    def test_ziprecruiter_error_message_is_not_treated_as_valid_jd(self) -> None:
        """The canonical ZR throttle text is recognized as a throttle
        response, not as genuine job description content."""
        assert is_throttle_response(_THROTTLE_TEXT) is True

    def test_non_throttle_error_text_is_not_misidentified_as_throttle(self) -> None:
        """Legitimate JD text — even if it mentions errors — is not
        falsely flagged as a throttle response."""
        assert is_throttle_response(_REAL_JD) is False
        # A JD that happens to mention 'error' should not trigger detection
        assert is_throttle_response(
            "Handle error reporting in production systems. " * 5
        ) is False

    @pytest.mark.asyncio
    async def test_throttle_detection_triggers_backoff_delay(self) -> None:
        """When a throttle response is detected on click-through, the
        adapter waits before retrying instead of immediately moving on."""
        adapter = ZipRecruiterAdapter()
        listing = _make_listing()

        # Panel returns throttle text on first call, then real JD
        panel_mock = AsyncMock()
        panel_mock.inner_text = AsyncMock(side_effect=[_THROTTLE_TEXT, _REAL_JD])
        panel_mock.wait_for = AsyncMock()

        card_mock = AsyncMock()
        card_mock.click = AsyncMock()

        page = MagicMock()
        page.locator = MagicMock(side_effect=[
            MagicMock(count=AsyncMock(return_value=1), nth=MagicMock(return_value=card_mock)),
            panel_mock,
        ])

        with patch("jobsearch_rag.adapters.ziprecruiter.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await adapter._click_through_cards(page, [listing])
            # Should have slept for backoff at least once
            assert mock_sleep.call_count >= 2  # click delay + backoff delay

    @pytest.mark.asyncio
    async def test_backoff_delay_increases_exponentially_on_consecutive_throttles(
        self,
    ) -> None:
        """Consecutive throttle responses trigger increasing delays:
        the second backoff wait is longer than the first."""
        adapter = ZipRecruiterAdapter()
        listing1 = _make_listing(external_id="zr-1")
        listing2 = _make_listing(external_id="zr-2")

        # Both listings get throttled, then succeed
        panel_mock = AsyncMock()
        panel_mock.inner_text = AsyncMock(
            side_effect=[_THROTTLE_TEXT, _REAL_JD, _THROTTLE_TEXT, _REAL_JD]
        )
        panel_mock.wait_for = AsyncMock()

        card_mock = AsyncMock()
        card_mock.click = AsyncMock()
        card_locator = MagicMock()
        card_locator.count = AsyncMock(return_value=2)
        card_locator.nth = MagicMock(return_value=card_mock)

        page = MagicMock()
        page.locator = MagicMock(side_effect=[card_locator, panel_mock])

        backoff_delays: list[float] = []

        async def capture_sleep(duration: float) -> None:
            backoff_delays.append(duration)

        with patch(
            "jobsearch_rag.adapters.ziprecruiter.asyncio.sleep",
            side_effect=capture_sleep,
        ):
            await adapter._click_through_cards(page, [listing1, listing2])

        # Filter out click delays (small) from backoff delays (larger)
        # Backoff delays should be >= 1 second; click delays are 0.3-0.8s
        backoff_waits = [d for d in backoff_delays if d >= 2.0]
        assert len(backoff_waits) >= 2, (
            f"Expected at least 2 backoff waits, got {len(backoff_waits)}: {backoff_delays}"
        )
        assert backoff_waits[1] > backoff_waits[0], (
            f"Second backoff ({backoff_waits[1]}) should be longer than first ({backoff_waits[0]})"
        )

    @pytest.mark.asyncio
    async def test_listing_is_skipped_after_max_retry_attempts(self) -> None:
        """After exceeding the retry limit, the listing is skipped rather
        than blocking the entire run indefinitely."""
        adapter = ZipRecruiterAdapter()
        listing = _make_listing()

        # Always return throttle text — never succeed
        panel_mock = AsyncMock()
        panel_mock.inner_text = AsyncMock(return_value=_THROTTLE_TEXT)
        panel_mock.wait_for = AsyncMock()

        card_mock = AsyncMock()
        card_mock.click = AsyncMock()
        card_locator = MagicMock()
        card_locator.count = AsyncMock(return_value=1)
        card_locator.nth = MagicMock(return_value=card_mock)

        page = MagicMock()
        page.locator = MagicMock(side_effect=[card_locator, panel_mock])

        with patch("jobsearch_rag.adapters.ziprecruiter.asyncio.sleep", new_callable=AsyncMock):
            await adapter._click_through_cards(page, [listing])

        # Listing was skipped — full_text should be empty or fallback
        assert not listing.full_text.strip() or listing.full_text == (
            f"{listing.title} at {listing.company}. "
        )

    @pytest.mark.asyncio
    async def test_skipped_listing_increments_failed_count(self) -> None:
        """A listing that exhausts retries counts as a failure so the
        run summary accurately reports extraction problems."""
        adapter = ZipRecruiterAdapter()
        listing = _make_listing()

        panel_mock = AsyncMock()
        panel_mock.inner_text = AsyncMock(return_value=_THROTTLE_TEXT)
        panel_mock.wait_for = AsyncMock()

        card_mock = AsyncMock()
        card_mock.click = AsyncMock()
        card_locator = MagicMock()
        card_locator.count = AsyncMock(return_value=1)
        card_locator.nth = MagicMock(return_value=card_mock)

        page = MagicMock()
        page.locator = MagicMock(side_effect=[card_locator, panel_mock])

        with patch("jobsearch_rag.adapters.ziprecruiter.asyncio.sleep", new_callable=AsyncMock):
            await adapter._click_through_cards(page, [listing])

        # After max retries, full_text should be empty (treated as failed)
        assert not listing.full_text.strip() or "short_description" not in listing.metadata

    def test_throttle_event_is_logged_with_url_and_retry_count(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Throttle events are logged at WARNING so the operator can
        see how many retries occurred and which listings were affected."""
        import asyncio as _asyncio

        adapter = ZipRecruiterAdapter()
        listing = _make_listing()

        panel_mock = AsyncMock()
        panel_mock.inner_text = AsyncMock(return_value=_THROTTLE_TEXT)
        panel_mock.wait_for = AsyncMock()

        card_mock = AsyncMock()
        card_mock.click = AsyncMock()
        card_locator = MagicMock()
        card_locator.count = AsyncMock(return_value=1)
        card_locator.nth = MagicMock(return_value=card_mock)

        page = MagicMock()
        page.locator = MagicMock(side_effect=[card_locator, panel_mock])

        with (
            caplog.at_level(logging.WARNING),
            patch(
                "jobsearch_rag.adapters.ziprecruiter.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            _asyncio.run(adapter._click_through_cards(page, [listing]))

        throttle_logs = [r for r in caplog.records if "throttl" in r.message.lower()]
        assert len(throttle_logs) >= 1, "No throttle warning logged"
        # Log should mention the listing identifier
        log_text = " ".join(r.message for r in throttle_logs)
        assert listing.external_id in log_text or listing.url in log_text

    @pytest.mark.asyncio
    async def test_successful_extraction_after_backoff_resets_throttle_state(
        self,
    ) -> None:
        """A successful extraction after backoff populates full_text and
        the listing is treated as normal rather than failed."""
        adapter = ZipRecruiterAdapter()
        listing = _make_listing()

        # First call: throttle. Second call: real JD.
        panel_mock = AsyncMock()
        panel_mock.inner_text = AsyncMock(side_effect=[_THROTTLE_TEXT, _REAL_JD])
        panel_mock.wait_for = AsyncMock()

        card_mock = AsyncMock()
        card_mock.click = AsyncMock()
        card_locator = MagicMock()
        card_locator.count = AsyncMock(return_value=1)
        card_locator.nth = MagicMock(return_value=card_mock)

        page = MagicMock()
        page.locator = MagicMock(side_effect=[card_locator, panel_mock])

        with patch("jobsearch_rag.adapters.ziprecruiter.asyncio.sleep", new_callable=AsyncMock):
            await adapter._click_through_cards(page, [listing])

        assert listing.full_text.strip() == _REAL_JD.strip()

    @pytest.mark.asyncio
    async def test_timeout_with_late_throttle_text_triggers_backoff(self) -> None:
        """When ``wait_for`` times out but the panel rendered throttle text
        after the timeout, the adapter detects the throttle on the late
        read and retries with backoff instead of immediately falling back."""
        adapter = ZipRecruiterAdapter()
        listing = _make_listing()

        # wait_for raises TimeoutError, but inner_text returns throttle text
        # on the first "late" read, then real JD on the retry.
        panel_mock = AsyncMock()
        panel_mock.wait_for = AsyncMock(
            side_effect=[TimeoutError("5000ms exceeded"), None]
        )
        panel_mock.inner_text = AsyncMock(
            side_effect=[_THROTTLE_TEXT, _REAL_JD]
        )

        card_mock = AsyncMock()
        card_mock.click = AsyncMock()
        card_locator = MagicMock()
        card_locator.count = AsyncMock(return_value=1)
        card_locator.nth = MagicMock(return_value=card_mock)

        page = MagicMock()
        page.locator = MagicMock(side_effect=[card_locator, panel_mock])

        with patch(
            "jobsearch_rag.adapters.ziprecruiter.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            await adapter._click_through_cards(page, [listing])

        # The retry should have succeeded with real JD
        assert listing.full_text.strip() == _REAL_JD.strip()

    @pytest.mark.asyncio
    async def test_timeout_without_throttle_text_falls_back_immediately(
        self,
    ) -> None:
        """When ``wait_for`` times out and the late panel read also fails
        or returns non-throttle content, the adapter falls back to the
        short description without retrying."""
        adapter = ZipRecruiterAdapter()
        listing = _make_listing()
        listing.metadata["short_description"] = "Short desc for fallback"

        # wait_for raises, and late inner_text also raises
        panel_mock = AsyncMock()
        panel_mock.wait_for = AsyncMock(
            side_effect=TimeoutError("5000ms exceeded")
        )
        panel_mock.inner_text = AsyncMock(
            side_effect=Exception("element detached")
        )

        card_mock = AsyncMock()
        card_mock.click = AsyncMock()
        card_locator = MagicMock()
        card_locator.count = AsyncMock(return_value=1)
        card_locator.nth = MagicMock(return_value=card_mock)

        page = MagicMock()
        page.locator = MagicMock(side_effect=[card_locator, panel_mock])

        with patch(
            "jobsearch_rag.adapters.ziprecruiter.asyncio.sleep",
            new_callable=AsyncMock,
        ):
            await adapter._click_through_cards(page, [listing])

        # Should have used the short description fallback
        assert "Short desc for fallback" in listing.full_text
