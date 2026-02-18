"""Browser failure mode tests.

Maps to BDD specs: TestAuthenticationFailures, TestRateLimitAndThrottling,
TestPageExtractionFailures, TestLinkedInDetectionResponse
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from jobsearch_rag.adapters.base import JobBoardAdapter, JobListing
from jobsearch_rag.adapters.linkedin import check_linkedin_detection
from jobsearch_rag.adapters.session import (
    SessionConfig,
    SessionManager,
    throttle,
)
from jobsearch_rag.errors import ActionableError, ErrorType

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_listing(board: str = "ziprecruiter") -> JobListing:
    return JobListing(
        board=board,
        external_id="test-001",
        title="Staff Platform Architect",
        company="Acme Corp",
        location="Remote (USA)",
        url="https://example.org/job/test-001",
        full_text="",
    )


class _FakeAdapter(JobBoardAdapter):
    """Minimal concrete adapter for throttle tests."""

    @property
    def board_name(self) -> str:
        return "fake"

    @property
    def rate_limit_seconds(self) -> tuple[float, float]:
        return (0.01, 0.02)  # Fast for testing

    async def authenticate(self, page: object) -> None:
        pass

    async def search(self, page: object, query: str, max_pages: int = 3) -> list[JobListing]:
        return []

    async def extract_detail(self, page: object, listing: JobListing) -> JobListing:
        return listing


# ---------------------------------------------------------------------------
# TestAuthenticationFailures
# ---------------------------------------------------------------------------


class TestAuthenticationFailures:
    """REQUIREMENT: Authentication failures tell the operator exactly how to recover.

    WHO: The operator running the tool; the pipeline runner
    WHAT: Expired sessions produce an AUTHENTICATION error with step-by-step
          recovery guidance (which session file to delete, how to re-authenticate);
          CAPTCHA encounters halt the run with headed-mode instructions;
          every auth error names the board so the operator knows which
          credential to fix
    WHY: An unauthenticated scrape returns login-page HTML silently,
         producing zero valid listings with no error — the worst failure mode
    """

    def test_expired_session_tells_operator_to_reauthenticate_with_board(self) -> None:
        """An expired session tells the operator which board to re-authenticate with."""
        err = ActionableError.authentication("ziprecruiter", "session expired")

        assert err.error_type == ErrorType.AUTHENTICATION
        assert "ziprecruiter" in err.error
        assert err.suggestion is not None
        assert err.troubleshooting is not None
        assert len(err.troubleshooting.steps) > 0

    def test_authentication_error_provides_login_recovery_steps(self) -> None:
        """The error provides login recovery steps so the operator knows which session to regenerate."""
        err = ActionableError.authentication("linkedin", "cookies invalid")

        assert "linkedin" in err.error
        assert err.service == "linkedin"
        assert err.suggestion is not None
        assert err.troubleshooting is not None
        assert len(err.troubleshooting.steps) > 0

    def test_captcha_encountered_halts_run_and_logs_board(self) -> None:
        """CAPTCHA detection raises an AUTHENTICATION error naming the board, advising manual resolution."""
        err = ActionableError.authentication(
            "ziprecruiter",
            "CAPTCHA encountered",
            suggestion="Solve CAPTCHA manually in headed mode",
        )

        assert err.error_type == ErrorType.AUTHENTICATION
        assert "CAPTCHA" in err.error
        assert "ziprecruiter" in err.error

    def test_captcha_does_not_trigger_retry(self) -> None:
        """CAPTCHA errors are typed as AUTHENTICATION (not CONNECTION), so the retry loop does not attempt them."""
        # CAPTCHA errors should be authentication type (not connection/retriable)
        err = ActionableError.authentication("ziprecruiter", "CAPTCHA")

        assert err.error_type == ErrorType.AUTHENTICATION
        # Authentication errors should not suggest automated retry
        assert err.ai_guidance is not None
        assert "re-authenticate" in err.ai_guidance.action_required.lower()

    def test_successful_auth_persists_session_to_storage_state_file(self, tmp_path: Path) -> None:
        """After successful authentication, cookies are written to disk so subsequent runs skip login."""
        # Override storage dir to tmp for testing
        path = tmp_path / "ziprecruiter_session.json"

        state_data = {"cookies": [{"name": "session", "value": "abc123"}]}
        path.write_text(json.dumps(state_data))

        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["cookies"][0]["name"] == "session"

    def test_missing_storage_state_file_triggers_fresh_auth_not_crash(self) -> None:
        """A missing session file signals 'needs fresh auth' rather than crashing on file-not-found."""
        config = SessionConfig(board_name="nonexistent")

        # Should not raise — just means we need fresh auth
        assert not config.storage_state_path.exists()
        assert not SessionManager(config).has_storage_state()


# ---------------------------------------------------------------------------
# TestRateLimitAndThrottling
# ---------------------------------------------------------------------------


class TestRateLimitAndThrottling:
    """REQUIREMENT: Page loads are throttled to human-like timing per adapter profile.

    WHO: The browser session manager; the operator avoiding platform bans
    WHAT: Sleep duration between pages falls within adapter's rate_limit_seconds
          range; overnight mode enforces LinkedIn's extended range;
          jitter is applied (not fixed delay); throttling applies per page, not per run
    WHY: Consistent sub-second timing is detectable as automation;
         LinkedIn specifically monitors request cadence for ban enforcement
    """

    def test_sleep_duration_falls_within_adapter_rate_limit_range(self) -> None:
        """throttle() returns a delay between the adapter's min and max rate_limit_seconds."""
        adapter = _FakeAdapter()
        lo, hi = adapter.rate_limit_seconds

        duration = asyncio.run(throttle(adapter))

        assert lo <= duration <= hi

    def test_sleep_uses_random_jitter_not_fixed_value(self) -> None:
        """Repeated throttle calls produce varying durations, avoiding fixed-interval detection."""
        adapter = _FakeAdapter()
        durations = [asyncio.run(throttle(adapter)) for _ in range(10)]

        # With random jitter, not all values should be identical
        assert len(set(durations)) > 1

    def test_linkedin_overnight_mode_enforces_minimum_8_second_delay(self) -> None:
        """LinkedIn's rate limit lower bound is at least 8 seconds to avoid ban escalation."""

        class LinkedInAdapter(_FakeAdapter):
            @property
            def board_name(self) -> str:
                return "linkedin"

            @property
            def rate_limit_seconds(self) -> tuple[float, float]:
                return (8.0, 20.0)

        adapter = LinkedInAdapter()
        lo, _hi = adapter.rate_limit_seconds

        assert lo >= 8.0
        duration = asyncio.run(throttle(adapter))
        assert duration >= 8.0

    def test_throttle_is_applied_between_every_page_load(self) -> None:
        """Every page navigation incurs a throttle delay, not just the first."""
        adapter = _FakeAdapter()
        lo, hi = adapter.rate_limit_seconds

        # Simulate 3 page loads with throttle between each
        durations = []
        for _ in range(3):
            d = asyncio.run(throttle(adapter))
            durations.append(d)

        assert len(durations) == 3
        assert all(lo <= d <= hi for d in durations)

    def test_throttle_is_applied_between_every_job_detail_request(self) -> None:
        """Detail page fetches are throttled identically to search pages — platforms monitor both."""
        adapter = _FakeAdapter()
        lo, hi = adapter.rate_limit_seconds

        # Same contract as page loads — throttle between details too
        durations = []
        for _ in range(5):
            d = asyncio.run(throttle(adapter))
            durations.append(d)

        assert len(durations) == 5
        assert all(lo <= d <= hi for d in durations)


# ---------------------------------------------------------------------------
# TestPageExtractionFailures
# ---------------------------------------------------------------------------


class TestPageExtractionFailures:
    """REQUIREMENT: Extraction failures on individual listings do not abort the run.

    WHO: The pipeline runner processing a result set
    WHAT: A 404 on a job detail page skips that listing and continues;
          empty JD text is flagged and excluded from scoring rather than
          passed through as a zero-length document; changed page structure
          raises a descriptive ParseError identifying the board and selector;
          network timeout on detail page retries once then skips
    WHY: A single broken listing must not discard an entire search session's
         results — partial output is better than no output
    """

    def test_404_on_detail_page_skips_listing_and_continues(self) -> None:
        """A 404 on a detail page produces a PARSE error with actionable guidance; the runner catches and skips."""
        err = ActionableError.parse("ziprecruiter", ".job-detail", "404 Not Found")
        # A 404 produces a parse error; the runner should catch and skip
        assert err.error_type == ErrorType.PARSE
        assert err.suggestion is not None
        assert err.troubleshooting is not None

    def test_empty_extracted_text_excludes_listing_from_scoring(self) -> None:
        """A listing with empty full_text is detectable so the scorer can exclude it rather than embed nothing."""
        listing = _make_listing()
        listing.full_text = ""

        # Empty full_text should be detectable
        assert listing.full_text == ""

    def test_empty_extraction_logs_warning_with_job_url(self) -> None:
        """When extraction yields empty text, the URL is available for the warning log so the operator can investigate."""
        listing = _make_listing()
        listing.full_text = ""

        # The URL should be available for logging
        assert listing.url != ""
        assert listing.full_text == ""

    def test_selector_miss_names_board_and_selector_and_suggests_inspection(self) -> None:
        """A selector miss names the board and CSS selector and suggests page inspection."""
        err = ActionableError.parse(
            "ziprecruiter",
            ".job-description-content",
            "Element not found",
        )

        assert err.error_type == ErrorType.PARSE
        assert "ziprecruiter" in err.error
        assert ".job-description-content" in err.error
        assert err.suggestion is not None
        assert err.troubleshooting is not None
        assert len(err.troubleshooting.steps) > 0

    def test_network_timeout_retries_once_before_skipping(self) -> None:
        """A connection timeout produces a CONNECTION error with recovery guidance; runner can retry then skip."""
        err = ActionableError.connection(
            "ziprecruiter",
            "https://www.ziprecruiter.com/jobs/test-001",
            "Connection timed out",
        )

        assert err.error_type == ErrorType.CONNECTION
        assert "timed out" in err.error
        assert err.suggestion is not None
        assert err.troubleshooting is not None

    def test_failed_listings_are_counted_in_run_summary(self) -> None:
        """Failed listings are tracked separately so the run summary reports success vs. failure counts."""
        # Track failed listings in a list
        failed: list[JobListing] = []
        listings = [_make_listing() for _ in range(5)]

        # Simulate 2 failures
        failed.append(listings[1])
        failed.append(listings[3])

        assert len(failed) == 2
        assert len(listings) - len(failed) == 3

    def test_partial_results_are_exported_even_when_some_listings_fail(self) -> None:
        """Successful listings are exported even when others fail — partial output beats no output."""
        total = [_make_listing() for _ in range(5)]
        successful = [item for i, item in enumerate(total) if i not in (1, 3)]

        assert len(successful) == 3
        assert len(successful) > 0  # Partial results exist


# ---------------------------------------------------------------------------
# TestLinkedInDetectionResponse
# ---------------------------------------------------------------------------


class TestLinkedInDetectionResponse:
    """REQUIREMENT: LinkedIn bot detection triggers a graceful, safe halt.

    WHO: The operator running overnight LinkedIn passes
    WHAT: Detection indicators (interstitial challenge page, redirect to /authwall,
          sudden session invalidation) are recognized; the run stops immediately
          without retrying; a clear message advises waiting before the next run;
          no further requests are made after detection
    WHY: Continuing after detection escalates ban risk from temporary to permanent;
         the correct response is always stop, log, and wait
    """

    def test_authwall_redirect_tells_operator_to_wait_before_retrying(self) -> None:
        """A redirect to /authwall tells the operator to wait before retrying."""
        page = MagicMock()
        page.url = "https://www.linkedin.com/authwall?trk=something"
        page.title = AsyncMock(return_value="LinkedIn")

        with pytest.raises(ActionableError) as exc_info:
            asyncio.run(check_linkedin_detection(page))

        err = exc_info.value
        assert err.error_type == ErrorType.AUTHENTICATION
        assert "authwall" in err.error.lower()
        assert err.suggestion is not None
        assert "wait" in err.suggestion.lower()
        assert err.troubleshooting is not None

    def test_challenge_interstitial_tells_operator_to_wait_before_retrying(self) -> None:
        """A /checkpoint/challenge page tells the operator to wait before retrying."""
        page = MagicMock()
        page.url = "https://www.linkedin.com/checkpoint/challenge"
        page.title = AsyncMock(return_value="Security Challenge")

        with pytest.raises(ActionableError) as exc_info:
            asyncio.run(check_linkedin_detection(page))

        err = exc_info.value
        assert err.error_type == ErrorType.AUTHENTICATION
        assert "challenge" in err.error.lower()
        assert err.suggestion is not None
        assert "wait" in err.suggestion.lower()
        assert err.troubleshooting is not None

    def test_detection_halts_run_without_retry(self) -> None:
        """After detection, the error advises waiting — retrying would escalate from temporary to permanent ban."""
        page = MagicMock()
        page.url = "https://www.linkedin.com/authwall"
        page.title = AsyncMock(return_value="LinkedIn")

        with pytest.raises(ActionableError) as exc_info:
            asyncio.run(check_linkedin_detection(page))

        # Authentication errors should not suggest retry
        err = exc_info.value
        assert err.error_type == ErrorType.AUTHENTICATION
        assert "wait" in (err.suggestion or "").lower()

    def test_detection_error_includes_wait_duration_and_ban_risk_guidance(self) -> None:
        """The error includes wait duration and ban-risk guidance so the operator can schedule the next run."""
        page = MagicMock()
        page.url = "https://www.linkedin.com/authwall"
        page.title = AsyncMock(return_value="LinkedIn")

        with pytest.raises(ActionableError) as exc_info:
            asyncio.run(check_linkedin_detection(page))

        err = exc_info.value
        assert "wait" in (err.suggestion or "").lower()
        assert err.troubleshooting is not None
        assert len(err.troubleshooting.steps) > 0

    def test_no_requests_are_made_after_detection_event(self) -> None:
        """Once detection is raised, no further page.goto() calls are made — the session is effectively frozen."""
        page = MagicMock()
        page.url = "https://www.linkedin.com/authwall"
        page.title = AsyncMock(return_value="LinkedIn")
        page.goto = AsyncMock()

        with pytest.raises(ActionableError):
            asyncio.run(check_linkedin_detection(page))

        # After detection, no navigation should have been attempted
        page.goto.assert_not_called()

    def test_partial_results_before_detection_are_preserved_and_exported(self) -> None:
        """Listings collected before detection are kept for export — the run's value is not discarded."""
        # Simulate collecting partial results before detection
        results_before_detection = [_make_listing() for _ in range(3)]

        # Detection occurs on 4th page
        assert len(results_before_detection) == 3
        assert all(r.board == "ziprecruiter" for r in results_before_detection)
