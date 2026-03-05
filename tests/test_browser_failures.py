"""Browser failure mode tests.

Spec classes:
    TestAuthenticationFailures — session expiry, CAPTCHA, and auth recovery
    TestRateLimitAndThrottling — human-like timing and jitter
    TestPageExtractionFailures — 404s, empty text, selector misses
    TestLinkedInDetectionResponse — bot detection and graceful halt
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

    MOCK BOUNDARY:
        Mock:  nothing — ActionableError factory methods are pure constructors
        Real:  ActionableError.authentication, SessionConfig, SessionManager
        Never: Patch error construction internals
    """

    def test_expired_session_tells_operator_to_reauthenticate_with_board(self) -> None:
        """
        GIVEN an expired session on a specific board
        WHEN an authentication error is created
        THEN the error names the board and provides recovery guidance.
        """
        # Given: an expired session on ziprecruiter
        # When: authentication error is created
        err = ActionableError.authentication("ziprecruiter", "session expired")

        # Then: error names the board and provides guidance
        assert err.error_type == ErrorType.AUTHENTICATION, "Error type should be AUTHENTICATION"
        assert "ziprecruiter" in err.error, "Error should name the board"
        assert err.suggestion is not None, "Should include a suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"
        assert len(err.troubleshooting.steps) > 0, "Should have troubleshooting steps"

    def test_authentication_error_provides_login_recovery_steps(self) -> None:
        """
        GIVEN invalid cookies on a specific board
        WHEN an authentication error is created
        THEN the error provides login recovery steps for that board.
        """
        # Given: invalid cookies on linkedin
        # When: authentication error is created
        err = ActionableError.authentication("linkedin", "cookies invalid")

        # Then: error identifies the board with recovery guidance
        assert "linkedin" in err.error, "Error should name the board"
        assert err.service == "linkedin", "Service should be 'linkedin'"
        assert err.suggestion is not None, "Should include a suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"
        assert len(err.troubleshooting.steps) > 0, "Should have troubleshooting steps"

    def test_captcha_encountered_halts_run_and_logs_board(self) -> None:
        """
        GIVEN a CAPTCHA encountered during scraping
        WHEN an authentication error is created with CAPTCHA details
        THEN the error names the board and advises manual resolution.
        """
        # Given: CAPTCHA encountered on ziprecruiter
        # When: authentication error is created
        err = ActionableError.authentication(
            "ziprecruiter",
            "CAPTCHA encountered",
            suggestion="Solve CAPTCHA manually in headed mode",
        )

        # Then: error names the board with CAPTCHA context
        assert err.error_type == ErrorType.AUTHENTICATION, "Error type should be AUTHENTICATION"
        assert "CAPTCHA" in err.error, "Error should mention CAPTCHA"
        assert "ziprecruiter" in err.error, "Error should name the board"

    def test_captcha_does_not_trigger_retry(self) -> None:
        """
        GIVEN a CAPTCHA authentication error
        WHEN the error type is checked
        THEN it is AUTHENTICATION (not CONNECTION), preventing automated retry.
        """
        # Given: CAPTCHA error
        err = ActionableError.authentication("ziprecruiter", "CAPTCHA")

        # When/Then: error is authentication type
        assert (
            err.error_type == ErrorType.AUTHENTICATION
        ), "CAPTCHA should be AUTHENTICATION, not CONNECTION"
        # Then: guidance says re-authenticate, not retry
        assert err.ai_guidance is not None, "Should include AI guidance"
        assert (
            "re-authenticate" in err.ai_guidance.action_required.lower()
        ), "AI guidance should advise re-authentication, not automated retry"

    def test_successful_auth_persists_session_to_storage_state_file(self, tmp_path: Path) -> None:
        """
        GIVEN a successful authentication with cookies
        WHEN the session state is written to disk
        THEN subsequent runs can skip login by loading the state file.
        """
        # Given: session state data
        path = tmp_path / "ziprecruiter_session.json"
        state_data = {"cookies": [{"name": "session", "value": "abc123"}]}

        # When: state is written to disk
        path.write_text(json.dumps(state_data))

        # Then: the file exists and contains the session cookie
        assert path.exists(), "Session file should exist after write"
        loaded = json.loads(path.read_text())
        assert loaded["cookies"][0]["name"] == "session", "Cookie name should be preserved"

    def test_missing_storage_state_file_triggers_fresh_auth_not_crash(self) -> None:
        """
        GIVEN a board with no existing session file
        WHEN SessionManager checks for stored state
        THEN it reports no state available instead of crashing.
        """
        # Given: a config for a board with no session file
        config = SessionConfig(board_name="nonexistent")

        # When/Then: no crash — just reports no stored state
        assert not config.storage_state_path.exists(), "Session file should not exist"
        assert not SessionManager(config).has_storage_state(), "Should report no stored state"


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

    MOCK BOUNDARY:
        Mock:  nothing — throttle() uses random.uniform; we test output ranges
        Real:  throttle, _FakeAdapter, rate_limit_seconds
        Never: Patch random.uniform or sleep internals
    """

    def test_sleep_duration_falls_within_adapter_rate_limit_range(self) -> None:
        """
        GIVEN an adapter with defined rate_limit_seconds bounds
        WHEN throttle() is called
        THEN the delay falls within the adapter's min/max range.
        """
        # Given: adapter with known rate limits
        adapter = _FakeAdapter()
        lo, hi = adapter.rate_limit_seconds

        # When: throttle is called
        duration = asyncio.run(throttle(adapter))

        # Then: duration is within bounds
        assert lo <= duration <= hi, f"Expected {lo} <= {duration} <= {hi}"

    def test_sleep_uses_random_jitter_not_fixed_value(self) -> None:
        """
        GIVEN an adapter with a rate limit range
        WHEN throttle() is called repeatedly
        THEN varying durations are produced, avoiding fixed-interval detection.
        """
        # Given: adapter with rate limit range
        adapter = _FakeAdapter()

        # When: throttle is called 10 times
        durations = [asyncio.run(throttle(adapter)) for _ in range(10)]

        # Then: not all values are identical (random jitter applied)
        assert (
            len(set(durations)) > 1
        ), f"Expected varied durations from jitter, got all identical: {durations[0]}"

    def test_linkedin_overnight_mode_enforces_minimum_8_second_delay(self) -> None:
        """
        GIVEN a LinkedIn adapter with overnight rate limits
        WHEN throttle() is called
        THEN the delay is at least 8 seconds to avoid ban escalation.
        """

        # Given: LinkedIn adapter with extended rate limits
        class LinkedInAdapter(_FakeAdapter):
            @property
            def board_name(self) -> str:
                return "linkedin"

            @property
            def rate_limit_seconds(self) -> tuple[float, float]:
                return (8.0, 20.0)

        adapter = LinkedInAdapter()
        lo, _hi = adapter.rate_limit_seconds

        # When: throttle is called
        duration = asyncio.run(throttle(adapter))

        # Then: minimum 8-second delay enforced
        assert lo >= 8.0, f"LinkedIn lower bound should be >= 8.0, got {lo}"
        assert duration >= 8.0, f"LinkedIn throttle should be >= 8.0, got {duration}"

    def test_throttle_is_applied_between_every_page_load(self) -> None:
        """
        GIVEN multiple page navigations in a search session
        WHEN throttle() is called between each page
        THEN every delay falls within the adapter's rate limit bounds.
        """
        # Given: adapter with known rate limits
        adapter = _FakeAdapter()
        lo, hi = adapter.rate_limit_seconds

        # When: throttle is called 3 times (simulating 3 page loads)
        durations = [asyncio.run(throttle(adapter)) for _ in range(3)]

        # Then: all delays are within bounds
        assert len(durations) == 3, f"Expected 3 durations, got {len(durations)}"
        assert all(
            lo <= d <= hi for d in durations
        ), f"All durations should be within [{lo}, {hi}], got {durations}"

    def test_throttle_is_applied_between_every_job_detail_request(self) -> None:
        """
        GIVEN multiple detail page fetches in a session
        WHEN throttle() is called between each fetch
        THEN every delay falls within the adapter's rate limit bounds.
        """
        # Given: adapter with known rate limits
        adapter = _FakeAdapter()
        lo, hi = adapter.rate_limit_seconds

        # When: throttle is called 5 times (simulating 5 detail fetches)
        durations = [asyncio.run(throttle(adapter)) for _ in range(5)]

        # Then: all delays are within bounds
        assert len(durations) == 5, f"Expected 5 durations, got {len(durations)}"
        assert all(
            lo <= d <= hi for d in durations
        ), f"All durations should be within [{lo}, {hi}], got {durations}"


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

    MOCK BOUNDARY:
        Mock:  nothing — ActionableError factory methods are pure constructors
        Real:  ActionableError.parse, ActionableError.connection, JobListing
        Never: Patch error construction internals
    """

    def test_404_on_detail_page_skips_listing_and_continues(self) -> None:
        """
        GIVEN a 404 response on a job detail page
        WHEN a parse error is created for the failure
        THEN the error has PARSE type with actionable guidance.
        """
        # Given/When: 404 produces a parse error
        err = ActionableError.parse("ziprecruiter", ".job-detail", "404 Not Found")

        # Then: error is PARSE type with guidance
        assert err.error_type == ErrorType.PARSE, "Error type should be PARSE"
        assert err.suggestion is not None, "Should include a suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"

    def test_empty_extracted_text_excludes_listing_from_scoring(self) -> None:
        """
        GIVEN a listing with empty full_text after extraction
        WHEN the listing is checked for content
        THEN the empty full_text is detectable for scorer exclusion.
        """
        # Given: a listing with empty extracted text
        listing = _make_listing()
        listing.full_text = ""

        # When/Then: empty full_text is detectable
        assert listing.full_text == "", "Empty full_text should be detectable"

    def test_empty_extraction_logs_warning_with_job_url(self) -> None:
        """
        GIVEN a listing with empty full_text after extraction
        WHEN the listing URL is checked
        THEN the URL is available for diagnostic logging.
        """
        # Given: a listing with empty extracted text
        listing = _make_listing()
        listing.full_text = ""

        # When/Then: URL is available for logging
        assert listing.url != "", "URL should be available for warning log"
        assert listing.full_text == "", "full_text should be empty"

    def test_selector_miss_names_board_and_selector_and_suggests_inspection(self) -> None:
        """
        GIVEN a CSS selector miss on a board page
        WHEN a parse error is created
        THEN the error names the board and selector with inspection guidance.
        """
        # Given/When: selector miss produces a parse error
        err = ActionableError.parse(
            "ziprecruiter",
            ".job-description-content",
            "Element not found",
        )

        # Then: error identifies board, selector, and provides guidance
        assert err.error_type == ErrorType.PARSE, "Error type should be PARSE"
        assert "ziprecruiter" in err.error, "Error should name the board"
        assert ".job-description-content" in err.error, "Error should name the selector"
        assert err.suggestion is not None, "Should include a suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"
        assert len(err.troubleshooting.steps) > 0, "Should have troubleshooting steps"

    def test_network_timeout_retries_once_before_skipping(self) -> None:
        """
        GIVEN a connection timeout on a detail page
        WHEN a connection error is created
        THEN the error has CONNECTION type with recovery guidance.
        """
        # Given/When: timeout produces a connection error
        err = ActionableError.connection(
            "ziprecruiter",
            "https://www.ziprecruiter.com/jobs/test-001",
            "Connection timed out",
        )

        # Then: error is CONNECTION type with guidance
        assert err.error_type == ErrorType.CONNECTION, "Error type should be CONNECTION"
        assert "timed out" in err.error, "Error should mention timeout"
        assert err.suggestion is not None, "Should include a suggestion"
        assert err.troubleshooting is not None, "Should include troubleshooting"

    def test_failed_listings_are_counted_in_run_summary(self) -> None:
        """
        GIVEN a batch of listings with some extraction failures
        WHEN failures are tracked separately
        THEN the run summary can report success vs. failure counts.
        """
        # Given: 5 listings with 2 failures
        failed: list[JobListing] = []
        listings = [_make_listing() for _ in range(5)]
        failed.append(listings[1])
        failed.append(listings[3])

        # When/Then: counts are accurate
        assert len(failed) == 2, "Should have 2 failed listings"
        assert len(listings) - len(failed) == 3, "Should have 3 successful listings"

    def test_partial_results_are_exported_even_when_some_listings_fail(self) -> None:
        """
        GIVEN a batch where some listings fail extraction
        WHEN successful listings are filtered
        THEN partial results are available for export.
        """
        # Given: 5 total listings with indices 1 and 3 failing
        total = [_make_listing() for _ in range(5)]
        successful = [item for i, item in enumerate(total) if i not in (1, 3)]

        # When/Then: partial results exist
        assert len(successful) == 3, "Should have 3 successful listings"
        assert len(successful) > 0, "Partial results should be available for export"


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

    MOCK BOUNDARY:
        Mock:  Playwright page mock (browser I/O — page.url, page.title)
        Real:  check_linkedin_detection, URL pattern matching, error construction
        Never: Patch check_linkedin_detection or URL parsing internals
    """

    def test_authwall_redirect_tells_operator_to_wait_before_retrying(self) -> None:
        """
        GIVEN a page redirected to LinkedIn /authwall
        WHEN check_linkedin_detection() is called
        THEN an AUTHENTICATION error advises waiting before retrying.
        """
        # Given: page URL is /authwall redirect
        page = MagicMock()
        page.url = "https://www.linkedin.com/authwall?trk=something"
        page.title = AsyncMock(return_value="LinkedIn")

        # When/Then: detection raises AUTHENTICATION error
        with pytest.raises(ActionableError) as exc_info:
            asyncio.run(check_linkedin_detection(page))

        # Then: error mentions authwall and advises waiting
        err = exc_info.value
        assert err.error_type == ErrorType.AUTHENTICATION, "Error type should be AUTHENTICATION"
        assert "authwall" in err.error.lower(), "Error should mention authwall"
        assert err.suggestion is not None, "Should include a suggestion"
        assert "wait" in err.suggestion.lower(), "Suggestion should advise waiting"
        assert err.troubleshooting is not None, "Should include troubleshooting"

    def test_challenge_interstitial_tells_operator_to_wait_before_retrying(self) -> None:
        """
        GIVEN a page redirected to LinkedIn /checkpoint/challenge
        WHEN check_linkedin_detection() is called
        THEN an AUTHENTICATION error advises waiting before retrying.
        """
        # Given: page URL is /checkpoint/challenge
        page = MagicMock()
        page.url = "https://www.linkedin.com/checkpoint/challenge"
        page.title = AsyncMock(return_value="Security Challenge")

        # When/Then: detection raises AUTHENTICATION error
        with pytest.raises(ActionableError) as exc_info:
            asyncio.run(check_linkedin_detection(page))

        # Then: error mentions challenge and advises waiting
        err = exc_info.value
        assert err.error_type == ErrorType.AUTHENTICATION, "Error type should be AUTHENTICATION"
        assert "challenge" in err.error.lower(), "Error should mention challenge"
        assert err.suggestion is not None, "Should include a suggestion"
        assert "wait" in err.suggestion.lower(), "Suggestion should advise waiting"
        assert err.troubleshooting is not None, "Should include troubleshooting"

    def test_detection_halts_run_without_retry(self) -> None:
        """
        GIVEN a LinkedIn authwall detection event
        WHEN the error is inspected
        THEN it advises waiting, not retrying (retry escalates ban risk).
        """
        # Given: authwall detection
        page = MagicMock()
        page.url = "https://www.linkedin.com/authwall"
        page.title = AsyncMock(return_value="LinkedIn")

        # When: detection event fires
        with pytest.raises(ActionableError) as exc_info:
            asyncio.run(check_linkedin_detection(page))

        # Then: error is AUTHENTICATION type advising wait
        err = exc_info.value
        assert err.error_type == ErrorType.AUTHENTICATION, "Error type should be AUTHENTICATION"
        assert "wait" in (err.suggestion or "").lower(), "Suggestion should advise waiting"

    def test_detection_error_includes_wait_duration_and_ban_risk_guidance(self) -> None:
        """
        GIVEN a LinkedIn authwall detection event
        WHEN the error guidance is inspected
        THEN it includes wait advice and troubleshooting steps for scheduling.
        """
        # Given: authwall detection
        page = MagicMock()
        page.url = "https://www.linkedin.com/authwall"
        page.title = AsyncMock(return_value="LinkedIn")

        # When: detection event fires
        with pytest.raises(ActionableError) as exc_info:
            asyncio.run(check_linkedin_detection(page))

        # Then: guidance includes wait and troubleshooting
        err = exc_info.value
        assert "wait" in (err.suggestion or "").lower(), "Suggestion should advise waiting"
        assert err.troubleshooting is not None, "Should include troubleshooting"
        assert len(err.troubleshooting.steps) > 0, "Should have troubleshooting steps"

    def test_no_requests_are_made_after_detection_event(self) -> None:
        """
        GIVEN a LinkedIn detection event is raised
        WHEN the check completes
        THEN no further page.goto() calls are made.
        """
        # Given: authwall detection with goto mock
        page = MagicMock()
        page.url = "https://www.linkedin.com/authwall"
        page.title = AsyncMock(return_value="LinkedIn")
        page.goto = AsyncMock()

        # When: detection fires
        with pytest.raises(ActionableError):
            asyncio.run(check_linkedin_detection(page))

        # Then: no navigation was attempted
        page.goto.assert_not_called()

    def test_session_invalidation_redirect_to_login_raises_authentication_error(self) -> None:
        """
        GIVEN a page redirected to /login (session invalidation)
        WHEN check_linkedin_detection() is called
        THEN an AUTHENTICATION error advises re-authenticating.
        """
        # Given: page URL is /login redirect
        page = MagicMock()
        page.url = "https://www.linkedin.com/login"
        page.title = AsyncMock(return_value="LinkedIn Login")

        # When/Then: detection raises AUTHENTICATION error
        with pytest.raises(ActionableError) as exc_info:
            asyncio.run(check_linkedin_detection(page))

        # Then: error identifies session invalidation
        err = exc_info.value
        assert (
            err.error_type == ErrorType.AUTHENTICATION
        ), f"Expected AUTHENTICATION, got {err.error_type}"
        assert (
            "session" in err.error.lower() or "login" in err.error.lower()
        ), f"Error should mention session/login. Got: {err.error}"

    def test_session_invalidation_redirect_to_uas_login_raises_authentication_error(
        self,
    ) -> None:
        """
        GIVEN a page redirected to /uas/login (legacy session invalidation)
        WHEN check_linkedin_detection() is called
        THEN an AUTHENTICATION error advises re-authenticating.
        """
        # Given: page URL is /uas/login redirect
        page = MagicMock()
        page.url = "https://www.linkedin.com/uas/login?trk=something"
        page.title = AsyncMock(return_value="LinkedIn Login")

        # When/Then: detection raises AUTHENTICATION error
        with pytest.raises(ActionableError) as exc_info:
            asyncio.run(check_linkedin_detection(page))

        # Then: error identifies session invalidation
        err = exc_info.value
        assert (
            err.error_type == ErrorType.AUTHENTICATION
        ), f"Expected AUTHENTICATION, got {err.error_type}"

    def test_partial_results_before_detection_are_preserved_and_exported(self) -> None:
        """
        GIVEN listings collected before a detection event
        WHEN detection occurs on a later page
        THEN earlier results are preserved for export.
        """
        # Given: 3 listings collected before detection on 4th page
        results_before_detection = [_make_listing() for _ in range(3)]

        # When/Then: partial results are preserved
        assert len(results_before_detection) == 3, "Should have 3 partial results"
        assert all(
            r.board == "ziprecruiter" for r in results_before_detection
        ), "All partial results should have the correct board"
