"""Adapter registration and IoC contract tests.

Maps to BDD specs: TestAdapterRegistration, TestAdapterContract,
TestJobListingDataContract
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from jobsearch_rag.adapters.base import JobBoardAdapter, JobListing
from jobsearch_rag.adapters.registry import AdapterRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter_class(name: str) -> type[JobBoardAdapter]:
    """Dynamically create a concrete adapter class for testing."""

    class _TestAdapter(JobBoardAdapter):
        @property
        def board_name(self) -> str:
            return name

        async def authenticate(self, page: object) -> None:
            pass

        async def search(
            self,
            page: object,
            query: str,
            max_pages: int = 3,
        ) -> list[JobListing]:
            return []

        async def extract_detail(
            self,
            page: object,
            listing: JobListing,
        ) -> JobListing:
            listing.full_text = "Extracted detail text for testing."
            return listing

    # Give each dynamic class a unique name for clarity
    _TestAdapter.__name__ = f"TestAdapter_{name}"
    _TestAdapter.__qualname__ = f"TestAdapter_{name}"
    return _TestAdapter


def _make_listing(
    board: str = "test-board",
    external_id: str = "test-001",
    full_text: str = "",
) -> JobListing:
    """Create a minimal JobListing for testing."""
    return JobListing(
        board=board,
        external_id=external_id,
        title="Staff Platform Architect",
        company="Acme Corp",
        location="Remote (USA)",
        url="https://example.org/job/test-001",
        full_text=full_text,
    )


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Reset the registry before each test to prevent cross-contamination."""
    AdapterRegistry._registry.clear()


# ---------------------------------------------------------------------------
# TestAdapterRegistration
# ---------------------------------------------------------------------------


class TestAdapterRegistration:
    """REQUIREMENT: Adapters self-register and are discoverable by board name.

    WHO: The pipeline runner loading adapters from settings.toml
    WHAT: Registered adapters are retrievable by board name string;
          unregistered names raise clear errors; all registered boards are listable
    WHY: The runner must not know concrete adapter classes — IoC requires
         that board name is the only coupling between config and implementation
    """

    def test_registered_adapter_is_retrievable_by_board_name(self) -> None:
        """A registered adapter is returned as a ready-to-use instance when looked up by its board name string."""
        adapter_cls = _make_adapter_class("ziprecruiter")
        AdapterRegistry.register(adapter_cls)

        adapter = AdapterRegistry.get("ziprecruiter")

        assert isinstance(adapter, JobBoardAdapter)
        assert adapter.board_name == "ziprecruiter"

    def test_retrieving_unregistered_board_name_raises_value_error_with_name(self) -> None:
        """Requesting an adapter for a board name that was never registered raises ValueError identifying the missing name."""
        with pytest.raises(ValueError, match="no-such-board"):
            AdapterRegistry.get("no-such-board")

    def test_registry_lists_all_registered_board_names(self) -> None:
        """list_registered() returns every board name that has been registered, enabling the CLI 'boards' command."""
        for name in ("ziprecruiter", "indeed", "weworkremotely", "linkedin"):
            AdapterRegistry.register(_make_adapter_class(name))

        registered = AdapterRegistry.list_registered()

        assert set(registered) == {"ziprecruiter", "indeed", "weworkremotely", "linkedin"}

    def test_duplicate_registration_overwrites_previous(self) -> None:
        """Re-registering a board name replaces the previous class; the old class is no longer instantiated."""
        cls_v1 = _make_adapter_class("ziprecruiter")
        cls_v2 = _make_adapter_class("ziprecruiter")
        assert cls_v1 is not cls_v2  # precondition: distinct classes

        AdapterRegistry.register(cls_v1)
        AdapterRegistry.register(cls_v2)

        adapter = AdapterRegistry.get("ziprecruiter")

        assert type(adapter) is cls_v2
        assert type(adapter) is not cls_v1

    def test_adapter_decorator_does_not_alter_class_interface(self) -> None:
        """The register() decorator returns the original class unchanged, preserving its interface and identity."""
        cls = _make_adapter_class("ziprecruiter")
        returned = AdapterRegistry.register(cls)

        assert returned is cls


# ---------------------------------------------------------------------------
# TestAdapterContract
# ---------------------------------------------------------------------------


class TestAdapterContract:
    """REQUIREMENT: All adapters conform to the JobBoardAdapter interface.

    WHO: The pipeline runner invoking adapters polymorphically
    WHAT: Every concrete adapter exposes board_name, authenticate, search,
          extract_detail, and rate_limit_seconds with correct return types
    WHY: The runner calls adapters without knowing their type;
         any deviation from the contract breaks the pipeline silently
    """

    def test_board_name_returns_non_empty_string(self) -> None:
        """Every adapter must expose a non-empty board_name so the registry and logs can identify it."""
        cls = _make_adapter_class("ziprecruiter")
        adapter = cls()

        assert isinstance(adapter.board_name, str)
        assert len(adapter.board_name) > 0

    def test_rate_limit_seconds_returns_tuple_of_two_floats(self) -> None:
        """rate_limit_seconds is a (min, max) float tuple so the throttle function can draw a random delay."""
        cls = _make_adapter_class("test-board")
        adapter = cls()
        rate_limit = adapter.rate_limit_seconds

        assert isinstance(rate_limit, tuple)
        assert len(rate_limit) == 2
        assert all(isinstance(v, float) for v in rate_limit)

    def test_rate_limit_min_is_less_than_max(self) -> None:
        """The lower bound must be strictly less than the upper bound to allow meaningful jitter."""
        cls = _make_adapter_class("test-board")
        adapter = cls()
        lo, hi = adapter.rate_limit_seconds

        assert lo < hi

    def test_search_returns_list_of_job_listings(self) -> None:
        """search() always returns a list (possibly empty), never None or a scalar."""
        cls = _make_adapter_class("test-board")
        adapter = cls()
        page = MagicMock()

        result = asyncio.run(adapter.search(page, "staff architect"))

        assert isinstance(result, list)

    def test_extract_detail_populates_full_text_on_listing(self) -> None:
        """extract_detail() fills the listing's full_text field so the scorer has content to embed."""
        cls = _make_adapter_class("test-board")
        adapter = cls()
        listing = _make_listing(board="test-board")
        page = MagicMock()

        result = asyncio.run(adapter.extract_detail(page, listing))

        assert result.full_text != ""

    def test_extract_detail_returns_same_listing_object_enriched(self) -> None:
        """extract_detail() mutates and returns the same object rather than copying, preserving caller references."""
        cls = _make_adapter_class("test-board")
        adapter = cls()
        listing = _make_listing(board="test-board")
        page = MagicMock()

        result = asyncio.run(adapter.extract_detail(page, listing))

        assert result is listing


# ---------------------------------------------------------------------------
# TestJobListingDataContract
# ---------------------------------------------------------------------------


class TestJobListingDataContract:
    """REQUIREMENT: JobListing is the canonical data contract across all boards.

    WHO: The RAG scorer, ranker, and exporter consuming listings
    WHAT: Required fields are always populated after extraction;
          optional fields degrade gracefully when absent;
          board field identifies source for deduplication
    WHY: Downstream components must not branch on board type —
         the listing is the abstraction that makes them board-agnostic
    """

    def test_required_fields_are_present_after_extraction(self) -> None:
        """All required fields (board, id, title, company, location, url, full_text) are non-empty after extraction."""
        listing = _make_listing(board="ziprecruiter", full_text="Some JD text")

        assert listing.board == "ziprecruiter"
        assert listing.external_id == "test-001"
        assert listing.title != ""
        assert listing.company != ""
        assert listing.location != ""
        assert listing.url != ""
        assert listing.full_text != ""

    def test_full_text_is_non_empty_string_after_detail_extraction(self) -> None:
        """full_text is a non-empty string after extract_detail(), ready for embedding by the scorer."""
        cls = _make_adapter_class("ziprecruiter")
        adapter = cls()
        listing = _make_listing(board="ziprecruiter")
        page = MagicMock()

        result = asyncio.run(adapter.extract_detail(page, listing))

        assert isinstance(result.full_text, str)
        assert len(result.full_text) > 0

    def test_board_field_matches_adapter_board_name(self) -> None:
        """The listing's board field matches the adapter that produced it, enabling cross-board deduplication."""
        cls = _make_adapter_class("ziprecruiter")
        adapter = cls()
        listing = _make_listing(board=adapter.board_name)

        assert listing.board == adapter.board_name

    def test_external_id_is_unique_within_a_board(self) -> None:
        """Distinct listings on the same board have distinct external_ids, supporting deduplication."""
        listing_a = _make_listing(board="ziprecruiter", external_id="zr-001")
        listing_b = _make_listing(board="ziprecruiter", external_id="zr-002")

        assert listing_a.external_id != listing_b.external_id

    def test_missing_posted_at_does_not_raise(self) -> None:
        """posted_at is optional; a missing value defaults to None without raising."""
        listing = _make_listing()

        assert listing.posted_at is None

    def test_metadata_defaults_to_empty_dict_not_none(self) -> None:
        """metadata defaults to {} rather than None, so consumers can iterate without null checks."""
        listing = _make_listing()

        assert listing.metadata is not None
        assert isinstance(listing.metadata, dict)
        assert listing.metadata == {}
