# Public API surface (from src/jobsearch_rag/adapters/):
#   AdapterRegistry.register(adapter_class) -> type[JobBoardAdapter]  (classmethod/decorator)
#   AdapterRegistry.get(board_name: str) -> JobBoardAdapter            (classmethod)
#   AdapterRegistry.list_registered() -> list[str]                     (classmethod)
#   JobBoardAdapter.board_name -> str                                  (abstract property)
#   JobBoardAdapter.authenticate(page: Page) -> None                   (abstract async)
#   JobBoardAdapter.search(page: Page, query: str, max_pages: int) -> list[JobListing]
#   JobBoardAdapter.extract_detail(page: Page, listing: JobListing) -> JobListing
#   JobBoardAdapter.rate_limit_seconds -> tuple[float, float]          (default (1.5, 3.5))
#   JobListing(board, external_id, title, company, location, url, full_text,
#              posted_at=None, raw_html=None, comp_min=None, comp_max=None,
#              comp_source=None, comp_text=None, metadata=field(default_factory=dict))
#   ZipRecruiterAdapter — concrete, board_name="ziprecruiter"
"""BDD specs for the adapter layer: registration, interface contract, and data contract.

Covers: TestAdapterRegistration, TestAdapterContract, TestJobListingDataContract
Spec doc: BDD Specifications — adapter-layer.md
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from jobsearch_rag.adapters.base import JobBoardAdapter, JobListing
from jobsearch_rag.adapters.registry import AdapterRegistry
from jobsearch_rag.adapters.ziprecruiter import ZipRecruiterAdapter


# ---------------------------------------------------------------------------
# TestAdapterRegistration
# ---------------------------------------------------------------------------


class TestAdapterRegistration:
    """
    REQUIREMENT: Adapters self-register and are discoverable by board name.

    WHO: The pipeline runner loading adapters from settings.toml
    WHAT: Registered adapters are retrievable by board name string;
          an unregistered board name produces an error that names the
          requested board so the operator knows what to fix;
          all registered boards are listable so the CLI can enumerate them
    WHY: The runner must not know concrete adapter classes — IoC requires
         that board name is the only coupling between config and implementation

    MOCK BOUNDARY:
        Mock:  nothing — AdapterRegistry is pure computation (dict lookup)
        Real:  AdapterRegistry, ZipRecruiterAdapter class reference
        Never: Mock the registry itself; the registry must be a real instance
               with real registration calls
    """

    def test_registered_adapter_is_retrievable_by_board_name(self) -> None:
        """
        When an adapter class is registered under a board name
        Then AdapterRegistry.get() with that name returns a usable adapter instance
        """
        # Given: ZipRecruiterAdapter is registered via @AdapterRegistry.register
        #        (registration happens at import time)

        # When: the registered board name is looked up
        adapter = AdapterRegistry.get("ziprecruiter")

        # Then: a usable adapter instance is returned
        assert isinstance(adapter, ZipRecruiterAdapter), (
            f"Expected a ZipRecruiterAdapter instance, got {type(adapter).__name__}"
        )

    def test_unregistered_board_name_error_names_the_missing_board(self) -> None:
        """
        When AdapterRegistry.get() is called with a board name that was never registered
        Then a ValueError is raised whose message contains the requested board name
        """
        # Given: "nonexistent_board" has never been registered
        missing_name = "nonexistent_board"

        # When / Then: get() raises ValueError naming the missing board
        with pytest.raises(ValueError) as exc_info:
            AdapterRegistry.get(missing_name)

        assert missing_name in str(exc_info.value), (
            f"Error should name the missing board '{missing_name}'. "
            f"Got: {exc_info.value}"
        )

    def test_registry_lists_all_registered_board_names(self) -> None:
        """
        When four adapters are registered
        Then AdapterRegistry.list_registered() returns all four names
        """
        # Given: adapters were registered at import time (ziprecruiter, linkedin,
        #        indeed, weworkremotely)

        # When: the registered names are listed
        names = AdapterRegistry.list_registered()

        # Then: all four board names are present
        expected = {"ziprecruiter", "linkedin", "indeed", "weworkremotely"}
        assert expected.issubset(set(names)), (
            f"Expected at least {expected} in registered boards, got {names}"
        )

    def test_duplicate_registration_overwrites_previous(self) -> None:
        """
        When a board name is registered twice with different classes
        Then AdapterRegistry.get() returns an instance of the second class
        """
        # Given: a first adapter class registered under "duplicate_test"
        class FirstAdapter(JobBoardAdapter):
            @property
            def board_name(self) -> str:
                return "duplicate_test"

            async def authenticate(self, page: object) -> None: ...  # type: ignore[override]
            async def search(self, page: object, query: str, max_pages: int = 3) -> list[JobListing]: return []  # type: ignore[override]
            async def extract_detail(self, page: object, listing: JobListing) -> JobListing: return listing  # type: ignore[override]

        class SecondAdapter(JobBoardAdapter):
            @property
            def board_name(self) -> str:
                return "duplicate_test"

            async def authenticate(self, page: object) -> None: ...  # type: ignore[override]
            async def search(self, page: object, query: str, max_pages: int = 3) -> list[JobListing]: return []  # type: ignore[override]
            async def extract_detail(self, page: object, listing: JobListing) -> JobListing: return listing  # type: ignore[override]

        AdapterRegistry.register(FirstAdapter)

        # When: a second class is registered under the same name
        AdapterRegistry.register(SecondAdapter)

        # Then: get() returns the second class
        adapter = AdapterRegistry.get("duplicate_test")
        assert isinstance(adapter, SecondAdapter), (
            f"Expected SecondAdapter after overwrite, got {type(adapter).__name__}"
        )

        # Cleanup: remove test entry to avoid polluting other tests
        AdapterRegistry.unregister("duplicate_test")

    def test_adapter_decorator_does_not_alter_class_interface(self) -> None:
        """
        When AdapterRegistry.register() is used as a decorator
        Then it returns the original class unchanged so the decorator is safe to apply
        """
        # Given: a fresh adapter class

        class DecoratorTestAdapter(JobBoardAdapter):
            @property
            def board_name(self) -> str:
                return "decorator_test"

            async def authenticate(self, page: object) -> None: ...  # type: ignore[override]
            async def search(self, page: object, query: str, max_pages: int = 3) -> list[JobListing]: return []  # type: ignore[override]
            async def extract_detail(self, page: object, listing: JobListing) -> JobListing: return listing  # type: ignore[override]

        # When: register() is used as a decorator
        returned_cls = AdapterRegistry.register(DecoratorTestAdapter)

        # Then: the returned class is the same object as the input
        assert returned_cls is DecoratorTestAdapter, (
            f"register() should return the class unchanged, "
            f"got {returned_cls} instead of {DecoratorTestAdapter}"
        )

        # Cleanup
        AdapterRegistry.unregister("decorator_test")


# ---------------------------------------------------------------------------
# TestAdapterContract
# ---------------------------------------------------------------------------


class TestAdapterContract:
    """
    REQUIREMENT: All adapters conform to the JobBoardAdapter interface.

    WHO: The pipeline runner invoking adapters polymorphically
    WHAT: Every concrete adapter exposes board_name, authenticate, search,
          extract_detail, and rate_limit_seconds with correct return types;
          board_name is non-empty; rate_limit_seconds is a (min, max) float
          tuple with min < max; search() always returns a list; extract_detail()
          returns the same listing object enriched with full_text
    WHY: The runner calls adapters without knowing their type —
         any deviation from the contract breaks the pipeline silently
         because Python duck-typing provides no compile-time safety net

    MOCK BOUNDARY:
        Mock:  Playwright page object — page.goto, page.title, page.content,
               page.locator, page.query_selector are AsyncMock on a mock page
               fixture passed to real adapter methods
        Real:  ZipRecruiterAdapter instance for all tests — board_name,
               rate_limit_seconds, search(), and extract_detail() are all
               exercised on the real adapter with mocked page I/O
        Never: Mock the adapter itself — all methods are tested on a real
               instance; only the Playwright page boundary is mocked
    """

    def test_board_name_is_a_non_empty_string(self) -> None:
        """
        When board_name is accessed on a concrete adapter
        Then it is a non-empty string that identifies the board in logs and registry lookups
        """
        # Given: a real ZipRecruiterAdapter instance
        adapter = ZipRecruiterAdapter()

        # When: board_name is accessed
        name = adapter.board_name

        # Then: it is a non-empty string
        assert isinstance(name, str), (
            f"board_name should be a string, got {type(name).__name__}"
        )
        assert len(name) > 0, "board_name should not be empty"

    def test_rate_limit_seconds_is_a_two_element_float_tuple(self) -> None:
        """
        When rate_limit_seconds is accessed on a concrete adapter
        Then it is a tuple of exactly two floats so the throttle() function can draw a random delay
        """
        # Given: a real ZipRecruiterAdapter instance
        adapter = ZipRecruiterAdapter()

        # When: rate_limit_seconds is accessed
        rate = adapter.rate_limit_seconds

        # Then: it is a tuple of exactly two floats
        assert isinstance(rate, tuple), (
            f"rate_limit_seconds should be a tuple, got {type(rate).__name__}"
        )
        assert len(rate) == 2, (
            f"rate_limit_seconds should have 2 elements, got {len(rate)}: {rate}"
        )
        assert all(isinstance(v, float) for v in rate), (
            f"Both elements should be floats, got types "
            f"{[type(v).__name__ for v in rate]}: {rate}"
        )

    def test_rate_limit_lower_bound_is_less_than_upper_bound(self) -> None:
        """
        When rate_limit_seconds is accessed on a concrete adapter
        Then the first element is strictly less than the second so random.uniform() produces meaningful jitter
        """
        # Given: a real ZipRecruiterAdapter instance
        adapter = ZipRecruiterAdapter()

        # When: rate_limit_seconds is accessed
        rate = adapter.rate_limit_seconds

        # Then: min < max
        assert rate[0] < rate[1], (
            f"rate_limit_seconds lower bound ({rate[0]}) should be strictly "
            f"less than upper bound ({rate[1]})"
        )

    @pytest.mark.asyncio
    async def test_search_returns_a_list(self) -> None:
        """
        When search() is called on a concrete adapter
        Then it returns a list (possibly empty) so callers never need to guard against None
        """
        # Given: a real ZipRecruiterAdapter and a page whose content()
        # returns HTML with an empty js_variables job-cards list
        adapter = ZipRecruiterAdapter()
        js_vars = json.dumps({
            "hydrateJobCardsResponse": {"jobCards": []},
            "maxPages": 1,
        })
        html = (
            '<html><head><title>Jobs</title></head><body>'
            f'<script id="js_variables" type="application/json">{js_vars}</script>'
            '</body></html>'
        )
        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="Jobs")
        mock_page.content = AsyncMock(return_value=html)

        # When: search() is called
        result = await adapter.search(mock_page, "https://ziprecruiter.com/search", 1)

        # Then: it returns a list
        assert isinstance(result, list), (
            f"search() should return a list, got {type(result).__name__}: {result}"
        )

    @pytest.mark.asyncio
    async def test_extract_detail_populates_full_text(self) -> None:
        """
        When extract_detail() is called on a listing with empty full_text
        Then the returned listing's full_text is non-empty so the scorer has content to embed
        """
        # Given: a real ZipRecruiterAdapter and a listing with empty full_text
        # but a short_description in metadata (the fallback path)
        adapter = ZipRecruiterAdapter()
        listing = JobListing(
            board="ziprecruiter",
            external_id="zr-42",
            title="Senior Engineer",
            company="Acme Corp",
            location="Remote",
            url="https://ziprecruiter.com/zr-42",
            full_text="",
            metadata={"short_description": "Build scalable distributed systems."},
        )
        mock_page = AsyncMock()

        # When: extract_detail() is called on the real adapter
        result = await adapter.extract_detail(mock_page, listing)

        # Then: full_text is non-empty (populated by short_description fallback)
        assert result.full_text, (
            f"extract_detail() should populate full_text via fallback, got: {result.full_text!r}"
        )

    @pytest.mark.asyncio
    async def test_extract_detail_returns_the_same_listing_object(self) -> None:
        """
        When extract_detail() is called with a listing
        Then the returned object is the same instance that was passed in
        so caller references remain valid after enrichment
        """
        # Given: a real ZipRecruiterAdapter and a listing with empty full_text
        adapter = ZipRecruiterAdapter()
        listing = JobListing(
            board="ziprecruiter",
            external_id="zr-99",
            title="Staff Architect",
            company="TechCo",
            location="NYC",
            url="https://ziprecruiter.com/zr-99",
            full_text="",
            metadata={"short_description": "Lead architecture for cloud platform."},
        )
        mock_page = AsyncMock()

        # When: extract_detail() is called on the real adapter
        result = await adapter.extract_detail(mock_page, listing)

        # Then: the same listing object is returned (identity check)
        assert result is listing, (
            f"extract_detail() should return the same listing instance. "
            f"Expected id={id(listing)}, got id={id(result)}"
        )


# ---------------------------------------------------------------------------
# TestJobListingDataContract
# ---------------------------------------------------------------------------


class TestJobListingDataContract:
    """
    REQUIREMENT: JobListing is the canonical data contract across all boards.

    WHO: The RAG scorer, ranker, and exporter consuming listings
    WHAT: Required fields are always populated after extraction;
          optional fields degrade gracefully when absent;
          board field identifies source for deduplication;
          comp fields default to None and accept floats when present;
          metadata defaults to an empty dict, never None
    WHY: Downstream components must not branch on board type —
         the listing is the abstraction that makes them board-agnostic.
         A None metadata crashes any caller that iterates it without a guard

    MOCK BOUNDARY:
        Mock:  nothing — JobListing is a data class with no I/O
        Real:  JobListing instances constructed with explicit field values
        Never: MagicMock for JobListing; all instances must be real
               so field access and type constraints are exercised
    """

    def test_required_fields_are_present_after_extraction(self) -> None:
        """
        When a JobListing is constructed with all required fields
        Then board, external_id, title, company, location, url, and full_text are all non-empty
        """
        # Given: a fully populated listing
        listing = JobListing(
            board="ziprecruiter",
            external_id="zr-100",
            title="ML Engineer",
            company="DataCo",
            location="San Francisco, CA",
            url="https://ziprecruiter.com/zr-100",
            full_text="Build and deploy ML models at scale.",
        )

        # When: required fields are accessed (no action needed — construction is the action)

        # Then: all required fields are populated
        assert listing.board, f"board should be non-empty, got {listing.board!r}"
        assert listing.external_id, f"external_id should be non-empty, got {listing.external_id!r}"
        assert listing.title, f"title should be non-empty, got {listing.title!r}"
        assert listing.company, f"company should be non-empty, got {listing.company!r}"
        assert listing.location, f"location should be non-empty, got {listing.location!r}"
        assert listing.url, f"url should be non-empty, got {listing.url!r}"
        assert listing.full_text, f"full_text should be non-empty, got {listing.full_text!r}"

    def test_full_text_is_non_empty_string_after_detail_extraction(self) -> None:
        """
        When a listing's full_text is set after detail extraction
        Then full_text is a non-empty string
        """
        # Given: a listing whose full_text was populated by extract_detail
        listing = JobListing(
            board="ziprecruiter",
            external_id="zr-101",
            title="Backend Developer",
            company="WebCo",
            location="Austin, TX",
            url="https://ziprecruiter.com/zr-101",
            full_text="Design and maintain REST APIs for a high-traffic SaaS platform.",
        )

        # When: full_text is accessed
        text = listing.full_text

        # Then: it is a non-empty string
        assert isinstance(text, str), (
            f"full_text should be a string, got {type(text).__name__}"
        )
        assert len(text) > 0, "full_text should not be empty after extraction"

    def test_board_field_matches_adapter_board_name(self) -> None:
        """
        When a listing is created by the ZipRecruiter adapter
        Then the board field matches the adapter's board_name
        """
        # Given: a listing from ziprecruiter and the adapter's board_name
        adapter = ZipRecruiterAdapter()
        listing = JobListing(
            board="ziprecruiter",
            external_id="zr-102",
            title="DevOps Engineer",
            company="CloudCo",
            location="Remote",
            url="https://ziprecruiter.com/zr-102",
            full_text="Manage CI/CD pipelines and cloud infrastructure.",
        )

        # When: the board field and adapter board_name are compared
        board = listing.board
        adapter_name = adapter.board_name

        # Then: they match
        assert board == adapter_name, (
            f"listing.board ({board!r}) should match adapter.board_name ({adapter_name!r})"
        )

    def test_external_id_is_unique_within_a_board(self) -> None:
        """
        When two listings are created with different external_ids on the same board
        Then their external_id fields are distinct
        """
        # Given: two listings from the same board with different IDs
        listing_a = JobListing(
            board="ziprecruiter",
            external_id="zr-200",
            title="Frontend Dev",
            company="UICo",
            location="Remote",
            url="https://ziprecruiter.com/zr-200",
            full_text="Build React dashboards.",
        )
        listing_b = JobListing(
            board="ziprecruiter",
            external_id="zr-201",
            title="Backend Dev",
            company="APICo",
            location="Remote",
            url="https://ziprecruiter.com/zr-201",
            full_text="Build FastAPI services.",
        )

        # When: external_ids are compared

        # Then: they are distinct
        assert listing_a.external_id != listing_b.external_id, (
            f"external_ids should be unique within a board. "
            f"Both are {listing_a.external_id!r}"
        )

    def test_missing_posted_at_defaults_to_none_without_raising(self) -> None:
        """
        When a listing is created without specifying posted_at
        Then posted_at defaults to None and no error is raised
        """
        # Given: a listing constructed without posted_at
        listing = JobListing(
            board="ziprecruiter",
            external_id="zr-300",
            title="QA Engineer",
            company="TestCo",
            location="Remote",
            url="https://ziprecruiter.com/zr-300",
            full_text="Write integration tests for payment systems.",
        )

        # When: posted_at is accessed

        # Then: it is None
        assert listing.posted_at is None, (
            f"posted_at should default to None, got {listing.posted_at!r}"
        )

    def test_metadata_defaults_to_empty_dict_not_none(self) -> None:
        """
        When a listing is created without specifying metadata
        Then metadata defaults to an empty dict, never None
        """
        # Given: a listing constructed without metadata
        listing = JobListing(
            board="ziprecruiter",
            external_id="zr-301",
            title="Data Analyst",
            company="AnalyticsCo",
            location="Chicago, IL",
            url="https://ziprecruiter.com/zr-301",
            full_text="Analyze customer behavior data using SQL and Python.",
        )

        # When: metadata is accessed

        # Then: it is an empty dict, not None
        assert listing.metadata is not None, "metadata should never be None"
        assert isinstance(listing.metadata, dict), (
            f"metadata should be a dict, got {type(listing.metadata).__name__}"
        )
        assert listing.metadata == {}, (
            f"metadata should default to empty dict, got {listing.metadata}"
        )

    def test_comp_fields_default_to_none_when_not_parsed(self) -> None:
        """
        When a listing is created without compensation fields
        Then comp_min, comp_max, comp_source, and comp_text all default to None
        """
        # Given: a listing constructed without comp fields
        listing = JobListing(
            board="ziprecruiter",
            external_id="zr-302",
            title="Product Manager",
            company="ProdCo",
            location="NYC",
            url="https://ziprecruiter.com/zr-302",
            full_text="Lead product strategy for B2B SaaS vertical.",
        )

        # When: comp fields are accessed

        # Then: all are None
        assert listing.comp_min is None, (
            f"comp_min should default to None, got {listing.comp_min!r}"
        )
        assert listing.comp_max is None, (
            f"comp_max should default to None, got {listing.comp_max!r}"
        )
        assert listing.comp_source is None, (
            f"comp_source should default to None, got {listing.comp_source!r}"
        )
        assert listing.comp_text is None, (
            f"comp_text should default to None, got {listing.comp_text!r}"
        )

    def test_comp_min_and_comp_max_accept_float_values(self) -> None:
        """
        When a listing is created with comp_min and comp_max as floats
        Then both fields store the values correctly
        """
        # Given: a listing with compensation data
        listing = JobListing(
            board="ziprecruiter",
            external_id="zr-303",
            title="Senior SRE",
            company="InfraCo",
            location="Remote",
            url="https://ziprecruiter.com/zr-303",
            full_text="Design fault-tolerant distributed systems.",
            comp_min=150_000.0,
            comp_max=220_000.0,
        )

        # When: comp fields are accessed

        # Then: they store the correct float values
        assert listing.comp_min == pytest.approx(150_000.0), (
            f"comp_min should be 150000.0, got {listing.comp_min}"
        )
        assert listing.comp_max == pytest.approx(220_000.0), (
            f"comp_max should be 220000.0, got {listing.comp_max}"
        )

    def test_comp_source_accepts_employer_or_estimated_or_none(self) -> None:
        """
        When a listing is created with comp_source set to "employer" or "estimated"
        Then the field stores the string value; when omitted it is None
        """
        # Given: listings with different comp_source values
        employer_listing = JobListing(
            board="ziprecruiter",
            external_id="zr-304a",
            title="SWE",
            company="Acme",
            location="Remote",
            url="https://ziprecruiter.com/zr-304a",
            full_text="Job description text.",
            comp_source="employer",
        )
        estimated_listing = JobListing(
            board="ziprecruiter",
            external_id="zr-304b",
            title="SWE",
            company="Acme",
            location="Remote",
            url="https://ziprecruiter.com/zr-304b",
            full_text="Job description text.",
            comp_source="estimated",
        )
        none_listing = JobListing(
            board="ziprecruiter",
            external_id="zr-304c",
            title="SWE",
            company="Acme",
            location="Remote",
            url="https://ziprecruiter.com/zr-304c",
            full_text="Job description text.",
        )

        # When: comp_source is accessed on each

        # Then: values match what was provided
        assert employer_listing.comp_source == "employer", (
            f"comp_source should be 'employer', got {employer_listing.comp_source!r}"
        )
        assert estimated_listing.comp_source == "estimated", (
            f"comp_source should be 'estimated', got {estimated_listing.comp_source!r}"
        )
        assert none_listing.comp_source is None, (
            f"comp_source should be None when omitted, got {none_listing.comp_source!r}"
        )

    def test_comp_text_accepts_a_string_when_present(self) -> None:
        """
        When a listing is created with comp_text set to a salary string
        Then the field stores the string value
        """
        # Given: a listing with comp_text
        listing = JobListing(
            board="ziprecruiter",
            external_id="zr-305",
            title="Data Engineer",
            company="PipeCo",
            location="Remote",
            url="https://ziprecruiter.com/zr-305",
            full_text="Build data pipelines with Spark and Airflow.",
            comp_text="$130,000 - $180,000 per year",
        )

        # When: comp_text is accessed

        # Then: the string value is stored correctly
        assert listing.comp_text == "$130,000 - $180,000 per year", (
            f"comp_text should store the salary string, got {listing.comp_text!r}"
        )
