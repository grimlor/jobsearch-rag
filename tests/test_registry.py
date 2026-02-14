"""Adapter registration and IoC contract tests.

Maps to BDD specs: TestAdapterRegistration, TestAdapterContract
"""

from __future__ import annotations

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

    def test_registered_adapter_is_retrievable_by_board_name(self) -> None: ...
    def test_retrieving_unregistered_board_name_raises_value_error_with_name(self) -> None: ...
    def test_registry_lists_all_registered_board_names(self) -> None: ...
    def test_duplicate_registration_overwrites_previous(self) -> None: ...
    def test_adapter_decorator_does_not_alter_class_interface(self) -> None: ...


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

    def test_board_name_returns_non_empty_string(self) -> None: ...
    def test_rate_limit_seconds_returns_tuple_of_two_floats(self) -> None: ...
    def test_rate_limit_min_is_less_than_max(self) -> None: ...
    def test_search_returns_list_of_job_listings(self) -> None: ...
    def test_extract_detail_populates_full_text_on_listing(self) -> None: ...
    def test_extract_detail_returns_same_listing_object_enriched(self) -> None: ...


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

    def test_required_fields_are_present_after_extraction(self) -> None: ...
    def test_full_text_is_non_empty_string_after_detail_extraction(self) -> None: ...
    def test_board_field_matches_adapter_board_name(self) -> None: ...
    def test_external_id_is_unique_within_a_board(self) -> None: ...
    def test_missing_posted_at_does_not_raise(self) -> None: ...
    def test_metadata_defaults_to_empty_dict_not_none(self) -> None: ...
