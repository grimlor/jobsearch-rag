"""
Config externalization tests — internal tuning knobs.

Maps to BDD spec: BDD Specifications — config-externalization-tuning.md
Implements: TestTopKRetrievalConfig, TestDistanceMetricConfig,
            TestSalaryBoundsConfig, TestHoursPerYearConfig,
            TestMaxFullTextCharsConfig, TestMaxSlugLengthConfig,
            TestViewportDimensionsConfig

These tests exercise config loading, validation, and wiring for
power-user tuning constants previously hardcoded in source.
All tests are expected to FAIL until Phase 3 implementation is complete.
"""

# Public API surface (from src/jobsearch_rag/config):
#   load_settings(path: str | Path) -> Settings
#   ScoringConfig(..., top_k_retrieval: int, salary_floor: float,
#                 salary_ceiling: float, hours_per_year: int)
#   ChromaConfig(persist_dir: str, distance_metric: str)
#   OutputConfig(..., max_slug_length: int)
#   AdaptersConfig(..., max_full_text_chars: int, viewport_width: int,
#                  viewport_height: int)
#
# Public API surface (from src/jobsearch_rag/rag/store):
#   VectorStore(persist_dir: str, distance_metric: str)
#
# Public API surface (from src/jobsearch_rag/rag/scorer):
#   Scorer(store, embedder, ..., top_k_retrieval: int)
#
# Public API surface (from src/jobsearch_rag/rag/comp_parser):
#   parse_compensation(text, source, *, salary_floor, salary_ceiling,
#                      hours_per_year) -> CompResult | None
#
# Public API surface (from src/jobsearch_rag/adapters/base):
#   JobListing(board, external_id, title, company, location, url,
#              full_text, ..., max_full_text_chars: int)
#
# Public API surface (from src/jobsearch_rag/text):
#   slugify(text, *, max_len: int) -> str
#
# Public API surface (from src/jobsearch_rag/adapters/session):
#   SessionConfig(board_name, ..., viewport_width, viewport_height)

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from jobsearch_rag.config import load_settings
from jobsearch_rag.errors import ActionableError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal valid TOML that includes ALL required fields (including the new ones).
_BASE_SETTINGS = """\
resume_path = "data/resume.md"
archetypes_path = "config/role_archetypes.toml"
global_rubric_path = "config/global_rubric.toml"

[boards]
enabled = ["testboard"]
session_storage_dir = "data"

[boards.testboard]
searches = ["https://example.org/search"]
max_pages = 2
headless = true
rate_limit_range = [1.5, 3.5]

[scoring]
archetype_weight = 0.5
fit_weight = 0.3
history_weight = 0.2
comp_weight = 0.15
negative_weight = 0.4
culture_weight = 0.2
base_salary = 220000
disqualify_on_llm_flag = true
min_score_threshold = 0.45
missing_comp_score = 0.5
chunk_overlap = 2000
dedup_similarity_threshold = 0.95
top_k_retrieval = 3
salary_floor = 10.0
salary_ceiling = 1000000.0
hours_per_year = 2080

[[scoring.comp_bands]]
ratio = 1.0
score = 1.0

[[scoring.comp_bands]]
ratio = 0.90
score = 0.7

[[scoring.comp_bands]]
ratio = 0.77
score = 0.4

[[scoring.comp_bands]]
ratio = 0.68
score = 0.0

[ollama]
base_url = "http://localhost:11434"
llm_model = "mistral:7b"
embed_model = "nomic-embed-text"
slow_llm_threshold_ms = 30000
classify_system_prompt = "You are a job listing classifier."
max_retries = 3
base_delay = 1.0
max_embed_chars = 8000
head_ratio = 0.6
retryable_status_codes = [408, 429, 500, 502, 503, 504]

[output]
default_format = "markdown"
output_dir = "./output"
open_top_n = 5
jd_dir = "output/jds"
decisions_dir = "data/decisions"
log_dir = "data/logs"
eval_history_path = "data/eval_history.jsonl"
max_slug_length = 80

[chroma]
persist_dir = "./data/chroma_db"
distance_metric = "cosine"

[security]
screen_prompt = "Review the following job description text."

[adapters]
cdp_timeout = 15.0
max_full_text_chars = 250000
viewport_width = 1440
viewport_height = 900

[adapters.browser_paths]
msedge = ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
"""


def _write_config(
    tmpdir: Path,
    settings_toml: str | None = None,
) -> Path:
    """
    Write settings.toml and return its path.

    Rewrites global_rubric_path to an absolute path inside *tmpdir* so
    load_settings() resolves it without depending on CWD.
    """
    content = settings_toml if settings_toml is not None else _BASE_SETTINGS

    rubric_path = tmpdir / "global_rubric.toml"
    rubric_path.write_text("", encoding="utf-8")
    content = content.replace(
        'global_rubric_path = "config/global_rubric.toml"',
        f'global_rubric_path = "{rubric_path}"',
    )

    settings_path = tmpdir / "settings.toml"
    settings_path.write_text(content, encoding="utf-8")
    return settings_path


def _remove_line(toml: str, key: str) -> str:
    """Remove the line containing *key* from a TOML string."""
    return "\n".join(line for line in toml.splitlines() if not line.strip().startswith(f"{key} ="))


def _replace_value(toml: str, key: str, new_value: str) -> str:
    """Replace the value of *key* in a TOML string."""
    lines = toml.splitlines()
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key} ="):
            result.append(f"{key} = {new_value}")
        else:
            result.append(line)
    return "\n".join(result)


# ---------------------------------------------------------------------------
# TestTopKRetrievalConfig
# ---------------------------------------------------------------------------


class TestTopKRetrievalConfig:
    """
    REQUIREMENT: The number of ChromaDB query results used for similarity
    scoring is configurable via settings.toml, not hardcoded in scorer.py.

    WHO: Power users tuning scoring precision/recall for different corpus sizes.
    WHAT: (1) load_settings() loads top_k_retrieval from [scoring].
          (2) Missing top_k_retrieval raises ActionableError naming the field.
          (3) top_k_retrieval < 1 raises ActionableError.
          (4) Scorer uses top_k_retrieval from config instead of hardcoded 3.
          (5) When collection count < top_k_retrieval, Scorer uses collection
              count (existing min() behavior preserved).
    WHY: Different corpus sizes may benefit from more or fewer retrieved
         documents — the hardcoded 3 prevents tuning.

    MOCK BOUNDARY:
        Mock:  ollama.AsyncClient (via conftest mock_embedder)
        Real:  load_settings(), Scorer, VectorStore (ChromaDB via tmp_path),
               config validation
        Never: internal scorer methods; config loading internals
    """

    def test_top_k_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [scoring] top_k_retrieval = 5
        When load_settings() is called
        Then ScoringConfig.top_k_retrieval is 5
        """
        # Given: settings with top_k_retrieval = 5
        toml = _replace_value(_BASE_SETTINGS, "top_k_retrieval", "5")
        path = _write_config(tmp_path, toml)

        # When: load settings
        settings = load_settings(path)

        # Then: value is loaded
        assert settings.scoring.top_k_retrieval == 5, (
            f"Expected top_k_retrieval=5, got {settings.scoring.top_k_retrieval}"
        )

    def test_missing_top_k_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml [scoring] has no top_k_retrieval field
        When load_settings() is called
        Then ActionableError is raised naming 'scoring.top_k_retrieval'
        """
        # Given: settings without top_k_retrieval
        toml = _remove_line(_BASE_SETTINGS, "top_k_retrieval")
        path = _write_config(tmp_path, toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError, match="top_k_retrieval") as exc_info:
            load_settings(path)
        assert "scoring" in str(exc_info.value).lower(), (
            f"Error should name the [scoring] section, got: {exc_info.value}"
        )

    def test_top_k_below_one_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [scoring] top_k_retrieval = 0
        When load_settings() is called
        Then ActionableError is raised stating value must be >= 1
        """
        # Given: top_k_retrieval = 0
        toml = _replace_value(_BASE_SETTINGS, "top_k_retrieval", "0")
        path = _write_config(tmp_path, toml)

        # When / Then: validation rejects it
        with pytest.raises(ActionableError, match="top_k_retrieval") as exc_info:
            load_settings(path)
        assert "1" in str(exc_info.value), (
            f"Error should state minimum of 1, got: {exc_info.value}"
        )

    def test_scorer_uses_configured_top_k(self, tmp_path: Path) -> None:
        """
        Given a Scorer constructed with top_k_retrieval = 5
        And a collection has 10 documents
        When score() queries that collection
        Then the query uses n_results = 5
        """
        import asyncio  # noqa: PLC0415
        from unittest.mock import AsyncMock  # noqa: PLC0415

        from jobsearch_rag.rag.scorer import Scorer  # noqa: PLC0415

        # Given: a mock store where all collections have 10 documents
        mock_store = MagicMock()
        mock_store.collection_count.return_value = 10
        mock_store.query.return_value = {
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
            "metadatas": [[{"name": "test"}]],
        }

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)
        mock_embedder.max_embed_chars = 8000
        mock_embedder.disqualify = AsyncMock(return_value=(False, None))

        scorer = Scorer(
            store=mock_store,
            embedder=mock_embedder,
            disqualify_on_llm_flag=False,
            top_k_retrieval=5,
        )

        # When: score() queries all collections
        asyncio.run(scorer.score("test job description"))

        # Then: every query call used n_results = 5
        for call in mock_store.query.call_args_list:
            assert call.kwargs["n_results"] == 5, (
                f"Expected n_results=5, got {call.kwargs['n_results']}"
            )

    def test_scorer_clamps_top_k_to_collection_count(self, tmp_path: Path) -> None:
        """
        Given a Scorer constructed with top_k_retrieval = 10
        And a collection has only 2 documents
        When score() queries that collection
        Then the query uses n_results = 2
        """
        import asyncio  # noqa: PLC0415
        from unittest.mock import AsyncMock  # noqa: PLC0415

        from jobsearch_rag.rag.scorer import Scorer  # noqa: PLC0415

        # Given: a mock store where all collections have 2 documents
        mock_store = MagicMock()
        mock_store.collection_count.return_value = 2
        mock_store.query.return_value = {
            "distances": [[0.1, 0.2]],
            "metadatas": [[{"name": "test"}]],
        }

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 384)
        mock_embedder.max_embed_chars = 8000
        mock_embedder.disqualify = AsyncMock(return_value=(False, None))

        scorer = Scorer(
            store=mock_store,
            embedder=mock_embedder,
            disqualify_on_llm_flag=False,
            top_k_retrieval=10,
        )

        # When: score() queries all collections
        asyncio.run(scorer.score("test job description"))

        # Then: every query call used n_results = 2 (clamped to collection count)
        for call in mock_store.query.call_args_list:
            assert call.kwargs["n_results"] == 2, (
                f"Expected n_results=2 (clamped), got {call.kwargs['n_results']}"
            )


# ---------------------------------------------------------------------------
# TestDistanceMetricConfig
# ---------------------------------------------------------------------------


class TestDistanceMetricConfig:
    """
    REQUIREMENT: The ChromaDB distance metric is configurable via settings.toml,
    not hardcoded in store.py.

    WHO: Power users experimenting with distance functions for different
         embedding models.
    WHAT: (1) load_settings() loads distance_metric from [chroma].
          (2) Missing distance_metric raises ActionableError naming the field.
          (3) Invalid distance_metric (not in {"cosine", "l2", "ip"}) raises
              ActionableError listing the valid options.
          (4) VectorStore passes the configured metric to
              get_or_create_collection().
    WHY: Different embedding models may perform better with different distance
         metrics — the hardcoded "cosine" prevents experimentation.

    MOCK BOUNDARY:
        Mock:  chromadb.PersistentClient (via conftest vector_store)
        Real:  load_settings(), VectorStore, config validation
        Never: chromadb internals; config loading internals
    """

    def test_distance_metric_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [chroma] distance_metric = "l2"
        When load_settings() is called
        Then ChromaConfig.distance_metric is "l2"
        """
        # Given: settings with distance_metric = "l2"
        toml = _replace_value(_BASE_SETTINGS, "distance_metric", '"l2"')
        path = _write_config(tmp_path, toml)

        # When: load settings
        settings = load_settings(path)

        # Then: value is loaded
        assert settings.chroma.distance_metric == "l2", (
            f"Expected distance_metric='l2', got '{settings.chroma.distance_metric}'"
        )

    def test_missing_distance_metric_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml [chroma] has no distance_metric field
        When load_settings() is called
        Then ActionableError is raised naming 'chroma.distance_metric'
        """
        # Given: settings without distance_metric
        toml = _remove_line(_BASE_SETTINGS, "distance_metric")
        path = _write_config(tmp_path, toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError, match="distance_metric") as exc_info:
            load_settings(path)
        assert "chroma" in str(exc_info.value).lower(), (
            f"Error should name the [chroma] section, got: {exc_info.value}"
        )

    def test_invalid_distance_metric_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [chroma] distance_metric = "euclidean"
        When load_settings() is called
        Then ActionableError is raised listing the valid options
        """
        # Given: invalid metric
        toml = _replace_value(_BASE_SETTINGS, "distance_metric", '"euclidean"')
        path = _write_config(tmp_path, toml)

        # When / Then: validation rejects it
        with pytest.raises(ActionableError, match="distance_metric") as exc_info:
            load_settings(path)
        error_msg = str(exc_info.value)
        assert "cosine" in error_msg, (
            f"Error should list valid options including 'cosine', got: {error_msg}"
        )

    def test_store_uses_configured_distance_metric(self, tmp_path: Path) -> None:
        """
        Given a VectorStore initialized with distance_metric = "l2"
        When get_or_create_collection() is called
        Then the collection metadata includes {"hnsw:space": "l2"}
        """
        from unittest.mock import patch  # noqa: PLC0415

        from jobsearch_rag.rag.store import VectorStore  # noqa: PLC0415

        # Given: a VectorStore with distance_metric = "l2"
        with patch("jobsearch_rag.rag.store.chromadb.PersistentClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_cls.return_value = mock_client

            store = VectorStore(persist_dir=str(tmp_path), distance_metric="l2")

            # When: create a collection
            store.get_or_create_collection("test_collection")

            # Then: metadata includes the configured metric
            call_args = mock_client.get_or_create_collection.call_args
            metadata = call_args.kwargs.get("metadata", call_args[1].get("metadata", {}))
            assert metadata.get("hnsw:space") == "l2", (
                f"Expected hnsw:space='l2', got metadata={metadata}"
            )


# ---------------------------------------------------------------------------
# TestSalaryBoundsConfig
# ---------------------------------------------------------------------------


class TestSalaryBoundsConfig:
    """
    REQUIREMENT: Salary floor and ceiling for compensation parsing are
    configurable via settings.toml, not hardcoded in comp_parser.py.

    WHO: Users in markets with different salary ranges (e.g., international,
         executive).
    WHAT: (1) load_settings() loads salary_floor from [scoring].
          (2) load_settings() loads salary_ceiling from [scoring].
          (3) Missing salary_floor raises ActionableError naming the field.
          (4) Missing salary_ceiling raises ActionableError naming the field.
          (5) salary_floor < 0 raises ActionableError.
          (6) salary_ceiling == salary_floor raises ActionableError.
          (7) salary_ceiling < salary_floor raises ActionableError.
          (8) comp_parser uses configured bounds instead of hardcoded
              10.0 / 1_000_000.0.
          (9) comp_parser rejects values below configured floor.
    WHY: Non-US markets or executive roles may have different valid salary
         ranges.

    MOCK BOUNDARY:
        Mock:  nothing — pure computation
        Real:  load_settings(), parse_compensation(), config validation
        Never: n/a
    """

    def test_salary_floor_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [scoring] salary_floor = 15.0
        When load_settings() is called
        Then ScoringConfig.salary_floor is 15.0
        """
        # Given: settings with salary_floor = 15.0
        toml = _replace_value(_BASE_SETTINGS, "salary_floor", "15.0")
        path = _write_config(tmp_path, toml)

        # When: load settings
        settings = load_settings(path)

        # Then: value is loaded
        assert settings.scoring.salary_floor == 15.0, (
            f"Expected salary_floor=15.0, got {settings.scoring.salary_floor}"
        )

    def test_salary_ceiling_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [scoring] salary_ceiling = 500000.0
        When load_settings() is called
        Then ScoringConfig.salary_ceiling is 500000.0
        """
        # Given: settings with salary_ceiling = 500000.0
        toml = _replace_value(_BASE_SETTINGS, "salary_ceiling", "500000.0")
        path = _write_config(tmp_path, toml)

        # When: load settings
        settings = load_settings(path)

        # Then: value is loaded
        assert settings.scoring.salary_ceiling == 500_000.0, (
            f"Expected salary_ceiling=500000.0, got {settings.scoring.salary_ceiling}"
        )

    def test_missing_salary_floor_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml [scoring] has no salary_floor field
        When load_settings() is called
        Then ActionableError is raised naming 'scoring.salary_floor'
        """
        # Given: settings without salary_floor
        toml = _remove_line(_BASE_SETTINGS, "salary_floor")
        path = _write_config(tmp_path, toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError, match="salary_floor") as exc_info:
            load_settings(path)
        assert "scoring" in str(exc_info.value).lower(), (
            f"Error should name the [scoring] section, got: {exc_info.value}"
        )

    def test_missing_salary_ceiling_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml [scoring] has no salary_ceiling field
        When load_settings() is called
        Then ActionableError is raised naming 'scoring.salary_ceiling'
        """
        # Given: settings without salary_ceiling
        toml = _remove_line(_BASE_SETTINGS, "salary_ceiling")
        path = _write_config(tmp_path, toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError, match="salary_ceiling") as exc_info:
            load_settings(path)
        assert "scoring" in str(exc_info.value).lower(), (
            f"Error should name the [scoring] section, got: {exc_info.value}"
        )

    def test_negative_salary_floor_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [scoring] salary_floor = -1.0
        When load_settings() is called
        Then ActionableError is raised stating floor must be >= 0
        """
        # Given: negative floor
        toml = _replace_value(_BASE_SETTINGS, "salary_floor", "-1.0")
        path = _write_config(tmp_path, toml)

        # When / Then: validation rejects it
        with pytest.raises(ActionableError, match="salary_floor") as exc_info:
            load_settings(path)
        assert "0" in str(exc_info.value), f"Error should mention 0, got: {exc_info.value}"

    def test_ceiling_equal_to_floor_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [scoring] salary_floor = 100.0 and salary_ceiling = 100.0
        When load_settings() is called
        Then ActionableError is raised stating ceiling must be > floor
        """
        # Given: ceiling == floor
        toml = _replace_value(_BASE_SETTINGS, "salary_floor", "100.0")
        toml = _replace_value(toml, "salary_ceiling", "100.0")
        path = _write_config(tmp_path, toml)

        # When / Then: validation rejects it
        with pytest.raises(ActionableError, match="salary_ceiling"):
            load_settings(path)

    def test_ceiling_below_floor_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [scoring] salary_floor = 200.0 and salary_ceiling = 100.0
        When load_settings() is called
        Then ActionableError is raised stating ceiling must be > floor
        """
        # Given: ceiling < floor
        toml = _replace_value(_BASE_SETTINGS, "salary_floor", "200.0")
        toml = _replace_value(toml, "salary_ceiling", "100.0")
        path = _write_config(tmp_path, toml)

        # When / Then: validation rejects it
        with pytest.raises(ActionableError, match="salary_ceiling"):
            load_settings(path)

    def test_comp_parser_uses_configured_bounds(self) -> None:
        """
        Given salary_floor = 50.0 and salary_ceiling = 2_000_000.0
        When parse_compensation() encounters "$40/hr" (annualizes to $83,200)
        Then the result is accepted (within configured range)
        """
        from jobsearch_rag.rag.comp_parser import parse_compensation  # noqa: PLC0415

        # Given: wide bounds that accept the value
        # When: parse hourly rate that annualizes into range
        result = parse_compensation(
            "$40/hr",
            salary_floor=50.0,
            salary_ceiling=2_000_000.0,
            hours_per_year=2080,
        )

        # Then: result is not None (accepted)
        assert result is not None, "Expected $40/hr to be accepted with floor=50, ceiling=2M"

    def test_comp_parser_rejects_below_configured_floor(self) -> None:
        """
        Given salary_floor = 100_000.0 and salary_ceiling = 1_000_000.0
        When parse_compensation() encounters "$80,000"
        Then the value is rejected as out of range (returns None)
        """
        from jobsearch_rag.rag.comp_parser import parse_compensation  # noqa: PLC0415

        # Given: floor higher than the salary value
        # When: parse a salary below the floor
        result = parse_compensation(
            "Salary: $80,000 per year",
            salary_floor=100_000.0,
            salary_ceiling=1_000_000.0,
            hours_per_year=2080,
        )

        # Then: result is None (rejected)
        assert result is None, "Expected $80,000 to be rejected with floor=100,000"


# ---------------------------------------------------------------------------
# TestHoursPerYearConfig
# ---------------------------------------------------------------------------


class TestHoursPerYearConfig:
    """
    REQUIREMENT: The hours-per-year factor for hourly→annual conversion is
    configurable via settings.toml, not hardcoded in comp_parser.py.

    WHO: Users with different work-hour assumptions (e.g., part-time,
         different countries).
    WHAT: (1) load_settings() loads hours_per_year from [scoring].
          (2) Missing hours_per_year raises ActionableError naming the field.
          (3) hours_per_year == 0 raises ActionableError.
          (4) hours_per_year < 0 raises ActionableError.
          (5) comp_parser uses configured hours_per_year instead of
              hardcoded 2080.
    WHY: 2080 assumes US full-time; other contexts use different annual hours.

    MOCK BOUNDARY:
        Mock:  nothing — pure computation
        Real:  load_settings(), parse_compensation(), config validation
        Never: n/a
    """

    def test_hours_per_year_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [scoring] hours_per_year = 1800
        When load_settings() is called
        Then ScoringConfig.hours_per_year is 1800
        """
        # Given: settings with hours_per_year = 1800
        toml = _replace_value(_BASE_SETTINGS, "hours_per_year", "1800")
        path = _write_config(tmp_path, toml)

        # When: load settings
        settings = load_settings(path)

        # Then: value is loaded
        assert settings.scoring.hours_per_year == 1800, (
            f"Expected hours_per_year=1800, got {settings.scoring.hours_per_year}"
        )

    def test_missing_hours_per_year_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml [scoring] has no hours_per_year field
        When load_settings() is called
        Then ActionableError is raised naming 'scoring.hours_per_year'
        """
        # Given: settings without hours_per_year
        toml = _remove_line(_BASE_SETTINGS, "hours_per_year")
        path = _write_config(tmp_path, toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError, match="hours_per_year") as exc_info:
            load_settings(path)
        assert "scoring" in str(exc_info.value).lower(), (
            f"Error should name the [scoring] section, got: {exc_info.value}"
        )

    def test_zero_hours_per_year_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [scoring] hours_per_year = 0
        When load_settings() is called
        Then ActionableError is raised stating value must be > 0
        """
        # Given: hours_per_year = 0
        toml = _replace_value(_BASE_SETTINGS, "hours_per_year", "0")
        path = _write_config(tmp_path, toml)

        # When / Then: validation rejects it
        with pytest.raises(ActionableError, match="hours_per_year"):
            load_settings(path)

    def test_negative_hours_per_year_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [scoring] hours_per_year = -100
        When load_settings() is called
        Then ActionableError is raised stating value must be > 0
        """
        # Given: negative value
        toml = _replace_value(_BASE_SETTINGS, "hours_per_year", "-100")
        path = _write_config(tmp_path, toml)

        # When / Then: validation rejects it
        with pytest.raises(ActionableError, match="hours_per_year"):
            load_settings(path)

    def test_comp_parser_uses_configured_hours_per_year(self) -> None:
        """
        Given hours_per_year = 1800
        When parse_compensation() encounters "$50/hr"
        Then comp_min and comp_max are annualized using 1800 (= $90,000)
        """
        from jobsearch_rag.rag.comp_parser import parse_compensation  # noqa: PLC0415

        # Given: configured hours_per_year = 1800
        # When: parse hourly rate
        result = parse_compensation(
            "$50/hr",
            salary_floor=10.0,
            salary_ceiling=1_000_000.0,
            hours_per_year=1800,
        )

        # Then: annualized using 1800 hours
        assert result is not None, "Expected $50/hr to parse successfully"
        expected = 50.0 * 1800
        assert result.comp_min == expected, f"Expected comp_min={expected}, got {result.comp_min}"
        assert result.comp_max == expected, f"Expected comp_max={expected}, got {result.comp_max}"


# ---------------------------------------------------------------------------
# TestMaxFullTextCharsConfig
# ---------------------------------------------------------------------------


class TestMaxFullTextCharsConfig:
    """
    REQUIREMENT: The maximum allowed full_text length for job listings is
    configurable via settings.toml, not hardcoded in base.py.

    WHO: Users encountering truncation or wanting stricter limits.
    WHAT: (1) load_settings() loads max_full_text_chars from [adapters].
          (2) Missing max_full_text_chars raises ActionableError naming the field.
          (3) max_full_text_chars <= 0 raises ActionableError.
          (4) JobListing rejects full_text exceeding configured
              max_full_text_chars.
          (5) JobListing accepts full_text within configured
              max_full_text_chars.
    WHY: Different boards may produce longer/shorter listings; the limit
         should be tunable.

    MOCK BOUNDARY:
        Mock:  nothing — dataclass validation
        Real:  load_settings(), JobListing construction, config validation
        Never: n/a
    """

    def test_max_full_text_chars_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [adapters] max_full_text_chars = 500000
        When load_settings() is called
        Then AdaptersConfig.max_full_text_chars is 500000
        """
        # Given: settings with max_full_text_chars = 500000
        toml = _replace_value(_BASE_SETTINGS, "max_full_text_chars", "500000")
        path = _write_config(tmp_path, toml)

        # When: load settings
        settings = load_settings(path)

        # Then: value is loaded
        assert settings.adapters.max_full_text_chars == 500_000, (
            f"Expected max_full_text_chars=500000, got {settings.adapters.max_full_text_chars}"
        )

    def test_missing_max_full_text_chars_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml [adapters] has no max_full_text_chars field
        When load_settings() is called
        Then ActionableError is raised naming 'adapters.max_full_text_chars'
        """
        # Given: settings without max_full_text_chars
        toml = _remove_line(_BASE_SETTINGS, "max_full_text_chars")
        path = _write_config(tmp_path, toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError, match="max_full_text_chars") as exc_info:
            load_settings(path)
        assert "adapters" in str(exc_info.value).lower(), (
            f"Error should name the [adapters] section, got: {exc_info.value}"
        )

    def test_zero_max_full_text_chars_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [adapters] max_full_text_chars = 0
        When load_settings() is called
        Then ActionableError is raised stating value must be > 0
        """
        # Given: zero value
        toml = _replace_value(_BASE_SETTINGS, "max_full_text_chars", "0")
        path = _write_config(tmp_path, toml)

        # When / Then: validation rejects it
        with pytest.raises(ActionableError, match="max_full_text_chars"):
            load_settings(path)

    def test_job_listing_uses_configured_max_full_text(self) -> None:
        """
        Given max_full_text_chars = 100
        When a JobListing is created with full_text of 150 chars
        Then ValueError is raised (exceeds configured limit)
        """
        from jobsearch_rag.adapters.base import JobListing  # noqa: PLC0415

        # Given: a listing with text exceeding the configured limit
        long_text = "x" * 150

        # When / Then: creation raises ValueError
        with pytest.raises(ValueError, match="full_text") as exc_info:
            JobListing(
                board="test",
                external_id="123",
                title="Test Job",
                company="TestCo",
                location="Remote",
                url="https://example.com/job/123",
                full_text=long_text,
                max_full_text_chars=100,
            )
        assert "150" in str(exc_info.value) or "100" in str(exc_info.value), (
            f"Error should mention the length or limit, got: {exc_info.value}"
        )

    def test_job_listing_accepts_text_within_configured_limit(self) -> None:
        """
        Given max_full_text_chars = 100
        When a JobListing is created with full_text of 50 chars
        Then the listing is created successfully
        """
        from jobsearch_rag.adapters.base import JobListing  # noqa: PLC0415

        # Given: a listing with text within the configured limit
        short_text = "x" * 50

        # When: creation succeeds
        listing = JobListing(
            board="test",
            external_id="123",
            title="Test Job",
            company="TestCo",
            location="Remote",
            url="https://example.com/job/123",
            full_text=short_text,
            max_full_text_chars=100,
        )

        # Then: listing is created with the text
        assert len(listing.full_text) == 50, (
            f"Expected full_text length=50, got {len(listing.full_text)}"
        )


# ---------------------------------------------------------------------------
# TestMaxSlugLengthConfig
# ---------------------------------------------------------------------------


class TestMaxSlugLengthConfig:
    """
    REQUIREMENT: The maximum slug length for output filenames is configurable
    via settings.toml, not hardcoded in text.py.

    WHO: Users on filesystems with path length constraints.
    WHAT: (1) load_settings() loads max_slug_length from [output].
          (2) Missing max_slug_length raises ActionableError naming the field.
          (3) max_slug_length <= 0 raises ActionableError.
          (4) slugify() uses configured max_slug_length instead of
              hardcoded 80.
    WHY: Some filesystems or tools have tighter path-length limits.

    MOCK BOUNDARY:
        Mock:  nothing — pure computation
        Real:  load_settings(), slugify(), config validation
        Never: n/a
    """

    def test_max_slug_length_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [output] max_slug_length = 40
        When load_settings() is called
        Then OutputConfig.max_slug_length is 40
        """
        # Given: settings with max_slug_length = 40
        toml = _replace_value(_BASE_SETTINGS, "max_slug_length", "40")
        path = _write_config(tmp_path, toml)

        # When: load settings
        settings = load_settings(path)

        # Then: value is loaded
        assert settings.output.max_slug_length == 40, (
            f"Expected max_slug_length=40, got {settings.output.max_slug_length}"
        )

    def test_missing_max_slug_length_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml [output] has no max_slug_length field
        When load_settings() is called
        Then ActionableError is raised naming 'output.max_slug_length'
        """
        # Given: settings without max_slug_length
        toml = _remove_line(_BASE_SETTINGS, "max_slug_length")
        path = _write_config(tmp_path, toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError, match="max_slug_length") as exc_info:
            load_settings(path)
        assert "output" in str(exc_info.value).lower(), (
            f"Error should name the [output] section, got: {exc_info.value}"
        )

    def test_zero_max_slug_length_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [output] max_slug_length = 0
        When load_settings() is called
        Then ActionableError is raised stating value must be > 0
        """
        # Given: zero value
        toml = _replace_value(_BASE_SETTINGS, "max_slug_length", "0")
        path = _write_config(tmp_path, toml)

        # When / Then: validation rejects it
        with pytest.raises(ActionableError, match="max_slug_length"):
            load_settings(path)

    def test_slugify_uses_configured_max_length(self) -> None:
        """
        Given max_slug_length = 20
        When slugify() is called with a long title
        Then the result is truncated to 20 characters
        """
        from jobsearch_rag.text import slugify  # noqa: PLC0415

        # Given: a long title
        long_title = "Senior Staff Platform Architect for Cloud Infrastructure"

        # When: slugify with max_len = 20
        result = slugify(long_title, max_len=20)

        # Then: truncated to 20 chars
        assert len(result) <= 20, f"Expected slug length <= 20, got {len(result)}: '{result}'"
        assert result == slugify(long_title, max_len=20), "slugify should be deterministic"


# ---------------------------------------------------------------------------
# TestViewportDimensionsConfig
# ---------------------------------------------------------------------------


class TestViewportDimensionsConfig:
    """
    REQUIREMENT: Browser viewport dimensions are configurable via settings.toml,
    not hardcoded as defaults on SessionConfig.

    WHO: Users running on different monitor sizes or needing specific viewport
         sizes for bot detection avoidance.
    WHAT: (1) load_settings() loads viewport_width from [adapters].
          (2) load_settings() loads viewport_height from [adapters].
          (3) Missing viewport_width raises ActionableError naming the field.
          (4) Missing viewport_height raises ActionableError naming the field.
          (5) viewport_width <= 0 raises ActionableError.
          (6) viewport_height <= 0 raises ActionableError.
          (7) SessionConfig receives viewport dimensions from config, not from
              hardcoded defaults.
    WHY: Some sites render differently at different viewport sizes; bot
         detection may flag unusual viewports.

    MOCK BOUNDARY:
        Mock:  Playwright browser (already mocked in session tests)
        Real:  load_settings(), SessionConfig construction, config validation
        Never: Playwright internals
    """

    def test_viewport_width_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [adapters] viewport_width = 1920
        When load_settings() is called
        Then AdaptersConfig.viewport_width is 1920
        """
        # Given: settings with viewport_width = 1920
        toml = _replace_value(_BASE_SETTINGS, "viewport_width", "1920")
        path = _write_config(tmp_path, toml)

        # When: load settings
        settings = load_settings(path)

        # Then: value is loaded
        assert settings.adapters.viewport_width == 1920, (
            f"Expected viewport_width=1920, got {settings.adapters.viewport_width}"
        )

    def test_viewport_height_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [adapters] viewport_height = 1080
        When load_settings() is called
        Then AdaptersConfig.viewport_height is 1080
        """
        # Given: settings with viewport_height = 1080
        toml = _replace_value(_BASE_SETTINGS, "viewport_height", "1080")
        path = _write_config(tmp_path, toml)

        # When: load settings
        settings = load_settings(path)

        # Then: value is loaded
        assert settings.adapters.viewport_height == 1080, (
            f"Expected viewport_height=1080, got {settings.adapters.viewport_height}"
        )

    def test_missing_viewport_width_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml [adapters] has no viewport_width field
        When load_settings() is called
        Then ActionableError is raised naming 'adapters.viewport_width'
        """
        # Given: settings without viewport_width
        toml = _remove_line(_BASE_SETTINGS, "viewport_width")
        path = _write_config(tmp_path, toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError, match="viewport_width") as exc_info:
            load_settings(path)
        assert "adapters" in str(exc_info.value).lower(), (
            f"Error should name the [adapters] section, got: {exc_info.value}"
        )

    def test_missing_viewport_height_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml [adapters] has no viewport_height field
        When load_settings() is called
        Then ActionableError is raised naming 'adapters.viewport_height'
        """
        # Given: settings without viewport_height
        toml = _remove_line(_BASE_SETTINGS, "viewport_height")
        path = _write_config(tmp_path, toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError, match="viewport_height") as exc_info:
            load_settings(path)
        assert "adapters" in str(exc_info.value).lower(), (
            f"Error should name the [adapters] section, got: {exc_info.value}"
        )

    def test_zero_viewport_width_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [adapters] viewport_width = 0
        When load_settings() is called
        Then ActionableError is raised stating value must be > 0
        """
        # Given: zero width
        toml = _replace_value(_BASE_SETTINGS, "viewport_width", "0")
        path = _write_config(tmp_path, toml)

        # When / Then: validation rejects it
        with pytest.raises(ActionableError, match="viewport_width"):
            load_settings(path)

    def test_zero_viewport_height_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [adapters] viewport_height = 0
        When load_settings() is called
        Then ActionableError is raised stating value must be > 0
        """
        # Given: zero height
        toml = _replace_value(_BASE_SETTINGS, "viewport_height", "0")
        path = _write_config(tmp_path, toml)

        # When / Then: validation rejects it
        with pytest.raises(ActionableError, match="viewport_height"):
            load_settings(path)

    def test_session_config_uses_configured_viewport(self) -> None:
        """
        Given AdaptersConfig with viewport_width = 1920 and viewport_height = 1080
        When a SessionConfig is constructed using those values
        Then SessionConfig.viewport_width is 1920 and viewport_height is 1080
        """
        from jobsearch_rag.adapters.session import SessionConfig  # noqa: PLC0415

        # Given: configured viewport dimensions
        # When: construct SessionConfig with those values
        config = SessionConfig(
            board_name="testboard",
            viewport_width=1920,
            viewport_height=1080,
        )

        # Then: dimensions match
        assert config.viewport_width == 1920, (
            f"Expected viewport_width=1920, got {config.viewport_width}"
        )
        assert config.viewport_height == 1080, (
            f"Expected viewport_height=1080, got {config.viewport_height}"
        )
