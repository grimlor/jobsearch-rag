"""
Config externalization tests — Phase 8a persona-specific values,
Phase 8b/8c operational and deployment parameters.

Maps to BDD spec: BDD Specifications — config-externalization.md
Implements: TestDisqualifierPromptConfig, TestScreenPromptConfig,
            TestClassifierSystemPromptConfig, TestCompScoreCurveConfig,
            TestEmbedderConfigExternalization, TestOutputPathConfig,
            TestScoringTunablesConfig, TestBoardBrowserConfig,
            TestEvalHistoryConfig, TestLoginUrlConfig,
            TestStealthConfig, TestAdaptersConfig

These tests exercise new config loading, validation, and wiring behaviors
for values that were previously hardcoded in source. All tests are expected
to FAIL until Phase 3 implementation is complete.

Spec classes:
    TestDisqualifierPromptConfig — disqualifier prompt synthesis from archetypes
    TestScreenPromptConfig — injection screening prompt from [security]
    TestClassifierSystemPromptConfig — classifier system message from [ollama]
    TestCompScoreCurveConfig — comp breakpoints and neutral score from [scoring]
    TestEmbedderConfigExternalization — embedder tuning from [ollama]
    TestOutputPathConfig — output paths from [output]
    TestScoringTunablesConfig — scoring tunables from [scoring]
    TestBoardBrowserConfig — browser/throttle settings from [boards]
    TestEvalHistoryConfig — eval history path from [output]
    TestLoginUrlConfig — per-board login URLs from [boards.<name>]
    TestStealthConfig — per-board stealth flag from [boards.<name>]
    TestAdaptersConfig — browser paths and CDP timeout from [adapters]
"""

# Public API surface (from src/jobsearch_rag/config):
#   load_settings(path: str | Path) -> Settings
#   Settings(enabled_boards, overnight_boards, boards, scoring, ollama, output, chroma, ...)
#   ScoringConfig(archetype_weight, fit_weight, history_weight, comp_weight,
#                 negative_weight, culture_weight, base_salary, disqualify_on_llm_flag,
#                 min_score_threshold)
#   OllamaConfig(base_url, llm_model, embed_model, slow_llm_threshold_ms)
#
# NEW symbols expected after Phase 3 implementation:
#   synthesize_disqualifier_prompt(archetypes_path: str | Path) -> str
#   DisqualifierConfig(system_prompt: str | None = None)
#   SecurityConfig(screen_prompt: str = <default>)
#   CompBand(ratio: float, score: float)
#   ScoringConfig.comp_bands: list[CompBand]
#   ScoringConfig.missing_comp_score: float
#   OllamaConfig.classify_system_prompt: str
#   Settings.disqualifier: DisqualifierConfig
#   Settings.security: SecurityConfig
#
# NEW symbols for Phase 8b remaining:
#   OutputConfig.eval_history_path: str
#   BoardConfig.login_url: str | None
#   BoardConfig.stealth: bool
#   AdaptersConfig(browser_paths: dict[str, list[str]], cdp_timeout: float)
#   Settings.adapters: AdaptersConfig
#
# Public API surface (from src/jobsearch_rag/rag/scorer):
#   Scorer(store, embedder, disqualify_on_llm_flag, disqualifier_prompt, screen_prompt)
#   Scorer.disqualify(jd_text) -> tuple[bool, str | None]
#   Scorer.score(jd_text) -> ScoreResult
#
# Public API surface (from src/jobsearch_rag/rag/embedder):
#   Embedder(base_url, embed_model, llm_model, ..., classify_system_prompt)
#   Embedder.classify(prompt) -> str
#
# Public API surface (from src/jobsearch_rag/rag/comp_parser):
#   compute_comp_score(comp_max, base_salary, breakpoints, default_score) -> float

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ollama import ResponseError

from jobsearch_rag.adapters import AdapterRegistry
from jobsearch_rag.adapters.base import JobListing
from jobsearch_rag.adapters.session import SessionConfig, SessionManager
from jobsearch_rag.cli import handle_login
from jobsearch_rag.config import (
    CompBand,
    OllamaConfig,
    load_settings,
    synthesize_disqualifier_prompt,
)
from jobsearch_rag.errors import ActionableError
from jobsearch_rag.pipeline.ranker import Ranker
from jobsearch_rag.pipeline.runner import PipelineRunner
from jobsearch_rag.rag.comp_parser import compute_comp_score
from jobsearch_rag.rag.embedder import Embedder
from jobsearch_rag.rag.scorer import Scorer, ScoreResult
from tests.conftest import make_mock_ollama_client, make_test_settings

if TYPE_CHECKING:
    from collections.abc import Callable

    from jobsearch_rag.rag.store import VectorStore


def _import_synthesize() -> Callable[..., str]:
    """Return synthesize_disqualifier_prompt."""
    return synthesize_disqualifier_prompt


def _import_comp_band() -> type[CompBand]:
    """Return CompBand class."""
    return CompBand


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal valid TOML that satisfies all required fields.
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
login_url = "https://www.testboard.com/login"
stealth = false

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
classify_system_prompt = "You are a job listing classifier. Respond concisely with your classification."
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

[chroma]
persist_dir = "./data/chroma_db"

[security]
screen_prompt = "Review the following job description text."

[adapters]
cdp_timeout = 15.0

[adapters.browser_paths]
msedge = ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
chrome = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
"""

# Minimal valid role_archetypes.toml with 2 archetypes
_ARCHETYPES_TOML = """\
[[archetypes]]
name = "Staff Platform Architect"
description = "Defines technical strategy for distributed systems."
signals_positive = ["cloud-native design", "cross-team influence"]
signals_negative = [
    "IC coding role disguised as architect",
    "Primarily SRE/on-call operations",
]

[[archetypes]]
name = "AI Systems Engineer"
description = "Builds AI infrastructure and tooling."
signals_positive = ["LLM integration", "evaluation frameworks"]
signals_negative = [
    "ML model development as primary function",
    "Data labeling or annotation only",
]
"""


def _write_config(
    tmpdir: Path,
    settings_extra: str = "",
    settings_toml: str | None = None,
    archetypes: str = _ARCHETYPES_TOML,
) -> Path:
    """
    Write settings.toml and role_archetypes.toml, return settings path.

    All generated files are written inside *tmpdir* so that parallel test
    workers (e.g. pytest-xdist) never race on shared filesystem paths.

    Use ``settings_extra`` for new sections that do not duplicate headers in
    ``_BASE_SETTINGS`` (e.g. ``[disqualifier]``, ``[security]``).

    Use ``settings_toml`` to provide the **complete** settings content when
    modifying existing sections (e.g. ``[scoring]``, ``[ollama]``) to avoid
    duplicate TOML section headers.
    """
    content = settings_toml if settings_toml is not None else _BASE_SETTINGS + settings_extra

    # Rewrite the global_rubric_path to an absolute path inside tmpdir so
    # load_settings() resolves it without depending on CWD.
    rubric_path = tmpdir / "global_rubric.toml"
    rubric_path.write_text("", encoding="utf-8")
    content = content.replace(
        'global_rubric_path = "config/global_rubric.toml"',
        f'global_rubric_path = "{rubric_path}"',
    )

    settings_path = tmpdir / "settings.toml"
    settings_path.write_text(content, encoding="utf-8")

    # Write archetypes file
    arch_path = tmpdir / "role_archetypes.toml"
    arch_path.write_text(archetypes, encoding="utf-8")

    return settings_path


# ---------------------------------------------------------------------------
# TestDisqualifierPromptConfig
# ---------------------------------------------------------------------------


class TestDisqualifierPromptConfig:
    """
    REQUIREMENT: The disqualifier prompt is synthesized from role_archetypes.toml
    when no freeform override is provided, so the LLM applies the correct
    user's role criteria — not a hardcoded persona.

    WHO: Any user targeting a different role than the original developer — the
         prompt must reflect *their* archetypes, not "Principal/Staff Platform
         Architect"
    WHAT: (1) The system synthesizes a disqualifier prompt from archetype names and
              their signals_negative fields when no [disqualifier] system_prompt is
              set in settings.toml.
          (2) The synthesized prompt includes all archetype names from
              role_archetypes.toml so the LLM knows what roles the user targets.
          (3) The synthesized prompt includes negative signals from all archetypes
              as disqualification criteria.
          (4) The synthesized prompt instructs the LLM to respond with the expected
              JSON schema {"disqualified": bool, "reason": str|null}.
          (5) The system uses a freeform [disqualifier] system_prompt from
              settings.toml verbatim when provided, bypassing synthesis.
          (6) The system raises ActionableError when role_archetypes.toml has no
              archetypes and no freeform override is provided.
          (7) The Scorer receives the synthesized or override prompt and passes it
              as the system message for disqualification LLM calls.

    WHY: The hardcoded "Principal/Staff Platform Architect" prompt is the single
         biggest barrier to other users — it would incorrectly disqualify roles
         that are good fits for their target

    MOCK BOUNDARY:
        Mock:  ollama.AsyncClient (via conftest mock_embedder)
        Real:  synthesize_disqualifier_prompt(), load_settings(),
               Scorer.disqualify(), VectorStore (ChromaDB via tmp_path)
        Never: Hardcode prompt text in tests — derive expected content from
               archetype fixture data; never mock the synthesis function itself
    """

    def test_synthesis_includes_all_archetype_names(self, tmp_path: Path) -> None:
        """
        Given role_archetypes.toml defines 2 archetypes
        When synthesize_disqualifier_prompt() is called
        Then the returned prompt contains both archetype names
        """
        # Given: archetypes file with 2 archetypes
        arch_path = tmp_path / "role_archetypes.toml"
        arch_path.write_text(_ARCHETYPES_TOML, encoding="utf-8")

        # When: prompt is synthesized
        synthesize = _import_synthesize()
        prompt = synthesize(arch_path)

        # Then: both archetype names appear in the prompt
        assert "Staff Platform Architect" in prompt, (
            f"Prompt should contain 'Staff Platform Architect'. Got: {prompt[:200]}"
        )
        assert "AI Systems Engineer" in prompt, (
            f"Prompt should contain 'AI Systems Engineer'. Got: {prompt[:200]}"
        )

    def test_synthesis_includes_negative_signals_as_disqualification_criteria(
        self, tmp_path: Path
    ) -> None:
        """
        Given role_archetypes.toml archetypes each have signals_negative entries
        When synthesize_disqualifier_prompt() is called
        Then the returned prompt contains the negative signal text from each archetype
        """
        # Given: archetypes file with negative signals
        arch_path = tmp_path / "role_archetypes.toml"
        arch_path.write_text(_ARCHETYPES_TOML, encoding="utf-8")

        # When: prompt is synthesized
        synthesize = _import_synthesize()
        prompt = synthesize(arch_path)

        # Then: negative signals from both archetypes appear
        assert "IC coding role disguised as architect" in prompt, (
            f"Prompt should contain architect negative signal. Got: {prompt[:300]}"
        )
        assert "ML model development as primary function" in prompt, (
            f"Prompt should contain AI eng negative signal. Got: {prompt[:300]}"
        )

    def test_synthesis_includes_json_response_schema(self, tmp_path: Path) -> None:
        """
        When synthesize_disqualifier_prompt() is called with valid archetypes
        Then the returned prompt instructs the LLM to respond with
        {"disqualified": bool, "reason": str|null}
        """
        # Given: valid archetypes
        arch_path = tmp_path / "role_archetypes.toml"
        arch_path.write_text(_ARCHETYPES_TOML, encoding="utf-8")

        # When: prompt is synthesized
        synthesize = _import_synthesize()
        prompt = synthesize(arch_path)

        # Then: the JSON schema instruction is present
        assert '"disqualified"' in prompt, (
            f"Prompt should instruct LLM to respond with 'disqualified' key. Got: {prompt[:300]}"
        )
        assert '"reason"' in prompt, (
            f"Prompt should instruct LLM to respond with 'reason' key. Got: {prompt[:300]}"
        )

    def test_freeform_override_bypasses_synthesis(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [disqualifier] system_prompt = "Custom prompt..."
        When load_settings() is called
        Then DisqualifierConfig.system_prompt contains the custom text verbatim
        """
        # Given: settings with a freeform disqualifier override
        settings_path = _write_config(
            tmp_path,
            settings_extra='\n[disqualifier]\nsystem_prompt = "My custom disqualifier prompt"\n',
        )

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: the freeform override is stored verbatim
        assert settings.disqualifier is not None
        assert settings.disqualifier.system_prompt == "My custom disqualifier prompt", (
            f"Expected freeform override, got {settings.disqualifier.system_prompt!r}"
        )

    def test_missing_disqualifier_section_triggers_synthesis(self, tmp_path: Path) -> None:
        """
        Given settings.toml has no [disqualifier] section
        When load_settings() is called
        Then DisqualifierConfig.system_prompt is None, signaling synthesis is needed
        """
        # Given: settings with no [disqualifier] section
        settings_path = _write_config(tmp_path)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: system_prompt is None (synthesis will be triggered at runtime)
        assert settings.disqualifier is not None
        assert settings.disqualifier.system_prompt is None, (
            f"Expected None for missing disqualifier section, got {settings.disqualifier.system_prompt!r}"
        )

    def test_empty_archetypes_without_override_raises_actionable_error(
        self, tmp_path: Path
    ) -> None:
        """
        Given role_archetypes.toml has no archetypes and no freeform override
        When synthesize_disqualifier_prompt() is called
        Then ActionableError is raised naming the empty archetype file
        """
        # Given: empty archetypes file
        arch_path = tmp_path / "role_archetypes.toml"
        arch_path.write_text("# empty\n", encoding="utf-8")

        # When / Then: synthesis raises ActionableError
        synthesize = _import_synthesize()

        with pytest.raises(ActionableError) as exc_info:
            synthesize(arch_path)

        assert "archetype" in str(exc_info.value).lower(), (
            f"Error should mention archetypes. Got: {exc_info.value}"
        )

    async def test_scorer_receives_synthesized_prompt_as_system_message(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a Scorer constructed with a synthesized prompt from archetypes
        When disqualify() is called on a JD
        Then the LLM classify call contains the synthesized prompt text
        """
        # Given: synthesize a prompt from archetypes
        arch_path = tmp_path / "role_archetypes.toml"
        arch_path.write_text(_ARCHETYPES_TOML, encoding="utf-8")

        synthesize = _import_synthesize()
        prompt = synthesize(arch_path)

        # Given: a Scorer constructed with the synthesized prompt
        scorer = Scorer(
            store=vector_store,
            embedder=mock_embedder,
            disqualifier_prompt=prompt,
        )

        # When: disqualify is called
        await scorer.disqualify("Some JD text")

        # Then: the disqualifier classify call (2nd call — after screening)
        # contains the synthesized prompt in the user message
        assert mock_embedder._client.chat.call_count >= 2, (  # type: ignore[union-attr]
            f"Expected at least 2 classify calls (screen + disqualify), "
            f"got {mock_embedder._client.chat.call_count}"  # type: ignore[union-attr]
        )
        disqualify_call = mock_embedder._client.chat.call_args_list[1]  # type: ignore[union-attr]
        call_kwargs = cast("dict[str, Any]", disqualify_call.kwargs)  # type: ignore[reportUnknownMemberType]
        call_args = cast("tuple[Any, ...]", disqualify_call.args)  # type: ignore[reportUnknownMemberType]
        messages: list[dict[str, Any]] = call_kwargs.get(
            "messages",
            cast("dict[str, Any]", call_args[1]).get("messages", []) if len(call_args) > 1 else [],
        )
        user_messages: list[dict[str, Any]] = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 1, (
            f"Expected 1 user message in disqualify call, got {len(user_messages)}"
        )
        user_content: str = user_messages[0]["content"]
        assert prompt in user_content, (
            f"User message should contain the synthesized prompt.\n"
            f"Expected to find: {prompt[:100]}...\nGot: {user_content[:200]}..."
        )


# ---------------------------------------------------------------------------
# TestScreenPromptConfig
# ---------------------------------------------------------------------------


class TestScreenPromptConfig:
    """
    REQUIREMENT: The injection screening prompt is configurable via
    settings.toml so operators can add few-shot examples or customize
    detection patterns without code changes.

    WHO: Security-conscious operators who want to reduce false positives
         (e.g. AI-mentioning JDs flagged as injection attempts) by adding
         few-shot examples to the screen prompt
    WHAT: (1) The system reads [security] screen_prompt from settings.toml
              when present.
          (2) The system uses the current default prompt when [security]
              screen_prompt is not set.
          (3) The Scorer passes the configured screen prompt to the injection
              screening layer and the LLM receives it as the system message.
    WHY: The bare prompt produces false positives on AI-mentioning JDs.
         Making it configurable enables few-shot example injection via config

    MOCK BOUNDARY:
        Mock:  ollama.AsyncClient (via conftest mock_embedder)
        Real:  load_settings(), Scorer._screen_jd_for_injection()
        Never: Mock Scorer internals; never assert on default prompt text
               (assert on config-driven behavior instead)
    """

    def test_custom_screen_prompt_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [security] screen_prompt = "Custom screen..."
        When load_settings() is called
        Then SecurityConfig.screen_prompt contains the custom text
        """
        # Given: settings with a custom screen prompt
        custom_toml = _BASE_SETTINGS.replace(
            'screen_prompt = "Review the following job description text."',
            'screen_prompt = "Custom screen prompt with few-shot examples"',
        )
        settings_path = _write_config(tmp_path, settings_toml=custom_toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: the custom screen prompt is accessible
        assert settings.security.screen_prompt == "Custom screen prompt with few-shot examples", (
            f"Expected custom screen prompt, got {settings.security.screen_prompt!r}"
        )

    def test_missing_security_section_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has no [security] section
        When load_settings() is called
        Then ActionableError is raised naming the missing section
        """
        # Given: settings with no [security] section
        no_security = _BASE_SETTINGS.replace(
            '[security]\nscreen_prompt = "Review the following job description text."\n',
            "",
        )
        settings_path = _write_config(tmp_path, settings_toml=no_security)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "security" in str(exc_info.value).lower(), (
            f"Error should name the missing section. Got: {exc_info.value}"
        )

    async def test_scorer_sends_configured_screen_prompt_to_llm(
        self,
        tmp_path: Path,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a Scorer constructed with a custom screen_prompt from config
        When _screen_jd_for_injection() is called on a JD
        Then the LLM classify call contains the custom screen prompt text
        """
        # Given: a Scorer with a custom screen prompt
        custom_prompt = "Custom injection screening prompt with examples"
        scorer = Scorer(
            store=vector_store,
            embedder=mock_embedder,
            screen_prompt=custom_prompt,
        )

        # When: screening is invoked (via disqualify which calls screening first)
        await scorer.disqualify("Some JD text to screen")

        # Then: the first classify call (screening) contains the custom prompt
        # in the user message
        first_call_args = mock_embedder._client.chat.call_args_list[0]  # type: ignore[union-attr]
        call_kwargs = cast("dict[str, Any]", first_call_args.kwargs)  # type: ignore[reportUnknownMemberType]
        call_args = cast("tuple[Any, ...]", first_call_args.args)  # type: ignore[reportUnknownMemberType]
        messages: list[dict[str, Any]] = call_kwargs.get(
            "messages",
            cast("dict[str, Any]", call_args[1]).get("messages", []) if len(call_args) > 1 else [],
        )
        user_messages: list[dict[str, Any]] = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 1, (
            f"Expected 1 user message in screening call, got {len(user_messages)}"
        )
        user_content: str = user_messages[0]["content"]
        assert custom_prompt in user_content, (
            f"User message should contain the custom screen prompt.\n"
            f"Expected to find: {custom_prompt}\nGot: {user_content[:200]}"
        )


# ---------------------------------------------------------------------------
# TestClassifierSystemPromptConfig
# ---------------------------------------------------------------------------


class TestClassifierSystemPromptConfig:
    """
    REQUIREMENT: The LLM classifier system message is configurable via
    settings.toml so users can adjust the classifier persona for their
    domain or language.

    WHO: Users who want to adjust the classifier persona for domain-specific
         framing (e.g. "You are an expert recruiter..." instead of generic)
    WHAT: (1) The system reads [ollama] classify_system_prompt from settings.toml
              when present.
          (2) The system uses the current default when the field is not set.
          (3) The Embedder sends the configured system message in all classify()
              calls.
    WHY: The generic "job listing classifier" message works but users may
         want domain-specific framing

    MOCK BOUNDARY:
        Mock:  ollama.AsyncClient (via conftest mock_embedder)
        Real:  load_settings(), Embedder.classify()
        Never: Mock the system message assembly separately from classify()
    """

    def test_custom_classify_prompt_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [ollama] classify_system_prompt = "You are an expert recruiter..."
        When load_settings() is called
        Then OllamaConfig.classify_system_prompt contains the custom text
        """
        # Given: settings with a custom classify system prompt
        custom_toml = _BASE_SETTINGS.replace(
            'classify_system_prompt = "You are a job listing classifier. Respond concisely with your classification."',
            'classify_system_prompt = "You are an expert recruiter. Respond concisely."',
        )
        settings_path = _write_config(tmp_path, settings_toml=custom_toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: the custom classify prompt is accessible
        assert (
            settings.ollama.classify_system_prompt
            == "You are an expert recruiter. Respond concisely."
        ), f"Expected custom classify prompt, got {settings.ollama.classify_system_prompt!r}"

    def test_missing_classify_prompt_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has no classify_system_prompt field
        When load_settings() is called
        Then ActionableError is raised naming the missing field
        """
        # Given: settings with no classify_system_prompt
        no_prompt = _BASE_SETTINGS.replace(
            'classify_system_prompt = "You are a job listing classifier. Respond concisely with your classification."\n',
            "",
        )
        settings_path = _write_config(tmp_path, settings_toml=no_prompt)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "classify_system_prompt" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )

    async def test_embedder_sends_configured_system_message_in_classify(self) -> None:
        """
        Given an Embedder constructed with a custom classify_system_prompt
        When classify() is called
        Then the mock LLM receives the custom text as the system message
        """
        # Given: an Embedder with a custom classify_system_prompt
        custom_system = "You are an expert recruiter. Respond concisely."
        mock_client = make_mock_ollama_client()
        embedder = Embedder(
            OllamaConfig(
                base_url="http://localhost:11434",
                embed_model="nomic-embed-text",
                llm_model="mistral:7b",
                slow_llm_threshold_ms=30_000,
                classify_system_prompt=custom_system,
                max_retries=1,
                base_delay=0.0,
                max_embed_chars=8_000,
                head_ratio=0.6,
                retryable_status_codes=[503],
            )
        )
        embedder._client = mock_client  # type: ignore[attr-defined]

        # When: classify is called
        await embedder.classify("Is this role suitable?")

        # Then: the LLM received the custom system message
        call_args = mock_client.chat.call_args
        messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) == 1, f"Expected 1 system message, got {len(system_messages)}"
        assert system_messages[0]["content"] == custom_system, (
            f"System message should be the custom prompt.\n"
            f"Expected: {custom_system}\nGot: {system_messages[0]['content']}"
        )


# ---------------------------------------------------------------------------
# TestCompScoreCurveConfig
# ---------------------------------------------------------------------------


class TestCompScoreCurveConfig:
    """
    REQUIREMENT: The compensation score breakpoints and neutral score are
    configurable via settings.toml so users targeting different seniority
    levels or markets can adjust the curve without code changes.

    WHO: Users targeting different seniority levels or markets where
         compensation expectations differ from the hardcoded 90%/77%/68%
         thresholds (e.g. "Senior ML Engineer" at $150K vs "Principal
         Platform Architect" at $220K)
    WHAT: (1) The system reads [scoring] comp_bands from settings.toml as a
              list of {ratio, score} pairs when present.
          (2) The system uses the current default breakpoints
              [{ratio=1.0, score=1.0}, {ratio=0.90, score=0.7},
              {ratio=0.77, score=0.4}, {ratio=0.68, score=0.0}] when not
              configured.
          (3) The system validates that breakpoint ratios are monotonically
              decreasing.
          (4) The system validates that breakpoint scores are monotonically
              decreasing.
          (5) The system raises ActionableError when breakpoint ratios are
              not monotonically decreasing.
          (6) The system raises ActionableError when fewer than 2 breakpoints
              are provided.
          (7) The system reads [scoring] missing_comp_score from settings.toml
              when present.
          (8) The system uses 0.5 as the default when missing_comp_score is
              not configured.
          (9) The system validates that missing_comp_score is in [0.0, 1.0].
          (10) compute_comp_score() applies the configured breakpoints to
               produce correct piecewise-linear interpolation.
          (11) compute_comp_score() returns missing_comp_score when comp_max
               is None.
          (12) compute_comp_score() returns the highest band score when the
               ratio exceeds the highest breakpoint.
          (13) compute_comp_score() returns the lowest band score when the
               ratio falls below the lowest breakpoint.
    WHY: A user targeting "Senior ML Engineer" at $150K has completely
         different compensation expectations than a "Principal Platform
         Architect" at $220K. The curve shape must adapt

    MOCK BOUNDARY:
        Mock:  nothing — this tests pure computation and config validation
        Real:  load_settings(), compute_comp_score(), CompBand dataclass
        Never: n/a — no I/O boundary
    """

    def test_custom_breakpoints_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [scoring] comp_bands with 3 custom breakpoints
        When load_settings() is called
        Then ScoringConfig.comp_bands contains the 3 custom CompBand entries
        """
        # Given: settings with custom comp_bands (replace defaults with 3 custom)
        stripped = re.sub(
            r"\[\[scoring\.comp_bands\]\]\nratio = [\d.]+\nscore = [\d.]+\n*",
            "",
            _BASE_SETTINGS,
        )
        custom_bands = """\

[[scoring.comp_bands]]
ratio = 1.0
score = 1.0

[[scoring.comp_bands]]
ratio = 0.80
score = 0.5

[[scoring.comp_bands]]
ratio = 0.60
score = 0.0
"""
        # Insert custom bands before [ollama]
        custom_toml = stripped.replace("[ollama]", custom_bands + "[ollama]")
        settings_path = _write_config(tmp_path, settings_toml=custom_toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: comp_bands has 3 entries with correct values
        bands = settings.scoring.comp_bands
        assert bands is not None
        assert len(bands) == 3, f"Expected 3 comp bands, got {len(bands)}"
        assert bands[0].ratio == 1.0, f"First band ratio should be 1.0, got {bands[0].ratio}"
        assert bands[0].score == 1.0, f"First band score should be 1.0, got {bands[0].score}"
        assert bands[1].ratio == 0.80, f"Second band ratio should be 0.80, got {bands[1].ratio}"
        assert bands[1].score == 0.5, f"Second band score should be 0.5, got {bands[1].score}"
        assert bands[2].ratio == 0.60, f"Third band ratio should be 0.60, got {bands[2].ratio}"
        assert bands[2].score == 0.0, f"Third band score should be 0.0, got {bands[2].score}"

    def test_missing_comp_bands_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has no comp_bands in [scoring]
        When load_settings() is called
        Then ActionableError is raised naming the missing field
        """
        # Given: settings with no comp_bands
        no_bands = re.sub(
            r"\[\[scoring\.comp_bands\]\]\nratio = [\d.]+\nscore = [\d.]+\n*",
            "",
            _BASE_SETTINGS,
        )
        settings_path = _write_config(tmp_path, settings_toml=no_bands)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "comp_bands" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )

    def test_non_decreasing_ratios_raise_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has comp_bands with ratios [0.9, 0.95, 0.7]
        When load_settings() is called
        Then ActionableError is raised naming the field and stating ratios
        must be monotonically decreasing
        """
        # Given: comp_bands with non-decreasing ratios (replace defaults)
        stripped = re.sub(
            r"\[\[scoring\.comp_bands\]\]\nratio = [\d.]+\nscore = [\d.]+\n*",
            "",
            _BASE_SETTINGS,
        )
        bad_bands = """\

[[scoring.comp_bands]]
ratio = 0.9
score = 0.7

[[scoring.comp_bands]]
ratio = 0.95
score = 0.5

[[scoring.comp_bands]]
ratio = 0.7
score = 0.0
"""
        custom_toml = stripped.replace("[ollama]", bad_bands + "[ollama]")
        settings_path = _write_config(tmp_path, settings_toml=custom_toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        error_msg = str(exc_info.value).lower()
        assert "ratio" in error_msg or "decreasing" in error_msg, (
            f"Error should mention ratio ordering. Got: {exc_info.value}"
        )

    def test_fewer_than_two_breakpoints_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has comp_bands with only 1 entry
        When load_settings() is called
        Then ActionableError is raised stating at least 2 breakpoints are required
        """
        # Given: comp_bands with only 1 entry (replace defaults)
        stripped = re.sub(
            r"\[\[scoring\.comp_bands\]\]\nratio = [\d.]+\nscore = [\d.]+\n*",
            "",
            _BASE_SETTINGS,
        )
        one_band = """\

[[scoring.comp_bands]]
ratio = 1.0
score = 1.0
"""
        custom_toml = stripped.replace("[ollama]", one_band + "[ollama]")
        settings_path = _write_config(tmp_path, settings_toml=custom_toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        error_msg = str(exc_info.value).lower()
        assert "2" in error_msg or "at least" in error_msg, (
            f"Error should mention minimum 2 breakpoints. Got: {exc_info.value}"
        )

    def test_custom_missing_comp_score_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [scoring] missing_comp_score = 0.3
        When load_settings() is called
        Then ScoringConfig.missing_comp_score is 0.3
        """
        # Given: settings with custom missing_comp_score
        custom_toml = _BASE_SETTINGS.replace(
            "missing_comp_score = 0.5",
            "missing_comp_score = 0.3",
        )
        settings_path = _write_config(tmp_path, settings_toml=custom_toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: missing_comp_score is 0.3
        assert settings.scoring.missing_comp_score == 0.3, (
            f"Expected missing_comp_score=0.3, got {settings.scoring.missing_comp_score}"
        )

    def test_missing_comp_score_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has no missing_comp_score in [scoring]
        When load_settings() is called
        Then ActionableError is raised naming the missing field
        """
        # Given: settings with no missing_comp_score
        no_score = _BASE_SETTINGS.replace("missing_comp_score = 0.5\n", "")
        settings_path = _write_config(tmp_path, settings_toml=no_score)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "missing_comp_score" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )

    def test_missing_comp_score_out_of_range_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [scoring] missing_comp_score = 1.5
        When load_settings() is called
        Then ActionableError is raised naming the field and valid range [0.0, 1.0]
        """
        # Given: missing_comp_score out of range
        bad_toml = _BASE_SETTINGS.replace(
            "missing_comp_score = 0.5",
            "missing_comp_score = 1.5",
        )
        settings_path = _write_config(tmp_path, settings_toml=bad_toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        error_msg = str(exc_info.value).lower()
        assert "missing_comp_score" in error_msg, (
            f"Error should name the field. Got: {exc_info.value}"
        )

    def test_compute_uses_configured_breakpoints(self) -> None:
        """
        Given comp_bands = [{ratio=1.0, score=1.0}, {ratio=0.80, score=0.5}, {ratio=0.60, score=0.0}]
        When compute_comp_score is called with comp_max at 80% of base
        Then score is 0.5
        """
        # Given: custom breakpoints
        comp_band_cls = _import_comp_band()

        breakpoints = [
            comp_band_cls(ratio=1.0, score=1.0),
            comp_band_cls(ratio=0.80, score=0.5),
            comp_band_cls(ratio=0.60, score=0.0),
        ]
        base_salary = 200_000
        comp_max = 160_000  # 80% of base

        # When: compute_comp_score is called with custom breakpoints
        score = compute_comp_score(comp_max, base_salary, breakpoints=breakpoints)

        # Then: score is 0.5 (at the 0.80 breakpoint)
        assert score == pytest.approx(0.5, abs=0.01), (
            f"Score at 80% ratio with custom breakpoints should be ~0.5, got {score}"
        )

    def test_compute_returns_configured_missing_score_for_none(self) -> None:
        """
        Given missing_comp_score = 0.3
        When compute_comp_score is called with comp_max = None
        Then score is 0.3
        """
        # Given: custom default score

        # When: compute with None comp_max and custom default
        score = compute_comp_score(None, 200_000, default_score=0.3)

        # Then: score is 0.3
        assert score == pytest.approx(0.3), (
            f"Missing comp data with custom default should return 0.3, got {score}"
        )

    def test_compute_returns_top_score_when_ratio_exceeds_highest_breakpoint(self) -> None:
        """
        Given custom breakpoints with highest ratio = 1.0
        When compute_comp_score is called with comp_max above base_salary
        Then score equals the highest breakpoint's score (1.0)
        """
        # Given: custom breakpoints
        comp_band_cls = _import_comp_band()
        breakpoints = [
            comp_band_cls(ratio=1.0, score=1.0),
            comp_band_cls(ratio=0.80, score=0.5),
            comp_band_cls(ratio=0.60, score=0.0),
        ]
        base_salary = 200_000
        comp_max = 250_000  # 125% of base — above highest breakpoint

        # When: compute_comp_score is called
        score = compute_comp_score(comp_max, base_salary, breakpoints=breakpoints)

        # Then: score equals the top breakpoint score
        assert score == pytest.approx(1.0), (
            f"Score above highest breakpoint should be 1.0, got {score}"
        )

    def test_compute_returns_bottom_score_when_ratio_below_lowest_breakpoint(self) -> None:
        """
        Given custom breakpoints with lowest ratio = 0.60
        When compute_comp_score is called with comp_max well below the lowest breakpoint
        Then score equals the lowest breakpoint's score (0.0)
        """
        # Given: custom breakpoints
        comp_band_cls = _import_comp_band()
        breakpoints = [
            comp_band_cls(ratio=1.0, score=1.0),
            comp_band_cls(ratio=0.80, score=0.5),
            comp_band_cls(ratio=0.60, score=0.0),
        ]
        base_salary = 200_000
        comp_max = 80_000  # 40% of base — below lowest breakpoint

        # When: compute_comp_score is called
        score = compute_comp_score(comp_max, base_salary, breakpoints=breakpoints)

        # Then: score equals the bottom breakpoint score
        assert score == pytest.approx(0.0), (
            f"Score below lowest breakpoint should be 0.0, got {score}"
        )


# ---------------------------------------------------------------------------
# TestEmbedderConfigExternalization
# ---------------------------------------------------------------------------

# _BASE_SETTINGS already includes all embedder config fields, so alias.
_EMBEDDER_SETTINGS = _BASE_SETTINGS


class TestEmbedderConfigExternalization:
    """
    REQUIREMENT: Embedder tuning values (retry, truncation, status codes)
    live in settings.toml under [ollama], not as hardcoded defaults in
    embedder.py.  The Embedder constructor accepts OllamaConfig directly.

    WHO: Operators running different Ollama deployments (GPU vs CPU,
         different models, varying network reliability)
    WHAT: (1) The system loads max_retries from [ollama].
          (2) The system loads base_delay from [ollama].
          (3) The system loads max_embed_chars from [ollama].
          (4) The system loads head_ratio from [ollama].
          (5) The system loads retryable_status_codes from [ollama].
          (6) The Embedder constructor accepts OllamaConfig and reads all
              values from it.
          (7) The Embedder uses config max_embed_chars for truncation.
          (8) The Embedder uses config head_ratio for head/tail split.
          (9) The Embedder retries only on configured retryable_status_codes.
          (10) The Embedder uses config max_retries for retry count.
          (11) The system rejects missing [ollama] max_retries.
          (12) The system rejects missing [ollama] base_delay.
          (13) The system rejects missing [ollama] max_embed_chars.
          (14) The system rejects missing [ollama] head_ratio.
          (15) The system rejects missing [ollama] retryable_status_codes.
    WHY: Different Ollama deployments need different retry settings and
         token limits — these are operator concerns, not code concerns

    MOCK BOUNDARY:
        Mock:  ollama.AsyncClient (via conftest make_mock_ollama_client)
        Real:  load_settings(), Embedder construction, truncation, retry logic
        Never: Mock config loading internals
    """

    # -- Config loading (WHAT 1-5) ------------------------------------------

    def test_max_retries_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [ollama] max_retries = 5
        When load_settings() is called
        Then settings.ollama.max_retries is 5
        """
        # Given: settings with custom max_retries
        toml = _EMBEDDER_SETTINGS.replace("max_retries = 3", "max_retries = 5")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: max_retries is 5
        assert settings.ollama.max_retries == 5, (
            f"Expected max_retries=5, got {settings.ollama.max_retries}"
        )

    def test_base_delay_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [ollama] base_delay = 2.5
        When load_settings() is called
        Then settings.ollama.base_delay is 2.5
        """
        # Given: settings with custom base_delay
        toml = _EMBEDDER_SETTINGS.replace("base_delay = 1.0", "base_delay = 2.5")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: base_delay is 2.5
        assert settings.ollama.base_delay == pytest.approx(2.5), (
            f"Expected base_delay=2.5, got {settings.ollama.base_delay}"
        )

    def test_max_embed_chars_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [ollama] max_embed_chars = 4000
        When load_settings() is called
        Then settings.ollama.max_embed_chars is 4000
        """
        # Given: settings with custom max_embed_chars
        toml = _EMBEDDER_SETTINGS.replace("max_embed_chars = 8000", "max_embed_chars = 4000")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: max_embed_chars is 4000
        assert settings.ollama.max_embed_chars == 4000, (
            f"Expected max_embed_chars=4000, got {settings.ollama.max_embed_chars}"
        )

    def test_head_ratio_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [ollama] head_ratio = 0.7
        When load_settings() is called
        Then settings.ollama.head_ratio is 0.7
        """
        # Given: settings with custom head_ratio
        toml = _EMBEDDER_SETTINGS.replace("head_ratio = 0.6", "head_ratio = 0.7")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: head_ratio is 0.7
        assert settings.ollama.head_ratio == pytest.approx(0.7), (
            f"Expected head_ratio=0.7, got {settings.ollama.head_ratio}"
        )

    def test_retryable_status_codes_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [ollama] retryable_status_codes = [503, 429]
        When load_settings() is called
        Then settings.ollama.retryable_status_codes is [503, 429]
        """
        # Given: settings with custom retryable_status_codes
        toml = _EMBEDDER_SETTINGS.replace(
            "retryable_status_codes = [408, 429, 500, 502, 503, 504]",
            "retryable_status_codes = [503, 429]",
        )
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: retryable_status_codes is [503, 429]
        assert settings.ollama.retryable_status_codes == [503, 429], (
            f"Expected [503, 429], got {settings.ollama.retryable_status_codes}"
        )

    # -- Embedder wiring (WHAT 6-10) ----------------------------------------

    async def test_embedder_accepts_ollama_config(self) -> None:
        """
        Given a valid OllamaConfig with all fields
        When Embedder(config) is constructed
        Then the embedder stores all config values
        """
        # Given: a complete OllamaConfig
        config = OllamaConfig(
            base_url="http://localhost:11434",
            llm_model="mistral:7b",
            embed_model="nomic-embed-text",
            slow_llm_threshold_ms=30_000,
            classify_system_prompt="You are a classifier.",
            max_retries=5,
            base_delay=2.0,
            max_embed_chars=4000,
            head_ratio=0.7,
            retryable_status_codes=[503, 429],
        )

        # When: Embedder is constructed from config
        embedder = Embedder(config)

        # Then: all config values are accessible
        assert embedder.base_url == "http://localhost:11434"
        assert embedder.embed_model == "nomic-embed-text"
        assert embedder.llm_model == "mistral:7b"
        assert embedder.max_retries == 5
        assert embedder.base_delay == 2.0
        assert embedder.max_embed_chars == 4000

    async def test_embedder_truncates_at_config_max_embed_chars(self) -> None:
        """
        Given an Embedder with max_embed_chars = 100
        When embed() is called with text longer than 100 chars
        Then the text is truncated to ~100 chars
        """
        # Given: Embedder with small max_embed_chars
        config = OllamaConfig(
            base_url="http://localhost:11434",
            llm_model="mistral:7b",
            embed_model="nomic-embed-text",
            slow_llm_threshold_ms=30_000,
            classify_system_prompt="test",
            max_retries=1,
            base_delay=0.0,
            max_embed_chars=100,
            head_ratio=0.6,
            retryable_status_codes=[503],
        )
        embedder = Embedder(config)
        mock_client = make_mock_ollama_client()
        embedder._client = mock_client  # type: ignore[attr-defined]

        # When: embed long text
        await embedder.embed("x" * 500)

        # Then: sent text is truncated to max_embed_chars
        call_args = mock_client.embed.call_args
        sent_text: str = call_args.kwargs.get("input") or call_args[1].get("input")
        assert len(sent_text) <= 100, (
            f"Text should be truncated to ~100 chars, got {len(sent_text)}"
        )

    async def test_embedder_uses_config_head_ratio(self) -> None:
        """
        Given an Embedder with head_ratio = 0.8 and max_embed_chars = 100
        When embed() truncates long text
        Then ~80% of the budget is from the head
        """
        # Given: Embedder with custom head_ratio
        config = OllamaConfig(
            base_url="http://localhost:11434",
            llm_model="mistral:7b",
            embed_model="nomic-embed-text",
            slow_llm_threshold_ms=30_000,
            classify_system_prompt="test",
            max_retries=1,
            base_delay=0.0,
            max_embed_chars=100,
            head_ratio=0.8,
            retryable_status_codes=[503],
        )
        embedder = Embedder(config)
        mock_client = make_mock_ollama_client()
        embedder._client = mock_client  # type: ignore[attr-defined]

        # When: embed text with distinct head ("H") and tail ("T") chars
        await embedder.embed("H" * 300 + "T" * 300)

        # Then: head is ~80% of budget (budget = 100 - 5 marker = 95, head = 76)
        call_args = mock_client.embed.call_args
        sent_text: str = call_args.kwargs.get("input") or call_args[1].get("input")
        parts = sent_text.split("\n[\u2026]\n")
        assert len(parts) == 2, f"Should have head and tail separated by marker, got {parts!r}"
        head_len = len(parts[0])
        # budget = 100 - 5 = 95; head = int(95 * 0.8) = 76
        assert head_len == 76, f"Head should be 76 chars (80% of 95 budget), got {head_len}"

    async def test_embedder_retries_only_configured_status_codes(self) -> None:
        """
        Given retryable_status_codes = [418]
        When embed() encounters a 503 error
        Then the error is NOT retried (503 is not in the configured set)
        """
        # Given: Embedder where only 418 is retryable
        config = OllamaConfig(
            base_url="http://localhost:11434",
            llm_model="mistral:7b",
            embed_model="nomic-embed-text",
            slow_llm_threshold_ms=30_000,
            classify_system_prompt="test",
            max_retries=3,
            base_delay=0.0,
            max_embed_chars=8000,
            head_ratio=0.6,
            retryable_status_codes=[418],
        )
        embedder = Embedder(config)
        mock_client = make_mock_ollama_client()
        mock_client.embed = AsyncMock(side_effect=ResponseError("server busy", status_code=503))
        embedder._client = mock_client  # type: ignore[attr-defined]

        # When / Then: 503 fails immediately (not retried)
        with pytest.raises(ActionableError):
            await embedder.embed("test")

        assert mock_client.embed.call_count == 1, (
            f"503 should NOT be retried when not in retryable set, "
            f"but was called {mock_client.embed.call_count} times"
        )

    async def test_embedder_uses_config_max_retries(self) -> None:
        """
        Given max_retries = 5
        When embed() encounters retryable errors on every attempt
        Then exactly 5 attempts are made
        """
        # Given: Embedder with 5 retries, all failing
        config = OllamaConfig(
            base_url="http://localhost:11434",
            llm_model="mistral:7b",
            embed_model="nomic-embed-text",
            slow_llm_threshold_ms=30_000,
            classify_system_prompt="test",
            max_retries=5,
            base_delay=0.0,
            max_embed_chars=8000,
            head_ratio=0.6,
            retryable_status_codes=[503],
        )
        embedder = Embedder(config)
        mock_client = make_mock_ollama_client()
        mock_client.embed = AsyncMock(side_effect=ResponseError("server busy", status_code=503))
        embedder._client = mock_client  # type: ignore[attr-defined]

        # When / Then: all retries exhausted
        with pytest.raises(ActionableError):
            await embedder.embed("test")

        # Then: exactly 5 attempts
        assert mock_client.embed.call_count == 5, (
            f"Should attempt exactly 5 times, got {mock_client.embed.call_count}"
        )

    # -- Missing field errors (WHAT 11-15) ----------------------------------

    def test_missing_max_retries_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml omits [ollama].max_retries
        When load_settings() is called
        Then ActionableError is raised naming max_retries
        """
        # Given: TOML without max_retries
        toml = _EMBEDDER_SETTINGS.replace("max_retries = 3\n", "")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "max_retries" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )

    def test_missing_base_delay_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml omits [ollama].base_delay
        When load_settings() is called
        Then ActionableError is raised naming base_delay
        """
        # Given: TOML without base_delay
        toml = _EMBEDDER_SETTINGS.replace("base_delay = 1.0\n", "")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "base_delay" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )

    def test_missing_max_embed_chars_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml omits [ollama].max_embed_chars
        When load_settings() is called
        Then ActionableError is raised naming max_embed_chars
        """
        # Given: TOML without max_embed_chars
        toml = _EMBEDDER_SETTINGS.replace("max_embed_chars = 8000\n", "")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "max_embed_chars" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )

    def test_missing_head_ratio_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml omits [ollama].head_ratio
        When load_settings() is called
        Then ActionableError is raised naming head_ratio
        """
        # Given: TOML without head_ratio
        toml = _EMBEDDER_SETTINGS.replace("head_ratio = 0.6\n", "")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "head_ratio" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )

    def test_missing_retryable_status_codes_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml omits [ollama].retryable_status_codes
        When load_settings() is called
        Then ActionableError is raised naming retryable_status_codes
        """
        # Given: TOML without retryable_status_codes
        toml = _EMBEDDER_SETTINGS.replace(
            "retryable_status_codes = [408, 429, 500, 502, 503, 504]\n", ""
        )
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "retryable_status_codes" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )


# ---------------------------------------------------------------------------
# Phase 8c — Remaining hardcoded defaults
# ---------------------------------------------------------------------------

# _BASE_SETTINGS now includes all Phase 8c fields, so alias for clarity.
_PHASE8C_SETTINGS = _BASE_SETTINGS


class TestOutputPathConfig:
    """
    REQUIREMENT: Output-related paths (jd_dir, decisions_dir, log_dir) live in
    settings.toml under [output], not as hardcoded module constants.

    WHO: Operators who need files written to non-default locations (shared
         volumes, absolute paths, CI-specific directories)
    WHAT: (1) The system loads jd_dir from [output].
          (2) The system loads decisions_dir from [output].
          (3) The system loads log_dir from [output].
          (4) The system rejects missing [output] jd_dir.
          (5) The system rejects missing [output] decisions_dir.
          (6) The system rejects missing [output] log_dir.
    WHY: Hardcoded paths prevent deployment flexibility; operators must be
         able to control where all data is written via config

    MOCK BOUNDARY:
        Mock:  nothing
        Real:  load_settings(), config validation
        Never: Mock config loading internals
    """

    # -- Config loading (WHAT 1-3) ------------------------------------------

    def test_jd_dir_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [output] jd_dir = "custom/jds"
        When load_settings() is called
        Then settings.output.jd_dir is "custom/jds"
        """
        # Given: settings with custom jd_dir
        toml = _PHASE8C_SETTINGS.replace('jd_dir = "output/jds"', 'jd_dir = "custom/jds"')
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: jd_dir is "custom/jds"
        assert settings.output.jd_dir == "custom/jds", (
            f"Expected jd_dir='custom/jds', got {settings.output.jd_dir}"
        )

    def test_decisions_dir_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [output] decisions_dir = "/tmp/decisions"
        When load_settings() is called
        Then settings.output.decisions_dir is "/tmp/decisions"
        """
        # Given: settings with custom decisions_dir
        toml = _PHASE8C_SETTINGS.replace(
            'decisions_dir = "data/decisions"', 'decisions_dir = "/tmp/decisions"'
        )
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: decisions_dir is "/tmp/decisions"
        assert settings.output.decisions_dir == "/tmp/decisions", (
            f"Expected decisions_dir='/tmp/decisions', got {settings.output.decisions_dir}"
        )

    def test_log_dir_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [output] log_dir = "logs/custom"
        When load_settings() is called
        Then settings.output.log_dir is "logs/custom"
        """
        # Given: settings with custom log_dir
        toml = _PHASE8C_SETTINGS.replace('log_dir = "data/logs"', 'log_dir = "logs/custom"')
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: log_dir is "logs/custom"
        assert settings.output.log_dir == "logs/custom", (
            f"Expected log_dir='logs/custom', got {settings.output.log_dir}"
        )

    # -- Missing field errors (WHAT 4-6) ------------------------------------

    def test_missing_jd_dir_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml omits [output].jd_dir
        When load_settings() is called
        Then ActionableError is raised naming jd_dir
        """
        # Given: TOML without jd_dir
        toml = _PHASE8C_SETTINGS.replace('jd_dir = "output/jds"\n', "")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "jd_dir" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )

    def test_missing_decisions_dir_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml omits [output].decisions_dir
        When load_settings() is called
        Then ActionableError is raised naming decisions_dir
        """
        # Given: TOML without decisions_dir
        toml = _PHASE8C_SETTINGS.replace('decisions_dir = "data/decisions"\n', "")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "decisions_dir" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )

    def test_missing_log_dir_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml omits [output].log_dir
        When load_settings() is called
        Then ActionableError is raised naming log_dir
        """
        # Given: TOML without log_dir
        toml = _PHASE8C_SETTINGS.replace('log_dir = "data/logs"\n', "")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "log_dir" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )


class TestScoringTunablesConfig:
    """
    REQUIREMENT: Scoring pipeline tunables (chunk_overlap,
    dedup_similarity_threshold) live in settings.toml under [scoring],
    not as hardcoded constants in scorer.py / ranker.py.

    WHO: Operators tuning recall vs. precision of the scoring and dedup
         pipeline
    WHAT: (1) The system loads chunk_overlap from [scoring].
          (2) The system loads dedup_similarity_threshold from [scoring].
          (3) The scorer uses config chunk_overlap for text chunking.
          (4) The ranker uses config dedup_similarity_threshold for dedup.
          (5) The system rejects missing [scoring] chunk_overlap.
          (6) The system rejects missing [scoring] dedup_similarity_threshold.
    WHY: Different corpora and embedding models may need different overlap
         and dedup sensitivity — these are operational concerns

    MOCK BOUNDARY:
        Mock:  nothing for config tests; ollama client (I/O) for scorer wiring
        Real:  load_settings(), Scorer.score(), Ranker.rank()
        Never: Mock config loading internals; call _chunk_text or _deduplicate_near directly
    """

    # -- Config loading (WHAT 1-2) ------------------------------------------

    def test_chunk_overlap_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [scoring] chunk_overlap = 1000
        When load_settings() is called
        Then settings.scoring.chunk_overlap is 1000
        """
        # Given: settings with custom chunk_overlap
        toml = _PHASE8C_SETTINGS.replace("chunk_overlap = 2000", "chunk_overlap = 1000")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: chunk_overlap is 1000
        assert settings.scoring.chunk_overlap == 1000, (
            f"Expected chunk_overlap=1000, got {settings.scoring.chunk_overlap}"
        )

    def test_dedup_similarity_threshold_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [scoring] dedup_similarity_threshold = 0.90
        When load_settings() is called
        Then settings.scoring.dedup_similarity_threshold is 0.90
        """
        # Given: settings with custom dedup_similarity_threshold
        toml = _PHASE8C_SETTINGS.replace(
            "dedup_similarity_threshold = 0.95", "dedup_similarity_threshold = 0.90"
        )
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: dedup_similarity_threshold is 0.90
        assert settings.scoring.dedup_similarity_threshold == pytest.approx(0.90), (
            f"Expected dedup_similarity_threshold=0.90, "
            f"got {settings.scoring.dedup_similarity_threshold}"
        )

    # -- Wiring (WHAT 3-4) --------------------------------------------------

    async def test_scorer_uses_config_chunk_overlap(
        self,
        mock_embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        """
        Given a Scorer with chunk_overlap = 50 and embedder.max_embed_chars = 100
        When score() is called with a 200-char JD
        Then the embedder receives 4 embed calls (4 overlapping chunks)
        """
        # Given: Scorer with custom chunk_overlap; embedder max_embed_chars = 100
        mock_embedder.max_embed_chars = 100  # type: ignore[misc]
        scorer = Scorer(
            store=vector_store,
            embedder=mock_embedder,
            chunk_overlap=50,
        )
        # Seed required collections so score() doesn't raise
        # Use 5-dim to match EMBED_FAKE from mock_embedder
        for coll_name in ("resume", "role_archetypes"):
            vector_store.add_documents(
                coll_name,
                ids=[f"{coll_name}-seed"],
                documents=[f"Seed document for {coll_name}"],
                embeddings=[[0.1] * 5],
            )

        # When: score a 200-char JD (should produce 4 chunks with step=50)
        text = "A" * 200
        await scorer.score(text)

        # Then: embedder.embed was called 4 times (once per chunk)
        embed_calls = mock_embedder._client.embed.call_count  # type: ignore[union-attr]
        assert embed_calls == 4, (
            f"Expected 4 embed calls (200 chars / step 50 = 4 chunks), got {embed_calls}"
        )

    def test_ranker_dedup_uses_config_threshold(self) -> None:
        """
        Given dedup_similarity_threshold = 0.80
        When ranker.rank() receives two listings with cosine similarity 0.85
        Then the lower-scored duplicate is collapsed
        """
        # Given: two listings with different URLs (not exact-match dupes)
        listing_a = JobListing(
            board="a",
            external_id="1",
            title="Senior Engineer",
            company="Acme",
            location="Remote",
            url="u1",
            full_text="Full text for listing A",
        )
        listing_b = JobListing(
            board="b",
            external_id="2",
            title="Senior Engineer",
            company="Beta",
            location="Remote",
            url="u2",
            full_text="Full text for listing B",
        )
        scores_a = ScoreResult(
            fit_score=0.8,
            archetype_score=0.9,
            history_score=0.5,
            disqualified=False,
        )
        scores_b = ScoreResult(
            fit_score=0.7,
            archetype_score=0.8,
            history_score=0.4,
            disqualified=False,
        )

        # Embeddings that produce cosine similarity ~0.85 (above 0.80 threshold)
        embed_a = [1.0, 0.0]
        embed_b = [0.85, 0.527]  # cosine sim ≈ 0.85
        embeddings = {"u1": embed_a, "u2": embed_b}

        # When: rank() with threshold = 0.80 and embeddings
        ranker = Ranker(
            archetype_weight=0.5,
            fit_weight=0.3,
            history_weight=0.2,
            dedup_similarity_threshold=0.80,
            min_score_threshold=0.0,  # don't filter by score
        )
        ranked, summary = ranker.rank(
            [(listing_a, scores_a), (listing_b, scores_b)],
            embeddings=embeddings,
        )

        # Then: the near-duplicate is collapsed
        assert len(ranked) == 1, (
            f"With threshold=0.80 and sim≈0.85, should dedup to 1 survivor, got {len(ranked)}"
        )
        assert summary.total_deduplicated >= 1, (
            f"Expected at least 1 dedup, got {summary.total_deduplicated}"
        )

    # -- Missing field errors (WHAT 5-6) ------------------------------------

    def test_missing_chunk_overlap_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml omits [scoring].chunk_overlap
        When load_settings() is called
        Then ActionableError is raised naming chunk_overlap
        """
        # Given: TOML without chunk_overlap
        toml = _PHASE8C_SETTINGS.replace("chunk_overlap = 2000\n", "")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "chunk_overlap" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )

    def test_missing_dedup_similarity_threshold_raises_actionable_error(
        self, tmp_path: Path
    ) -> None:
        """
        Given settings.toml omits [scoring].dedup_similarity_threshold
        When load_settings() is called
        Then ActionableError is raised naming dedup_similarity_threshold
        """
        # Given: TOML without dedup_similarity_threshold
        toml = _PHASE8C_SETTINGS.replace("dedup_similarity_threshold = 0.95\n", "")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "dedup_similarity_threshold" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )


class TestBoardBrowserConfig:
    """
    REQUIREMENT: Browser-related defaults (session_storage_dir,
    per-board rate_limit_range, throttle_max_retries, throttle_base_delay)
    live in settings.toml, not as hardcoded constants in adapter code.

    WHO: Operators running against different boards with varying anti-bot
         sensitivity, or deploying on non-standard filesystem layouts
    WHAT: (1) The system loads session_storage_dir from [boards].
          (2) The system loads rate_limit_range from per-board config.
          (3) Boards without rate_limit_range get the default [1.5, 3.5].
          (4) The system loads throttle_max_retries from per-board config.
          (5) The system loads throttle_base_delay from per-board config.
          (6) Boards without throttle settings get None (not all boards
              need throttle backoff).
          (7) The system rejects missing [boards] session_storage_dir.
          (8) The system rejects missing rate_limit_range for an enabled board.
    WHY: Anti-bot tuning is operational — different boards need different
         timing, and the cookie storage path must be operator-configurable

    MOCK BOUNDARY:
        Mock:  nothing
        Real:  load_settings(), config validation
        Never: Mock config loading internals
    """

    # -- Config loading (WHAT 1-6) ------------------------------------------

    def test_session_storage_dir_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [boards] session_storage_dir = "/opt/sessions"
        When load_settings() is called
        Then settings.session_storage_dir is "/opt/sessions"
        """
        # Given: settings with custom session_storage_dir
        toml = _PHASE8C_SETTINGS.replace(
            'session_storage_dir = "data"', 'session_storage_dir = "/opt/sessions"'
        )
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: session_storage_dir is "/opt/sessions"
        assert settings.session_storage_dir == "/opt/sessions", (
            f"Expected session_storage_dir='/opt/sessions', got {settings.session_storage_dir}"
        )

    def test_rate_limit_range_loaded_from_board_config(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [boards.testboard] rate_limit_range = [5.0, 10.0]
        When load_settings() is called
        Then settings.boards["testboard"].rate_limit_range is (5.0, 10.0)
        """
        # Given: settings with custom rate_limit_range
        toml = _PHASE8C_SETTINGS.replace(
            "rate_limit_range = [1.5, 3.5]", "rate_limit_range = [5.0, 10.0]"
        )
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: rate_limit_range is (5.0, 10.0)
        board = settings.boards["testboard"]
        assert board.rate_limit_range == (5.0, 10.0), (
            f"Expected rate_limit_range=(5.0, 10.0), got {board.rate_limit_range}"
        )

    def test_board_without_throttle_settings_gets_none(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [boards.testboard] without throttle settings
        When load_settings() is called
        Then settings.boards["testboard"].throttle_max_retries is None
        And settings.boards["testboard"].throttle_base_delay is None
        """
        # Given: standard board config with no throttle settings
        settings_path = _write_config(tmp_path, settings_toml=_PHASE8C_SETTINGS)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: throttle settings are None
        board = settings.boards["testboard"]
        assert board.throttle_max_retries is None, (
            f"Expected throttle_max_retries=None, got {board.throttle_max_retries}"
        )
        assert board.throttle_base_delay is None, (
            f"Expected throttle_base_delay=None, got {board.throttle_base_delay}"
        )

    def test_throttle_settings_loaded_from_board_config(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [boards.testboard] throttle_max_retries = 5
        And throttle_base_delay = 3.0
        When load_settings() is called
        Then settings.boards["testboard"].throttle_max_retries is 5
        And settings.boards["testboard"].throttle_base_delay is 3.0
        """
        # Given: board config with throttle settings
        toml = _PHASE8C_SETTINGS.replace(
            "rate_limit_range = [1.5, 3.5]",
            "rate_limit_range = [1.5, 3.5]\nthrottle_max_retries = 5\nthrottle_base_delay = 3.0",
        )
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: throttle settings are loaded
        board = settings.boards["testboard"]
        assert board.throttle_max_retries == 5, (
            f"Expected throttle_max_retries=5, got {board.throttle_max_retries}"
        )
        assert board.throttle_base_delay == pytest.approx(3.0), (
            f"Expected throttle_base_delay=3.0, got {board.throttle_base_delay}"
        )

    # -- Missing field errors (WHAT 7-8) ------------------------------------

    def test_missing_session_storage_dir_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml omits [boards].session_storage_dir
        When load_settings() is called
        Then ActionableError is raised naming session_storage_dir
        """
        # Given: TOML without session_storage_dir
        toml = _PHASE8C_SETTINGS.replace('session_storage_dir = "data"\n', "")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "session_storage_dir" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )

    def test_missing_rate_limit_range_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml omits [boards.testboard].rate_limit_range
        When load_settings() is called
        Then ActionableError is raised naming rate_limit_range
        """
        # Given: TOML without rate_limit_range
        toml = _PHASE8C_SETTINGS.replace("rate_limit_range = [1.5, 3.5]\n", "")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "rate_limit_range" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )


# ---------------------------------------------------------------------------
# Phase 8b Remaining — Operational Config Externalization
# ---------------------------------------------------------------------------


class TestEvalHistoryConfig:
    """
    REQUIREMENT: The eval history JSONL path is configurable via
    settings.toml so operators can control where evaluation history
    is persisted.

    WHO: Operators deploying on non-default directory layouts or wanting
         eval history alongside other output artifacts
    WHAT: (1) The system loads eval_history_path from [output].
          (2) The system rejects missing [output] eval_history_path.
    WHY: The hardcoded path prevents operators from controlling where
         eval artifacts land — and the path uses forward slashes that
         may behave unexpectedly on Windows

    MOCK BOUNDARY:
        Mock:  nothing — pure config loading and validation
        Real:  load_settings(), config validation
        Never: Mock config loading internals
    """

    def test_eval_history_path_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [output] eval_history_path = "output/eval.jsonl"
        When load_settings() is called
        Then OutputConfig.eval_history_path is "output/eval.jsonl"
        """
        # Given: settings with custom eval_history_path
        toml = _BASE_SETTINGS.replace(
            'eval_history_path = "data/eval_history.jsonl"',
            'eval_history_path = "output/eval.jsonl"',
        )
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: eval_history_path is "output/eval.jsonl"
        assert settings.output.eval_history_path == "output/eval.jsonl", (  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # Phase 3 field
            f"Expected eval_history_path='output/eval.jsonl', "
            f"got {settings.output.eval_history_path!r}"  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # Phase 3 field
        )

    def test_missing_eval_history_path_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml [output] has no eval_history_path field
        When load_settings() is called
        Then ActionableError is raised naming the missing field
        """
        # Given: TOML without eval_history_path
        toml = _BASE_SETTINGS.replace('eval_history_path = "data/eval_history.jsonl"\n', "")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "eval_history_path" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )


class TestLoginUrlConfig:
    """
    REQUIREMENT: Per-board login URLs are configurable via settings.toml
    so operators can point to custom login endpoints or SSO portals
    without modifying source code.

    WHO: Operators behind corporate proxies or using SSO-wrapped board
         login pages
    WHAT: (1) The system loads login_url from per-board config when present.
          (2) Boards without login_url get None (field is optional).
          (3) handle_login() uses the configured login_url when present.
          (4) handle_login() falls back to the generic URL pattern when
              login_url is absent from both config and _LOGIN_URLS.
    WHY: Login URLs are deployment-specific — corporate SSO wrappers,
         regional domains, or custom auth flows require different endpoints

    MOCK BOUNDARY:
        Mock:  async_playwright (I/O — launches browser),
               builtins.input (I/O — terminal prompt)
        Real:  handle_login() dispatch, BoardConfig, Settings construction
        Never: Mock load_settings or config loading internals
    """

    # -- Config loading (WHAT 1-2) ------------------------------------------

    def test_login_url_loaded_from_board_config(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [boards.testboard] login_url = "https://custom.example.com/login"
        When load_settings() is called
        Then BoardConfig.login_url is "https://custom.example.com/login"
        """
        # Given: settings with custom login_url
        toml = _BASE_SETTINGS.replace(
            'login_url = "https://www.testboard.com/login"',
            'login_url = "https://custom.example.com/login"',
        )
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: login_url is "https://custom.example.com/login"
        board = settings.boards["testboard"]
        assert board.login_url == "https://custom.example.com/login", (  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # Phase 3 field
            f"Expected login_url='https://custom.example.com/login', got {board.login_url!r}"  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # Phase 3 field
        )

    def test_board_without_login_url_gets_none(self, tmp_path: Path) -> None:
        """
        Given settings.toml [boards.testboard] has no login_url field
        When load_settings() is called
        Then BoardConfig.login_url is None
        """
        # Given: settings without login_url
        toml = _BASE_SETTINGS.replace('login_url = "https://www.testboard.com/login"\n', "")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: login_url is None
        board = settings.boards["testboard"]
        assert board.login_url is None, (  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # Phase 3 field
            f"Expected login_url=None for board without login_url, got {board.login_url!r}"  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # Phase 3 field
        )

    # -- Wiring (WHAT 3-4) --------------------------------------------------

    def test_handle_login_uses_configured_url(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Given settings.toml has [boards.testboard] login_url = "https://custom.example.com/login"
        When handle_login(args) is called for that board
        Then page.goto receives "https://custom.example.com/login"
        """
        # Given: settings.toml with custom login_url on testboard
        monkeypatch.chdir(tmp_path)
        toml = _BASE_SETTINGS.replace(
            'login_url = "https://www.testboard.com/login"',
            'login_url = "https://custom.example.com/login"',
        )
        (tmp_path / "config").mkdir(exist_ok=True)
        _write_config(tmp_path / "config", settings_toml=toml)
        # Write resume so load_settings() doesn't fail on missing file
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / "data" / "resume.md").write_text("# Resume\n")

        # Given: mock Playwright at the I/O boundary
        mock_page = MagicMock()
        mock_page.goto = AsyncMock()
        mock_context = MagicMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.storage_state = AsyncMock(return_value={"cookies": [], "origins": []})
        mock_context.close = AsyncMock()
        mock_browser = MagicMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()
        mock_pw = MagicMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw.stop = AsyncMock()
        mock_pw_cm = MagicMock()
        mock_pw_cm.start = AsyncMock(return_value=mock_pw)

        # When: handle_login runs with real load_settings()
        with (
            patch(
                "jobsearch_rag.adapters.session.async_playwright",
                return_value=mock_pw_cm,
            ),
            patch("builtins.input", return_value=""),
        ):
            args = argparse.Namespace(board="testboard", browser=None)
            handle_login(args)

        # Then: page.goto received the config login URL
        url_arg = mock_page.goto.call_args[0][0]
        assert url_arg == "https://custom.example.com/login", (
            f"Expected page.goto to receive 'https://custom.example.com/login', got {url_arg!r}"
        )

    def test_handle_login_falls_back_to_generic_url(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Given settings.toml has a board with no login_url and board not in _LOGIN_URLS
        When handle_login(args) is called for that board
        Then page.goto receives "https://www.<board>.com"
        """
        # Given: settings.toml with "unknownboard" — no login_url, not in _LOGIN_URLS
        monkeypatch.chdir(tmp_path)
        toml = (
            _BASE_SETTINGS.replace(
                'enabled = ["testboard"]',
                'enabled = ["unknownboard"]',
            )
            .replace(
                "[boards.testboard]\n",
                "[boards.unknownboard]\n",
            )
            .replace(
                'login_url = "https://www.testboard.com/login"\n',
                "",
            )
        )
        (tmp_path / "config").mkdir(exist_ok=True)
        _write_config(tmp_path / "config", settings_toml=toml)
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / "data" / "resume.md").write_text("# Resume\n")

        # Given: mock Playwright at the I/O boundary
        mock_page = MagicMock()
        mock_page.goto = AsyncMock()
        mock_context = MagicMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.storage_state = AsyncMock(return_value={"cookies": [], "origins": []})
        mock_context.close = AsyncMock()
        mock_browser = MagicMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()
        mock_pw = MagicMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw.stop = AsyncMock()
        mock_pw_cm = MagicMock()
        mock_pw_cm.start = AsyncMock(return_value=mock_pw)

        # When: handle_login runs with real load_settings()
        with (
            patch(
                "jobsearch_rag.adapters.session.async_playwright",
                return_value=mock_pw_cm,
            ),
            patch("builtins.input", return_value=""),
        ):
            args = argparse.Namespace(board="unknownboard", browser=None)
            handle_login(args)

        # Then: page.goto received the generic URL pattern
        url_arg = mock_page.goto.call_args[0][0]
        assert url_arg == "https://www.unknownboard.com", (
            f"Expected fallback to 'https://www.unknownboard.com', got {url_arg!r}"
        )


class TestStealthConfig:
    """
    REQUIREMENT: The stealth flag is configurable per board via settings.toml
    so operators can enable playwright-stealth for any board that needs it,
    not just the hardcoded "linkedin" check.

    WHO: Operators running against boards with bot detection that requires
         stealth patches (currently LinkedIn; potentially others in future)
    WHAT: (1) The system loads stealth from per-board config as a bool.
          (2) Boards without stealth default to False.
          (3) The runner passes the configured stealth value to SessionConfig
              instead of comparing board_name == "linkedin".
    WHY: Hardcoding stealth to a single board name couples detection-evasion
         policy to source code — new boards requiring stealth need a code
         change instead of a config change

    MOCK BOUNDARY:
        Mock:  SessionManager (I/O — launches browser)
        Real:  load_settings(), BoardConfig construction, SessionConfig construction
        Never: Mock config loading internals; never mock the runner to test config flow
    """

    # -- Config loading (WHAT 1-2) ------------------------------------------

    def test_stealth_loaded_from_board_config(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [boards.testboard] stealth = true
        When load_settings() is called
        Then BoardConfig.stealth is True
        """
        # Given: settings with stealth = true
        toml = _BASE_SETTINGS.replace("stealth = false", "stealth = true")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: stealth is True
        board = settings.boards["testboard"]
        assert board.stealth is True, f"Expected stealth=True, got {board.stealth!r}"  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # Phase 3 field

    def test_board_without_stealth_defaults_to_false(self, tmp_path: Path) -> None:
        """
        Given settings.toml [boards.testboard] has no stealth field
        When load_settings() is called
        Then BoardConfig.stealth is False
        """
        # Given: settings without stealth
        toml = _BASE_SETTINGS.replace("stealth = false\n", "")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: stealth defaults to False
        board = settings.boards["testboard"]
        assert board.stealth is False, f"Expected stealth=False by default, got {board.stealth!r}"  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # Phase 3 field

    # -- Wiring (WHAT 3) ----------------------------------------------------

    async def test_runner_passes_config_stealth_to_session(self) -> None:
        """
        Given a board config with stealth = true
        When the runner constructs a SessionConfig for that board
        Then SessionConfig.stealth is True (not derived from board name)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Given: settings where "testboard" (not "linkedin") has stealth=True
            settings = make_test_settings(tmpdir)
            # Patch the board config to include stealth=True
            board_cfg = settings.boards["testboard"]
            object.__setattr__(board_cfg, "stealth", True)

            # Given: a real PipelineRunner with mocked Ollama client
            mock_client = make_mock_ollama_client()
            with patch(
                "jobsearch_rag.rag.embedder.ollama_sdk.AsyncClient",
                return_value=mock_client,
            ):
                runner = PipelineRunner(settings)

            # Seed store so auto-indexing is skipped
            for name in ("resume", "role_archetypes", "global_positive_signals"):
                runner.store.add_documents(
                    name,
                    ids=[f"{name}-seed"],
                    documents=[f"Seed document for {name}"],
                    embeddings=[[0.1] * 768],
                )

            # Given: mock adapter and SessionManager at I/O boundary
            mock_adapter = MagicMock()
            mock_adapter.board_name = "testboard"
            mock_adapter.rate_limit_seconds = (0.0, 0.0)
            mock_adapter.authenticate = AsyncMock()
            mock_adapter.search = AsyncMock(return_value=[])

            captured_configs: list[Any] = []

            class _CapturingSessionManager:
                """Capture SessionConfig passed to SessionManager."""

                def __init__(self, config: Any) -> None:
                    captured_configs.append(config)

                async def __aenter__(self) -> _CapturingSessionManager:
                    return self

                async def __aexit__(self, *args: object) -> None:
                    pass

                async def new_page(self) -> MagicMock:
                    return MagicMock()

                async def save_storage_state(self) -> Path:
                    return Path(tmpdir) / "session.json"

            with (
                patch.dict(
                    AdapterRegistry._registry,  # pyright: ignore[reportPrivateUsage]
                    {"testboard": lambda: mock_adapter},
                ),  # type: ignore[dict-item]
                patch(
                    "jobsearch_rag.pipeline.runner.SessionManager",
                    _CapturingSessionManager,
                ),
            ):
                # When: the pipeline runs
                await runner.run()

            # Then: SessionConfig.stealth was True (from board config, not board name)
            assert len(captured_configs) == 1, (
                f"Expected 1 SessionManager creation, got {len(captured_configs)}"
            )
            session_config = captured_configs[0]
            assert session_config.stealth is True, (
                f"Expected stealth=True from board config, got {session_config.stealth!r}. "
                f"Board name is 'testboard', not 'linkedin' — stealth must come from config."
            )


class TestAdaptersConfig:
    r"""
    REQUIREMENT: Browser binary paths and CDP timeout are configurable via
    settings.toml so operators on Windows and Linux can specify browser
    locations without code changes, and adjust CDP startup timing for
    slower machines. All fields are required — there are no OS-specific
    defaults or fallback paths.

    WHO: Operators on Windows (Edge at C:\Program Files\...\msedge.exe),
         Linux (chromium at /usr/bin/chromium-browser), or machines with
         slow startup where the CDP timeout must be tuned
    WHAT: (1) The system loads browser_paths from [adapters] as a dict of
              {channel: [path, ...]}.
          (2) The system loads cdp_timeout from [adapters].
          (3) Missing [adapters] section raises ActionableError.
          (4) Missing browser_paths in [adapters] raises ActionableError.
          (5) Missing cdp_timeout in [adapters] raises ActionableError.
          (6) cdp_timeout <= 0.0 raises ActionableError (must be positive).
          (7) browser_paths channel value that is not a list raises
              ActionableError naming the channel.
          (8) SessionManager launches the browser from a config-provided path
              when that path exists.
          (9) SessionManager raises ActionableError when the requested
              channel has no entry in browser_paths.
          (10) SessionManager waits the configured cdp_timeout duration
              before raising TimeoutError on a slow CDP endpoint.
          (11) The system skips a non-existent config-provided browser path
              and uses the next valid path in the list.
    WHY: _BROWSER_PATHS contains only macOS paths — Windows and Linux users
         cannot launch CDP browsers without editing source. Making all paths
         config-driven with no OS-specific defaults ensures every platform
         is a first-class citizen. The CDP timeout must also be explicit so
         slow machines and CI environments can set an appropriate value.

    MOCK BOUNDARY:
        Mock:  subprocess.Popen (I/O — launches browser process),
               urllib.request.urlopen (I/O — polls CDP endpoint),
               playwright connect_over_cdp (I/O — connects to browser)
        Real:  load_settings(), SessionManager.__aenter__(), SessionConfig,
               config validation, binary resolution logic
        Never: Call _find_browser_binary() or _launch_cdp() directly from tests;
               mock config loading internals
    """

    # -- Config loading (WHAT 1-2) ------------------------------------------

    def test_browser_paths_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [adapters] browser_paths with msedge and chrome entries
        When load_settings() is called
        Then AdaptersConfig.browser_paths contains the configured paths
        """
        # Given: settings with browser_paths
        settings_path = _write_config(tmp_path, settings_toml=_BASE_SETTINGS)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: browser_paths has msedge and chrome
        adapters = settings.adapters  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType,reportUnknownVariableType] # Phase 3 field
        assert "msedge" in adapters.browser_paths, (  # pyright: ignore[reportUnknownMemberType] # Phase 3 field
            f"Expected 'msedge' in browser_paths. Got: {adapters.browser_paths!r}"  # pyright: ignore[reportUnknownMemberType] # Phase 3 field
        )
        assert "chrome" in adapters.browser_paths, (  # pyright: ignore[reportUnknownMemberType] # Phase 3 field
            f"Expected 'chrome' in browser_paths. Got: {adapters.browser_paths!r}"  # pyright: ignore[reportUnknownMemberType] # Phase 3 field
        )
        assert isinstance(adapters.browser_paths["msedge"], list), (  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType] # Phase 3 field
            f"browser_paths['msedge'] should be a list, got {type(adapters.browser_paths['msedge'])}"  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType] # Phase 3 field
        )

    def test_cdp_timeout_loaded_from_settings(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [adapters] cdp_timeout = 30.0
        When load_settings() is called
        Then AdaptersConfig.cdp_timeout is 30.0
        """
        # Given: settings with custom cdp_timeout
        toml = _BASE_SETTINGS.replace("cdp_timeout = 15.0", "cdp_timeout = 30.0")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When: settings are loaded
        settings = load_settings(settings_path)

        # Then: cdp_timeout is 30.0
        assert settings.adapters.cdp_timeout == 30.0, (  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # Phase 3 field
            f"Expected cdp_timeout=30.0, got {settings.adapters.cdp_timeout}"  # pyright: ignore[reportAttributeAccessIssue,reportUnknownMemberType] # Phase 3 field
        )

    # -- Missing field errors (WHAT 3-6) ------------------------------------

    def test_missing_adapters_section_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has no [adapters] section
        When load_settings() is called
        Then ActionableError is raised naming the missing section
        """
        # Given: TOML without [adapters] section
        toml = re.sub(
            r"\[adapters\].*$",
            "",
            _BASE_SETTINGS,
            flags=re.DOTALL,
        )
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "adapters" in str(exc_info.value).lower(), (
            f"Error should name the missing section. Got: {exc_info.value}"
        )

    def test_missing_browser_paths_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml [adapters] has no browser_paths field
        When load_settings() is called
        Then ActionableError is raised naming the missing field
        """
        # Given: TOML with [adapters] but no browser_paths
        toml = re.sub(
            r"\[adapters\.browser_paths\].*$",
            "",
            _BASE_SETTINGS,
            flags=re.DOTALL,
        )
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "browser_paths" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )

    def test_missing_cdp_timeout_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml [adapters] has no cdp_timeout field
        When load_settings() is called
        Then ActionableError is raised naming the missing field
        """
        # Given: TOML without cdp_timeout
        toml = _BASE_SETTINGS.replace("cdp_timeout = 15.0\n", "")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        assert "cdp_timeout" in str(exc_info.value).lower(), (
            f"Error should name the missing field. Got: {exc_info.value}"
        )

    def test_negative_cdp_timeout_raises_actionable_error(self, tmp_path: Path) -> None:
        """
        Given settings.toml has [adapters] cdp_timeout = -1.0
        When load_settings() is called
        Then ActionableError is raised naming the field and stating it must be positive
        """
        # Given: negative cdp_timeout
        toml = _BASE_SETTINGS.replace("cdp_timeout = 15.0", "cdp_timeout = -1.0")
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        error_msg = str(exc_info.value).lower()
        assert "cdp_timeout" in error_msg, f"Error should name the field. Got: {exc_info.value}"
        assert "positive" in error_msg or "> 0" in error_msg, (
            f"Error should state value must be positive. Got: {exc_info.value}"
        )

    def test_browser_paths_channel_not_a_list_raises_actionable_error(
        self, tmp_path: Path
    ) -> None:
        """
        Given settings.toml [adapters.browser_paths] has a channel set to a string instead of a list
        When load_settings() is called
        Then ActionableError is raised naming the channel
        """
        # Given: browser_paths.msedge is a bare string, not a list
        toml = _BASE_SETTINGS.replace(
            'msedge = ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]',
            'msedge = "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"',
        )
        settings_path = _write_config(tmp_path, settings_toml=toml)

        # When / Then: load_settings raises ActionableError
        with pytest.raises(ActionableError) as exc_info:
            load_settings(settings_path)

        error_msg = str(exc_info.value).lower()
        assert "msedge" in error_msg, f"Error should name the channel. Got: {exc_info.value}"

    # -- SessionManager wiring (WHAT 8-10) ----------------------------------

    async def test_session_launches_browser_from_config_path(self, tmp_path: Path) -> None:
        """
        Given config browser_paths has msedge = ["/custom/path/msedge"]
        And that path exists on disk
        When SessionManager opens a CDP session with browser_channel="msedge"
        Then subprocess.Popen receives the config path as the binary
        """
        # Given: a custom binary path that "exists"
        custom_binary = str(tmp_path / "custom" / "msedge")
        (tmp_path / "custom").mkdir()
        (tmp_path / "custom" / "msedge").touch()

        # Given: a SessionConfig with browser_channel and config browser_paths
        config = SessionConfig(  # pyright: ignore[reportCallIssue]  # Phase 3 fields
            board_name="testboard",
            headless=True,
            browser_channel="msedge",
            storage_dir=tmp_path,
            browser_paths={"msedge": [custom_binary]},  # pyright: ignore[reportCallIssue]  # Phase 3 field
            cdp_timeout=15.0,  # pyright: ignore[reportCallIssue]  # Phase 3 field
        )

        # Given: mock Playwright and subprocess at I/O boundaries
        mock_browser = MagicMock()
        mock_browser.contexts = []
        mock_browser.new_context = AsyncMock(
            return_value=MagicMock(
                new_page=AsyncMock(return_value=MagicMock()),
                storage_state=AsyncMock(return_value={"cookies": [], "origins": []}),
                close=AsyncMock(),
            )
        )
        mock_browser.close = AsyncMock()

        mock_pw = MagicMock()
        mock_pw.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
        mock_pw.stop = AsyncMock()
        mock_pw_cm = MagicMock()
        mock_pw_cm.start = AsyncMock(return_value=mock_pw)

        captured_popen_args: list[Any] = []

        def _capturing_popen(cmd: Any, **kwargs: Any) -> MagicMock:
            captured_popen_args.append(cmd)
            proc = MagicMock()
            proc.poll.return_value = None
            proc.wait.return_value = 0
            return proc

        with (
            patch(
                "jobsearch_rag.adapters.session.async_playwright",
                return_value=mock_pw_cm,
            ),
            patch(
                "jobsearch_rag.adapters.session.subprocess.Popen",
                side_effect=_capturing_popen,
            ),
            patch("jobsearch_rag.adapters.session.urllib.request.urlopen"),
            patch(
                "jobsearch_rag.adapters.session.tempfile.mkdtemp",
                return_value=str(tmp_path / "cdp-profile"),
            ),
        ):
            # When: SessionManager opens a CDP session
            async with SessionManager(config) as _session:
                pass

        # Then: subprocess.Popen received the config path
        assert len(captured_popen_args) == 1, (
            f"Expected 1 Popen call, got {len(captured_popen_args)}"
        )
        binary_used = captured_popen_args[0][0]
        assert binary_used == custom_binary, (
            f"Expected Popen to receive '{custom_binary}', got {binary_used!r}"
        )

    async def test_session_raises_error_when_channel_not_in_config(self, tmp_path: Path) -> None:
        """
        Given config browser_paths has no entry for "chrome"
        When SessionManager opens a CDP session with browser_channel="chrome"
        Then ActionableError is raised naming the missing channel and
        suggesting the operator add it to [adapters] browser_paths
        """
        # Given: browser_paths has only msedge, not chrome
        config = SessionConfig(  # pyright: ignore[reportCallIssue]  # Phase 3 fields
            board_name="testboard",
            headless=True,
            browser_channel="chrome",
            storage_dir=tmp_path,
            browser_paths={"msedge": ["/some/path/msedge"]},  # pyright: ignore[reportCallIssue]  # Phase 3 field
            cdp_timeout=15.0,  # pyright: ignore[reportCallIssue]  # Phase 3 field
        )

        mock_pw = MagicMock()
        mock_pw.stop = AsyncMock()
        mock_pw_cm = MagicMock()
        mock_pw_cm.start = AsyncMock(return_value=mock_pw)

        with (
            patch(
                "jobsearch_rag.adapters.session.async_playwright",
                return_value=mock_pw_cm,
            ),
            patch(
                "jobsearch_rag.adapters.session.shutil.which",
                return_value="/usr/bin/google-chrome",
            ),
            pytest.raises(ActionableError) as exc_info,
        ):
            # When / Then: SessionManager raises ActionableError
            async with SessionManager(config) as _session:
                pass

        error_msg = str(exc_info.value).lower()
        assert "chrome" in error_msg, (
            f"Error should name the missing channel. Got: {exc_info.value}"
        )

    async def test_session_waits_configured_cdp_timeout(self, tmp_path: Path) -> None:
        """
        Given a SessionConfig with cdp_timeout = 0.5
        When SessionManager opens a CDP session and the endpoint never responds
        Then the system raises TimeoutError after approximately 0.5 seconds
        """
        # Given: a fake binary that "exists" on disk
        fake_binary = str(tmp_path / "msedge")
        (tmp_path / "msedge").touch()

        # Given: a SessionConfig with short cdp_timeout
        config = SessionConfig(  # pyright: ignore[reportCallIssue]  # Phase 3 fields
            board_name="testboard",
            headless=True,
            browser_channel="msedge",
            storage_dir=tmp_path,
            browser_paths={"msedge": [fake_binary]},  # pyright: ignore[reportCallIssue]  # Phase 3 field
            cdp_timeout=0.5,  # pyright: ignore[reportCallIssue]  # Phase 3 field
        )

        # Given: mock Playwright and subprocess, but urlopen always fails (CDP never ready)
        mock_pw = MagicMock()
        mock_pw.stop = AsyncMock()
        mock_pw_cm = MagicMock()
        mock_pw_cm.start = AsyncMock(return_value=mock_pw)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.wait.return_value = 0

        with (
            patch(
                "jobsearch_rag.adapters.session.async_playwright",
                return_value=mock_pw_cm,
            ),
            patch(
                "jobsearch_rag.adapters.session.subprocess.Popen",
                return_value=mock_proc,
            ),
            patch(
                "jobsearch_rag.adapters.session.urllib.request.urlopen",
                side_effect=OSError("connection refused"),
            ),
            patch(
                "jobsearch_rag.adapters.session.tempfile.mkdtemp",
                return_value=str(tmp_path / "cdp-profile"),
            ),
        ):
            # When / Then: CDP never becomes ready, timeout fires at ~0.5s
            with pytest.raises(TimeoutError) as exc_info:
                async with SessionManager(config) as _session:
                    pass

            # Then: the error mentions the timeout
            assert "0.5" in str(exc_info.value) or "cdp" in str(exc_info.value).lower(), (
                f"Error should reference the timeout duration. Got: {exc_info.value}"
            )

    async def test_session_skips_nonexistent_config_path_and_uses_next(
        self, tmp_path: Path
    ) -> None:
        """
        Given config browser_paths has two entries: one non-existent, one valid
        When SessionManager opens a CDP session with browser_channel="msedge"
        Then the non-existent path is skipped and the valid path is used
        """
        # Given: first path does not exist, second path does
        missing_binary = str(tmp_path / "nonexistent" / "msedge")
        valid_binary = str(tmp_path / "real" / "msedge")
        (tmp_path / "real").mkdir()
        (tmp_path / "real" / "msedge").touch()

        config = SessionConfig(  # pyright: ignore[reportCallIssue]  # Phase 3 fields
            board_name="testboard",
            headless=True,
            browser_channel="msedge",
            storage_dir=tmp_path,
            browser_paths={"msedge": [missing_binary, valid_binary]},  # pyright: ignore[reportCallIssue]  # Phase 3 field
            cdp_timeout=15.0,  # pyright: ignore[reportCallIssue]  # Phase 3 field
        )

        # Given: mock Playwright and subprocess at I/O boundaries
        mock_browser = MagicMock()
        mock_browser.contexts = []
        mock_browser.new_context = AsyncMock(
            return_value=MagicMock(
                new_page=AsyncMock(return_value=MagicMock()),
                storage_state=AsyncMock(return_value={"cookies": [], "origins": []}),
                close=AsyncMock(),
            )
        )
        mock_browser.close = AsyncMock()

        mock_pw = MagicMock()
        mock_pw.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
        mock_pw.stop = AsyncMock()
        mock_pw_cm = MagicMock()
        mock_pw_cm.start = AsyncMock(return_value=mock_pw)

        captured_popen_args: list[Any] = []

        def _capturing_popen(cmd: Any, **kwargs: Any) -> MagicMock:
            captured_popen_args.append(cmd)
            proc = MagicMock()
            proc.poll.return_value = None
            proc.wait.return_value = 0
            return proc

        with (
            patch(
                "jobsearch_rag.adapters.session.async_playwright",
                return_value=mock_pw_cm,
            ),
            patch(
                "jobsearch_rag.adapters.session.subprocess.Popen",
                side_effect=_capturing_popen,
            ),
            patch("jobsearch_rag.adapters.session.urllib.request.urlopen"),
            patch(
                "jobsearch_rag.adapters.session.tempfile.mkdtemp",
                return_value=str(tmp_path / "cdp-profile"),
            ),
        ):
            # When: SessionManager opens a CDP session
            async with SessionManager(config) as _session:
                pass

        # Then: the valid binary was used, not the missing one
        assert len(captured_popen_args) == 1, (
            f"Expected 1 Popen call, got {len(captured_popen_args)}"
        )
        binary_used = captured_popen_args[0][0]
        assert binary_used == valid_binary, (
            f"Expected Popen to use '{valid_binary}' (skip missing), got {binary_used!r}"
        )
