from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat.baselines.live_eviction import LiveEvictionStats, run_live_eviction
from vorn_mat.benchmarks.niah import (
    EVICTION_ANTAGONIST_FIXTURE,
    EVICTION_ANTAGONIST_SUITE_ID,
    load_eviction_antagonist_cases,
)
from vorn_mat.local_exec import select_live_eviction_plan


class FakeLiveGenerator:
    def __init__(self, answers: dict[str, str], stats: LiveEvictionStats):
        self.answers = answers
        self.stats = stats

    def generate_with_live_eviction(self, prompt: str, config):
        return self.answers[prompt], self.stats


def test_load_eviction_antagonist_cases_covers_required_edge_kinds():
    cases = load_eviction_antagonist_cases()

    assert EVICTION_ANTAGONIST_FIXTURE.name == "niah_eviction_antagonists.jsonl"
    assert len(cases) == 4
    assert {case.metadata["edge_kind"] for case in cases} == {
        "needle_at_start",
        "needle_near_end",
        "shorter_than_budget",
        "multi_question",
    }
    assert {case.metadata["suite_id"] for case in cases} == {
        EVICTION_ANTAGONIST_SUITE_ID
    }


def test_packaged_antagonist_cases_preserve_kind_metadata_and_order():
    cases = load_eviction_antagonist_cases()

    assert [case.case_id for case in cases] == [
        "eviction-antagonist-needle-start",
        "eviction-antagonist-needle-near-end",
        "eviction-antagonist-short-context",
        "eviction-antagonist-multi-question",
    ]
    assert {case.metadata["kind"] for case in cases} == {"eviction_antagonist"}


def test_packaged_antagonist_suite_composes_with_existing_live_eviction_plan():
    plan = select_live_eviction_plan()

    assert plan.run.run_id == "step2-niah-vorn-live"
    assert plan.run.cache_budget_tokens == 256
    assert plan.baseline.name == "vorn_live"


def test_run_live_eviction_antagonist_suite_propagates_suite_metadata():
    cases = load_eviction_antagonist_cases()
    answers = {case.prompt: case.expected_answer for case in cases}
    result, traces = run_live_eviction(
        select_live_eviction_plan(),
        cases,
        generator=FakeLiveGenerator(
            answers,
            LiveEvictionStats(
                prompt_token_count=48,
                generated_token_count=2,
                mean_kept_token_count=20.0,
                final_kept_token_count=21,
                eviction_steps=2,
                mean_retention_ratio=20.0 / 48.0,
                retention_policy="vorn",
                summary_contract="canonical_hidden_state_float32_per_token_from_layer_L_star",
                summary_fingerprint="abc123",
            ),
        ),
    )

    assert result.run_id == "step2-niah-vorn-live"
    assert result.metrics == {"needle_hit_rate": 1.0}
    assert result.metadata["suite_id"] == EVICTION_ANTAGONIST_SUITE_ID
    assert result.metadata["case_count"] == "4"
    assert result.metadata["edge_kinds"] == (
        "multi_question,needle_at_start,needle_near_end,shorter_than_budget"
    )
    assert len(traces) == 4


def test_run_live_eviction_antagonist_suite_exposes_edge_kind_per_trace():
    cases = load_eviction_antagonist_cases()
    answers = {case.prompt: case.expected_answer for case in cases}
    _result, traces = run_live_eviction(
        select_live_eviction_plan(),
        cases,
        generator=FakeLiveGenerator(
            answers,
            LiveEvictionStats(
                prompt_token_count=48,
                generated_token_count=2,
                mean_kept_token_count=20.0,
                final_kept_token_count=21,
                eviction_steps=2,
                mean_retention_ratio=20.0 / 48.0,
                retention_policy="vorn",
                summary_contract="canonical_hidden_state_float32_per_token_from_layer_L_star",
                summary_fingerprint="abc123",
            ),
        ),
    )

    assert {trace.edge_kind for trace in traces} == {
        "needle_at_start",
        "needle_near_end",
        "shorter_than_budget",
        "multi_question",
    }
