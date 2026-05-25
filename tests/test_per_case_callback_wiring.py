"""Per-case on_case callback wiring across baseline runners.

The persistence invariant (Layne directive 2026-05-23: store every completed
case before the next case runs) requires that run_vanilla / run_vorn /
run_live_eviction fire an on_case callback once per completed case, in order,
with the CaseObservation for that case. These tests cover that contract.

The mid-run-kill survival test simulates a KeyboardInterrupt at case N and
asserts that the on_case callback fired exactly N times before propagating
the interrupt: that is the property the per-case JSONL persistence layer
exploits to preserve completed work across mid-run kills.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat import (
    build_execution_plans,
    build_step1_run_matrix,
    build_week1_run_matrix,
    run_vanilla,
)
from vorn_mat.baselines.live_eviction import (
    LiveEvictionStats,
    LiveEvictionTextGenerator,
    SEMANTIC_SUMMARY_CONTRACT,
    run_live_eviction,
)
from vorn_mat.baselines.vanilla import TextGenerator
from vorn_mat.baselines.vorn import (
    RetentionStats,
    VornTextGenerator,
    run_vorn,
)
from vorn_mat.benchmarks.common import BenchmarkCase
from vorn_mat.local_exec import select_live_eviction_plan
from vorn_mat.results import CaseObservation


class _FakeVanillaGenerator(TextGenerator):
    def __init__(self, answers: dict[str, str], on_generate: Callable[[str], None] | None = None):
        self.answers = answers
        self.on_generate = on_generate

    def generate(self, prompt: str) -> str:
        if self.on_generate is not None:
            self.on_generate(prompt)
        return self.answers[prompt]


class _FakeVornGenerator(VornTextGenerator):
    def __init__(self, answers: dict[str, str], stats: RetentionStats):
        self.answers = answers
        self.stats = stats

    def generate_with_retention(self, prompt: str, config):
        return self.answers[prompt], self.stats


class _FakeLiveGenerator(LiveEvictionTextGenerator):
    def __init__(self, answers: dict[str, str], stats: LiveEvictionStats):
        self.answers = answers
        self.stats = stats

    def generate_with_live_eviction(self, prompt: str, config, **kwargs):
        return self.answers[prompt], self.stats


def _three_cases() -> tuple[BenchmarkCase, ...]:
    return (
        BenchmarkCase("case-1", "prompt-1", "Paris", {}),
        BenchmarkCase("case-2", "prompt-2", "5", {}),
        BenchmarkCase("case-3", "prompt-3", "Tokyo", {}),
    )


def test_run_vanilla_fires_on_case_callback_per_case_in_order():
    plan = build_execution_plans(build_week1_run_matrix())[0]
    cases = _three_cases()
    generator = _FakeVanillaGenerator(
        {"prompt-1": "Paris", "prompt-2": "5", "prompt-3": "Tokyo"}
    )

    captured: list[CaseObservation] = []
    run_vanilla(plan, cases, generator, on_case=captured.append)

    assert [obs.fixture_id for obs in captured] == ["case-1", "case-2", "case-3"]
    assert [obs.correct for obs in captured] == [True, True, True]
    assert [obs.prediction for obs in captured] == ["Paris", "5", "Tokyo"]


def test_run_vanilla_on_case_callback_fires_before_next_case_runs():
    """The persistence invariant: case N is captured BEFORE case N+1 runs.

    This is the property Layne's directive turns on: 'store each and every
    result before moving to next one.' We assert it by interleaving the
    on_case callback with the generator's per-prompt hook and checking that
    on_case for case N has fired before generate() is called for case N+1.
    """
    plan = build_execution_plans(build_week1_run_matrix())[0]
    cases = _three_cases()

    timeline: list[str] = []
    generator = _FakeVanillaGenerator(
        {"prompt-1": "Paris", "prompt-2": "5", "prompt-3": "Tokyo"},
        on_generate=lambda prompt: timeline.append(f"generate:{prompt}"),
    )

    def _persist(observation: CaseObservation) -> None:
        timeline.append(f"persist:{observation.fixture_id}")

    run_vanilla(plan, cases, generator, on_case=_persist)

    assert timeline == [
        "generate:prompt-1",
        "persist:case-1",
        "generate:prompt-2",
        "persist:case-2",
        "generate:prompt-3",
        "persist:case-3",
    ]


def test_run_vanilla_mid_run_interrupt_preserves_completed_observations():
    """Mid-run kill at case 2 of 3 leaves exactly 2 observations captured.

    KeyboardInterrupt is the simulation surface for any mid-run termination
    (Modal container kill, server-side OOM, manual kill, network blip).
    """
    plan = build_execution_plans(build_week1_run_matrix())[0]
    cases = _three_cases()

    captured: list[CaseObservation] = []

    def _persist(observation: CaseObservation) -> None:
        captured.append(observation)
        if observation.fixture_id == "case-2":
            raise KeyboardInterrupt("simulated mid-run kill")

    generator = _FakeVanillaGenerator(
        {"prompt-1": "Paris", "prompt-2": "5", "prompt-3": "Tokyo"}
    )

    with pytest.raises(KeyboardInterrupt):
        run_vanilla(plan, cases, generator, on_case=_persist)

    assert [obs.fixture_id for obs in captured] == ["case-1", "case-2"]


def test_run_vorn_fires_on_case_callback_per_case_in_order():
    _, vorn_plan = build_execution_plans(build_step1_run_matrix())
    cases = _three_cases()
    generator = _FakeVornGenerator(
        {"prompt-1": "Paris", "prompt-2": "5", "prompt-3": "Tokyo"},
        stats=RetentionStats(
            original_token_count=100,
            kept_token_count=50,
            kept_positions=(0, 1),
            dropped_positions=(2, 3),
        ),
    )

    captured: list[CaseObservation] = []
    run_vorn(vorn_plan, cases, generator, on_case=captured.append)

    assert [obs.fixture_id for obs in captured] == ["case-1", "case-2", "case-3"]
    assert [obs.prediction for obs in captured] == ["Paris", "5", "Tokyo"]


def test_run_live_eviction_fires_on_case_callback_per_case_in_order():
    plan = select_live_eviction_plan(cache_budget_tokens=256, retention_policy="vorn")
    cases = _three_cases()
    generator = _FakeLiveGenerator(
        {"prompt-1": "Paris", "prompt-2": "5", "prompt-3": "Tokyo"},
        stats=LiveEvictionStats(
            prompt_token_count=100,
            generated_token_count=8,
            mean_kept_token_count=64.0,
            final_kept_token_count=64,
            eviction_steps=2,
            mean_retention_ratio=0.64,
            retention_policy="vorn",
            summary_contract=SEMANTIC_SUMMARY_CONTRACT,
            summary_fingerprint="abc",
        ),
    )

    captured: list[CaseObservation] = []
    run_live_eviction(plan, cases, generator, on_case=captured.append)

    assert [obs.fixture_id for obs in captured] == ["case-1", "case-2", "case-3"]
    assert [obs.prediction for obs in captured] == ["Paris", "5", "Tokyo"]
