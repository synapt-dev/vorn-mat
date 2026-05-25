from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import vorn_mat.baselines.live_eviction as live_module
import vorn_mat.baselines.vanilla as vanilla_module
from vorn_mat import build_execution_plans, build_week1_run_matrix, run_vanilla
from vorn_mat.baselines.live_eviction import (
    LiveEvictionStats,
    SEMANTIC_SUMMARY_CONTRACT,
    run_live_eviction,
)
from vorn_mat.benchmarks.common import BenchmarkCase
from vorn_mat.local_exec import select_live_eviction_plan
from vorn_mat.memory import (
    CaseMemoryStats,
    capture_case_memory_stats,
    reset_case_memory_stats,
)
from vorn_mat.results import CaseObservation, append_observation, load_observations


def test_capture_case_memory_stats_reads_cuda_peak_counters(monkeypatch):
    calls: list[str] = []

    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        reset_peak_memory_stats=lambda: calls.append("reset"),
        synchronize=lambda: calls.append("sync"),
        max_memory_allocated=lambda: 256 * 1024 * 1024,
        max_memory_reserved=lambda: 512 * 1024 * 1024,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))

    reset_case_memory_stats()
    stats = capture_case_memory_stats()

    assert calls == ["reset", "sync"]
    assert stats == CaseMemoryStats(
        peak_memory_allocated_mb=256.0,
        peak_memory_reserved_mb=512.0,
    )


def test_case_observation_memory_fields_round_trip_jsonl(tmp_path: Path):
    path = tmp_path / "cell.observations.jsonl"
    observation = CaseObservation(
        fixture_id="case-1",
        correct=True,
        prediction="9375710",
        peak_memory_allocated_mb=1234.5,
        peak_memory_reserved_mb=2345.5,
    )

    append_observation(path, observation)

    assert load_observations(path) == (observation,)


def test_run_vanilla_attaches_per_case_memory_to_observations(monkeypatch):
    plan = build_execution_plans(build_week1_run_matrix())[0]
    cases = (
        BenchmarkCase("case-1", "prompt-1", "Paris", {}),
        BenchmarkCase("case-2", "prompt-2", "5", {}),
    )

    class FakeGenerator:
        def generate(self, prompt: str) -> str:
            return {"prompt-1": "Paris", "prompt-2": "5"}[prompt]

    captures = iter(
        (
            CaseMemoryStats(peak_memory_allocated_mb=101.0, peak_memory_reserved_mb=201.0),
            CaseMemoryStats(peak_memory_allocated_mb=102.0, peak_memory_reserved_mb=202.0),
        )
    )
    reset_calls = 0

    def fake_reset() -> None:
        nonlocal reset_calls
        reset_calls += 1

    monkeypatch.setattr(vanilla_module, "reset_case_memory_stats", fake_reset)
    monkeypatch.setattr(vanilla_module, "capture_case_memory_stats", lambda: next(captures))

    result, _traces = run_vanilla(plan, cases, FakeGenerator())

    assert reset_calls == 2
    assert [obs.peak_memory_allocated_mb for obs in result.observations] == [101.0, 102.0]
    assert [obs.peak_memory_reserved_mb for obs in result.observations] == [201.0, 202.0]


def test_run_live_eviction_carries_memory_to_callback_and_final_report(monkeypatch):
    plan = select_live_eviction_plan(cache_budget_tokens=256, retention_policy="sentence_tova")
    cases = (
        BenchmarkCase("case-1", "prompt-1", "Paris", {}),
        BenchmarkCase("case-2", "prompt-2", "5", {}),
    )

    class FakeGenerator:
        def generate_with_live_eviction(self, prompt: str, config, **kwargs):
            return (
                {"prompt-1": "Paris", "prompt-2": "5"}[prompt],
                LiveEvictionStats(
                    prompt_token_count=100,
                    generated_token_count=8,
                    mean_kept_token_count=64.0,
                    final_kept_token_count=64,
                    eviction_steps=2,
                    mean_retention_ratio=0.64,
                    retention_policy="sentence_tova",
                    summary_contract=SEMANTIC_SUMMARY_CONTRACT,
                    summary_fingerprint="abc",
                ),
            )

    captures = iter(
        (
            CaseMemoryStats(peak_memory_allocated_mb=301.0, peak_memory_reserved_mb=401.0),
            CaseMemoryStats(peak_memory_allocated_mb=302.0, peak_memory_reserved_mb=402.0),
        )
    )
    monkeypatch.setattr(live_module, "reset_case_memory_stats", lambda: None)
    monkeypatch.setattr(live_module, "capture_case_memory_stats", lambda: next(captures))

    persisted: list[CaseObservation] = []
    result, _traces = run_live_eviction(
        plan,
        cases,
        FakeGenerator(),
        on_case=persisted.append,
    )

    assert [obs.peak_memory_allocated_mb for obs in persisted] == [301.0, 302.0]
    assert [obs.peak_memory_reserved_mb for obs in persisted] == [401.0, 402.0]
    assert result.observations == tuple(persisted)
