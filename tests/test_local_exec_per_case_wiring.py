"""run_local_*_smoke wrappers wire per-case persistence to .observations.jsonl.

The wrapper layer is what the entrypoint scripts call. It must:
- derive the per-case JSONL path from output_path via observations_path()
- pass an on_case callback into run_vanilla / run_vorn / run_live_eviction
- so that each completed case lands on disk BEFORE the next case runs

These tests monkeypatch the runner to count callback invocations and verify
the per-case ledger lands at the expected path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import vorn_mat.local_exec as local_exec
from vorn_mat.benchmarks.common import BenchmarkCase
from vorn_mat.results import (
    CaseObservation,
    RunResult,
    load_observations,
    observations_path,
)


def _stub_runner_that_fires_on_case(captured_observations: list[CaseObservation]):
    """Returns a stub run_vanilla-shaped function that fires on_case per fixture."""

    def _stub(plan, cases, generator, *, on_case=None, **kwargs):
        for case in cases:
            obs = CaseObservation(
                fixture_id=case.case_id,
                correct=True,
                prediction="stub",
            )
            captured_observations.append(obs)
            if on_case is not None:
                on_case(obs)
        result = RunResult(
            run_id="week1-niah-vanilla",
            benchmark="niah",
            baseline="vanilla",
            metrics={"needle_hit_rate": 1.0},
            metadata={
                "model": "stub-model",
                "gpu": "A100-80GB",
                "canonical_layer": "16",
                "recent_token_window": "16",
                "eviction_unit": "token_position",
            },
            observations=tuple(captured_observations),
        )
        return result, ()

    return _stub


def test_run_local_vanilla_smoke_writes_per_case_ledger_to_observations_path(
    tmp_path: Path, monkeypatch
):
    cases = (
        BenchmarkCase("case-1", "prompt-1", "Paris", {}),
        BenchmarkCase("case-2", "prompt-2", "5", {}),
        BenchmarkCase("case-3", "prompt-3", "Tokyo", {}),
    )
    monkeypatch.setattr(local_exec, "load_cases", lambda benchmark, path: cases)
    captured: list[CaseObservation] = []
    monkeypatch.setattr(
        local_exec, "run_vanilla", _stub_runner_that_fires_on_case(captured)
    )

    class _NoopGenerator:
        def generate(self, prompt: str) -> str:
            return "stub"

    output_path = tmp_path / "results" / "smoke-cell.jsonl"
    result, _traces = local_exec.run_local_vanilla_smoke(
        benchmark="niah",
        dataset_path=tmp_path / "ignored.jsonl",
        output_path=output_path,
        case_limit=3,
        generator=_NoopGenerator(),
    )

    ledger = observations_path(output_path)
    assert ledger.exists()
    loaded = load_observations(ledger)
    assert [obs.fixture_id for obs in loaded] == ["case-1", "case-2", "case-3"]
    # Summary envelope still lands at output_path (backward compat)
    assert output_path.exists()
    assert result.run_id == "week1-niah-vanilla"
