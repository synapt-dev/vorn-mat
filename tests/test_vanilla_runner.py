from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat import (
    append_result,
    build_execution_plans,
    build_week1_run_matrix,
    load_results,
    run_vanilla,
)
from vorn_mat.benchmarks.common import BenchmarkCase


class FakeGenerator:
    def __init__(self, answers: dict[str, str]):
        self.answers = answers

    def generate(self, prompt: str) -> str:
        return self.answers[prompt]


def test_vanilla_runner_executes_plan_and_returns_result():
    plan = build_execution_plans(build_week1_run_matrix())[0]  # ruler + vanilla
    cases = (
        BenchmarkCase("case-1", "prompt-1", "Paris", {}),
        BenchmarkCase("case-2", "prompt-2", "4", {}),
    )
    generator = FakeGenerator({"prompt-1": "Paris", "prompt-2": "5"})

    result, traces = run_vanilla(plan, cases, generator)

    assert result.run_id == "week1-ruler-vanilla"
    assert result.benchmark == "ruler"
    assert result.baseline == "vanilla"
    assert result.metrics == {"task_accuracy": 0.5}
    assert result.metadata["model"] == plan.run.model
    assert result.metadata["canonical_layer"] == "16"
    assert [observation.fixture_id for observation in result.observations] == [
        "case-1",
        "case-2",
    ]
    assert [observation.correct for observation in result.observations] == [True, False]
    assert [trace.case_id for trace in traces] == ["case-1", "case-2"]
    assert [trace.prediction for trace in traces] == ["Paris", "5"]


def test_vanilla_runner_round_trips_through_result_sink(tmp_path: Path):
    plan = build_execution_plans(build_week1_run_matrix())[0]
    cases = (BenchmarkCase("case-1", "prompt-1", "Paris", {}),)
    generator = FakeGenerator({"prompt-1": "Paris"})
    result, _traces = run_vanilla(plan, cases, generator)

    path = tmp_path / "vanilla-results.jsonl"
    append_result(path, result)

    assert load_results(path) == [result]
