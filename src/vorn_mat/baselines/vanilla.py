"""Vanilla baseline runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..benchmarks import BenchmarkCase, score_predictions
from ..benchmarks.common import build_case_observations
from ..results import RunResult
from ..runner import ExecutionPlan


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class PredictionTrace:
    case_id: str
    prediction: str


def run_vanilla(
    plan: ExecutionPlan,
    cases: tuple[BenchmarkCase, ...],
    generator: TextGenerator,
) -> tuple[RunResult, tuple[PredictionTrace, ...]]:
    predictions = tuple(generator.generate(case.prompt) for case in cases)
    traces = tuple(
        PredictionTrace(case_id=case.case_id, prediction=prediction)
        for case, prediction in zip(cases, predictions, strict=True)
    )
    metrics = score_predictions(plan.benchmark.name, cases, predictions)
    observations = build_case_observations(cases, predictions)
    result = RunResult(
        run_id=plan.run.run_id,
        benchmark=plan.benchmark.name,
        baseline=plan.baseline.name,
        metrics=metrics,
        metadata={
            "model": plan.run.model,
            "gpu": plan.run.gpu,
            "canonical_layer": str(plan.run.canonical_layer),
            "recent_token_window": str(plan.run.recent_token_window),
            "eviction_unit": plan.run.eviction_unit,
        },
        observations=observations,
    )
    return result, traces
