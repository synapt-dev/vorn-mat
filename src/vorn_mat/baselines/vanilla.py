"""Vanilla baseline runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from ..benchmarks import BenchmarkCase, score_predictions
from ..benchmarks.common import build_case_observation
from ..results import CaseObservation, RunResult
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
    *,
    on_case: Callable[[CaseObservation], None] | None = None,
) -> tuple[RunResult, tuple[PredictionTrace, ...]]:
    predictions: list[str] = []
    traces: list[PredictionTrace] = []
    observations: list[CaseObservation] = []
    for case in cases:
        prediction = generator.generate(case.prompt)
        predictions.append(prediction)
        traces.append(PredictionTrace(case_id=case.case_id, prediction=prediction))
        observation = build_case_observation(case, prediction)
        observations.append(observation)
        if on_case is not None:
            on_case(observation)
    metrics = score_predictions(plan.benchmark.name, cases, tuple(predictions))
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
        observations=tuple(observations),
    )
    return result, tuple(traces)
