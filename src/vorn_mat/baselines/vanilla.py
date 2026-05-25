"""Vanilla baseline runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from ..benchmarks import BenchmarkCase, score_predictions
from ..benchmarks.common import build_case_observation
from ..memory import capture_case_memory_stats, reset_case_memory_stats
from ..progress import (
    ProgressLogger,
    format_case_progress,
    format_complete,
    format_dataset_loaded,
)
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
    progress_logger: ProgressLogger | None = None,
) -> tuple[RunResult, tuple[PredictionTrace, ...]]:
    n_cases = len(cases)
    if progress_logger is not None:
        progress_logger(format_dataset_loaded(n_cases))
    predictions: list[str] = []
    traces: list[PredictionTrace] = []
    observations: list[CaseObservation] = []
    running_hits = 0
    for case_index, case in enumerate(cases, start=1):
        reset_case_memory_stats()
        prediction = generator.generate(case.prompt)
        memory_stats = capture_case_memory_stats()
        predictions.append(prediction)
        traces.append(PredictionTrace(case_id=case.case_id, prediction=prediction))
        observation = build_case_observation(
            case,
            prediction,
            peak_memory_allocated_mb=memory_stats.peak_memory_allocated_mb,
            peak_memory_reserved_mb=memory_stats.peak_memory_reserved_mb,
        )
        observations.append(observation)
        if on_case is not None:
            on_case(observation)
        if observation.correct:
            running_hits += 1
        if progress_logger is not None:
            progress_logger(
                format_case_progress(
                    case_index,
                    n_cases,
                    observation.correct,
                    running_hits / case_index,
                )
            )
    metrics = score_predictions(plan.benchmark.name, cases, tuple(predictions))
    if progress_logger is not None:
        progress_logger(
            format_complete(n_cases, running_hits / n_cases if n_cases else 0.0)
        )
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
