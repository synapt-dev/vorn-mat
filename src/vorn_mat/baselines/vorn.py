"""Vorn-scored prompt-retention proxy for Step 1 signal collection.

This is intentionally narrower than full live KV-cache surgery. It uses the
same canonical residual-space scoring contract from the research spec, but
applies it to prompt-token retention so we can get the first real signal
without hand-waving over rotary-position bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

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


@dataclass(frozen=True)
class VornRetentionConfig:
    canonical_layer: int
    recent_token_window: int
    cache_budget_tokens: int
    always_keep_prefix_tokens: int = 1
    preserve_recent_window: bool = True


@dataclass(frozen=True)
class RetentionStats:
    original_token_count: int
    kept_token_count: int
    kept_positions: tuple[int, ...]
    dropped_positions: tuple[int, ...]


@dataclass(frozen=True)
class VornPredictionTrace:
    case_id: str
    prediction: str
    original_token_count: int
    kept_token_count: int


class VornTextGenerator(Protocol):
    def generate_with_retention(
        self,
        prompt: str,
        config: VornRetentionConfig,
    ) -> tuple[str, RetentionStats]: ...


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = _normalize(left.astype(np.float32, copy=False))
    right_norm = _normalize(right.astype(np.float32, copy=False))
    return float(np.dot(left_norm, right_norm))


def compute_vorn_direction(
    token_summaries: np.ndarray,
    *,
    recent_token_window: int,
) -> np.ndarray:
    """Estimate the current vorn from the last W token summaries."""
    if token_summaries.ndim != 2:
        raise ValueError("token_summaries must be rank-2")
    if token_summaries.shape[0] == 0:
        raise ValueError("token_summaries must be non-empty")
    if recent_token_window <= 0:
        raise ValueError("recent_token_window must be positive")

    window = min(recent_token_window, token_summaries.shape[0])
    pooled = np.mean(token_summaries[-window:], axis=0)
    return _normalize(pooled.astype(np.float32, copy=False))


def select_retained_positions(
    token_summaries: np.ndarray,
    vorn_direction: np.ndarray,
    *,
    cache_budget_tokens: int,
    always_keep_prefix_tokens: int = 1,
    preserve_recent_window: int = 0,
) -> tuple[int, ...]:
    """Select prompt token positions to retain under a fixed budget.

    Required positions are kept first:
    - prefix tokens (to avoid deleting the prompt preamble completely)
    - the recent window that produced the current vorn estimate
    The remaining slots are filled by cosine similarity against the current
    direction signal.
    """
    if token_summaries.ndim != 2:
        raise ValueError("token_summaries must be rank-2")
    token_count = token_summaries.shape[0]
    if token_count == 0:
        raise ValueError("token_summaries must be non-empty")
    if cache_budget_tokens <= 0:
        raise ValueError("cache_budget_tokens must be positive")

    required: set[int] = set(range(min(always_keep_prefix_tokens, token_count)))
    if preserve_recent_window > 0:
        recent_start = max(0, token_count - preserve_recent_window)
        required.update(range(recent_start, token_count))

    if cache_budget_tokens < len(required):
        raise ValueError(
            "cache_budget_tokens must fit the required prefix + recent-window positions"
        )
    if cache_budget_tokens >= token_count:
        return tuple(range(token_count))

    scores = [
        cosine_similarity(token_summaries[position], vorn_direction)
        for position in range(token_count)
    ]
    ranked_candidates = sorted(
        (position for position in range(token_count) if position not in required),
        key=lambda position: (scores[position], -position),
        reverse=True,
    )

    keep = set(required)
    for position in ranked_candidates:
        if len(keep) >= cache_budget_tokens:
            break
        keep.add(position)
    return tuple(sorted(keep))


def run_vorn(
    plan: ExecutionPlan,
    cases: tuple[BenchmarkCase, ...],
    generator: VornTextGenerator,
    *,
    on_case: Callable[[CaseObservation], None] | None = None,
    progress_logger: ProgressLogger | None = None,
) -> tuple[RunResult, tuple[VornPredictionTrace, ...]]:
    """Run the Step 1 proxy baseline on a case slice."""
    if plan.run.cache_budget_tokens is None:
        raise ValueError("vorn baseline requires cache_budget_tokens")

    config = VornRetentionConfig(
        canonical_layer=plan.run.canonical_layer,
        recent_token_window=plan.run.recent_token_window,
        cache_budget_tokens=plan.run.cache_budget_tokens,
    )

    n_cases = len(cases)
    if progress_logger is not None:
        progress_logger(format_dataset_loaded(n_cases))

    predictions: list[str] = []
    traces: list[VornPredictionTrace] = []
    observations: list[CaseObservation] = []
    total_original = 0
    total_kept = 0
    running_hits = 0

    for case_index, case in enumerate(cases, start=1):
        reset_case_memory_stats()
        prediction, stats = generator.generate_with_retention(case.prompt, config)
        memory_stats = capture_case_memory_stats()
        predictions.append(prediction)
        traces.append(
            VornPredictionTrace(
                case_id=case.case_id,
                prediction=prediction,
                original_token_count=stats.original_token_count,
                kept_token_count=stats.kept_token_count,
            )
        )
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
        total_original += stats.original_token_count
        total_kept += stats.kept_token_count

    metrics = score_predictions(plan.benchmark.name, cases, tuple(predictions))
    if progress_logger is not None:
        progress_logger(
            format_complete(n_cases, running_hits / n_cases if n_cases else 0.0)
        )
    mean_retention_ratio = (
        (total_kept / total_original) if total_original else 0.0
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
            "cache_budget_tokens": str(plan.run.cache_budget_tokens),
            "compression_mode": plan.run.compression_mode or "unknown",
            "mean_retention_ratio": f"{mean_retention_ratio:.4f}",
        },
        observations=tuple(observations),
    )
    return result, tuple(traces)
