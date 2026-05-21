"""Benchmark registry for Week 1 baseline reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .common import BenchmarkCase
from .niah import (
    benchmark_case_from_ruler_record as benchmark_case_from_ruler_record,
    load_niah_cases,
    load_ruler_hf_niah_slice as load_ruler_hf_niah_slice,
    score_niah_predictions,
)
from .ruler import load_ruler_cases, score_ruler_predictions


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    dataset_id: str
    metric_name: str
    split: str
    task_family: str


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "ruler": BenchmarkSpec(
        name="ruler",
        dataset_id="ruler",
        metric_name="task_accuracy",
        split="validation",
        task_family="long_context",
    ),
    "niah": BenchmarkSpec(
        name="niah",
        dataset_id="niah",
        metric_name="needle_hit_rate",
        split="validation",
        task_family="needle_in_haystack",
    ),
}


def get_benchmark(name: str) -> BenchmarkSpec:
    try:
        return BENCHMARKS[name]
    except KeyError as exc:
        raise ValueError(f"unknown benchmark: {name}") from exc


def load_cases(name: str, path: Path) -> tuple[BenchmarkCase, ...]:
    if name == "ruler":
        return load_ruler_cases(path)
    if name == "niah":
        return load_niah_cases(path)
    raise ValueError(f"unknown benchmark: {name}")


def score_predictions(
    name: str,
    cases: tuple[BenchmarkCase, ...],
    predictions: tuple[str, ...],
) -> dict[str, float]:
    if name == "ruler":
        return score_ruler_predictions(cases, predictions)
    if name == "niah":
        return score_niah_predictions(cases, predictions)
    raise ValueError(f"unknown benchmark: {name}")


__all__ = [
    "BENCHMARKS",
    "BenchmarkCase",
    "BenchmarkSpec",
    "benchmark_case_from_ruler_record",
    "get_benchmark",
    "load_cases",
    "load_niah_cases",
    "load_ruler_cases",
    "load_ruler_hf_niah_slice",
    "score_niah_predictions",
    "score_predictions",
    "score_ruler_predictions",
]
