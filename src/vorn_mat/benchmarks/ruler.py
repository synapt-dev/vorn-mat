"""RULER adapter."""

from __future__ import annotations

from pathlib import Path

from .common import BenchmarkCase, exact_match_rate, load_cases_from_jsonl


def load_ruler_cases(path: Path) -> tuple[BenchmarkCase, ...]:
    return load_cases_from_jsonl(path)


def score_ruler_predictions(
    cases: tuple[BenchmarkCase, ...],
    predictions: tuple[str, ...],
) -> dict[str, float]:
    return {"task_accuracy": exact_match_rate(cases, predictions)}
