"""Shared benchmark data shapes and utilities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..results import CaseObservation


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    prompt: str
    expected_answer: str
    metadata: dict[str, str]


def load_cases_from_jsonl(path: Path) -> tuple[BenchmarkCase, ...]:
    cases: list[BenchmarkCase] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        cases.append(
            BenchmarkCase(
                case_id=payload["case_id"],
                prompt=payload["prompt"],
                expected_answer=payload["expected_answer"],
                metadata={str(k): str(v) for k, v in payload.get("metadata", {}).items()},
            )
        )
    return tuple(cases)


def normalize_answer(text: str) -> str:
    return " ".join(text.strip().lower().split())


def exact_match_rate(
    cases: tuple[BenchmarkCase, ...],
    predictions: tuple[str, ...],
) -> float:
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must have the same length")
    if not cases:
        return 0.0

    hits = 0
    for case, prediction in zip(cases, predictions, strict=True):
        if is_prediction_correct(case, prediction):
            hits += 1
    return hits / len(cases)


def build_case_observations(
    cases: tuple[BenchmarkCase, ...],
    predictions: tuple[str, ...],
) -> tuple[CaseObservation, ...]:
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must have the same length")
    return tuple(
        CaseObservation(
            fixture_id=case.case_id,
            correct=is_prediction_correct(case, prediction),
            prediction=prediction,
        )
        for case, prediction in zip(cases, predictions, strict=True)
    )


def is_prediction_correct(case: BenchmarkCase, prediction: str) -> bool:
    acceptable_answers = _acceptable_answers(case)
    normalized_prediction = normalize_answer(prediction)
    if normalized_prediction in acceptable_answers:
        return True
    return _matches_ruler_niah_numeric_answer(case, prediction, acceptable_answers)


def _acceptable_answers(case: BenchmarkCase) -> set[str]:
    payload = case.metadata.get("acceptable_answers")
    if not payload:
        return {normalize_answer(case.expected_answer)}

    try:
        raw = json.loads(payload)
    except json.JSONDecodeError:
        return {normalize_answer(case.expected_answer)}

    if not isinstance(raw, list) or not raw:
        return {normalize_answer(case.expected_answer)}

    return {normalize_answer(str(item)) for item in raw}


def _matches_ruler_niah_numeric_answer(
    case: BenchmarkCase,
    prediction: str,
    acceptable_answers: set[str],
) -> bool:
    if case.metadata.get("dataset_id") != "rbiswasfc/ruler":
        return False

    dataset_config = case.metadata.get("dataset_config", "")
    if not dataset_config.startswith("niah_"):
        return False

    if not acceptable_answers or not all(answer.isdigit() for answer in acceptable_answers):
        return False

    for answer in acceptable_answers:
        if re.search(rf"(?<!\d){re.escape(answer)}(?!\d)", prediction):
            return True
    return False
