"""NIAH adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import BenchmarkCase, exact_match_rate, load_cases_from_jsonl


EVICTION_ANTAGONIST_SUITE_ID = "niah_eviction_antagonists_v1"
EVICTION_ANTAGONIST_FIXTURE = (
    Path(__file__).resolve().parents[3] / "examples" / "niah_eviction_antagonists.jsonl"
)


def load_niah_cases(path: Path) -> tuple[BenchmarkCase, ...]:
    return load_cases_from_jsonl(path)


def load_eviction_antagonist_cases() -> tuple[BenchmarkCase, ...]:
    return load_cases_from_jsonl(EVICTION_ANTAGONIST_FIXTURE)


def score_niah_predictions(
    cases: tuple[BenchmarkCase, ...],
    predictions: tuple[str, ...],
) -> dict[str, float]:
    return {"needle_hit_rate": exact_match_rate(cases, predictions)}


def benchmark_case_from_ruler_record(
    record: dict[str, Any],
    *,
    dataset_id: str,
    dataset_config: str,
    split: str,
) -> BenchmarkCase:
    outputs = record.get("outputs") or []
    if not outputs:
        raise ValueError("RULER NIAH record is missing outputs")
    return BenchmarkCase(
        case_id=f"{dataset_config}-{record['index']}",
        prompt=str(record["input"]),
        expected_answer=str(outputs[0]),
        metadata={
            "acceptable_answers": json.dumps([str(item) for item in outputs]),
            "dataset_id": dataset_id,
            "dataset_config": dataset_config,
            "split": split,
            "source_index": str(record["index"]),
            "context_length": str(record.get("length", "")),
        },
    )


def load_ruler_hf_niah_slice(
    dataset_config: str,
    *,
    dataset_id: str = "rbiswasfc/ruler",
    split: str = "validation",
    case_limit: int = 50,
    case_offset_start: int = 0,
) -> tuple[BenchmarkCase, ...]:
    """Load a real NIAH slice from a published RULER dataset mirror."""
    if case_limit <= 0:
        raise ValueError("case_limit must be positive")
    if case_offset_start < 0:
        raise ValueError("case_offset_start must be non-negative")

    from datasets import load_dataset

    split_slice = (
        f"{split}[:{case_limit}]"
        if case_offset_start == 0
        else f"{split}[{case_offset_start}:{case_offset_start + case_limit}]"
    )
    dataset = load_dataset(
        dataset_id,
        dataset_config,
        split=split_slice,
    )
    return tuple(
        benchmark_case_from_ruler_record(
            dict(row),
            dataset_id=dataset_id,
            dataset_config=dataset_config,
            split=split_slice,
        )
        for row in dataset
    )
