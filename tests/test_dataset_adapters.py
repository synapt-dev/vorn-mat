from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat import load_cases, score_predictions
from vorn_mat.benchmarks.common import BenchmarkCase
from vorn_mat.benchmarks.niah import benchmark_case_from_ruler_record


def _write_cases(path: Path) -> None:
    rows = [
        {
            "case_id": "case-1",
            "prompt": "What is the capital of France?",
            "expected_answer": "Paris",
            "metadata": {"topic": "geography"},
        },
        {
            "case_id": "case-2",
            "prompt": "What is 2 + 2?",
            "expected_answer": "4",
            "metadata": {"topic": "arithmetic"},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_ruler_adapter_loads_cases_and_scores_accuracy(tmp_path: Path):
    path = tmp_path / "ruler.jsonl"
    _write_cases(path)

    cases = load_cases("ruler", path)
    metrics = score_predictions("ruler", cases, ("Paris", "5"))

    assert [case.case_id for case in cases] == ["case-1", "case-2"]
    assert cases[0].metadata == {"topic": "geography"}
    assert metrics == {"task_accuracy": 0.5}


def test_niah_adapter_loads_cases_and_scores_hit_rate(tmp_path: Path):
    path = tmp_path / "niah.jsonl"
    _write_cases(path)

    cases = load_cases("niah", path)
    metrics = score_predictions("niah", cases, ("paris", "4"))

    assert [case.case_id for case in cases] == ["case-1", "case-2"]
    assert metrics == {"needle_hit_rate": 1.0}


def test_niah_scoring_accepts_any_configured_answer():
    cases = (
        BenchmarkCase(
            case_id="vt-1",
            prompt="prompt",
            expected_answer="FITJT",
            metadata={"acceptable_answers": '["FITJT", "VGCAO", "ZJQUQ"]'},
        ),
    )

    metrics = score_predictions("niah", cases, ("ZJQUQ",))

    assert metrics == {"needle_hit_rate": 1.0}


def test_niah_scoring_accepts_numeric_answer_embedded_in_sentence():
    cases = (
        BenchmarkCase(
            case_id="niah-1",
            prompt="prompt",
            expected_answer="9375710",
            metadata={
                "acceptable_answers": '["9375710"]',
                "dataset_id": "rbiswasfc/ruler",
                "dataset_config": "niah_multikey_1_4k",
            },
        ),
    )

    metrics = score_predictions(
        "niah",
        cases,
        ("The special magic number mentioned in the text is **9375710**.",),
    )

    assert metrics == {"needle_hit_rate": 1.0}


def test_niah_scoring_does_not_widen_non_numeric_answers_to_substring_match():
    cases = (
        BenchmarkCase(
            case_id="niah-2",
            prompt="prompt",
            expected_answer="forest",
            metadata={
                "acceptable_answers": '["forest"]',
                "dataset_id": "rbiswasfc/ruler",
                "dataset_config": "niah_multikey_1_4k",
            },
        ),
    )

    metrics = score_predictions(
        "niah",
        cases,
        ("The answer is forest.",),
    )

    assert metrics == {"needle_hit_rate": 0.0}


def test_ruler_case_builder_preserves_all_outputs():
    case = benchmark_case_from_ruler_record(
        {
            "index": 7,
            "input": "prompt",
            "outputs": ["appliance", "meter", "forest"],
            "length": 4096,
        },
        dataset_id="rbiswasfc/ruler",
        dataset_config="cwe_4k",
        split="validation",
    )

    assert case.expected_answer == "appliance"
    assert case.metadata["acceptable_answers"] == '["appliance", "meter", "forest"]'
