from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat.benchmarks.common import (
    BenchmarkCase,
    build_case_observation,
    exact_match_rate,
    is_prediction_correct,
    score_prediction_text,
)


def _niah_case(expected_answer: str = "1234567") -> BenchmarkCase:
    return BenchmarkCase(
        case_id="niah-case-1",
        prompt="prompt",
        expected_answer=expected_answer,
        metadata={
            "acceptable_answers": f'["{expected_answer}"]',
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "niah_multikey_1_4k",
        },
    )


def test_score_prediction_text_strips_closed_think_blocks() -> None:
    prediction = "<think>I should search for 9999999.</think>\n\n1234567"

    assert score_prediction_text(prediction) == "1234567"


def test_score_prediction_text_strips_multiple_think_blocks_case_insensitively() -> None:
    prediction = (
        "<THINK>first reasoning</think>\n"
        "intermediate "
        "<think>second reasoning</THINK>\n"
        "1234567"
    )

    assert score_prediction_text(prediction) == "intermediate \n1234567"


def test_score_prediction_text_drops_unclosed_think_block_conservatively() -> None:
    prediction = "<think>the answer might be 1234567"

    assert score_prediction_text(prediction) == ""


def test_qwen_think_block_is_removed_for_correctness_but_raw_prediction_is_preserved() -> None:
    case = _niah_case()
    prediction = "<think>Reasoning mentions a decoy 7654321.</think>\n\n1234567"

    observation = build_case_observation(case, prediction)

    assert is_prediction_correct(case, prediction) is True
    assert exact_match_rate((case,), (prediction,)) == 1.0
    assert observation.correct is True
    assert observation.prediction == prediction
    assert observation.scored_prediction == "1234567"


def test_unclosed_think_block_is_not_scored_even_if_it_mentions_answer() -> None:
    case = _niah_case()
    prediction = "<think>The answer is probably 1234567"

    observation = build_case_observation(case, prediction)

    assert is_prediction_correct(case, prediction) is False
    assert exact_match_rate((case,), (prediction,)) == 0.0
    assert observation.prediction == prediction
    assert observation.scored_prediction == ""
