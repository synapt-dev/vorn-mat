from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat import benchmark_case_from_ruler_record


def test_benchmark_case_from_ruler_record_maps_real_fields():
    case = benchmark_case_from_ruler_record(
        {
            "index": 7,
            "input": "needle prompt",
            "outputs": ["1234567"],
            "length": 4096,
        },
        dataset_id="rbiswasfc/ruler",
        dataset_config="niah_multikey_1_4k",
        split="validation",
    )

    assert case.case_id == "niah_multikey_1_4k-7"
    assert case.prompt == "needle prompt"
    assert case.expected_answer == "1234567"
    assert case.metadata == {
        "acceptable_answers": '["1234567"]',
        "dataset_id": "rbiswasfc/ruler",
        "dataset_config": "niah_multikey_1_4k",
        "split": "validation",
        "source_index": "7",
        "context_length": "4096",
    }
