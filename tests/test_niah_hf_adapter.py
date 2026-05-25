from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat import benchmark_case_from_ruler_record, load_ruler_hf_niah_slice


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


def test_load_ruler_hf_niah_slice_uses_offset_split(monkeypatch):
    calls = []

    def fake_load_dataset(dataset_id, dataset_config, split):
        calls.append((dataset_id, dataset_config, split))
        return (
            {
                "index": 34,
                "input": "needle prompt",
                "outputs": ["9375710"],
                "length": 4096,
            },
        )

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    cases = load_ruler_hf_niah_slice(
        "niah_multikey_1_4k",
        split="validation",
        case_limit=16,
        case_offset_start=34,
    )

    assert calls == [("rbiswasfc/ruler", "niah_multikey_1_4k", "validation[34:50]")]
    assert cases[0].case_id == "niah_multikey_1_4k-34"
    assert cases[0].metadata["split"] == "validation[34:50]"


def test_load_ruler_hf_niah_slice_keeps_default_prefix_slice(monkeypatch):
    calls = []

    def fake_load_dataset(dataset_id, dataset_config, split):
        calls.append((dataset_id, dataset_config, split))
        return (
            {
                "index": 0,
                "input": "needle prompt",
                "outputs": ["9375710"],
                "length": 4096,
            },
        )

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    load_ruler_hf_niah_slice(
        "niah_multikey_1_4k",
        split="validation",
        case_limit=50,
    )

    assert calls == [("rbiswasfc/ruler", "niah_multikey_1_4k", "validation[:50]")]
