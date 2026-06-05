from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat import (
    PASSAGE_RETRIEVAL_EN_MAX_NEW_TOKENS,
    PASSAGE_RETRIEVAL_EN_PROMPT,
    PASSAGE_RETRIEVAL_EN_EXPANDED_COMPARISON_CELLS,
    PASSAGE_RETRIEVAL_EN_PREREGISTERED_CELLS,
    build_passage_retrieval_en_cell_specs,
    build_passage_retrieval_en_expanded_comparison_specs,
    get_benchmark,
    load_longbench_passage_retrieval_en_slice,
    official_retrieval_score,
    render_passage_retrieval_en_prompt,
    score_passage_retrieval_prediction,
    score_predictions,
)
from vorn_mat.benchmarks.common import build_case_observation
from vorn_mat.benchmarks.longbench import (
    PASSAGE_RETRIEVAL_EN_CONFIG,
    PASSAGE_RETRIEVAL_EN_FILE,
    PASSAGE_RETRIEVAL_EN_PROMPT_TEMPLATE_ID,
    benchmark_case_from_longbench_passage_retrieval_record,
)


def _record(index: int, *, answer: str = "Paragraph 15") -> dict[str, object]:
    return {
        "_id": f"case-{index}",
        "input": f"abstract {index}",
        "context": f"Paragraph 1: filler\nParagraph 15: target {index}",
        "answers": [answer],
        "length": 9000 + index,
        "dataset": PASSAGE_RETRIEVAL_EN_CONFIG,
        "language": "en",
        "all_classes": None,
    }


def test_passage_retrieval_prompt_matches_longbench_contract() -> None:
    rendered = render_passage_retrieval_en_prompt(
        context="Paragraph 1: context",
        input_text="abstract",
    )

    assert rendered == PASSAGE_RETRIEVAL_EN_PROMPT.format(
        context="Paragraph 1: context",
        input="abstract",
    )
    assert rendered.endswith("The answer is: ")
    assert PASSAGE_RETRIEVAL_EN_MAX_NEW_TOKENS == 32


@pytest.mark.parametrize(
    ("prediction", "ground_truth", "expected"),
    [
        ("Paragraph 15", "Paragraph 15", 1.0),
        ("Paragraph 14", "Paragraph 15", 0.0),
        ("Paragraph 15 Paragraph 14", "Paragraph 15", 0.5),
        ("No paragraph named.", "Paragraph 15", 0.0),
    ],
)
def test_official_retrieval_score_matches_longbench_examples(
    prediction: str,
    ground_truth: str,
    expected: float,
) -> None:
    assert official_retrieval_score(prediction, ground_truth) == expected


def test_binary_paragraph_hit_is_stricter_than_official_fractional_score() -> None:
    score = score_passage_retrieval_prediction(
        "Paragraph 15 Paragraph 14",
        "Paragraph 15",
    )

    assert score.official_score == 0.5
    assert score.binary_paragraph_hit is False
    assert score.numbers_extracted == ("15", "14")


def test_record_builder_preserves_preregistered_metadata() -> None:
    case = benchmark_case_from_longbench_passage_retrieval_record(
        _record(0),
        source_index=0,
        split="test[:50]",
    )

    assert case.case_id == "case-0"
    assert case.expected_answer == "Paragraph 15"
    assert case.prompt == render_passage_retrieval_en_prompt(
        context="Paragraph 1: filler\nParagraph 15: target 0",
        input_text="abstract 0",
    )
    assert case.metadata["dataset_config"] == PASSAGE_RETRIEVAL_EN_CONFIG
    assert case.metadata["prompt_template_id"] == PASSAGE_RETRIEVAL_EN_PROMPT_TEMPLATE_ID
    assert case.metadata["max_new_tokens"] == "32"
    assert case.metadata["answers"] == '["Paragraph 15"]'
    assert case.metadata["input"] == "abstract 0"
    assert case.metadata["context"].startswith("Paragraph 1:")


def test_longbench_observation_persists_official_and_binary_scores() -> None:
    case = benchmark_case_from_longbench_passage_retrieval_record(
        _record(0),
        source_index=0,
    )

    observation = build_case_observation(case, "Paragraph 15 Paragraph 14")

    assert observation.correct is False
    assert observation.expected_answer == "Paragraph 15"
    assert observation.official_score == 0.5
    assert observation.binary_paragraph_hit is False
    assert observation.numbers_extracted == ("15", "14")
    assert observation.case_metadata is not None
    assert observation.case_metadata["longbench_id"] == "case-0"


def test_longbench_registry_scores_official_mean_and_binary_hit_rate() -> None:
    cases = (
        benchmark_case_from_longbench_passage_retrieval_record(_record(0), source_index=0),
        benchmark_case_from_longbench_passage_retrieval_record(
            _record(1, answer="Paragraph 8"),
            source_index=1,
        ),
    )

    metrics = score_predictions(
        "longbench_passage_retrieval_en",
        cases,
        ("Paragraph 15 Paragraph 14", "Paragraph 8"),
    )

    assert metrics == {
        "mean_official_score": 0.75,
        "longbench_percent": 75.0,
        "binary_paragraph_hit_rate": 0.5,
    }
    assert get_benchmark("longbench_passage_retrieval_en").metric_name == (
        "mean_official_score"
    )


def test_load_longbench_slice_reads_official_data_zip_member(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_zip = tmp_path / "data.zip"
    rows = [_record(0), _record(1), _record(2)]
    with zipfile.ZipFile(data_zip, "w") as archive:
        archive.writestr(
            PASSAGE_RETRIEVAL_EN_FILE,
            "\n".join(json.dumps(row) for row in rows) + "\n",
        )

    def fake_hf_hub_download(dataset_id, filename, *, repo_type, revision):
        assert dataset_id == "THUDM/LongBench"
        assert filename == "data.zip"
        assert repo_type == "dataset"
        assert revision
        return str(data_zip)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )

    cases = load_longbench_passage_retrieval_en_slice(
        case_limit=2,
        case_offset_start=1,
    )

    assert [case.case_id for case in cases] == ["case-1", "case-2"]
    assert cases[0].metadata["split"] == "test[1:3]"
    assert cases[0].metadata["source_index"] == "1"


def test_load_longbench_slice_rejects_out_of_range_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_zip = tmp_path / "data.zip"
    with zipfile.ZipFile(data_zip, "w") as archive:
        archive.writestr(PASSAGE_RETRIEVAL_EN_FILE, json.dumps(_record(0)) + "\n")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=lambda *args, **kwargs: str(data_zip)),
    )

    with pytest.raises(ValueError, match="exceeds available rows"):
        load_longbench_passage_retrieval_en_slice(case_limit=2)


def test_preregistered_cell_list_matches_config316_table() -> None:
    assert [
        (cell.family, cell.model_id, cell.method_label, cell.retention_policy)
        for cell in PASSAGE_RETRIEVAL_EN_PREREGISTERED_CELLS
    ] == [
        (
            "Mistral",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "sentence vorn",
            "sentence_vorn",
        ),
        (
            "Mistral",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "sentence attention",
            "sentence_tova",
        ),
        (
            "Llama 3.1",
            "meta-llama/Llama-3.1-8B-Instruct",
            "sentence vorn",
            "sentence_vorn",
        ),
        (
            "Llama 3.1",
            "meta-llama/Llama-3.1-8B-Instruct",
            "sentence attention",
            "sentence_tova",
        ),
        (
            "Gemma 4",
            "google/gemma-4-E4B-it",
            "sentence vorn",
            "sentence_vorn",
        ),
        (
            "Gemma 4",
            "google/gemma-4-E4B-it",
            "sentence attention",
            "sentence_tova",
        ),
        ("Qwen 3-NT 8B", "Qwen/Qwen3-8B", "sentence vorn", "sentence_vorn"),
        (
            "Qwen 3-NT 8B",
            "Qwen/Qwen3-8B",
            "sentence attention",
            "sentence_tova",
        ),
    ]


def test_build_preregistered_cell_specs_are_run_ready_and_unique() -> None:
    specs = build_passage_retrieval_en_cell_specs(results_root="/vol/results/vorn-mat")

    assert len(specs) == 8
    assert len({spec["output_path"] for spec in specs}) == 8
    assert {spec["retention_policy"] for spec in specs} == {
        "sentence_vorn",
        "sentence_tova",
    }
    assert {spec["cache_budget_tokens"] for spec in specs} == {1024}
    assert {spec["case_limit"] for spec in specs} == {50}
    assert {spec["case_offset_start"] for spec in specs} == {0}
    assert {spec["max_new_tokens"] for spec in specs} == {32}
    assert {spec["gpu"] for spec in specs} == {"A100-80GB"}
    assert {spec["modal_profile"] for spec in specs} == {"layne1penney"}
    assert {spec["preregistration"] for spec in specs} == {"config#316"}
    assert {spec["sentence_pooling"] for spec in specs} == {"max"}
    assert {spec["sentence_top_k"] for spec in specs} == {3}
    assert {spec["always_keep_prefix_tokens"] for spec in specs} == {1}
    assert {spec["preserve_recent_window"] for spec in specs} == {True}
    assert {spec["random_seed"] for spec in specs} == {17}
    assert {
        (spec["model_id"], spec["retention_policy"])
        for spec in specs
    } == {
        ("mistralai/Mistral-7B-Instruct-v0.3", "sentence_vorn"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "sentence_tova"),
        ("meta-llama/Llama-3.1-8B-Instruct", "sentence_vorn"),
        ("meta-llama/Llama-3.1-8B-Instruct", "sentence_tova"),
        ("google/gemma-4-E4B-it", "sentence_vorn"),
        ("google/gemma-4-E4B-it", "sentence_tova"),
        ("Qwen/Qwen3-8B", "sentence_vorn"),
        ("Qwen/Qwen3-8B", "sentence_tova"),
    }


def test_build_preregistered_cell_specs_supports_attempt_label_isolation() -> None:
    specs = build_passage_retrieval_en_cell_specs(
        results_root="/vol/results/vorn-mat",
        attempt_label="config316-h100",
        gpu="H100",
    )

    assert all("config316-h100-" in str(spec["output_path"]) for spec in specs)
    assert {spec["gpu"] for spec in specs} == {"H100"}


def test_build_preregistered_cell_specs_supports_h200_substrate_metadata() -> None:
    specs = build_passage_retrieval_en_cell_specs(
        results_root="/vol/results/vorn-mat",
        attempt_label="config316-h200",
        gpu="H200",
    )

    assert all("config316-h200-" in str(spec["output_path"]) for spec in specs)
    assert {spec["gpu"] for spec in specs} == {"H200"}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"case_limit": 49}, "case_limit=50"),
        ({"case_offset_start": 1}, "case_offset_start=0"),
        ({"modal_profile": ""}, "modal_profile"),
        ({"attempt_label": ""}, "attempt_label"),
    ],
)
def test_build_preregistered_cell_specs_rejects_contract_drift(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_passage_retrieval_en_cell_specs(
            results_root="/vol/results/vorn-mat",
            **kwargs,
        )


def test_expanded_comparison_cell_list_matches_config321_table() -> None:
    assert len(PASSAGE_RETRIEVAL_EN_EXPANDED_COMPARISON_CELLS) == 16
    assert {
        (cell.model_id, cell.retention_policy)
        for cell in PASSAGE_RETRIEVAL_EN_EXPANDED_COMPARISON_CELLS
    } == {
        (model_id, retention_policy)
        for model_id in {
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Llama-3.1-8B-Instruct",
            "google/gemma-4-E4B-it",
            "Qwen/Qwen3-8B",
        }
        for retention_policy in {
            "sentence_snapkv",
            "sentence_l2_norm",
            "sentence_streaming_llm",
            "vanilla",
        }
    }


def test_build_expanded_comparison_specs_are_run_ready_and_unique() -> None:
    specs = build_passage_retrieval_en_expanded_comparison_specs(
        results_root="/vol/results/vorn-mat",
    )

    assert len(specs) == 16
    assert len({spec["output_path"] for spec in specs}) == 16
    assert {spec["retention_policy"] for spec in specs} == {
        "sentence_snapkv",
        "sentence_l2_norm",
        "sentence_streaming_llm",
        "vanilla",
    }
    assert {spec["cache_budget_tokens"] for spec in specs} == {1024}
    assert {spec["case_limit"] for spec in specs} == {50}
    assert {spec["case_offset_start"] for spec in specs} == {0}
    assert {spec["max_new_tokens"] for spec in specs} == {32}
    assert {spec["gpu"] for spec in specs} == {"H200"}
    assert {spec["modal_profile"] for spec in specs} == {"layne1penney"}
    assert {spec["preregistration"] for spec in specs} == {"config#321"}
    assert {spec["sentence_pooling"] for spec in specs} == {"max"}
    assert {spec["sentence_top_k"] for spec in specs} == {3}
    assert {spec["always_keep_prefix_tokens"] for spec in specs} == {1}
    assert {spec["preserve_recent_window"] for spec in specs} == {True}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"case_limit": 49}, "case_limit=50"),
        ({"case_offset_start": 1}, "case_offset_start=0"),
        ({"gpu": "A100-80GB"}, "gpu=H200"),
        ({"modal_profile": ""}, "modal_profile"),
        ({"attempt_label": ""}, "attempt_label"),
    ],
)
def test_build_expanded_comparison_specs_rejects_contract_drift(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_passage_retrieval_en_expanded_comparison_specs(
            results_root="/vol/results/vorn-mat",
            **kwargs,
        )
