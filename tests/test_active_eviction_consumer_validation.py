from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat.active_eviction_consumer_validation import (
    build_semus_and_scores_from_phase0,
    load_consumer_validation_delta_table,
    run_consumer_validation,
    write_consumer_validation_artifacts,
)
from vorn_mat.benchmarks.common import BenchmarkCase
from vorn_mat.counterfactual_intervention_runner import sha256_text


_RENDERED_PROMPT = (
    "[INST] Setup. "
    "Protect answer. "
    "High magic clue. "
    "Long neutral sentence with many extra words. "
    "Length match. "
    "Low filler. "
    "Question? [/INST]"
)


class FakeGenerator:
    def __init__(self, rendered_prompt: str = _RENDERED_PROMPT):
        self.rendered_prompt = rendered_prompt
        self.generated_prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.generated_prompts.append(prompt)
        return "blue"

    def generate_rendered_prompt(self, rendered_prompt: str) -> str:
        self.generated_prompts.append(rendered_prompt)
        if "High magic clue." in rendered_prompt:
            return "blue"
        return "wrong"

    def render_prompt_text_with_offsets(
        self,
        prompt: str,
    ) -> tuple[str, tuple[tuple[int, int], ...]]:
        return self.rendered_prompt, ()

    def count_rendered_prompt_tokens(self, rendered_prompt: str) -> int:
        return len(rendered_prompt.split())


class OOMGenerator(FakeGenerator):
    def generate_rendered_prompt(self, rendered_prompt: str) -> str:
        raise RuntimeError("CUDA out of memory during generation")


def test_build_semus_and_scores_from_phase0_marks_protected_answer_overlap():
    phase0_case = _phase0_case()

    semus, scores = build_semus_and_scores_from_phase0(
        phase0_case,
        protected_semu_ids=(0, 1),
    )

    assert semus[0].protected_classes == ("protected_prompt_region",)
    assert semus[1].protected_classes == ("expected_answer_overlap",)
    assert semus[2].protected_classes == ()
    assert scores[2].vorn_score == 0.9
    assert scores[2].vorn_rank == 1


def test_run_consumer_validation_emits_locked_success_records_and_mask_dry_runs():
    snapshots = iter(
        [
            {"peak_memory_allocated_gb": 0.5, "peak_memory_reserved_gb": 0.75},
            {"peak_memory_allocated_gb": 1.0, "peak_memory_reserved_gb": 1.25},
            {"peak_memory_allocated_gb": 1.1, "peak_memory_reserved_gb": 1.35},
            {"peak_memory_allocated_gb": 1.2, "peak_memory_reserved_gb": 1.45},
            {"peak_memory_allocated_gb": 1.3, "peak_memory_reserved_gb": 1.55},
        ]
    )
    reset_calls = []
    report = run_consumer_validation(
        run_id="consumer-validation-test",
        family="Mistral",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_revision="main",
        tokenizer_revision="main",
        case=_case(),
        phase0_case=_phase0_case(),
        generator=FakeGenerator(),
        protected_semu_ids=(0, 1),
        selector_arms=(
            "vorn_high",
            "vorn_low",
            "length_high",
            "random_length_matched",
        ),
        expected_selected_semu_ids={
            "vorn_high": 2,
            "vorn_low": 5,
            "length_high": 3,
            "random_length_matched": 4,
        },
        cost_per_second=0.01,
        reset_telemetry=lambda: reset_calls.append("reset"),
        telemetry_snapshot=lambda: next(snapshots),
    )

    assert report.full_context.binary_hit is True
    assert report.full_context.peak_memory_allocated_mb == 512.0
    assert len(reset_calls) == 5
    assert [record.record_status for record in report.records] == [
        "PROMPT_QUALITY_SUCCESS",
        "PROMPT_QUALITY_SUCCESS",
        "PROMPT_QUALITY_SUCCESS",
        "PROMPT_QUALITY_SUCCESS",
    ]
    assert [record.intervention.semu_id for record in report.records] == [2, 5, 3, 4]
    assert report.records[0].quality_record is not None
    assert report.records[0].quality_record.binary_hit is False
    assert report.records[0].quality_record.delta_quality == 1.0
    assert report.records[0].quality_record.peak_memory_allocated_mb == 1024.0
    assert report.records[1].quality_record is not None
    assert report.records[1].quality_record.binary_hit is True
    assert len(report.mask_prompt_records) == 2
    assert all(
        prompt_record.mask_text_policy
        == "repeat_MASKED_SEMU_token_to_original_semu_token_length"
        for prompt_record in report.mask_prompt_records
    )
    assert all(
        prompt_record.original_prompt_hash
        != prompt_record.counterfactual_prompt_hash
        for prompt_record in report.mask_prompt_records
    )


def test_run_consumer_validation_rejects_locked_semu_drift_before_generation():
    generator = FakeGenerator()

    with pytest.raises(ValueError, match="selected SEMU drift"):
        run_consumer_validation(
            run_id="consumer-validation-test",
            family="Mistral",
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            model_revision="main",
            tokenizer_revision="main",
            case=_case(),
            phase0_case=_phase0_case(),
            generator=generator,
            protected_semu_ids=(0, 1),
            selector_arms=("vorn_high",),
            expected_selected_semu_ids={"vorn_high": 5},
        )

    assert generator.generated_prompts == []


def test_run_consumer_validation_rejects_prompt_hash_mismatch_before_generation():
    generator = FakeGenerator(rendered_prompt="unexpected prompt")

    with pytest.raises(ValueError, match="rendered prompt hash mismatch"):
        run_consumer_validation(
            run_id="consumer-validation-test",
            family="Mistral",
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            model_revision="main",
            tokenizer_revision="main",
            case=_case(),
            phase0_case=_phase0_case(),
            generator=generator,
            protected_semu_ids=(0, 1),
            selector_arms=("vorn_high",),
        )

    assert generator.generated_prompts == []


def test_run_consumer_validation_records_capacity_missing_terminal_path():
    report = run_consumer_validation(
        run_id="consumer-validation-test",
        family="Mistral",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_revision="main",
        tokenizer_revision="main",
        case=_case(),
        phase0_case=_phase0_case(),
        generator=OOMGenerator(),
        protected_semu_ids=(0, 1),
        selector_arms=("vorn_high",),
        hardware="H200",
        modal_profile="layne1penney",
    )

    record = report.records[0]
    assert record.record_status == "CAPACITY_MISSING"
    assert record.quality_record is None
    assert record.prompt_record is not None
    assert record.failure is not None
    assert record.failure.capacity_missing_class == "oom_after_retry"
    assert record.failure.hardware == "H200"
    assert record.failure.modal_profile == "layne1penney"


def test_artifact_writer_and_delta_loader_round_trip(tmp_path: Path):
    report = run_consumer_validation(
        run_id="consumer-validation-test",
        family="Mistral",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_revision="main",
        tokenizer_revision="main",
        case=_case(),
        phase0_case=_phase0_case(),
        generator=FakeGenerator(),
        protected_semu_ids=(0, 1),
        selector_arms=("vorn_high", "vorn_low"),
    )
    jsonl_path = tmp_path / "consumer-validation.jsonl"
    md_path = tmp_path / "consumer-validation.md"

    write_consumer_validation_artifacts(
        report,
        jsonl_path=jsonl_path,
        md_path=md_path,
    )
    lines = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    delta_table = load_consumer_validation_delta_table(jsonl_path)

    assert lines[0]["record_type"] == "FULL_CONTEXT_CONTROL"
    assert lines[1]["record_status"] == "PROMPT_QUALITY_SUCCESS"
    assert delta_table["status_counts"] == {"PROMPT_QUALITY_SUCCESS": 2}
    assert len(delta_table["successful_rows"]) == 2
    assert delta_table["successful_rows"][0]["selector_arm"] == "vorn_high"
    assert delta_table["failure_rows"] == []
    assert "Record status counts" in md_path.read_text()


def _case() -> BenchmarkCase:
    return BenchmarkCase("case-1", "raw prompt", "blue", {})


def _phase0_case() -> dict[str, object]:
    return {
        "case_id": "case-1",
        "prompt_hash": sha256_text(_RENDERED_PROMPT),
        "prompt_token_count": len(_RENDERED_PROMPT.split()),
        "semu_matrix": (
            _semu_payload(0, "[INST] Setup.", 0.99, 0, 0, 2),
            _semu_payload(1, "Protect answer.", 0.95, 2, 2, 4, answer_overlap=True),
            _semu_payload(2, "High magic clue.", 0.90, 1, 4, 8),
            _semu_payload(
                3,
                "Long neutral sentence with many extra words.",
                0.50,
                3,
                8,
                16,
            ),
            _semu_payload(4, "Length match.", 0.40, 4, 16, 20),
            _semu_payload(5, "Low filler.", 0.10, 5, 20, 22),
        ),
    }


def _semu_payload(
    semu_id: int,
    text: str,
    final_score: float,
    final_rank: int,
    token_start: int,
    token_end: int,
    *,
    answer_overlap: bool = False,
) -> dict[str, object]:
    char_start = _RENDERED_PROMPT.index(text)
    char_end = char_start + len(text)
    return {
        "semu_id": semu_id,
        "char_start": char_start,
        "char_end": char_end,
        "token_start": token_start,
        "token_end": token_end,
        "text_preview": text,
        "final_score": final_score,
        "final_rank": final_rank,
        "answer_overlap": answer_overlap,
    }
