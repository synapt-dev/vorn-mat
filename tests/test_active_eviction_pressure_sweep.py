from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import vorn_mat.active_eviction_pressure_sweep as sweep
from vorn_mat.active_eviction_consumer_validation import (
    build_semus_and_scores_from_phase0,
    load_phase0_case,
)
from vorn_mat.benchmarks.common import BenchmarkCase
from vorn_mat.counterfactual_intervention_runner import sha256_text


_RENDERED_PROMPT = (
    "[INST] Setup. "
    "Protect answer. "
    "Alpha top vorn. "
    "Beta also top. "
    "Gamma third. "
    "Very very long neutral sentence with many extra words here. "
    "Medium length sentence. "
    "Short low. "
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
        if "Alpha top vorn." not in rendered_prompt:
            return "wrong"
        return "blue"

    def render_prompt_text_with_offsets(
        self,
        prompt: str,
    ) -> tuple[str, tuple[tuple[int, int], ...]]:
        return self.rendered_prompt, ()

    def count_rendered_prompt_tokens(self, rendered_prompt: str) -> int:
        return len(rendered_prompt.split())


class CountingFailureGenerator(FakeGenerator):
    def count_rendered_prompt_tokens(self, rendered_prompt: str) -> int:
        raise RuntimeError("token count failed before model call")


def test_select_semus_for_arm_supports_pressure_n_and_locks_n1_smoke_ids():
    semus, scores = build_semus_and_scores_from_phase0(
        _phase0_case(),
        protected_semu_ids=(0, 1),
    )

    assert [
        semu.semu_id
        for semu in sweep.select_semus_for_arm(
            selector_arm="vorn_high",
            pressure_n=1,
            semus=semus,
            scores=scores,
            run_id="pilot-b",
            case_id="case-1",
            model_id="mistral",
            base_seed=534588164691762844,
        )
    ] == [2]
    assert [
        semu.semu_id
        for semu in sweep.select_semus_for_arm(
            selector_arm="vorn_low",
            pressure_n=1,
            semus=semus,
            scores=scores,
            run_id="pilot-b",
            case_id="case-1",
            model_id="mistral",
            base_seed=534588164691762844,
        )
    ] == [7]
    assert [
        semu.semu_id
        for semu in sweep.select_semus_for_arm(
            selector_arm="length_high",
            pressure_n=1,
            semus=semus,
            scores=scores,
            run_id="pilot-b",
            case_id="case-1",
            model_id="mistral",
            base_seed=534588164691762844,
        )
    ] == [5]

    vorn_high = sweep.select_semus_for_arm(
        selector_arm="vorn_high",
        pressure_n=3,
        semus=semus,
        scores=scores,
        run_id="pilot-b",
        case_id="case-1",
        model_id="mistral",
        base_seed=534588164691762844,
    )
    random_matched = sweep.select_semus_for_arm(
        selector_arm="random_length_matched",
        pressure_n=3,
        semus=semus,
        scores=scores,
        run_id="pilot-b",
        case_id="case-1",
        model_id="mistral",
        base_seed=534588164691762844,
        reference_semus=vorn_high,
    )

    assert len(vorn_high) == 3
    assert len(random_matched) == 3
    assert {semu.semu_id for semu in vorn_high}.isdisjoint(
        {semu.semu_id for semu in random_matched}
    )
    assert all(semu.semu_id not in {0, 1} for semu in random_matched)


def test_real_fixture_n1_selection_matches_consumer_validation_smoke_ids():
    phase0_case = load_phase0_case(
        Path("results/vorn-active-eviction-phase0-2026-06-03.json"),
        "niah_multikey_1_4k-1",
    )
    semus, scores = build_semus_and_scores_from_phase0(
        phase0_case,
        protected_semu_ids=(0, 1, 2, 29, 209, 210),
    )
    kwargs = {
        "pressure_n": 1,
        "semus": semus,
        "scores": scores,
        "run_id": "vorn-active-eviction-pilot-b-pressure-sweep-2026-06-04",
        "case_id": "niah_multikey_1_4k-1",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "base_seed": 534588164691762844,
    }

    vorn_high = sweep.select_semus_for_arm(selector_arm="vorn_high", **kwargs)
    selected = {
        "vorn_high": vorn_high,
        "vorn_low": sweep.select_semus_for_arm(selector_arm="vorn_low", **kwargs),
        "length_high": sweep.select_semus_for_arm(
            selector_arm="length_high",
            **kwargs,
        ),
        "random_length_matched": sweep.select_semus_for_arm(
            selector_arm="random_length_matched",
            reference_semus=vorn_high,
            **kwargs,
        ),
    }

    assert {
        arm: tuple(semu.semu_id for semu in semus_for_arm)
        for arm, semus_for_arm in selected.items()
    } == {
        "vorn_high": (17,),
        "vorn_low": (201,),
        "length_high": (125,),
        "random_length_matched": (111,),
    }


def test_select_semus_for_arm_supports_snapkv_high_and_records_snapkv_metadata():
    semus, scores = build_semus_and_scores_from_phase0(
        _phase0_case_with_snapkv(),
        protected_semu_ids=(0, 1),
    )

    selected = sweep.select_semus_for_arm(
        selector_arm="snapkv_high",
        pressure_n=3,
        semus=semus,
        scores=scores,
        run_id="pilot-snapkv",
        case_id="case-1",
        model_id="mistral",
        base_seed=534588164691762844,
    )
    report = sweep.run_pressure_sweep(
        run_id="pilot-snapkv",
        family="Mistral",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_revision="main",
        tokenizer_revision="main",
        case=_case(),
        phase0_case=_phase0_case_with_snapkv(),
        generator=FakeGenerator(),
        protected_semu_ids=(0, 1),
        selector_arms=("snapkv_high",),
        pressure_ns=(3,),
        base_seed=534588164691762844,
    )

    assert tuple(semu.semu_id for semu in selected) == (6, 3, 2)
    assert report.records[0].intervention.selected_semu_ids == (6, 3, 2)
    assert report.records[0].intervention.selected_semu_scores == (0.99, 0.90, 0.80)
    assert report.records[0].intervention.selected_semu_ranks == (1, 2, 3)


def test_select_semus_for_arm_fails_closed_when_snapkv_scores_missing():
    semus, scores = build_semus_and_scores_from_phase0(
        _phase0_case(),
        protected_semu_ids=(0, 1),
    )

    with pytest.raises(sweep.CounterfactualContractError, match="SnapKV scores"):
        sweep.select_semus_for_arm(
            selector_arm="snapkv_high",
            pressure_n=1,
            semus=semus,
            scores=scores,
            run_id="pilot-snapkv",
            case_id="case-1",
            model_id="mistral",
            base_seed=534588164691762844,
        )


def test_render_pressure_prompt_deletes_multiple_spans_and_rejects_overlap():
    semus, _scores = build_semus_and_scores_from_phase0(
        _phase0_case(),
        protected_semu_ids=(0, 1),
    )

    counterfactual_prompt, prompt_record = sweep.render_pressure_prompt(
        rendered_prompt=_RENDERED_PROMPT,
        semus=(semus[2], semus[4]),
        deletion_mode="delete",
        original_token_count=len(_RENDERED_PROMPT.split()),
        counterfactual_token_count=23,
    )

    assert "Alpha top vorn." not in counterfactual_prompt
    assert "Gamma third." not in counterfactual_prompt
    assert "Beta also top." in counterfactual_prompt
    assert prompt_record.sentence_id_alignment_audit["semu_ids"] == [2, 4]
    assert prompt_record.counterfactual_token_count == 23

    overlapping = (
        semus[2],
        sweep.SemanticUnit(
            semu_id=99,
            granularity="sentence",
            char_start=semus[2].char_start + 1,
            char_end=semus[2].char_end,
            token_start=semus[2].token_start,
            token_end=semus[2].token_end,
            text="overlap",
        ),
    )
    with pytest.raises(ValueError, match="overlapping SEMU spans"):
        sweep.render_pressure_prompt(
            rendered_prompt=_RENDERED_PROMPT,
            semus=overlapping,
            deletion_mode="delete",
        )


def test_run_pressure_sweep_preflights_all_prompts_before_generation():
    generator = CountingFailureGenerator()

    with pytest.raises(RuntimeError, match="token count failed"):
        sweep.run_pressure_sweep(
            run_id="pressure-sweep-test",
            family="Mistral",
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            model_revision="main",
            tokenizer_revision="main",
            case=_case(),
            phase0_case=_phase0_case(),
            generator=generator,
            protected_semu_ids=(0, 1),
            selector_arms=("vorn_high", "vorn_low"),
            pressure_ns=(1, 3),
            base_seed=534588164691762844,
        )

    assert generator.generated_prompts == []


def test_run_pressure_sweep_emits_arm_by_pressure_records():
    snapshots = iter(
        [
            {"peak_memory_allocated_gb": 0.5, "peak_memory_reserved_gb": 0.75},
            {"peak_memory_allocated_gb": 1.0, "peak_memory_reserved_gb": 1.25},
            {"peak_memory_allocated_gb": 1.1, "peak_memory_reserved_gb": 1.35},
            {"peak_memory_allocated_gb": 1.2, "peak_memory_reserved_gb": 1.45},
            {"peak_memory_allocated_gb": 1.3, "peak_memory_reserved_gb": 1.55},
        ]
    )
    reset_calls: list[str] = []

    report = sweep.run_pressure_sweep(
        run_id="pressure-sweep-test",
        family="Mistral",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_revision="main",
        tokenizer_revision="main",
        case=_case(),
        phase0_case=_phase0_case(),
        generator=FakeGenerator(),
        protected_semu_ids=(0, 1),
        selector_arms=("vorn_high", "vorn_low"),
        pressure_ns=(1, 3),
        base_seed=534588164691762844,
        cost_per_second=0.01,
        reset_telemetry=lambda: reset_calls.append("reset"),
        telemetry_snapshot=lambda: next(snapshots),
    )

    assert report.full_context.binary_hit is True
    assert len(reset_calls) == 5
    assert len(report.records) == 4
    assert [
        (record.intervention.selector_arm, record.intervention.pressure_n)
        for record in report.records
    ] == [
        ("vorn_high", 1),
        ("vorn_high", 3),
        ("vorn_low", 1),
        ("vorn_low", 3),
    ]
    assert report.records[0].intervention.selected_semu_ids == (2,)
    assert report.records[1].intervention.selected_semu_ids == (2, 3, 4)
    assert report.records[0].quality_record is not None
    assert report.records[0].quality_record.binary_hit is False
    assert report.records[0].quality_record.delta_quality == 1.0
    assert report.records[2].quality_record is not None
    assert report.records[2].quality_record.binary_hit is True


def test_pressure_sweep_artifact_writer_and_loader_round_trip(tmp_path: Path):
    report = sweep.run_pressure_sweep(
        run_id="pressure-sweep-test",
        family="Mistral",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_revision="main",
        tokenizer_revision="main",
        case=_case(),
        phase0_case=_phase0_case(),
        generator=FakeGenerator(),
        protected_semu_ids=(0, 1),
        selector_arms=("vorn_high", "vorn_low"),
        pressure_ns=(1, 3),
        base_seed=534588164691762844,
    )
    jsonl_path = tmp_path / "pressure-sweep.jsonl"
    md_path = tmp_path / "pressure-sweep.md"

    sweep.write_pressure_sweep_artifacts(
        report,
        jsonl_path=jsonl_path,
        md_path=md_path,
    )
    payloads = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    delta_table = sweep.load_pressure_sweep_delta_table(jsonl_path)

    assert payloads[0]["record_type"] == "FULL_CONTEXT_CONTROL"
    assert payloads[1]["intervention"]["selected_semu_ids"] == [2]
    assert delta_table["status_counts"] == {"PROMPT_QUALITY_SUCCESS": 4}
    assert delta_table["successful_rows"][0]["pressure_n"] == 1
    assert delta_table["successful_rows"][0]["selected_semu_ids"] == [2]
    assert "Pressure N" in md_path.read_text()


def _case() -> BenchmarkCase:
    return BenchmarkCase("case-1", "raw prompt", "blue", {})


def _phase0_case() -> dict[str, object]:
    return {
        "case_id": "case-1",
        "prompt_hash": sha256_text(_RENDERED_PROMPT),
        "prompt_token_count": len(_RENDERED_PROMPT.split()),
        "semu_matrix": (
            _semu_payload(0, "[INST] Setup.", 0.99, 0, 0, 2),
            _semu_payload(1, "Protect answer.", 0.95, 1, 2, 4, answer_overlap=True),
            _semu_payload(2, "Alpha top vorn.", 0.90, 2, 4, 7),
            _semu_payload(3, "Beta also top.", 0.80, 3, 7, 10),
            _semu_payload(4, "Gamma third.", 0.70, 4, 10, 12),
            _semu_payload(
                5,
                "Very very long neutral sentence with many extra words here.",
                0.20,
                5,
                12,
                22,
            ),
            _semu_payload(6, "Medium length sentence.", 0.10, 6, 22, 25),
            _semu_payload(7, "Short low.", 0.01, 7, 25, 27),
        ),
    }


def _phase0_case_with_snapkv() -> dict[str, object]:
    return {
        "case_id": "case-1",
        "prompt_hash": sha256_text(_RENDERED_PROMPT),
        "prompt_token_count": len(_RENDERED_PROMPT.split()),
        "semu_matrix": (
            _semu_payload(0, "[INST] Setup.", 0.99, 0, 0, 2, snapkv_score=0.99, snapkv_rank=1),
            _semu_payload(
                1,
                "Protect answer.",
                0.95,
                1,
                2,
                4,
                answer_overlap=True,
                snapkv_score=0.98,
                snapkv_rank=2,
            ),
            _semu_payload(2, "Alpha top vorn.", 0.90, 2, 4, 7, snapkv_score=0.80, snapkv_rank=3),
            _semu_payload(3, "Beta also top.", 0.80, 3, 7, 10, snapkv_score=0.90, snapkv_rank=2),
            _semu_payload(4, "Gamma third.", 0.70, 4, 10, 12, snapkv_score=0.30, snapkv_rank=5),
            _semu_payload(
                5,
                "Very very long neutral sentence with many extra words here.",
                0.20,
                5,
                12,
                22,
                snapkv_score=0.20,
                snapkv_rank=6,
            ),
            _semu_payload(6, "Medium length sentence.", 0.10, 6, 22, 25, snapkv_score=0.99, snapkv_rank=1),
            _semu_payload(7, "Short low.", 0.01, 7, 25, 27, snapkv_score=0.10, snapkv_rank=7),
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
    snapkv_score: float | None = None,
    snapkv_rank: int | None = None,
) -> dict[str, object]:
    char_start = _RENDERED_PROMPT.index(text)
    char_end = char_start + len(text)
    payload: dict[str, object] = {
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
    if snapkv_score is not None:
        payload["snapkv_score"] = snapkv_score
    if snapkv_rank is not None:
        payload["snapkv_rank"] = snapkv_rank
    return payload
