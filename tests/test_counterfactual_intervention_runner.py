from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat.counterfactual_intervention_runner import (
    CounterfactualContractError,
    CounterfactualPromptRecord,
    CounterfactualQualityRecord,
    CounterfactualRunRecord,
    CounterfactualSEMUIntervention,
    SCHEMA_VERSION,
    SEMUScore,
    SemanticUnit,
    TargetedDropFailure,
    build_intervention,
    build_sentence_semus,
    deterministic_seed,
    render_counterfactual_prompt,
    run_record_to_json_line,
    select_semu_for_arm,
)
from vorn_mat.text_spans import token_span_from_offsets


def _nonspace_offsets(text: str) -> tuple[tuple[int, int], ...]:
    return tuple((match.start(), match.end()) for match in re.finditer(r"\S+", text))


def _manual_semus() -> tuple[SemanticUnit, ...]:
    return (
        SemanticUnit(
            semu_id=0,
            granularity="sentence",
            char_start=0,
            char_end=12,
            token_start=0,
            token_end=2,
            text="Protected.",
            protected_classes=("system_prompt",),
        ),
        SemanticUnit(
            semu_id=1,
            granularity="sentence",
            char_start=13,
            char_end=25,
            token_start=2,
            token_end=5,
            text="Short clue.",
        ),
        SemanticUnit(
            semu_id=2,
            granularity="sentence",
            char_start=26,
            char_end=50,
            token_start=5,
            token_end=11,
            text="Load-bearing answer.",
        ),
        SemanticUnit(
            semu_id=3,
            granularity="sentence",
            char_start=51,
            char_end=80,
            token_start=11,
            token_end=20,
            text="Recent filler with extra words.",
        ),
    )


def _manual_scores() -> tuple[SEMUScore, ...]:
    return (
        SEMUScore(semu_id=0, vorn_score=0.99, vorn_rank=1, attention_score=0.99),
        SEMUScore(semu_id=1, vorn_score=0.10, vorn_rank=4, attention_score=0.20),
        SEMUScore(semu_id=2, vorn_score=0.80, vorn_rank=2, attention_score=0.90),
        SEMUScore(semu_id=3, vorn_score=0.30, vorn_rank=3, attention_score=0.50),
    )


def test_build_sentence_semus_maps_protected_and_answer_overlap():
    prompt = "System prompt stays. The short clue is red. The answer is blue."
    offsets = _nonspace_offsets(prompt)
    blue_start = prompt.index("blue")
    blue_token_span = token_span_from_offsets(
        offsets,
        char_start=blue_start,
        char_end=blue_start + len("blue"),
    )
    assert blue_token_span is not None

    semus = build_sentence_semus(
        rendered_prompt=prompt,
        offsets=offsets,
        protected_char_spans=((0, len("System prompt stays.")),),
        answer_token_spans=(blue_token_span,),
    )

    assert [semu.text for semu in semus] == [
        "System prompt stays.",
        "The short clue is red.",
        "The answer is blue.",
    ]
    assert semus[0].protected_classes == ("protected_prompt_region",)
    assert not semus[1].is_protected
    assert semus[2].answer_overlap


def test_selector_arms_skip_protected_semus_and_choose_expected_targets():
    semus = _manual_semus()
    scores = _manual_scores()

    assert (
        select_semu_for_arm(
            selector_arm="vorn_high",
            semus=semus,
            scores=scores,
            run_id="run-a",
            case_id="case-1",
            model_id="mistral",
        ).semu_id
        == 2
    )
    assert (
        select_semu_for_arm(
            selector_arm="vorn_low",
            semus=semus,
            scores=scores,
            run_id="run-a",
            case_id="case-1",
            model_id="mistral",
        ).semu_id
        == 1
    )
    assert (
        select_semu_for_arm(
            selector_arm="attention_high",
            semus=semus,
            scores=scores,
            run_id="run-a",
            case_id="case-1",
            model_id="mistral",
        ).semu_id
        == 2
    )
    assert (
        select_semu_for_arm(
            selector_arm="recency_high",
            semus=semus,
            scores=scores,
            run_id="run-a",
            case_id="case-1",
            model_id="mistral",
        ).semu_id
        == 3
    )
    assert (
        select_semu_for_arm(
            selector_arm="length_high",
            semus=semus,
            scores=scores,
            run_id="run-a",
            case_id="case-1",
            model_id="mistral",
        ).semu_id
        == 3
    )


def test_attention_selector_fails_closed_when_attention_scores_are_missing():
    scores = tuple(
        SEMUScore(semu_id=score.semu_id, vorn_score=score.vorn_score, vorn_rank=score.vorn_rank)
        for score in _manual_scores()
    )

    with pytest.raises(CounterfactualContractError, match="capacity-missing"):
        select_semu_for_arm(
            selector_arm="attention_high",
            semus=_manual_semus(),
            scores=scores,
            run_id="run-a",
            case_id="case-1",
            model_id="mistral",
        )


def test_snapkv_selector_uses_snapkv_scores_and_records_snapkv_metadata():
    semus = _manual_semus()
    scores = (
        SEMUScore(semu_id=0, vorn_score=0.99, vorn_rank=1, snapkv_score=0.99, snapkv_rank=1),
        SEMUScore(semu_id=1, vorn_score=0.10, vorn_rank=4, snapkv_score=0.95, snapkv_rank=1),
        SEMUScore(semu_id=2, vorn_score=0.80, vorn_rank=2, snapkv_score=0.40, snapkv_rank=3),
        SEMUScore(semu_id=3, vorn_score=0.30, vorn_rank=3, snapkv_score=0.70, snapkv_rank=2),
    )

    selected = select_semu_for_arm(
        selector_arm="snapkv_high",
        semus=semus,
        scores=scores,
        run_id="run-a",
        case_id="case-1",
        model_id="mistral",
    )
    intervention = build_intervention(
        family="Mistral",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_revision="main",
        tokenizer_revision="main",
        case_id="case-1",
        checkpoint="T0_PRE_GENERATION",
        selector_arm="snapkv_high",
        semu=selected,
        score=scores[selected.semu_id],
        deletion_mode="delete",
        run_id="pilot-a",
    )

    assert selected.semu_id == 1
    assert intervention.original_score == 0.95
    assert intervention.original_rank == 1


def test_snapkv_selector_fails_closed_when_snapkv_scores_are_missing():
    with pytest.raises(CounterfactualContractError, match="SnapKV scores"):
        select_semu_for_arm(
            selector_arm="snapkv_high",
            semus=_manual_semus(),
            scores=_manual_scores(),
            run_id="run-a",
            case_id="case-1",
            model_id="mistral",
        )


def test_random_length_matched_is_deterministic_and_excludes_reference_semu():
    semus = _manual_semus()
    scores = _manual_scores()
    reference = semus[2]

    first = select_semu_for_arm(
        selector_arm="random_length_matched",
        semus=semus,
        scores=scores,
        run_id="run-a",
        case_id="case-1",
        model_id="mistral",
        reference_semu=reference,
    )
    second = select_semu_for_arm(
        selector_arm="random_length_matched",
        semus=semus,
        scores=scores,
        run_id="run-a",
        case_id="case-1",
        model_id="mistral",
        reference_semu=reference,
    )

    assert first == second
    assert first.semu_id != reference.semu_id


def test_random_length_matched_requires_reference_semu():
    with pytest.raises(CounterfactualContractError, match="requires reference_semu"):
        select_semu_for_arm(
            selector_arm="random_length_matched",
            semus=_manual_semus(),
            scores=_manual_scores(),
            run_id="run-a",
            case_id="case-1",
            model_id="mistral",
        )


def test_build_intervention_records_seed_and_refuses_protected_semu():
    semu = _manual_semus()[2]
    score = _manual_scores()[2]

    intervention = build_intervention(
        family="Mistral",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_revision="main",
        tokenizer_revision="main",
        case_id="case-1",
        checkpoint="T0_PRE_GENERATION",
        selector_arm="vorn_high",
        semu=semu,
        score=score,
        deletion_mode="delete",
        run_id="pilot-a",
    )

    assert intervention.original_char_span == (26, 50)
    assert intervention.original_score == 0.80
    assert intervention.random_seed == deterministic_seed(
        run_id="pilot-a",
        case_id="case-1",
        selector_arm="vorn_high",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
    )

    with pytest.raises(CounterfactualContractError, match="protected SEMU selected"):
        build_intervention(
            family="Mistral",
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            model_revision="main",
            tokenizer_revision="main",
            case_id="case-1",
            checkpoint="T0_PRE_GENERATION",
            selector_arm="vorn_high",
            semu=_manual_semus()[0],
            score=_manual_scores()[0],
            deletion_mode="delete",
            run_id="pilot-a",
        )


def test_render_counterfactual_prompt_delete_and_mask_modes():
    prompt = "Alpha one. Beta two. Gamma three."
    semu = SemanticUnit(
        semu_id=1,
        granularity="sentence",
        char_start=11,
        char_end=20,
        token_start=2,
        token_end=4,
        text="Beta two.",
    )

    deleted, delete_record = render_counterfactual_prompt(
        rendered_prompt=prompt,
        semu=semu,
        deletion_mode="delete",
        original_token_count=6,
        counterfactual_token_count=4,
    )
    masked, mask_record = render_counterfactual_prompt(
        rendered_prompt=prompt,
        semu=semu,
        deletion_mode="mask",
    )

    assert deleted == "Alpha one. Gamma three."
    assert delete_record.sentence_id_alignment_audit["semu_id"] == 1
    assert delete_record.original_token_count == 6
    assert delete_record.counterfactual_token_count == 4
    assert masked == "Alpha one. [MASKED_SEMU] [MASKED_SEMU] Gamma three."
    assert mask_record.mask_text_policy


def _manual_intervention() -> CounterfactualSEMUIntervention:
    return CounterfactualSEMUIntervention(
        family="Mistral",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_revision="main",
        tokenizer_revision="main",
        case_id="case-1",
        checkpoint="T0_PRE_GENERATION",
        selector_arm="vorn_high",
        semu_id=2,
        protected_class="none",
        original_char_span=(26, 50),
        original_token_span=(5, 11),
        original_score=0.80,
        original_rank=2,
        deletion_mode="delete",
        random_seed=123,
    )


def _manual_prompt_record() -> CounterfactualPromptRecord:
    return CounterfactualPromptRecord(
        original_prompt_hash="original-hash",
        counterfactual_prompt_hash="counterfactual-hash",
        deletion_text_policy="delete_original_char_span_and_collapse_single_gap_space",
        mask_text_policy="",
        original_token_count=100,
        counterfactual_token_count=94,
        sentence_id_alignment_audit={
            "semu_id": 2,
            "original_char_span": [26, 50],
            "original_token_span": [5, 11],
            "deletion_mode": "delete",
            "granularity": "sentence",
        },
    )


def _manual_quality_record() -> CounterfactualQualityRecord:
    return CounterfactualQualityRecord(
        full_context_prediction="blue",
        counterfactual_prediction="red",
        primary_metric=0.0,
        binary_hit=False,
        delta_quality=-1.0,
        runtime_seconds=3.5,
        estimated_cost_usd=0.02,
        peak_memory_allocated_mb=1024.0,
        peak_memory_reserved_mb=2048.0,
    )


def _manual_failure(*, capacity_missing_class: str = "none") -> TargetedDropFailure:
    return TargetedDropFailure(
        failure_reason="CUDA out of memory",
        failure_stage="model_generation",
        hardware="H200",
        modal_profile="layne1penney",
        capacity_missing_class=capacity_missing_class,
        retry_count=1,
        artifact_partial_path="results/partial.jsonl",
    )


def test_success_run_record_requires_prompt_and_quality_records():
    payload = json.loads(
        run_record_to_json_line(
            CounterfactualRunRecord(
                schema_version=SCHEMA_VERSION,
                record_status="PROMPT_QUALITY_SUCCESS",
                intervention=_manual_intervention(),
                prompt_record=_manual_prompt_record(),
                quality_record=_manual_quality_record(),
            )
        )
    )

    assert payload["schema_version"] == "vorn-active-eviction-counterfactual/v1"
    assert payload["record_status"] == "PROMPT_QUALITY_SUCCESS"
    assert payload["intervention"]["selector_arm"] == "vorn_high"
    assert payload["intervention"]["original_token_span"] == [5, 11]
    assert payload["prompt_record"]["counterfactual_token_count"] == 94
    assert payload["quality_record"]["delta_quality"] == -1.0
    assert payload["failure"] is None


def test_failure_run_records_require_failure_and_reject_measured_quality():
    runtime_payload = json.loads(
        run_record_to_json_line(
            CounterfactualRunRecord(
                schema_version=SCHEMA_VERSION,
                record_status="RUNTIME_FAILURE",
                intervention=_manual_intervention(),
                prompt_record=_manual_prompt_record(),
                failure=_manual_failure(),
            )
        )
    )
    capacity_payload = json.loads(
        run_record_to_json_line(
            CounterfactualRunRecord(
                schema_version=SCHEMA_VERSION,
                record_status="CAPACITY_MISSING",
                intervention=_manual_intervention(),
                failure=_manual_failure(capacity_missing_class="oom_after_retry"),
            )
        )
    )

    assert runtime_payload["record_status"] == "RUNTIME_FAILURE"
    assert runtime_payload["prompt_record"]["original_token_count"] == 100
    assert runtime_payload["quality_record"] is None
    assert runtime_payload["failure"]["capacity_missing_class"] == "none"
    assert capacity_payload["record_status"] == "CAPACITY_MISSING"
    assert capacity_payload["prompt_record"] is None
    assert capacity_payload["failure"]["capacity_missing_class"] == "oom_after_retry"

    with pytest.raises(CounterfactualContractError, match="cannot include quality_record"):
        CounterfactualRunRecord(
            schema_version=SCHEMA_VERSION,
            record_status="RUNTIME_FAILURE",
            intervention=_manual_intervention(),
            quality_record=_manual_quality_record(),
            failure=_manual_failure(),
        )


def test_run_record_rejects_ambiguous_no_terminal_record():
    with pytest.raises(CounterfactualContractError, match="requires prompt_record"):
        CounterfactualRunRecord(
            schema_version=SCHEMA_VERSION,
            record_status="PROMPT_QUALITY_SUCCESS",
            intervention=_manual_intervention(),
        )

    with pytest.raises(CounterfactualContractError, match="requires failure"):
        CounterfactualRunRecord(
            schema_version=SCHEMA_VERSION,
            record_status="RUNTIME_FAILURE",
            intervention=_manual_intervention(),
        )

    with pytest.raises(CounterfactualContractError, match="capacity_missing_class"):
        CounterfactualRunRecord(
            schema_version=SCHEMA_VERSION,
            record_status="CAPACITY_MISSING",
            intervention=_manual_intervention(),
            failure=_manual_failure(capacity_missing_class=""),
        )


def test_success_run_record_rejects_failure_branch():
    intervention = CounterfactualSEMUIntervention(
        family="Mistral",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        model_revision="main",
        tokenizer_revision="main",
        case_id="case-1",
        checkpoint="T0_PRE_GENERATION",
        selector_arm="vorn_high",
        semu_id=2,
        protected_class="none",
        original_char_span=(26, 50),
        original_token_span=(5, 11),
        original_score=0.80,
        original_rank=2,
        deletion_mode="delete",
        random_seed=123,
    )

    with pytest.raises(CounterfactualContractError, match="cannot include failure"):
        CounterfactualRunRecord(
            schema_version=SCHEMA_VERSION,
            record_status="PROMPT_QUALITY_SUCCESS",
            intervention=intervention,
            prompt_record=_manual_prompt_record(),
            quality_record=_manual_quality_record(),
            failure=_manual_failure(),
        )
