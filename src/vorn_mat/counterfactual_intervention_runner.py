"""Counterfactual SEMU-intervention contract for active-vorn Pilot A.

This module is the narrow substrate between Phase 0 SEMU ranking and the
future model-execution runner. It intentionally does not launch Modal jobs or
call a model. Instead, it defines the deterministic selection, prompt-rendering,
and artifact-record contracts that a measured runner must satisfy before Pilot
A cells are authorized.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Literal, Sequence

from .text_spans import sentence_char_spans, token_span_from_offsets

SchemaVersion = Literal["vorn-active-eviction-counterfactual/v1"]
RecordStatus = Literal[
    "PROMPT_QUALITY_SUCCESS",
    "RUNTIME_FAILURE",
    "CAPACITY_MISSING",
]
SEMUGranularity = Literal["sentence"]
SelectorArm = Literal[
    "vorn_high",
    "vorn_low",
    "attention_high",
    "snapkv_high",
    "recency_high",
    "length_high",
    "random_length_matched",
]
Checkpoint = Literal["T0_PRE_GENERATION", "T1_CONTINUATION"]
DeletionMode = Literal["delete", "mask"]

SCHEMA_VERSION: SchemaVersion = "vorn-active-eviction-counterfactual/v1"
MASK_TOKEN = "[MASKED_SEMU]"


class CounterfactualContractError(ValueError):
    """Raised when the counterfactual contract would otherwise be ambiguous."""


@dataclass(frozen=True)
class SemanticUnit:
    semu_id: int
    granularity: SEMUGranularity
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    text: str
    protected_classes: tuple[str, ...] = ()
    answer_overlap: bool = False

    @property
    def token_length(self) -> int:
        return self.token_end - self.token_start

    @property
    def is_protected(self) -> bool:
        return bool(self.protected_classes)


@dataclass(frozen=True)
class SEMUScore:
    semu_id: int
    vorn_score: float
    vorn_rank: int
    attention_score: float | None = None
    snapkv_score: float | None = None
    snapkv_rank: int | None = None


@dataclass(frozen=True)
class CounterfactualSEMUIntervention:
    family: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    case_id: str
    checkpoint: Checkpoint
    selector_arm: SelectorArm
    semu_id: int
    protected_class: str
    original_char_span: tuple[int, int]
    original_token_span: tuple[int, int]
    original_score: float
    original_rank: int
    deletion_mode: DeletionMode
    random_seed: int


@dataclass(frozen=True)
class CounterfactualPromptRecord:
    original_prompt_hash: str
    counterfactual_prompt_hash: str
    deletion_text_policy: str
    mask_text_policy: str
    original_token_count: int | None
    counterfactual_token_count: int | None
    sentence_id_alignment_audit: dict[str, object]


@dataclass(frozen=True)
class CounterfactualQualityRecord:
    full_context_prediction: str
    counterfactual_prediction: str
    primary_metric: float
    binary_hit: bool
    delta_quality: float
    runtime_seconds: float
    estimated_cost_usd: float
    peak_memory_allocated_mb: float | None = None
    peak_memory_reserved_mb: float | None = None


@dataclass(frozen=True)
class TargetedDropFailure:
    failure_reason: str
    failure_stage: str
    hardware: str
    modal_profile: str
    capacity_missing_class: str
    retry_count: int
    artifact_partial_path: str


@dataclass(frozen=True)
class CounterfactualRunRecord:
    schema_version: SchemaVersion
    record_status: RecordStatus
    intervention: CounterfactualSEMUIntervention
    prompt_record: CounterfactualPromptRecord | None = None
    quality_record: CounterfactualQualityRecord | None = None
    failure: TargetedDropFailure | None = None

    def __post_init__(self) -> None:
        if self.record_status == "PROMPT_QUALITY_SUCCESS":
            if self.prompt_record is None or self.quality_record is None:
                raise CounterfactualContractError(
                    "PROMPT_QUALITY_SUCCESS requires prompt_record and quality_record"
                )
            if self.failure is not None:
                raise CounterfactualContractError(
                    "PROMPT_QUALITY_SUCCESS cannot include failure"
                )
            return

        if self.record_status in {"RUNTIME_FAILURE", "CAPACITY_MISSING"}:
            if self.failure is None:
                raise CounterfactualContractError(
                    f"{self.record_status} requires failure"
                )
            if self.quality_record is not None:
                raise CounterfactualContractError(
                    f"{self.record_status} cannot include quality_record"
                )
            if (
                self.record_status == "CAPACITY_MISSING"
                and not self.failure.capacity_missing_class
            ):
                raise CounterfactualContractError(
                    "CAPACITY_MISSING requires capacity_missing_class"
                )
            return

        raise CounterfactualContractError(
            f"unknown counterfactual record_status: {self.record_status}"
        )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deterministic_seed(
    *,
    run_id: str,
    case_id: str,
    selector_arm: SelectorArm,
    model_id: str,
) -> int:
    digest = hashlib.sha256(
        f"{run_id}|{case_id}|{selector_arm}|{model_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def build_sentence_semus(
    *,
    rendered_prompt: str,
    offsets: Sequence[tuple[int, int]],
    protected_char_spans: Sequence[tuple[int, int]] = (),
    answer_token_spans: Sequence[tuple[int, int]] = (),
) -> tuple[SemanticUnit, ...]:
    semus: list[SemanticUnit] = []
    for semu_id, (char_start, char_end) in enumerate(
        sentence_char_spans(rendered_prompt)
    ):
        token_span = token_span_from_offsets(
            offsets,
            char_start=char_start,
            char_end=char_end,
        )
        if token_span is None:
            continue
        token_start, token_end = token_span
        protected = (
            ("protected_prompt_region",)
            if any(
                _overlaps((char_start, char_end), span)
                for span in protected_char_spans
            )
            else ()
        )
        semus.append(
            SemanticUnit(
                semu_id=len(semus),
                granularity="sentence",
                char_start=char_start,
                char_end=char_end,
                token_start=token_start,
                token_end=token_end,
                text=rendered_prompt[char_start:char_end],
                protected_classes=protected,
                answer_overlap=any(
                    _overlaps((token_start, token_end), span)
                    for span in answer_token_spans
                ),
            )
        )
    return tuple(semus)


def select_semu_for_arm(
    *,
    selector_arm: SelectorArm,
    semus: Sequence[SemanticUnit],
    scores: Sequence[SEMUScore],
    run_id: str,
    case_id: str,
    model_id: str,
    reference_semu: SemanticUnit | None = None,
) -> SemanticUnit:
    eligible = [semu for semu in semus if not semu.is_protected]
    if not eligible:
        raise CounterfactualContractError("no eligible SEMUs after protected filters")
    score_by_semu = {score.semu_id: score for score in scores}
    missing_scores = [semu.semu_id for semu in eligible if semu.semu_id not in score_by_semu]
    if missing_scores:
        raise CounterfactualContractError(
            f"missing score records for SEMUs: {missing_scores}"
        )

    if selector_arm == "vorn_high":
        return max(
            eligible,
            key=lambda semu: (
                score_by_semu[semu.semu_id].vorn_score,
                -score_by_semu[semu.semu_id].vorn_rank,
                -semu.semu_id,
            ),
        )
    if selector_arm == "vorn_low":
        return min(
            eligible,
            key=lambda semu: (
                score_by_semu[semu.semu_id].vorn_score,
                score_by_semu[semu.semu_id].vorn_rank,
                semu.semu_id,
            ),
        )
    if selector_arm == "attention_high":
        attention_eligible = [
            semu
            for semu in eligible
            if score_by_semu[semu.semu_id].attention_score is not None
        ]
        if not attention_eligible:
            raise CounterfactualContractError("attention scores are capacity-missing")
        return max(
            attention_eligible,
            key=lambda semu: (
                score_by_semu[semu.semu_id].attention_score,
                -semu.semu_id,
            ),
        )
    if selector_arm == "snapkv_high":
        snapkv_eligible = [
            semu
            for semu in eligible
            if score_by_semu[semu.semu_id].snapkv_score is not None
            and score_by_semu[semu.semu_id].snapkv_rank is not None
        ]
        if not snapkv_eligible:
            raise CounterfactualContractError("SnapKV scores are capacity-missing")
        return max(
            snapkv_eligible,
            key=lambda semu: (
                score_by_semu[semu.semu_id].snapkv_score,
                -(score_by_semu[semu.semu_id].snapkv_rank or 0),
                -semu.semu_id,
            ),
        )
    if selector_arm == "recency_high":
        return max(eligible, key=lambda semu: (semu.char_start, -semu.semu_id))
    if selector_arm == "length_high":
        return max(eligible, key=lambda semu: (semu.token_length, -semu.semu_id))
    if selector_arm == "random_length_matched":
        if reference_semu is None:
            raise CounterfactualContractError(
                "random_length_matched requires reference_semu"
            )
        return _deterministic_length_matched_choice(
            eligible=eligible,
            reference_semu=reference_semu,
            seed=deterministic_seed(
                run_id=run_id,
                case_id=case_id,
                selector_arm=selector_arm,
                model_id=model_id,
            ),
        )
    raise CounterfactualContractError(f"unknown selector arm: {selector_arm}")


def build_intervention(
    *,
    family: str,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    case_id: str,
    checkpoint: Checkpoint,
    selector_arm: SelectorArm,
    semu: SemanticUnit,
    score: SEMUScore,
    deletion_mode: DeletionMode,
    run_id: str,
) -> CounterfactualSEMUIntervention:
    if semu.is_protected:
        raise CounterfactualContractError(
            f"protected SEMU selected: {semu.semu_id} {semu.protected_classes}"
        )
    original_score, original_rank = selector_score_and_rank(selector_arm, score)
    return CounterfactualSEMUIntervention(
        family=family,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        case_id=case_id,
        checkpoint=checkpoint,
        selector_arm=selector_arm,
        semu_id=semu.semu_id,
        protected_class="none",
        original_char_span=(semu.char_start, semu.char_end),
        original_token_span=(semu.token_start, semu.token_end),
        original_score=original_score,
        original_rank=original_rank,
        deletion_mode=deletion_mode,
        random_seed=deterministic_seed(
            run_id=run_id,
            case_id=case_id,
            selector_arm=selector_arm,
            model_id=model_id,
        ),
    )


def selector_score_and_rank(
    selector_arm: SelectorArm,
    score: SEMUScore,
) -> tuple[float, int]:
    if selector_arm == "snapkv_high":
        if score.snapkv_score is None or score.snapkv_rank is None:
            raise CounterfactualContractError("SnapKV scores are capacity-missing")
        return float(score.snapkv_score), int(score.snapkv_rank)
    return float(score.vorn_score), int(score.vorn_rank)


def render_counterfactual_prompt(
    *,
    rendered_prompt: str,
    semu: SemanticUnit,
    deletion_mode: DeletionMode,
    original_token_count: int | None = None,
    counterfactual_token_count: int | None = None,
) -> tuple[str, CounterfactualPromptRecord]:
    if semu.is_protected:
        raise CounterfactualContractError(
            f"cannot render protected SEMU intervention: {semu.semu_id}"
        )
    if deletion_mode == "delete":
        counterfactual_prompt = _delete_char_span(
            rendered_prompt,
            semu.char_start,
            semu.char_end,
        )
        deletion_policy = "delete_original_char_span_and_collapse_single_gap_space"
        mask_policy = ""
    elif deletion_mode == "mask":
        mask_text = _mask_text_for_semu(semu)
        counterfactual_prompt = (
            rendered_prompt[: semu.char_start]
            + mask_text
            + rendered_prompt[semu.char_end :]
        )
        deletion_policy = "replace_original_char_span_with_neutral_mask"
        mask_policy = "repeat_MASKED_SEMU_token_to_original_semu_token_length"
    else:
        raise CounterfactualContractError(f"unknown deletion mode: {deletion_mode}")

    return (
        counterfactual_prompt,
        CounterfactualPromptRecord(
            original_prompt_hash=sha256_text(rendered_prompt),
            counterfactual_prompt_hash=sha256_text(counterfactual_prompt),
            deletion_text_policy=deletion_policy,
            mask_text_policy=mask_policy,
            original_token_count=original_token_count,
            counterfactual_token_count=counterfactual_token_count,
            sentence_id_alignment_audit={
                "semu_id": semu.semu_id,
                "original_char_span": [semu.char_start, semu.char_end],
                "original_token_span": [semu.token_start, semu.token_end],
                "deletion_mode": deletion_mode,
                "granularity": semu.granularity,
            },
        ),
    )


def run_record_to_json_payload(record: CounterfactualRunRecord) -> dict[str, Any]:
    return asdict(record)


def run_record_to_json_line(record: CounterfactualRunRecord) -> str:
    return json.dumps(run_record_to_json_payload(record), sort_keys=True)


def _deterministic_length_matched_choice(
    *,
    eligible: Sequence[SemanticUnit],
    reference_semu: SemanticUnit,
    seed: int,
) -> SemanticUnit:
    alternates = [semu for semu in eligible if semu.semu_id != reference_semu.semu_id]
    if not alternates:
        raise CounterfactualContractError(
            "random_length_matched requires an eligible non-reference SEMU"
        )
    min_distance = min(
        abs(semu.token_length - reference_semu.token_length)
        for semu in alternates
    )
    candidates = [
        semu
        for semu in alternates
        if abs(semu.token_length - reference_semu.token_length) == min_distance
    ]
    candidates = sorted(candidates, key=lambda semu: semu.semu_id)
    return candidates[seed % len(candidates)]


def _delete_char_span(text: str, start: int, end: int) -> str:
    left = text[:start]
    right = text[end:]
    if left.endswith(" ") and right.startswith(" "):
        right = right[1:]
    return left + right


def _mask_text_for_semu(semu: SemanticUnit) -> str:
    return " ".join(MASK_TOKEN for _ in range(max(1, semu.token_length)))


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[1] > right[0] and right[1] > left[0]
