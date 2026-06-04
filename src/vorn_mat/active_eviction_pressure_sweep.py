"""Pressure-scaling executor for vorn-active-eviction Pilot B."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Literal, Mapping, Protocol, Sequence

from .active_eviction_consumer_validation import (
    FullContextControlRecord,
    build_semus_and_scores_from_phase0,
)
from .benchmarks.common import BenchmarkCase, is_prediction_correct
from .counterfactual_intervention_runner import (
    CounterfactualContractError,
    CounterfactualPromptRecord,
    CounterfactualQualityRecord,
    RecordStatus,
    SCHEMA_VERSION,
    SEMUScore,
    SemanticUnit,
    SelectorArm,
    TargetedDropFailure,
    sha256_text,
)

PressureSweepSchemaVersion = Literal["vorn-active-eviction-pressure-sweep/v1"]
PRESSURE_SWEEP_SCHEMA_VERSION: PressureSweepSchemaVersion = (
    "vorn-active-eviction-pressure-sweep/v1"
)


class PressureSweepGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...

    def generate_rendered_prompt(self, rendered_prompt: str) -> str: ...

    def render_prompt_text_with_offsets(
        self,
        prompt: str,
    ) -> tuple[str, tuple[tuple[int, int], ...]]: ...

    def count_rendered_prompt_tokens(self, rendered_prompt: str) -> int: ...


@dataclass(frozen=True)
class PressureSweepIntervention:
    family: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    case_id: str
    checkpoint: str
    selector_arm: SelectorArm
    pressure_n: int
    selected_semu_ids: tuple[int, ...]
    selected_char_spans: tuple[tuple[int, int], ...]
    selected_token_spans: tuple[tuple[int, int], ...]
    selected_semu_scores: tuple[float, ...]
    selected_semu_ranks: tuple[int, ...]
    deletion_mode: str
    random_seed: int


@dataclass(frozen=True)
class PressureSweepRunRecord:
    schema_version: PressureSweepSchemaVersion
    record_status: RecordStatus
    intervention: PressureSweepIntervention
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
            f"unknown pressure-sweep record_status: {self.record_status}"
        )


@dataclass(frozen=True)
class PressureSweepReport:
    run_id: str
    case_id: str
    full_context: FullContextControlRecord
    records: tuple[PressureSweepRunRecord, ...]


def select_semus_for_arm(
    *,
    selector_arm: SelectorArm,
    pressure_n: int,
    semus: Sequence[SemanticUnit],
    scores: Sequence[SEMUScore],
    run_id: str,
    case_id: str,
    model_id: str,
    base_seed: int,
    reference_semus: Sequence[SemanticUnit] | None = None,
) -> tuple[SemanticUnit, ...]:
    if pressure_n < 1:
        raise CounterfactualContractError(f"pressure_n must be >= 1: {pressure_n}")

    eligible = [semu for semu in semus if not semu.is_protected]
    if len(eligible) < pressure_n:
        raise CounterfactualContractError(
            f"pressure_n={pressure_n} exceeds eligible SEMUs={len(eligible)}"
        )
    score_by_semu = {score.semu_id: score for score in scores}
    missing_scores = [semu.semu_id for semu in eligible if semu.semu_id not in score_by_semu]
    if missing_scores:
        raise CounterfactualContractError(
            f"missing score records for SEMUs: {missing_scores}"
        )

    if selector_arm == "vorn_high":
        ordered = sorted(
            eligible,
            key=lambda semu: (
                -score_by_semu[semu.semu_id].vorn_score,
                score_by_semu[semu.semu_id].vorn_rank,
                semu.semu_id,
            ),
        )
        return tuple(ordered[:pressure_n])

    if selector_arm == "vorn_low":
        ordered = sorted(
            eligible,
            key=lambda semu: (
                score_by_semu[semu.semu_id].vorn_score,
                score_by_semu[semu.semu_id].vorn_rank,
                semu.semu_id,
            ),
        )
        return tuple(ordered[:pressure_n])

    if selector_arm == "length_high":
        ordered = sorted(
            eligible,
            key=lambda semu: (-semu.token_length, semu.semu_id),
        )
        return tuple(ordered[:pressure_n])

    if selector_arm == "random_length_matched":
        if reference_semus is None:
            raise CounterfactualContractError(
                "random_length_matched requires reference_semus"
            )
        if len(reference_semus) != pressure_n:
            raise CounterfactualContractError(
                "random_length_matched reference_semus length must equal pressure_n"
            )
        return _select_length_matched_set(
            eligible=eligible,
            reference_semus=reference_semus,
            run_id=run_id,
            case_id=case_id,
            model_id=model_id,
            selector_arm=selector_arm,
            base_seed=base_seed,
            pressure_n=pressure_n,
        )

    raise CounterfactualContractError(f"unsupported pressure-sweep arm: {selector_arm}")


def render_pressure_prompt(
    *,
    rendered_prompt: str,
    semus: Sequence[SemanticUnit],
    deletion_mode: str,
    original_token_count: int | None = None,
    counterfactual_token_count: int | None = None,
) -> tuple[str, CounterfactualPromptRecord]:
    if not semus:
        raise CounterfactualContractError("pressure prompt requires at least one SEMU")
    if deletion_mode != "delete":
        raise CounterfactualContractError(
            f"pressure sweep supports deletion_mode='delete', got {deletion_mode!r}"
        )
    if any(semu.is_protected for semu in semus):
        protected = [semu.semu_id for semu in semus if semu.is_protected]
        raise CounterfactualContractError(f"cannot delete protected SEMUs: {protected}")

    ordered = sorted(semus, key=lambda semu: (semu.char_start, semu.char_end))
    _reject_overlapping_spans(ordered)
    counterfactual_prompt = rendered_prompt
    for semu in reversed(ordered):
        counterfactual_prompt = _delete_char_span(
            counterfactual_prompt,
            semu.char_start,
            semu.char_end,
        )

    return (
        counterfactual_prompt,
        CounterfactualPromptRecord(
            original_prompt_hash=sha256_text(rendered_prompt),
            counterfactual_prompt_hash=sha256_text(counterfactual_prompt),
            deletion_text_policy="delete_selected_char_spans_descending_and_collapse_single_gap_space",
            mask_text_policy="",
            original_token_count=original_token_count,
            counterfactual_token_count=counterfactual_token_count,
            sentence_id_alignment_audit={
                "semu_ids": [semu.semu_id for semu in ordered],
                "original_char_spans": [
                    [semu.char_start, semu.char_end] for semu in ordered
                ],
                "original_token_spans": [
                    [semu.token_start, semu.token_end] for semu in ordered
                ],
                "deletion_mode": deletion_mode,
                "granularity": ordered[0].granularity,
            },
        ),
    )


def run_pressure_sweep(
    *,
    run_id: str,
    family: str,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    case: BenchmarkCase,
    phase0_case: dict[str, object],
    generator: PressureSweepGenerator,
    protected_semu_ids: Sequence[int],
    selector_arms: Sequence[SelectorArm],
    pressure_ns: Sequence[int],
    base_seed: int,
    expected_selected_semu_ids: Mapping[tuple[str, int], Sequence[int]] | None = None,
    deletion_mode: str = "delete",
    hardware: str = "local",
    modal_profile: str = "local",
    cost_per_second: float = 0.0,
    reset_telemetry: Callable[[], None] | None = None,
    telemetry_snapshot: Callable[[], Mapping[str, object]] | None = None,
) -> PressureSweepReport:
    rendered_prompt, _offsets = generator.render_prompt_text_with_offsets(case.prompt)
    expected_prompt_hash = str(phase0_case["prompt_hash"])
    prompt_hash = sha256_text(rendered_prompt)
    if prompt_hash != expected_prompt_hash:
        raise ValueError(
            "rendered prompt hash mismatch: "
            f"expected {expected_prompt_hash}, observed {prompt_hash}"
        )

    semus, scores = build_semus_and_scores_from_phase0(
        phase0_case,
        protected_semu_ids=protected_semu_ids,
    )
    score_by_semu = {score.semu_id: score for score in scores}
    original_token_count = int(phase0_case["prompt_token_count"])

    selected: dict[tuple[str, int], tuple[SemanticUnit, ...]] = {}
    prompt_plans: dict[tuple[str, int], tuple[str, CounterfactualPromptRecord]] = {}
    for arm in selector_arms:
        for pressure_n in pressure_ns:
            reference_semus = selected.get(("vorn_high", pressure_n))
            selected_semus = select_semus_for_arm(
                selector_arm=arm,
                pressure_n=pressure_n,
                semus=semus,
                scores=scores,
                run_id=run_id,
                case_id=case.case_id,
                model_id=model_id,
                base_seed=base_seed,
                reference_semus=reference_semus,
            )
            key = (arm, pressure_n)
            selected[key] = selected_semus
            if expected_selected_semu_ids is not None:
                expected_ids = expected_selected_semu_ids.get(key)
                observed_ids = tuple(semu.semu_id for semu in selected_semus)
                if expected_ids is not None and tuple(expected_ids) != observed_ids:
                    raise ValueError(
                        "selected SEMU list drift before model execution: "
                        f"{arm} N={pressure_n} expected {tuple(expected_ids)}, "
                        f"observed {observed_ids}"
                    )
            prompt, prompt_record = render_pressure_prompt(
                rendered_prompt=rendered_prompt,
                semus=selected_semus,
                deletion_mode=deletion_mode,
                original_token_count=original_token_count,
                counterfactual_token_count=None,
            )
            token_count = generator.count_rendered_prompt_tokens(prompt)
            prompt_plans[key] = (
                prompt,
                CounterfactualPromptRecord(
                    original_prompt_hash=prompt_record.original_prompt_hash,
                    counterfactual_prompt_hash=prompt_record.counterfactual_prompt_hash,
                    deletion_text_policy=prompt_record.deletion_text_policy,
                    mask_text_policy=prompt_record.mask_text_policy,
                    original_token_count=prompt_record.original_token_count,
                    counterfactual_token_count=token_count,
                    sentence_id_alignment_audit=prompt_record.sentence_id_alignment_audit,
                ),
            )

    if reset_telemetry is not None:
        reset_telemetry()
    full_start = time.perf_counter()
    full_prediction = generator.generate(case.prompt)
    full_runtime = time.perf_counter() - full_start
    full_memory = _memory_fields(telemetry_snapshot)
    full_hit = is_prediction_correct(case, full_prediction)
    full_metric = 1.0 if full_hit else 0.0
    full_context = FullContextControlRecord(
        schema_version="vorn-active-eviction-consumer-control/v1",
        record_type="FULL_CONTEXT_CONTROL",
        run_id=run_id,
        family=family,
        model_id=model_id,
        case_id=case.case_id,
        prompt_hash=prompt_hash,
        prediction=full_prediction,
        primary_metric=full_metric,
        binary_hit=full_hit,
        runtime_seconds=full_runtime,
        estimated_cost_usd=full_runtime * cost_per_second,
        peak_memory_allocated_mb=full_memory[0],
        peak_memory_reserved_mb=full_memory[1],
    )

    records: list[PressureSweepRunRecord] = []
    for arm in selector_arms:
        for pressure_n in pressure_ns:
            key = (arm, pressure_n)
            selected_semus = selected[key]
            counterfactual_prompt, prompt_record = prompt_plans[key]
            intervention = _build_pressure_intervention(
                family=family,
                model_id=model_id,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                case_id=case.case_id,
                selector_arm=arm,
                pressure_n=pressure_n,
                selected_semus=selected_semus,
                score_by_semu=score_by_semu,
                deletion_mode=deletion_mode,
                run_id=run_id,
                base_seed=base_seed,
            )
            try:
                if reset_telemetry is not None:
                    reset_telemetry()
                start = time.perf_counter()
                counterfactual_prediction = generator.generate_rendered_prompt(
                    counterfactual_prompt
                )
                runtime = time.perf_counter() - start
                memory_fields = _memory_fields(telemetry_snapshot)
            except Exception as exc:  # pragma: no cover - exercised by fake tests
                records.append(
                    _failure_record(
                        intervention=intervention,
                        prompt_record=prompt_record,
                        exc=exc,
                        hardware=hardware,
                        modal_profile=modal_profile,
                    )
                )
                continue

            counterfactual_hit = is_prediction_correct(case, counterfactual_prediction)
            counterfactual_metric = 1.0 if counterfactual_hit else 0.0
            records.append(
                PressureSweepRunRecord(
                    schema_version=PRESSURE_SWEEP_SCHEMA_VERSION,
                    record_status="PROMPT_QUALITY_SUCCESS",
                    intervention=intervention,
                    prompt_record=prompt_record,
                    quality_record=CounterfactualQualityRecord(
                        full_context_prediction=full_prediction,
                        counterfactual_prediction=counterfactual_prediction,
                        primary_metric=counterfactual_metric,
                        binary_hit=counterfactual_hit,
                        delta_quality=full_metric - counterfactual_metric,
                        runtime_seconds=runtime,
                        estimated_cost_usd=runtime * cost_per_second,
                        peak_memory_allocated_mb=memory_fields[0],
                        peak_memory_reserved_mb=memory_fields[1],
                    ),
                )
            )

    return PressureSweepReport(
        run_id=run_id,
        case_id=case.case_id,
        full_context=full_context,
        records=tuple(records),
    )


def write_pressure_sweep_artifacts(
    report: PressureSweepReport,
    *,
    jsonl_path: Path,
    md_path: Path,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w") as handle:
        handle.write(json.dumps(asdict(report.full_context), sort_keys=True))
        handle.write("\n")
        for record in report.records:
            handle.write(json.dumps(asdict(record), sort_keys=True))
            handle.write("\n")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_pressure_sweep_summary(report))


def load_pressure_sweep_delta_table(jsonl_path: Path) -> dict[str, object]:
    payloads = [
        json.loads(line)
        for line in jsonl_path.read_text().splitlines()
        if line.strip()
    ]
    if not payloads:
        raise ValueError(f"empty pressure-sweep artifact: {jsonl_path}")

    full_context = payloads[0]
    if full_context.get("record_type") != "FULL_CONTEXT_CONTROL":
        raise ValueError("first pressure-sweep JSONL row must be FULL_CONTEXT_CONTROL")

    status_counts: dict[str, int] = {}
    successful_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    for payload in payloads[1:]:
        status = str(payload["record_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        intervention = dict(payload["intervention"])
        quality = payload.get("quality_record")
        if quality is not None:
            successful_rows.append(
                {
                    "selector_arm": intervention["selector_arm"],
                    "pressure_n": intervention["pressure_n"],
                    "selected_semu_ids": intervention["selected_semu_ids"],
                    "selected_semu_ranks": intervention["selected_semu_ranks"],
                    "selected_semu_scores": intervention["selected_semu_scores"],
                    "deletion_mode": intervention["deletion_mode"],
                    "full_context_hit": full_context["binary_hit"],
                    "counterfactual_hit": quality["binary_hit"],
                    "delta_quality": quality["delta_quality"],
                    "runtime_seconds": quality["runtime_seconds"],
                    "estimated_cost_usd": quality["estimated_cost_usd"],
                    "peak_memory_allocated_mb": quality.get(
                        "peak_memory_allocated_mb"
                    ),
                    "peak_memory_reserved_mb": quality.get(
                        "peak_memory_reserved_mb"
                    ),
                }
            )
            continue

        failure = payload.get("failure")
        if failure is None:
            raise ValueError("pressure-sweep row has no quality_record or failure")
        failure_rows.append(
            {
                "selector_arm": intervention["selector_arm"],
                "pressure_n": intervention["pressure_n"],
                "selected_semu_ids": intervention["selected_semu_ids"],
                "record_status": status,
                "failure_reason": failure["failure_reason"],
                "capacity_missing_class": failure["capacity_missing_class"],
            }
        )

    return {
        "full_context": full_context,
        "row_count": len(payloads) - 1,
        "status_counts": status_counts,
        "successful_rows": successful_rows,
        "failure_rows": failure_rows,
    }


def render_pressure_sweep_summary(report: PressureSweepReport) -> str:
    status_counts: dict[str, int] = {}
    rows = []
    for record in report.records:
        status_counts[record.record_status] = (
            status_counts.get(record.record_status, 0) + 1
        )
        if record.quality_record is None:
            continue
        rows.append(
            "| {arm} | {pressure_n} | {semu_ids} | {hit} | {delta:.3f} |".format(
                arm=record.intervention.selector_arm,
                pressure_n=record.intervention.pressure_n,
                semu_ids=",".join(str(i) for i in record.intervention.selected_semu_ids),
                hit=record.quality_record.binary_hit,
                delta=record.quality_record.delta_quality,
            )
        )

    lines = [
        "# Vorn-Active Eviction Pressure Sweep",
        "",
        f"- Run id: `{report.run_id}`",
        f"- Case id: `{report.case_id}`",
        f"- Full-context hit: `{report.full_context.binary_hit}`",
        f"- Record status counts: `{json.dumps(status_counts, sort_keys=True)}`",
        "",
        "| Arm | Pressure N | SEMU ids | Counterfactual hit | Delta quality |",
        "|---|---:|---|---|---:|",
    ]
    lines.extend(rows or ["| n/a | n/a | n/a | n/a | n/a |"])
    lines.append("")
    return "\n".join(lines)


def _select_length_matched_set(
    *,
    eligible: Sequence[SemanticUnit],
    reference_semus: Sequence[SemanticUnit],
    run_id: str,
    case_id: str,
    model_id: str,
    selector_arm: str,
    base_seed: int,
    pressure_n: int,
) -> tuple[SemanticUnit, ...]:
    reference_ids = {semu.semu_id for semu in reference_semus}
    remaining = [semu for semu in eligible if semu.semu_id not in reference_ids]
    if len(remaining) < pressure_n:
        raise CounterfactualContractError(
            "random_length_matched lacks enough non-reference eligible SEMUs"
        )

    selected: list[SemanticUnit] = []
    for index, reference_semu in enumerate(reference_semus):
        min_distance = min(
            abs(semu.token_length - reference_semu.token_length)
            for semu in remaining
        )
        candidates = [
            semu
            for semu in remaining
            if abs(semu.token_length - reference_semu.token_length) == min_distance
        ]
        candidates = sorted(candidates, key=lambda semu: semu.semu_id)
        # The first length-matched draw is anchored directly to the locked
        # Pilot B seed so N=1 remains regression-compatible with the #36 smoke
        # cell. Subsequent draws add context to avoid repeating the same tie
        # break across the whole pressure set.
        seed = (
            base_seed
            if index == 0
            else _pressure_seed(
                run_id=run_id,
                case_id=case_id,
                selector_arm=selector_arm,
                model_id=model_id,
                base_seed=base_seed,
                pressure_n=pressure_n,
                index=index,
            )
        )
        chosen = candidates[seed % len(candidates)]
        selected.append(chosen)
        remaining = [semu for semu in remaining if semu.semu_id != chosen.semu_id]
    return tuple(selected)


def _build_pressure_intervention(
    *,
    family: str,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    case_id: str,
    selector_arm: SelectorArm,
    pressure_n: int,
    selected_semus: Sequence[SemanticUnit],
    score_by_semu: Mapping[int, SEMUScore],
    deletion_mode: str,
    run_id: str,
    base_seed: int,
) -> PressureSweepIntervention:
    scores = [score_by_semu[semu.semu_id] for semu in selected_semus]
    return PressureSweepIntervention(
        family=family,
        model_id=model_id,
        model_revision=model_revision,
        tokenizer_revision=tokenizer_revision,
        case_id=case_id,
        checkpoint="T0_PRE_GENERATION",
        selector_arm=selector_arm,
        pressure_n=pressure_n,
        selected_semu_ids=tuple(semu.semu_id for semu in selected_semus),
        selected_char_spans=tuple(
            (semu.char_start, semu.char_end) for semu in selected_semus
        ),
        selected_token_spans=tuple(
            (semu.token_start, semu.token_end) for semu in selected_semus
        ),
        selected_semu_scores=tuple(score.vorn_score for score in scores),
        selected_semu_ranks=tuple(score.vorn_rank for score in scores),
        deletion_mode=deletion_mode,
        random_seed=_pressure_seed(
            run_id=run_id,
            case_id=case_id,
            selector_arm=selector_arm,
            model_id=model_id,
            base_seed=base_seed,
            pressure_n=pressure_n,
            index=0,
        ),
    )


def _failure_record(
    *,
    intervention: PressureSweepIntervention,
    prompt_record: CounterfactualPromptRecord,
    exc: Exception,
    hardware: str,
    modal_profile: str,
) -> PressureSweepRunRecord:
    message = str(exc)
    is_capacity_missing = "out of memory" in message.lower() or "oom" in message.lower()
    status = "CAPACITY_MISSING" if is_capacity_missing else "RUNTIME_FAILURE"
    return PressureSweepRunRecord(
        schema_version=PRESSURE_SWEEP_SCHEMA_VERSION,
        record_status=status,
        intervention=intervention,
        prompt_record=prompt_record,
        failure=TargetedDropFailure(
            failure_reason=type(exc).__name__,
            failure_stage="counterfactual_generation",
            hardware=hardware,
            modal_profile=modal_profile,
            capacity_missing_class="oom_after_retry" if is_capacity_missing else "none",
            retry_count=0,
            artifact_partial_path="",
        ),
    )


def _reject_overlapping_spans(ordered_semus: Sequence[SemanticUnit]) -> None:
    for left, right in zip(ordered_semus, ordered_semus[1:]):
        if left.char_end > right.char_start:
            raise ValueError(
                "overlapping SEMU spans: "
                f"{left.semu_id}=({left.char_start},{left.char_end}) "
                f"{right.semu_id}=({right.char_start},{right.char_end})"
            )


def _delete_char_span(text: str, start: int, end: int) -> str:
    left = text[:start]
    right = text[end:]
    if left.endswith(" ") and right.startswith(" "):
        right = right[1:]
    return left + right


def _pressure_seed(
    *,
    run_id: str,
    case_id: str,
    selector_arm: str,
    model_id: str,
    base_seed: int,
    pressure_n: int,
    index: int,
) -> int:
    digest = hashlib.sha256(
        (
            f"{base_seed}|{run_id}|{case_id}|{selector_arm}|"
            f"{model_id}|{pressure_n}|{index}"
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _memory_fields(
    telemetry_snapshot: Callable[[], Mapping[str, object]] | None,
) -> tuple[float | None, float | None]:
    if telemetry_snapshot is None:
        return None, None
    snapshot = telemetry_snapshot()
    allocated_gb = snapshot.get("peak_memory_allocated_gb")
    reserved_gb = snapshot.get("peak_memory_reserved_gb")
    allocated_mb = (
        float(allocated_gb) * 1024 if allocated_gb is not None else None
    )
    reserved_mb = float(reserved_gb) * 1024 if reserved_gb is not None else None
    return allocated_mb, reserved_mb
