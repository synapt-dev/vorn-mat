"""Consumer-validation executor for vorn-active-eviction Pilot A."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Callable, Mapping, Protocol, Sequence

from .benchmarks.common import BenchmarkCase, is_prediction_correct
from .counterfactual_intervention_runner import (
    MASK_TOKEN,
    SCHEMA_VERSION,
    CounterfactualPromptRecord,
    CounterfactualQualityRecord,
    CounterfactualRunRecord,
    CounterfactualSEMUIntervention,
    SEMUScore,
    SemanticUnit,
    TargetedDropFailure,
    build_intervention,
    render_counterfactual_prompt,
    run_record_to_json_payload,
    select_semu_for_arm,
    sha256_text,
)


class ConsumerValidationGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...

    def generate_rendered_prompt(self, rendered_prompt: str) -> str: ...

    def render_prompt_text_with_offsets(
        self,
        prompt: str,
    ) -> tuple[str, tuple[tuple[int, int], ...]]: ...

    def count_rendered_prompt_tokens(self, rendered_prompt: str) -> int: ...


@dataclass(frozen=True)
class FullContextControlRecord:
    schema_version: str
    record_type: str
    run_id: str
    family: str
    model_id: str
    case_id: str
    prompt_hash: str
    prediction: str
    primary_metric: float
    binary_hit: bool
    runtime_seconds: float
    estimated_cost_usd: float
    peak_memory_allocated_mb: float | None = None
    peak_memory_reserved_mb: float | None = None


@dataclass(frozen=True)
class ConsumerValidationReport:
    run_id: str
    case_id: str
    full_context: FullContextControlRecord
    records: tuple[CounterfactualRunRecord, ...]
    mask_prompt_records: tuple[CounterfactualPromptRecord, ...]


def load_phase0_case(path: Path, case_id: str) -> dict[str, object]:
    payload = json.loads(path.read_text())
    for case in payload["cases"]:
        if case["case_id"] == case_id:
            return dict(case)
    raise ValueError(f"case_id not found in Phase 0 artifact: {case_id}")


def build_semus_and_scores_from_phase0(
    phase0_case: dict[str, object],
    *,
    protected_semu_ids: Sequence[int],
) -> tuple[tuple[SemanticUnit, ...], tuple[SEMUScore, ...]]:
    protected = set(protected_semu_ids)
    semus: list[SemanticUnit] = []
    scores: list[SEMUScore] = []
    for raw in phase0_case["semu_matrix"]:
        semu_payload = dict(raw)
        semu_id = int(semu_payload["semu_id"])
        protected_classes: tuple[str, ...] = ()
        if semu_id in protected:
            protected_label = (
                "expected_answer_overlap"
                if bool(semu_payload.get("answer_overlap"))
                else "protected_prompt_region"
            )
            protected_classes = (protected_label,)
        semus.append(
            SemanticUnit(
                semu_id=semu_id,
                granularity="sentence",
                char_start=int(semu_payload["char_start"]),
                char_end=int(semu_payload["char_end"]),
                token_start=int(semu_payload["token_start"]),
                token_end=int(semu_payload["token_end"]),
                text=str(semu_payload.get("text_preview", "")),
                protected_classes=protected_classes,
                answer_overlap=bool(semu_payload.get("answer_overlap")),
            )
        )
        scores.append(
            SEMUScore(
                semu_id=semu_id,
                vorn_score=float(semu_payload["final_score"]),
                vorn_rank=int(semu_payload["final_rank"]),
                snapkv_score=(
                    float(semu_payload["snapkv_score"])
                    if semu_payload.get("snapkv_score") is not None
                    else None
                ),
                snapkv_rank=(
                    int(semu_payload["snapkv_rank"])
                    if semu_payload.get("snapkv_rank") is not None
                    else None
                ),
            )
        )
    return tuple(semus), tuple(scores)


def run_consumer_validation(
    *,
    run_id: str,
    family: str,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    case: BenchmarkCase,
    phase0_case: dict[str, object],
    generator: ConsumerValidationGenerator,
    protected_semu_ids: Sequence[int],
    selector_arms: Sequence[str],
    expected_selected_semu_ids: Mapping[str, int] | None = None,
    deletion_mode: str = "delete",
    mask_dry_run_arms: Sequence[str] = ("vorn_high", "vorn_low"),
    hardware: str = "local",
    modal_profile: str = "local",
    cost_per_second: float = 0.0,
    reset_telemetry: Callable[[], None] | None = None,
    telemetry_snapshot: Callable[[], Mapping[str, object]] | None = None,
) -> ConsumerValidationReport:
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
    selected: dict[str, SemanticUnit] = {}

    for arm in selector_arms:
        reference_semu = selected.get("vorn_high")
        semu = select_semu_for_arm(
            selector_arm=arm,
            semus=semus,
            scores=scores,
            run_id=run_id,
            case_id=case.case_id,
            model_id=model_id,
            reference_semu=reference_semu,
        )
        selected[arm] = semu
        if expected_selected_semu_ids is not None:
            expected_semu_id = expected_selected_semu_ids.get(arm)
            if expected_semu_id is not None and semu.semu_id != expected_semu_id:
                raise ValueError(
                    "selected SEMU drift before model execution: "
                    f"{arm} expected {expected_semu_id}, observed {semu.semu_id}"
                )

    original_token_count = int(phase0_case["prompt_token_count"])
    mask_prompt_records = _build_mask_dry_run_records(
        generator=generator,
        rendered_prompt=rendered_prompt,
        selected=selected,
        mask_dry_run_arms=mask_dry_run_arms,
        original_token_count=original_token_count,
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

    records: list[CounterfactualRunRecord] = []

    for arm in selector_arms:
        semu = selected[arm]
        score = score_by_semu[semu.semu_id]
        intervention = build_intervention(
            family=family,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            case_id=case.case_id,
            checkpoint="T0_PRE_GENERATION",
            selector_arm=arm,
            semu=semu,
            score=score,
            deletion_mode=deletion_mode,
            run_id=run_id,
        )
        counterfactual_prompt, prompt_record = _render_prompt_with_count(
            generator=generator,
            rendered_prompt=rendered_prompt,
            semu=semu,
            deletion_mode=deletion_mode,
            original_token_count=original_token_count,
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
            CounterfactualRunRecord(
                schema_version=SCHEMA_VERSION,
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

    return ConsumerValidationReport(
        run_id=run_id,
        case_id=case.case_id,
        full_context=full_context,
        records=tuple(records),
        mask_prompt_records=mask_prompt_records,
    )


def write_consumer_validation_artifacts(
    report: ConsumerValidationReport,
    *,
    jsonl_path: Path,
    md_path: Path,
) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w") as handle:
        handle.write(json.dumps(asdict(report.full_context), sort_keys=True))
        handle.write("\n")
        for record in report.records:
            handle.write(json.dumps(run_record_to_json_payload(record), sort_keys=True))
            handle.write("\n")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_consumer_validation_summary(report))


def load_consumer_validation_delta_table(jsonl_path: Path) -> dict[str, object]:
    """Read the emitted JSONL artifact into the consumer-validation analysis shape."""
    payloads = [
        json.loads(line)
        for line in jsonl_path.read_text().splitlines()
        if line.strip()
    ]
    if not payloads:
        raise ValueError(f"empty consumer-validation artifact: {jsonl_path}")

    full_context = payloads[0]
    if full_context.get("record_type") != "FULL_CONTEXT_CONTROL":
        raise ValueError(
            "first consumer-validation JSONL row must be FULL_CONTEXT_CONTROL"
        )

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
                    "semu_id": intervention["semu_id"],
                    "original_rank": intervention["original_rank"],
                    "original_score": intervention["original_score"],
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
            raise ValueError(
                "counterfactual artifact row has no quality_record or failure"
            )
        failure_rows.append(
            {
                "selector_arm": intervention["selector_arm"],
                "semu_id": intervention["semu_id"],
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


def render_consumer_validation_summary(report: ConsumerValidationReport) -> str:
    status_counts: dict[str, int] = {}
    successful_rows = []
    for record in report.records:
        status_counts[record.record_status] = (
            status_counts.get(record.record_status, 0) + 1
        )
        if record.quality_record is None:
            continue
        successful_rows.append(
            "| {arm} | {semu_id} | {rank} | {score:.6f} | {hit} | {delta:.3f} |".format(
                arm=record.intervention.selector_arm,
                semu_id=record.intervention.semu_id,
                rank=record.intervention.original_rank,
                score=record.intervention.original_score,
                hit=record.quality_record.binary_hit,
                delta=record.quality_record.delta_quality,
            )
        )

    lines = [
        "# Vorn-Active Eviction Consumer Validation",
        "",
        f"- Run id: `{report.run_id}`",
        f"- Case id: `{report.case_id}`",
        f"- Full-context hit: `{report.full_context.binary_hit}`",
        f"- Record status counts: `{json.dumps(status_counts, sort_keys=True)}`",
        f"- Mask dry-run records: `{len(report.mask_prompt_records)}`",
        "",
        "| Arm | SEMU id | Rank | Score | Counterfactual hit | Delta quality |",
        "|---|---:|---:|---:|---|---:|",
    ]
    lines.extend(successful_rows or ["| n/a | n/a | n/a | n/a | n/a | n/a |"])
    lines.append("")
    return "\n".join(lines)


def _build_mask_dry_run_records(
    *,
    generator: ConsumerValidationGenerator,
    rendered_prompt: str,
    selected: Mapping[str, SemanticUnit],
    mask_dry_run_arms: Sequence[str],
    original_token_count: int,
) -> tuple[CounterfactualPromptRecord, ...]:
    records: list[CounterfactualPromptRecord] = []
    for arm in mask_dry_run_arms:
        if arm not in selected:
            continue
        semu = selected[arm]
        mask_prompt, prompt_record = _render_prompt_with_count(
            generator=generator,
            rendered_prompt=rendered_prompt,
            semu=semu,
            deletion_mode="mask",
            original_token_count=original_token_count,
        )
        _validate_mask_dry_run(
            mask_prompt=mask_prompt,
            prompt_record=prompt_record,
            semu=semu,
        )
        records.append(prompt_record)
    return tuple(records)


def _validate_mask_dry_run(
    *,
    mask_prompt: str,
    prompt_record: CounterfactualPromptRecord,
    semu: SemanticUnit,
) -> None:
    if prompt_record.original_prompt_hash == prompt_record.counterfactual_prompt_hash:
        raise ValueError("mask dry-run did not change prompt hash")
    if prompt_record.mask_text_policy != (
        "repeat_MASKED_SEMU_token_to_original_semu_token_length"
    ):
        raise ValueError(
            "mask dry-run used unexpected mask_text_policy: "
            f"{prompt_record.mask_text_policy}"
        )

    audit = prompt_record.sentence_id_alignment_audit
    expected_char_span = [semu.char_start, semu.char_end]
    expected_token_span = [semu.token_start, semu.token_end]
    if (
        audit.get("semu_id") != semu.semu_id
        or audit.get("original_char_span") != expected_char_span
        or audit.get("original_token_span") != expected_token_span
        or audit.get("deletion_mode") != "mask"
        or audit.get("granularity") != semu.granularity
    ):
        raise ValueError(
            "mask dry-run corrupted sentence-id alignment audit for "
            f"SEMU {semu.semu_id}"
        )

    mask_count = mask_prompt.count(MASK_TOKEN)
    if mask_count != semu.token_length:
        raise ValueError(
            "mask dry-run token-count mismatch: "
            f"expected {semu.token_length} {MASK_TOKEN} tokens, observed {mask_count}"
        )


def _render_prompt_with_count(
    *,
    generator: ConsumerValidationGenerator,
    rendered_prompt: str,
    semu: SemanticUnit,
    deletion_mode: str,
    original_token_count: int,
) -> tuple[str, CounterfactualPromptRecord]:
    counterfactual_prompt, prompt_record = render_counterfactual_prompt(
        rendered_prompt=rendered_prompt,
        semu=semu,
        deletion_mode=deletion_mode,
        original_token_count=original_token_count,
        counterfactual_token_count=None,
    )
    token_count = generator.count_rendered_prompt_tokens(counterfactual_prompt)
    prompt_record = CounterfactualPromptRecord(
        original_prompt_hash=prompt_record.original_prompt_hash,
        counterfactual_prompt_hash=prompt_record.counterfactual_prompt_hash,
        deletion_text_policy=prompt_record.deletion_text_policy,
        mask_text_policy=prompt_record.mask_text_policy,
        original_token_count=prompt_record.original_token_count,
        counterfactual_token_count=token_count,
        sentence_id_alignment_audit=prompt_record.sentence_id_alignment_audit,
    )
    return counterfactual_prompt, prompt_record


def _failure_record(
    *,
    intervention: CounterfactualSEMUIntervention,
    prompt_record: CounterfactualPromptRecord,
    exc: Exception,
    hardware: str,
    modal_profile: str,
) -> CounterfactualRunRecord:
    message = str(exc)
    is_capacity_missing = "out of memory" in message.lower() or "oom" in message.lower()
    status = "CAPACITY_MISSING" if is_capacity_missing else "RUNTIME_FAILURE"
    return CounterfactualRunRecord(
        schema_version=SCHEMA_VERSION,
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
