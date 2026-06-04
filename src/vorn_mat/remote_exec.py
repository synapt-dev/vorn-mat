"""Remote execution helpers for real Modal-backed baseline runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from .baselines.live_eviction import run_live_eviction
from .baselines.vanilla import run_vanilla
from .active_eviction_consumer_validation import (
    ConsumerValidationReport,
    load_phase0_case,
    run_consumer_validation,
    write_consumer_validation_artifacts,
)
from .active_eviction_pressure_sweep import (
    PressureSweepReport,
    run_pressure_sweep,
    write_pressure_sweep_artifacts,
)
from .benchmarks import load_ruler_hf_niah_slice
from .local_exec import (
    LocalModelConfig,
    TransformersObservationGenerator,
    TransformersLiveEvictionGenerator,
    TransformersScoreDistributionObservationGenerator,
    TransformersTextGenerator,
    attach_runtime_telemetry,
    capture_runtime_telemetry,
    reset_runtime_telemetry,
    select_live_eviction_plan,
    select_week1_plan,
)
from .modal_app import DEFAULT_MODAL_GPU
from .observation import ObservationReport
from .plan import (
    A100_80GB_PER_SECOND,
    DEFAULT_LIVE_EVICTION_CACHE_BUDGET,
    DEFAULT_MODEL,
    per_second_rate_for_gpu,
)
from .progress import default_progress_logger
from .results import RunResult, append_observation, append_result, observations_path
from .score_distribution_observation import ScoreDistributionObservationReport


@dataclass(frozen=True)
class ModalVanillaRunRequest:
    dataset_config: str = "niah_multikey_1_4k"
    split: str = "validation"
    case_limit: int = 50
    benchmark: str = "niah"
    output_path: str | None = None
    max_new_tokens: int = 32
    model_id: str = DEFAULT_MODEL
    gpu: str = "A100-80GB"


@dataclass(frozen=True)
class ModalVanillaRunReport:
    result: RunResult
    dataset_config: str
    split: str
    case_count: int
    elapsed_seconds: float
    estimated_cost_usd: float


@dataclass(frozen=True)
class ModalLiveEvictionRunRequest:
    dataset_config: str = "niah_multikey_1_4k"
    split: str = "validation"
    case_limit: int = 50
    case_offset_start: int = 0
    benchmark: str = "niah"
    output_path: str | None = None
    max_new_tokens: int = 32
    cache_budget_tokens: int = DEFAULT_LIVE_EVICTION_CACHE_BUDGET
    retention_policy: str = "vorn"
    random_seed: int = 17
    always_keep_prefix_tokens: int = 1
    preserve_recent_window: bool = True
    sentence_pooling: str = "max"
    sentence_top_k: int = 3
    eviction_trigger: str = "budget_threshold"
    sentence_boundary_lookahead_tokens: int = 25
    force_eviction_overflow_ratio: float = 1.2
    model_id: str = DEFAULT_MODEL
    gpu: str = "A100-80GB"


@dataclass(frozen=True)
class ModalLiveEvictionRunReport:
    result: RunResult
    dataset_config: str
    split: str
    case_count: int
    case_offset_start: int
    elapsed_seconds: float
    estimated_cost_usd: float
    cache_budget_tokens: int
    retention_policy: str
    always_keep_prefix_tokens: int
    preserve_recent_window: bool
    sentence_pooling: str
    sentence_top_k: int
    eviction_trigger: str
    sentence_boundary_lookahead_tokens: int
    force_eviction_overflow_ratio: float
    model_id: str


@dataclass(frozen=True)
class ModalVanillaObservationRequest:
    dataset_config: str = "niah_multikey_1_4k"
    split: str = "validation"
    case_limit: int = 50
    max_new_tokens: int = 32
    canonical_layer: int = 16
    recent_token_window: int = 16
    top_k: int = 10
    attention_last_n_layers: int = 4
    model_id: str = DEFAULT_MODEL


@dataclass(frozen=True)
class ModalScoreDistributionObservationRequest:
    dataset_config: str = "niah_multikey_1_8k"
    split: str = "validation"
    case_limit: int = 50
    max_new_tokens: int = 32
    canonical_layer: int = 16
    recent_token_window: int = 16
    cache_budget_tokens: int = DEFAULT_LIVE_EVICTION_CACHE_BUDGET
    retention_policy: str = "vorn"
    always_keep_prefix_tokens: int = 1
    preserve_recent_window: bool = True
    sentence_pooling: str = "max"
    sentence_top_k: int = 3
    model_id: str = DEFAULT_MODEL


@dataclass(frozen=True)
class ModalConsumerValidationRunRequest:
    dataset_config: str = "niah_multikey_1_4k"
    split: str = "validation"
    case_limit: int = 1
    case_offset_start: int = 1
    output_jsonl_path: str | None = None
    output_summary_path: str | None = None
    phase0_case: dict[str, object] | None = None
    phase0_artifact_path: str | None = None
    run_id: str = "vorn-active-eviction-pilot-a-consumer-validation-2026-06-04"
    family: str = "Mistral"
    model_id: str = DEFAULT_MODEL
    model_revision: str = "main"
    tokenizer_revision: str = "main"
    max_new_tokens: int = 32
    protected_semu_ids: tuple[int, ...] = (0, 1, 2, 29, 209, 210)
    selector_arms: tuple[str, ...] = (
        "vorn_high",
        "vorn_low",
        "length_high",
        "random_length_matched",
    )
    expected_selected_semu_ids: tuple[tuple[str, int], ...] = (
        ("vorn_high", 17),
        ("vorn_low", 201),
        ("length_high", 125),
        ("random_length_matched", 111),
    )
    deletion_mode: str = "delete"
    mask_dry_run_arms: tuple[str, ...] = ("vorn_high", "vorn_low")
    gpu: str = DEFAULT_MODAL_GPU
    modal_profile: str = "layne1penney"
    cost_per_second: float = A100_80GB_PER_SECOND


@dataclass(frozen=True)
class ModalConsumerValidationRunReport:
    report: ConsumerValidationReport
    dataset_config: str
    split: str
    case_count: int
    case_offset_start: int
    case_id: str
    elapsed_seconds: float
    estimated_cost_usd: float
    output_jsonl_path: str | None
    output_summary_path: str | None


@dataclass(frozen=True)
class ModalPressureSweepRunRequest:
    dataset_config: str = "niah_multikey_1_4k"
    split: str = "validation"
    case_limit: int = 1
    case_offset_start: int = 1
    output_jsonl_path: str | None = None
    output_summary_path: str | None = None
    phase0_case: dict[str, object] | None = None
    phase0_artifact_path: str | None = None
    run_id: str = "vorn-active-eviction-pilot-b-pressure-sweep-2026-06-04"
    family: str = "Mistral"
    model_id: str = DEFAULT_MODEL
    model_revision: str = "main"
    tokenizer_revision: str = "main"
    max_new_tokens: int = 32
    protected_semu_ids: tuple[int, ...] = (0, 1, 2, 29, 209, 210)
    selector_arms: tuple[str, ...] = (
        "vorn_high",
        "vorn_low",
        "length_high",
        "random_length_matched",
    )
    pressure_ns: tuple[int, ...] = (1, 3, 5, 10, 20, 50)
    base_seed: int = 534588164691762844
    expected_selected_semu_ids: tuple[tuple[str, int, tuple[int, ...]], ...] = ()
    deletion_mode: str = "delete"
    gpu: str = DEFAULT_MODAL_GPU
    modal_profile: str = "layne1penney"
    cost_per_second: float = A100_80GB_PER_SECOND


@dataclass(frozen=True)
class ModalPressureSweepRunReport:
    report: PressureSweepReport
    dataset_config: str
    split: str
    case_count: int
    case_offset_start: int
    case_id: str
    elapsed_seconds: float
    estimated_cost_usd: float
    output_jsonl_path: str | None
    output_summary_path: str | None


def run_modal_vanilla_niah(request: ModalVanillaRunRequest) -> ModalVanillaRunReport:
    """Run a real vanilla-only NIAH slice on remote GPU infrastructure."""
    per_second_rate = per_second_rate_for_gpu(request.gpu)
    start = time.perf_counter()
    reset_runtime_telemetry()
    cases = load_ruler_hf_niah_slice(
        request.dataset_config,
        split=request.split,
        case_limit=request.case_limit,
    )
    plan = select_week1_plan(request.benchmark, baseline="vanilla")
    generator = TransformersTextGenerator(
        LocalModelConfig(
            model_id=request.model_id,
            max_new_tokens=request.max_new_tokens,
        )
    )
    ledger = (
        observations_path(Path(request.output_path)) if request.output_path else None
    )
    result, _traces = run_vanilla(
        plan,
        cases,
        generator,
        on_case=(
            (lambda observation: append_observation(ledger, observation))
            if ledger is not None
            else None
        ),
        progress_logger=default_progress_logger,
    )
    elapsed_seconds = time.perf_counter() - start
    estimated_cost_usd = elapsed_seconds * per_second_rate

    metadata = dict(result.metadata)
    metadata.update(
        {
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": request.dataset_config,
            "split": request.split,
            "case_count": str(len(cases)),
            "model": request.model_id,
            "model_id": request.model_id,
            "elapsed_seconds": f"{elapsed_seconds:.3f}",
            "estimated_cost_usd": f"{estimated_cost_usd:.4f}",
        }
    )
    enriched_result = RunResult(
        run_id=result.run_id,
        benchmark=result.benchmark,
        baseline=result.baseline,
        metrics=result.metrics,
        metadata=metadata,
        preprocessing_elapsed_seconds=result.preprocessing_elapsed_seconds,
        preprocessing_cost_usd=(
            result.preprocessing_elapsed_seconds * per_second_rate
        ),
        observations=result.observations,
    )
    enriched_result = attach_runtime_telemetry(enriched_result)

    if request.output_path:
        append_result(Path(request.output_path), enriched_result)

    return ModalVanillaRunReport(
        result=enriched_result,
        dataset_config=request.dataset_config,
        split=request.split,
        case_count=len(cases),
        elapsed_seconds=elapsed_seconds,
        estimated_cost_usd=estimated_cost_usd,
    )


def run_modal_live_eviction_niah(
    request: ModalLiveEvictionRunRequest,
) -> ModalLiveEvictionRunReport:
    """Run the live eviction-only NIAH arm on remote GPU infrastructure."""
    per_second_rate = per_second_rate_for_gpu(request.gpu)
    start = time.perf_counter()
    reset_runtime_telemetry()
    cases = load_ruler_hf_niah_slice(
        request.dataset_config,
        split=request.split,
        case_limit=request.case_limit,
        case_offset_start=request.case_offset_start,
    )
    plan = select_live_eviction_plan(
        cache_budget_tokens=request.cache_budget_tokens,
        retention_policy=request.retention_policy,
        random_seed=request.random_seed,
        always_keep_prefix_tokens=request.always_keep_prefix_tokens,
        preserve_recent_window=request.preserve_recent_window,
        sentence_pooling=request.sentence_pooling,
        sentence_top_k=request.sentence_top_k,
        eviction_trigger=request.eviction_trigger,
        sentence_boundary_lookahead_tokens=request.sentence_boundary_lookahead_tokens,
        force_eviction_overflow_ratio=request.force_eviction_overflow_ratio,
    )
    generator = TransformersLiveEvictionGenerator(
        LocalModelConfig(
            model_id=request.model_id,
            max_new_tokens=request.max_new_tokens,
        )
    )
    ledger = (
        observations_path(Path(request.output_path)) if request.output_path else None
    )
    result, _traces = run_live_eviction(
        plan,
        cases,
        generator,
        on_case=(
            (lambda observation: append_observation(ledger, observation))
            if ledger is not None
            else None
        ),
        progress_logger=default_progress_logger,
    )
    elapsed_seconds = time.perf_counter() - start
    estimated_cost_usd = elapsed_seconds * per_second_rate

    metadata = dict(result.metadata)
    metadata.update(
        {
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": request.dataset_config,
            "split": request.split,
            "case_count": str(len(cases)),
            "case_offset_start": str(request.case_offset_start),
            "model": request.model_id,
            "model_id": request.model_id,
            "elapsed_seconds": f"{elapsed_seconds:.3f}",
            "estimated_cost_usd": f"{estimated_cost_usd:.4f}",
            "cache_budget_tokens": str(request.cache_budget_tokens),
            "retention_policy": request.retention_policy,
            "random_seed": str(request.random_seed),
            "always_keep_prefix_tokens": str(request.always_keep_prefix_tokens),
            "preserve_recent_window": str(request.preserve_recent_window).lower(),
            "sentence_pooling": request.sentence_pooling,
            "sentence_top_k": str(request.sentence_top_k),
            "eviction_trigger": request.eviction_trigger,
            "sentence_boundary_lookahead_tokens": str(
                request.sentence_boundary_lookahead_tokens
            ),
            "force_eviction_overflow_ratio": (
                f"{request.force_eviction_overflow_ratio:.2f}"
            ),
        }
    )
    enriched_result = RunResult(
        run_id=result.run_id,
        benchmark=result.benchmark,
        baseline=result.baseline,
        metrics=result.metrics,
        metadata=metadata,
        preprocessing_elapsed_seconds=result.preprocessing_elapsed_seconds,
        preprocessing_cost_usd=(
            result.preprocessing_elapsed_seconds * per_second_rate
        ),
        observations=result.observations,
    )
    enriched_result = attach_runtime_telemetry(enriched_result)

    if request.output_path:
        append_result(Path(request.output_path), enriched_result)

    return ModalLiveEvictionRunReport(
        result=enriched_result,
        dataset_config=request.dataset_config,
        split=request.split,
        case_count=len(cases),
        case_offset_start=request.case_offset_start,
        elapsed_seconds=elapsed_seconds,
        estimated_cost_usd=estimated_cost_usd,
        cache_budget_tokens=request.cache_budget_tokens,
        retention_policy=request.retention_policy,
        always_keep_prefix_tokens=request.always_keep_prefix_tokens,
        preserve_recent_window=request.preserve_recent_window,
        sentence_pooling=request.sentence_pooling,
        sentence_top_k=request.sentence_top_k,
        eviction_trigger=request.eviction_trigger,
        sentence_boundary_lookahead_tokens=request.sentence_boundary_lookahead_tokens,
        force_eviction_overflow_ratio=request.force_eviction_overflow_ratio,
        model_id=request.model_id,
    )


def run_modal_consumer_validation_niah(
    request: ModalConsumerValidationRunRequest,
) -> ModalConsumerValidationRunReport:
    """Run the locked active-eviction consumer-validation smoke on Modal."""
    if request.case_limit != 1:
        raise ValueError("consumer validation requires exactly one fixture")

    start = time.perf_counter()
    cases = load_ruler_hf_niah_slice(
        request.dataset_config,
        split=request.split,
        case_limit=request.case_limit,
        case_offset_start=request.case_offset_start,
    )
    if len(cases) != 1:
        raise ValueError(f"expected one consumer-validation case, got {len(cases)}")
    case = cases[0]
    if request.phase0_case is not None:
        phase0_case = dict(request.phase0_case)
    elif request.phase0_artifact_path is not None:
        phase0_case = load_phase0_case(
            Path(request.phase0_artifact_path),
            case.case_id,
        )
    else:
        raise ValueError(
            "consumer validation requires phase0_case or phase0_artifact_path"
        )

    generator = TransformersTextGenerator(
        LocalModelConfig(
            model_id=request.model_id,
            max_new_tokens=request.max_new_tokens,
        )
    )
    report = run_consumer_validation(
        run_id=request.run_id,
        family=request.family,
        model_id=request.model_id,
        model_revision=request.model_revision,
        tokenizer_revision=request.tokenizer_revision,
        case=case,
        phase0_case=phase0_case,
        generator=generator,
        protected_semu_ids=request.protected_semu_ids,
        selector_arms=request.selector_arms,
        expected_selected_semu_ids=dict(request.expected_selected_semu_ids),
        deletion_mode=request.deletion_mode,
        mask_dry_run_arms=request.mask_dry_run_arms,
        hardware=request.gpu,
        modal_profile=request.modal_profile,
        cost_per_second=request.cost_per_second,
        reset_telemetry=reset_runtime_telemetry,
        telemetry_snapshot=capture_runtime_telemetry,
    )
    elapsed_seconds = time.perf_counter() - start
    if request.output_jsonl_path and request.output_summary_path:
        write_consumer_validation_artifacts(
            report,
            jsonl_path=Path(request.output_jsonl_path),
            md_path=Path(request.output_summary_path),
        )

    return ModalConsumerValidationRunReport(
        report=report,
        dataset_config=request.dataset_config,
        split=request.split,
        case_count=len(cases),
        case_offset_start=request.case_offset_start,
        case_id=case.case_id,
        elapsed_seconds=elapsed_seconds,
        estimated_cost_usd=elapsed_seconds * request.cost_per_second,
        output_jsonl_path=request.output_jsonl_path,
        output_summary_path=request.output_summary_path,
    )


def run_modal_pressure_sweep_niah(
    request: ModalPressureSweepRunRequest,
) -> ModalPressureSweepRunReport:
    """Run the locked active-eviction pressure-scaling smoke on Modal."""
    if request.case_limit != 1:
        raise ValueError("pressure sweep requires exactly one fixture")

    start = time.perf_counter()
    cases = load_ruler_hf_niah_slice(
        request.dataset_config,
        split=request.split,
        case_limit=request.case_limit,
        case_offset_start=request.case_offset_start,
    )
    if len(cases) != 1:
        raise ValueError(f"expected one pressure-sweep case, got {len(cases)}")
    case = cases[0]
    if request.phase0_case is not None:
        phase0_case = dict(request.phase0_case)
    elif request.phase0_artifact_path is not None:
        phase0_case = load_phase0_case(
            Path(request.phase0_artifact_path),
            case.case_id,
        )
    else:
        raise ValueError("pressure sweep requires phase0_case or phase0_artifact_path")

    generator = TransformersTextGenerator(
        LocalModelConfig(
            model_id=request.model_id,
            max_new_tokens=request.max_new_tokens,
        )
    )
    expected_selected = {
        (arm, pressure_n): ids
        for arm, pressure_n, ids in request.expected_selected_semu_ids
    }
    report = run_pressure_sweep(
        run_id=request.run_id,
        family=request.family,
        model_id=request.model_id,
        model_revision=request.model_revision,
        tokenizer_revision=request.tokenizer_revision,
        case=case,
        phase0_case=phase0_case,
        generator=generator,
        protected_semu_ids=request.protected_semu_ids,
        selector_arms=request.selector_arms,
        pressure_ns=request.pressure_ns,
        base_seed=request.base_seed,
        expected_selected_semu_ids=expected_selected or None,
        deletion_mode=request.deletion_mode,
        hardware=request.gpu,
        modal_profile=request.modal_profile,
        cost_per_second=request.cost_per_second,
        reset_telemetry=reset_runtime_telemetry,
        telemetry_snapshot=capture_runtime_telemetry,
    )
    elapsed_seconds = time.perf_counter() - start
    if request.output_jsonl_path and request.output_summary_path:
        write_pressure_sweep_artifacts(
            report,
            jsonl_path=Path(request.output_jsonl_path),
            md_path=Path(request.output_summary_path),
        )

    return ModalPressureSweepRunReport(
        report=report,
        dataset_config=request.dataset_config,
        split=request.split,
        case_count=len(cases),
        case_offset_start=request.case_offset_start,
        case_id=case.case_id,
        elapsed_seconds=elapsed_seconds,
        estimated_cost_usd=elapsed_seconds * request.cost_per_second,
        output_jsonl_path=request.output_jsonl_path,
        output_summary_path=request.output_summary_path,
    )


def run_modal_vanilla_observation_niah(
    request: ModalVanillaObservationRequest,
) -> ObservationReport:
    """Run pure vanilla observation on a real NIAH slice."""
    start = time.perf_counter()
    cases = load_ruler_hf_niah_slice(
        request.dataset_config,
        split=request.split,
        case_limit=request.case_limit,
    )
    generator = TransformersObservationGenerator(
        LocalModelConfig(
            model_id=request.model_id,
            max_new_tokens=request.max_new_tokens,
        )
    )
    observed_cases = tuple(
        generator.observe_vanilla_case(
            case,
            canonical_layer=request.canonical_layer,
            recent_token_window=request.recent_token_window,
            top_k=request.top_k,
            attention_last_n_layers=request.attention_last_n_layers,
        )
        for case in cases
    )
    elapsed_seconds = time.perf_counter() - start
    estimated_cost_usd = elapsed_seconds * A100_80GB_PER_SECOND
    return ObservationReport(
        dataset_config=request.dataset_config,
        split=f"{request.split}[:{request.case_limit}]",
        case_count=len(observed_cases),
        elapsed_seconds=elapsed_seconds,
        estimated_cost_usd=estimated_cost_usd,
        cases=observed_cases,
    )


def run_modal_score_distribution_observation_niah(
    request: ModalScoreDistributionObservationRequest,
) -> ScoreDistributionObservationReport:
    """Run live budgeted score-shape observation on a real NIAH slice."""
    start = time.perf_counter()
    cases = load_ruler_hf_niah_slice(
        request.dataset_config,
        split=request.split,
        case_limit=request.case_limit,
    )
    generator = TransformersScoreDistributionObservationGenerator(
        LocalModelConfig(
            model_id=request.model_id,
            max_new_tokens=request.max_new_tokens,
        )
    )
    observed_cases = tuple(
        generator.observe_live_case(
            case,
            config=generator_observation_config(request),
        )
        for case in cases
    )
    elapsed_seconds = time.perf_counter() - start
    estimated_cost_usd = elapsed_seconds * A100_80GB_PER_SECOND
    return ScoreDistributionObservationReport(
        dataset_config=request.dataset_config,
        split=f"{request.split}[:{request.case_limit}]",
        case_count=len(observed_cases),
        cache_budget_tokens=request.cache_budget_tokens,
        retention_policy=request.retention_policy,
        always_keep_prefix_tokens=request.always_keep_prefix_tokens,
        preserve_recent_window=request.preserve_recent_window,
        sentence_pooling=request.sentence_pooling,
        sentence_top_k=request.sentence_top_k,
        model_id=request.model_id,
        elapsed_seconds=elapsed_seconds,
        estimated_cost_usd=estimated_cost_usd,
        cases=observed_cases,
    )


def generator_observation_config(
    request: ModalScoreDistributionObservationRequest,
):
    from .baselines.live_eviction import LiveEvictionConfig

    return LiveEvictionConfig(
        canonical_layer=request.canonical_layer,
        recent_token_window=request.recent_token_window,
        cache_budget_tokens=request.cache_budget_tokens,
        retention_policy=request.retention_policy,
        always_keep_prefix_tokens=request.always_keep_prefix_tokens,
        preserve_recent_window=request.preserve_recent_window,
        sentence_pooling=request.sentence_pooling,
        sentence_top_k=request.sentence_top_k,
    )
