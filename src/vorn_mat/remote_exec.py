"""Remote execution helpers for real Modal-backed baseline runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from .baselines.live_eviction import run_live_eviction
from .baselines.vanilla import run_vanilla
from .benchmarks import load_ruler_hf_niah_slice
from .local_exec import (
    LocalModelConfig,
    TransformersObservationGenerator,
    TransformersLiveEvictionGenerator,
    TransformersScoreDistributionObservationGenerator,
    TransformersTextGenerator,
    select_live_eviction_plan,
    select_week1_plan,
)
from .observation import ObservationReport
from .plan import (
    A100_80GB_PER_SECOND,
    DEFAULT_LIVE_EVICTION_CACHE_BUDGET,
    DEFAULT_MODEL,
    per_second_rate_for_gpu,
)
from .results import RunResult, append_result
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


def run_modal_vanilla_niah(request: ModalVanillaRunRequest) -> ModalVanillaRunReport:
    """Run a real vanilla-only NIAH slice on remote GPU infrastructure."""
    per_second_rate = per_second_rate_for_gpu(request.gpu)
    start = time.perf_counter()
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
    result, _traces = run_vanilla(plan, cases, generator)
    elapsed_seconds = time.perf_counter() - start
    estimated_cost_usd = elapsed_seconds * per_second_rate

    metadata = dict(result.metadata)
    metadata.update(
        {
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": request.dataset_config,
            "split": request.split,
            "case_count": str(len(cases)),
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
    cases = load_ruler_hf_niah_slice(
        request.dataset_config,
        split=request.split,
        case_limit=request.case_limit,
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
    result, _traces = run_live_eviction(plan, cases, generator)
    elapsed_seconds = time.perf_counter() - start
    estimated_cost_usd = elapsed_seconds * per_second_rate

    metadata = dict(result.metadata)
    metadata.update(
        {
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": request.dataset_config,
            "split": request.split,
            "case_count": str(len(cases)),
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

    if request.output_path:
        append_result(Path(request.output_path), enriched_result)

    return ModalLiveEvictionRunReport(
        result=enriched_result,
        dataset_config=request.dataset_config,
        split=request.split,
        case_count=len(cases),
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
