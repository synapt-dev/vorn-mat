"""Week 1 experiment manifest for the vorn-aligned MAT lane.

This module turns the merged research spec into a stable code contract:
- default model and scoring choices
- baseline reproduction matrix
- Modal cost envelope
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_CANONICAL_LAYER = 16
DEFAULT_RECENT_WINDOW = 16
DEFAULT_STEP1_CASE_LIMIT = 50
DEFAULT_STEP1_CACHE_BUDGET = 256
DEFAULT_LIVE_EVICTION_CASE_LIMIT = 50
DEFAULT_LIVE_EVICTION_CACHE_BUDGET = 256
DEFAULT_RANDOM_LIVE_EVICTION_SEED = 17

A100_80GB_PER_SECOND = 0.000694
H100_PER_SECOND = 0.001097


@dataclass(frozen=True)
class ExperimentDefaults:
    model: str = DEFAULT_MODEL
    canonical_layer: int = DEFAULT_CANONICAL_LAYER
    recent_token_window: int = DEFAULT_RECENT_WINDOW
    eviction_unit: str = "token_position"
    retrieval_space: str = "canonical_residual_layer"
    gpu: str = "A100-80GB"


def per_second_rate_for_gpu(gpu: str) -> float:
    """Return the modeled per-second cost rate for a supported Modal GPU."""
    if gpu == "A100-80GB":
        return A100_80GB_PER_SECOND
    if gpu == "H100":
        return H100_PER_SECOND
    raise ValueError(f"unsupported gpu for cost modeling: {gpu}")


@dataclass(frozen=True)
class Week1Run:
    run_id: str
    model: str
    benchmark: str
    baseline: str
    gpu: str
    canonical_layer: int
    recent_token_window: int
    eviction_unit: str
    case_limit: int | None = None
    cache_budget_tokens: int | None = None
    retention_policy: str | None = None
    random_seed: int | None = None
    always_keep_prefix_tokens: int = 1
    preserve_recent_window: bool = True
    sentence_pooling: str | None = None
    sentence_top_k: int | None = None
    eviction_trigger: str = "budget_threshold"
    sentence_boundary_lookahead_tokens: int = 25
    force_eviction_overflow_ratio: float = 1.2
    experiment_stage: str = "week1"
    compression_mode: str | None = None


@dataclass(frozen=True)
class Step1Defaults:
    benchmark: str = "niah"
    case_limit: int = DEFAULT_STEP1_CASE_LIMIT
    cache_budget_tokens: int = DEFAULT_STEP1_CACHE_BUDGET
    compression_mode: str = "prompt_retention_proxy"


@dataclass(frozen=True)
class LiveEvictionDefaults:
    benchmark: str = "niah"
    case_limit: int = DEFAULT_LIVE_EVICTION_CASE_LIMIT
    cache_budget_tokens: int = DEFAULT_LIVE_EVICTION_CACHE_BUDGET
    baseline: str = "vorn_live"
    retention_policy: str = "vorn"
    random_seed: int = DEFAULT_RANDOM_LIVE_EVICTION_SEED
    always_keep_prefix_tokens: int = 1
    preserve_recent_window: bool = True
    eviction_unit: str = "token_position"
    sentence_pooling: str | None = None
    sentence_top_k: int | None = None
    eviction_trigger: str = "budget_threshold"
    sentence_boundary_lookahead_tokens: int = 25
    force_eviction_overflow_ratio: float = 1.2
    compression_mode: str = "live_eviction_only"


def middle_layer_index(num_layers: int) -> int:
    """Choose a canonical middle layer under 0-based indexing."""
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    return num_layers // 2


def per_hour_rate(per_second_rate: float) -> float:
    """Convert a Modal per-second GPU rate to hourly cost."""
    return per_second_rate * 3600


def estimate_week1_cost_window() -> tuple[float, float]:
    """Return the A100-only cost window locked in the merged spec."""
    lower_hours = 210
    upper_hours = 350
    hourly_rate = per_hour_rate(A100_80GB_PER_SECOND)
    return (lower_hours * hourly_rate, upper_hours * hourly_rate)


def build_week1_run_matrix(
    defaults: ExperimentDefaults = ExperimentDefaults(),
) -> tuple[Week1Run, ...]:
    """Build the Week 1 baseline reproduction matrix.

    The first implementation pass only covers baseline reproduction:
    vanilla, H2O, and StreamingLLM on RULER and NIAH.
    """
    runs: list[Week1Run] = []
    benchmarks = ("ruler", "niah")
    baselines = ("vanilla", "h2o", "streamingllm")

    for benchmark in benchmarks:
        for baseline in baselines:
            runs.append(
                Week1Run(
                    run_id=f"week1-{benchmark}-{baseline}",
                    model=defaults.model,
                    benchmark=benchmark,
                    baseline=baseline,
                    gpu=defaults.gpu,
                    canonical_layer=defaults.canonical_layer,
                    recent_token_window=defaults.recent_token_window,
                    eviction_unit=defaults.eviction_unit,
                    experiment_stage="week1",
                )
            )
    return tuple(runs)


def build_step1_run_matrix(
    defaults: ExperimentDefaults = ExperimentDefaults(),
    step1: Step1Defaults = Step1Defaults(),
) -> tuple[Week1Run, Week1Run]:
    """Build the smallest publishable Step 1 comparison pair.

    Step 1 is intentionally narrower than Week 1 baseline reproduction:
    one benchmark, one cache budget, and two arms:
    - vanilla full-context reference
    - vorn-scored prompt-retention proxy for the first real signal
    """
    vanilla = Week1Run(
        run_id=f"step1-{step1.benchmark}-vanilla",
        model=defaults.model,
        benchmark=step1.benchmark,
        baseline="vanilla",
        gpu=defaults.gpu,
        canonical_layer=defaults.canonical_layer,
        recent_token_window=defaults.recent_token_window,
        eviction_unit=defaults.eviction_unit,
        case_limit=step1.case_limit,
        cache_budget_tokens=None,
        experiment_stage="step1",
        compression_mode=None,
    )
    vorn = Week1Run(
        run_id=f"step1-{step1.benchmark}-vorn",
        model=defaults.model,
        benchmark=step1.benchmark,
        baseline="vorn",
        gpu=defaults.gpu,
        canonical_layer=defaults.canonical_layer,
        recent_token_window=defaults.recent_token_window,
        eviction_unit=defaults.eviction_unit,
        case_limit=step1.case_limit,
        cache_budget_tokens=step1.cache_budget_tokens,
        experiment_stage="step1",
        compression_mode=step1.compression_mode,
    )
    return (vanilla, vorn)


def build_live_eviction_run(
    defaults: ExperimentDefaults = ExperimentDefaults(),
    live: LiveEvictionDefaults = LiveEvictionDefaults(),
) -> Week1Run:
    """Build the first rotary-safe live-eviction mechanism run."""
    run_id = f"step2-{live.benchmark}-{live.baseline.replace('_', '-')}"
    if (
        live.baseline == "vorn_live"
        and live.cache_budget_tokens == DEFAULT_LIVE_EVICTION_CACHE_BUDGET
        and live.retention_policy == "vorn"
        and live.always_keep_prefix_tokens == 1
        and live.preserve_recent_window is True
    ):
        run_id = f"step2-{live.benchmark}-vorn-live"
    elif (
        live.baseline == "vorn_live"
        and live.retention_policy == "vorn"
        and live.always_keep_prefix_tokens == 0
        and live.preserve_recent_window is False
    ):
        run_id = f"step2-{live.benchmark}-vorn-live-b{live.cache_budget_tokens}-noguards"
    elif (
        live.baseline == "sentence_vorn_live"
        and live.retention_policy == "sentence_vorn"
        and live.always_keep_prefix_tokens == 0
        and live.preserve_recent_window is False
        and live.eviction_trigger == "budget_threshold"
    ):
        run_id = (
            f"step2-{live.benchmark}-sentence-vorn-live-"
            f"b{live.cache_budget_tokens}-noguards"
        )
    elif (
        live.baseline == "sentence_vorn_live"
        and live.retention_policy == "sentence_vorn"
        and live.eviction_trigger == "sentence_boundary"
    ):
        suffix = "sentbound"
        if live.always_keep_prefix_tokens == 0 and live.preserve_recent_window is False:
            suffix = "sentbound-noguards"
        run_id = (
            f"step2-{live.benchmark}-sentence-vorn-live-"
            f"b{live.cache_budget_tokens}-{suffix}"
        )
    elif (
        live.baseline == "word_vorn_live"
        and live.retention_policy == "word_vorn"
        and live.always_keep_prefix_tokens == 0
        and live.preserve_recent_window is False
    ):
        run_id = (
            f"step2-{live.benchmark}-word-vorn-live-"
            f"b{live.cache_budget_tokens}-noguards"
        )
    elif (
        live.baseline == "adaptive_vorn_live"
        and live.retention_policy == "adaptive_vorn"
        and live.always_keep_prefix_tokens == 0
        and live.preserve_recent_window is False
    ):
        run_id = (
            f"step2-{live.benchmark}-adaptive-vorn-live-"
            f"b{live.cache_budget_tokens}-noguards"
        )
    else:
        run_id = f"{run_id}-b{live.cache_budget_tokens}"
    return Week1Run(
        run_id=run_id,
        model=defaults.model,
        benchmark=live.benchmark,
        baseline=live.baseline,
        gpu=defaults.gpu,
        canonical_layer=defaults.canonical_layer,
        recent_token_window=defaults.recent_token_window,
        eviction_unit=live.eviction_unit,
        case_limit=live.case_limit,
        cache_budget_tokens=live.cache_budget_tokens,
        retention_policy=live.retention_policy,
        random_seed=live.random_seed,
        always_keep_prefix_tokens=live.always_keep_prefix_tokens,
        preserve_recent_window=live.preserve_recent_window,
        sentence_pooling=live.sentence_pooling,
        sentence_top_k=live.sentence_top_k,
        eviction_trigger=live.eviction_trigger,
        sentence_boundary_lookahead_tokens=live.sentence_boundary_lookahead_tokens,
        force_eviction_overflow_ratio=live.force_eviction_overflow_ratio,
        experiment_stage="step2",
        compression_mode=live.compression_mode,
    )
