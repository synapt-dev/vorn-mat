"""Rotary-safe live eviction arm for vorn-MAT mechanism validation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from ..benchmarks import BenchmarkCase, score_predictions
from ..benchmarks.common import build_case_observation, build_case_observations
from ..progress import (
    ProgressLogger,
    format_case_progress,
    format_complete,
    format_dataset_loaded,
)
from ..results import CaseObservation, RunResult
from ..runner import ExecutionPlan
from ..text_spans import sentence_char_spans, token_span_from_offsets, word_char_spans

SEMANTIC_SUMMARY_CONTRACT = (
    "canonical_hidden_state_float32_per_token_from_layer_L_star"
)
TOVA_ATTENTION_CONTRACT = "last_token_attention_mean_over_final_layer_heads"
H2O_ATTENTION_CONTRACT = "accumulated_last_token_attention_mean_over_final_layer_heads"
SUMMARIZE_COMPACT_CONTRACT = (
    "whole_context_summary_without_question_then_answer_with_question"
)
ADAPTIVE_SELECTOR_CONTRACT = (
    "choose_token_or_sentence_by_peak_zscore_over_current_alignment_scores"
)


@dataclass(frozen=True)
class LiveEvictionConfig:
    canonical_layer: int
    recent_token_window: int
    cache_budget_tokens: int
    retention_policy: str = "vorn"
    random_seed: int = 17
    always_keep_prefix_tokens: int = 1
    preserve_recent_window: bool = True
    prefix_suffix_prefix_tokens: int = 64
    eviction_unit: str = "token_position"
    sentence_pooling: str = "max"
    sentence_top_k: int = 3
    eviction_trigger: str = "budget_threshold"
    sentence_boundary_lookahead_tokens: int = 25
    force_eviction_overflow_ratio: float = 1.2


@dataclass(frozen=True)
class LiveEvictionStats:
    prompt_token_count: int
    generated_token_count: int
    mean_kept_token_count: float
    final_kept_token_count: int
    eviction_steps: int
    mean_retention_ratio: float
    retention_policy: str
    summary_contract: str
    summary_fingerprint: str
    preprocessing_elapsed_seconds: float = 0.0
    adaptive_token_steps: int = 0
    adaptive_sentence_steps: int = 0
    adaptive_selector_contract: str = ""


@dataclass(frozen=True)
class LiveEvictionPredictionTrace:
    case_id: str
    prediction: str
    mean_retention_ratio: float
    eviction_steps: int
    edge_kind: str = ""


class LiveEvictionTextGenerator(Protocol):
    def generate_with_live_eviction(
        self,
        prompt: str,
        config: LiveEvictionConfig,
        *,
        case_id: str = "",
        expected_answer: str | None = None,
    ) -> tuple[str, LiveEvictionStats]: ...


def extract_canonical_token_summaries(
    hidden_states: tuple[Any, ...] | list[Any],
    *,
    canonical_layer: int,
) -> np.ndarray:
    """Return one deterministic semantic summary vector per token position."""
    layer_index = canonical_layer + 1  # hidden_states[0] is the embedding output
    if layer_index >= len(hidden_states):
        raise ValueError(
            f"canonical_layer {canonical_layer} is unavailable for this model"
        )
    token_summaries = hidden_states[layer_index][0]
    return token_summaries.detach().float().cpu().numpy().astype(
        np.float32,
        copy=False,
    )


def summary_fingerprint(token_summaries: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(token_summaries, dtype=np.float32)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def extract_last_token_attention_scores(
    attentions: tuple[Any, ...] | list[Any],
) -> np.ndarray:
    """Return one scalar attention score per retained position.

    This uses the final layer's attention weights for the most recent query token,
    averaged across heads into a single token-position score vector.
    """
    if not attentions:
        raise ValueError("attentions must be non-empty")
    final_layer = attentions[-1]
    if getattr(final_layer, "ndim", None) != 4:
        raise ValueError("attention tensor must be rank-4")
    scores = final_layer[0, :, -1, :].mean(dim=0)
    return scores.detach().float().cpu().numpy().astype(np.float32, copy=False)


def select_attention_score_retained_positions(
    attention_scores: np.ndarray,
    *,
    cache_budget_tokens: int,
    always_keep_prefix_tokens: int = 1,
    preserve_recent_window: int = 0,
) -> tuple[int, ...]:
    """Select retained positions by scalar attention score with required guards."""
    if attention_scores.ndim != 1:
        raise ValueError("attention_scores must be rank-1")
    token_count = attention_scores.shape[0]
    if token_count == 0:
        raise ValueError("attention_scores must be non-empty")
    if cache_budget_tokens <= 0:
        raise ValueError("cache_budget_tokens must be positive")

    required: set[int] = set(range(min(always_keep_prefix_tokens, token_count)))
    if preserve_recent_window > 0:
        recent_start = max(0, token_count - preserve_recent_window)
        required.update(range(recent_start, token_count))

    if cache_budget_tokens < len(required):
        raise ValueError(
            "cache_budget_tokens must fit the required prefix + recent-window positions"
        )
    if cache_budget_tokens >= token_count:
        return tuple(range(token_count))

    ranked_candidates = sorted(
        (position for position in range(token_count) if position not in required),
        key=lambda position: (float(attention_scores[position]), -position),
        reverse=True,
    )

    keep = set(required)
    for position in ranked_candidates:
        if len(keep) >= cache_budget_tokens:
            break
        keep.add(position)
    return tuple(sorted(keep))


def select_scalar_score_retained_positions(
    scores: np.ndarray,
    *,
    cache_budget_tokens: int,
    always_keep_prefix_tokens: int = 1,
    preserve_recent_window: int = 0,
) -> tuple[int, ...]:
    """Select retained positions from any rank-1 score vector with guards."""
    if scores.ndim != 1:
        raise ValueError("scores must be rank-1")
    token_count = scores.shape[0]
    if token_count == 0:
        raise ValueError("scores must be non-empty")
    if cache_budget_tokens <= 0:
        raise ValueError("cache_budget_tokens must be positive")

    required: set[int] = set(range(min(always_keep_prefix_tokens, token_count)))
    if preserve_recent_window > 0:
        recent_start = max(0, token_count - preserve_recent_window)
        required.update(range(recent_start, token_count))

    if cache_budget_tokens < len(required):
        raise ValueError(
            "cache_budget_tokens must fit the required prefix + recent-window positions"
        )
    if cache_budget_tokens >= token_count:
        return tuple(range(token_count))

    ranked_candidates = sorted(
        (position for position in range(token_count) if position not in required),
        key=lambda position: (float(scores[position]), -position),
        reverse=True,
    )

    keep = set(required)
    for position in ranked_candidates:
        if len(keep) >= cache_budget_tokens:
            break
        keep.add(position)
    return tuple(sorted(keep))


def accumulate_attention_scores(
    running_scores: np.ndarray,
    current_scores: np.ndarray,
) -> np.ndarray:
    """Accumulate per-position attention mass for heavy-hitter retention."""
    if running_scores.ndim != 1 or current_scores.ndim != 1:
        raise ValueError("attention score vectors must be rank-1")
    if running_scores.shape != current_scores.shape:
        raise ValueError("attention score vectors must have the same shape")
    return np.asarray(running_scores + current_scores, dtype=np.float32)


def select_random_retained_positions(
    *,
    token_count: int,
    cache_budget_tokens: int,
    always_keep_prefix_tokens: int = 1,
    preserve_recent_window: int = 0,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Select retained positions by deterministic random sampling."""
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if cache_budget_tokens <= 0:
        raise ValueError("cache_budget_tokens must be positive")

    required: set[int] = set(range(min(always_keep_prefix_tokens, token_count)))
    if preserve_recent_window > 0:
        recent_start = max(0, token_count - preserve_recent_window)
        required.update(range(recent_start, token_count))

    if cache_budget_tokens < len(required):
        raise ValueError(
            "cache_budget_tokens must fit the required prefix + recent-window positions"
        )
    if cache_budget_tokens >= token_count:
        return tuple(range(token_count))

    candidates = [
        position for position in range(token_count) if position not in required
    ]
    keep = set(required)
    extra_slots = cache_budget_tokens - len(required)
    if extra_slots > 0:
        chosen = rng.choice(candidates, size=extra_slots, replace=False)
        keep.update(int(position) for position in chosen.tolist())
    return tuple(sorted(keep))


def select_sliding_window_retained_positions(
    *,
    token_count: int,
    cache_budget_tokens: int,
) -> tuple[int, ...]:
    """Select the most recent token positions under a fixed budget."""
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if cache_budget_tokens <= 0:
        raise ValueError("cache_budget_tokens must be positive")
    if cache_budget_tokens >= token_count:
        return tuple(range(token_count))
    start = token_count - cache_budget_tokens
    return tuple(range(start, token_count))


def select_prefix_suffix_retained_positions(
    *,
    token_count: int,
    cache_budget_tokens: int,
    prefix_token_count: int = 64,
) -> tuple[int, ...]:
    """Keep a fixed prompt prefix and the most recent suffix under one budget."""
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if cache_budget_tokens <= 0:
        raise ValueError("cache_budget_tokens must be positive")
    if prefix_token_count <= 0:
        raise ValueError("prefix_token_count must be positive")
    if cache_budget_tokens >= token_count:
        return tuple(range(token_count))

    prefix_count = min(prefix_token_count, cache_budget_tokens, token_count)
    suffix_count = cache_budget_tokens - prefix_count
    keep = set(range(prefix_count))
    if suffix_count > 0:
        keep.update(range(token_count - suffix_count, token_count))
    return tuple(sorted(keep))


def text_ends_at_sentence_boundary(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    spans = sentence_char_spans(stripped)
    return bool(spans) and spans[-1][1] == len(stripped) and stripped.endswith((".", "!", "?"))


def should_trigger_sentence_boundary_eviction(
    *,
    rendered_text: str,
    token_count: int,
    cache_budget_tokens: int,
    lookahead_tokens: int = 25,
    force_overflow_ratio: float = 1.2,
) -> bool:
    if cache_budget_tokens <= 0:
        raise ValueError("cache_budget_tokens must be positive")
    if lookahead_tokens < 0:
        raise ValueError("lookahead_tokens must be non-negative")
    if force_overflow_ratio < 1.0:
        raise ValueError("force_overflow_ratio must be >= 1.0")

    if token_count > int(cache_budget_tokens * force_overflow_ratio):
        return True
    if token_count + lookahead_tokens <= cache_budget_tokens:
        return False
    return text_ends_at_sentence_boundary(rendered_text)


def assign_sentence_ids_from_offsets(
    rendered_prompt: str,
    offsets: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    """Map each prompt token position to a sentence id in rendered-prompt order."""
    return assign_unit_ids_from_spans(sentence_char_spans(rendered_prompt), offsets)


def assign_word_ids_from_offsets(
    rendered_prompt: str,
    offsets: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    """Map each prompt token position to a word id in rendered-prompt order."""
    return assign_unit_ids_from_spans(word_char_spans(rendered_prompt), offsets)
 

def assign_unit_ids_from_spans(
    spans: Sequence[tuple[int, int]],
    offsets: Sequence[tuple[int, int]],
) -> tuple[int, ...]:
    """Map each prompt token position to a semantic-unit id in prompt order."""
    if not offsets:
        return ()
    if not spans:
        return tuple(0 for _ in offsets)

    unit_ids = [-1] * len(offsets)
    for unit_id, (char_start, char_end) in enumerate(spans):
        token_span = token_span_from_offsets(
            offsets,
            char_start=char_start,
            char_end=char_end,
        )
        if token_span is None:
            continue
        start, end = token_span
        for token_index in range(start, end):
            unit_ids[token_index] = unit_id

    current_unit = 0
    for token_index, unit_id in enumerate(unit_ids):
        if unit_id == -1:
            unit_ids[token_index] = current_unit
        else:
            current_unit = unit_id
    return tuple(unit_ids)


def build_unit_ids_for_active_positions(
    *,
    prompt_unit_ids: Sequence[int],
    active_absolute_positions: Sequence[int],
) -> tuple[int, ...]:
    """Group prompt tokens by unit and generated tokens as singleton units."""
    if not active_absolute_positions:
        return ()
    prompt_token_count = len(prompt_unit_ids)
    prompt_unit_count = max(prompt_unit_ids, default=-1) + 1
    return tuple(
        prompt_unit_ids[position]
        if 0 <= position < prompt_token_count
        else prompt_unit_count + position
        for position in active_absolute_positions
    )


def build_sentence_unit_ids_for_active_positions(
    *,
    prompt_sentence_ids: Sequence[int],
    active_absolute_positions: Sequence[int],
) -> tuple[int, ...]:
    """Group prompt tokens by sentence and generated tokens as singleton units."""
    return build_unit_ids_for_active_positions(
        prompt_unit_ids=prompt_sentence_ids,
        active_absolute_positions=active_absolute_positions,
    )


def build_word_unit_ids_for_active_positions(
    *,
    prompt_word_ids: Sequence[int],
    active_absolute_positions: Sequence[int],
) -> tuple[int, ...]:
    """Group prompt tokens by word and generated tokens as singleton units."""
    return build_unit_ids_for_active_positions(
        prompt_unit_ids=prompt_word_ids,
        active_absolute_positions=active_absolute_positions,
    )


def select_sentence_retained_positions(
    alignment_scores: np.ndarray,
    *,
    unit_ids: Sequence[int],
    cache_budget_tokens: int,
    always_keep_prefix_tokens: int = 1,
    preserve_recent_window: int = 0,
    pooling: str = "max",
    top_k: int = 3,
) -> tuple[int, ...]:
    """Drop whole semantic units until the retained token count fits the budget."""
    if alignment_scores.ndim != 1:
        raise ValueError("alignment_scores must be rank-1")
    token_count = alignment_scores.shape[0]
    if token_count == 0:
        raise ValueError("alignment_scores must be non-empty")
    if len(unit_ids) != token_count:
        raise ValueError("unit_ids must match the token count")
    if cache_budget_tokens <= 0:
        raise ValueError("cache_budget_tokens must be positive")
    if pooling not in {"max", "mean", "topk"}:
        raise ValueError(f"unknown sentence pooling: {pooling}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if cache_budget_tokens >= token_count:
        return tuple(range(token_count))

    units: list[tuple[int, tuple[int, ...]]] = []
    for local_position, unit_id in enumerate(unit_ids):
        if units and units[-1][0] == unit_id:
            units[-1] = (unit_id, units[-1][1] + (local_position,))
        else:
            units.append((unit_id, (local_position,)))

    required_positions = set(range(min(always_keep_prefix_tokens, token_count)))
    if preserve_recent_window > 0:
        recent_start = max(0, token_count - preserve_recent_window)
        required_positions.update(range(recent_start, token_count))

    required_units = {
        unit_id
        for unit_id, positions in units
        if required_positions.intersection(positions)
    }
    required_token_count = sum(
        len(positions) for unit_id, positions in units if unit_id in required_units
    )
    if cache_budget_tokens < required_token_count:
        raise ValueError(
            "cache_budget_tokens must fit the required prefix/recent sentence units"
        )

    def _unit_score(positions: tuple[int, ...]) -> float:
        values = np.asarray(
            [float(alignment_scores[position]) for position in positions],
            dtype=np.float32,
        )
        if pooling == "max":
            return float(values.max())
        if pooling == "mean":
            return float(values.mean())
        ordered = np.sort(values)[::-1]
        return float(ordered[: min(top_k, ordered.shape[0])].mean())

    total_kept = token_count
    keep_units = {unit_id for unit_id, _positions in units}
    droppable = sorted(
        (
            (unit_id, positions)
            for unit_id, positions in units
            if unit_id not in required_units
        ),
        key=lambda item: (_unit_score(item[1]), -item[1][0]),
    )

    for unit_id, positions in droppable:
        if total_kept <= cache_budget_tokens:
            break
        keep_units.remove(unit_id)
        total_kept -= len(positions)

    kept_positions = [
        position
        for unit_id, positions in units
        if unit_id in keep_units
        for position in positions
    ]
    return tuple(kept_positions)


def aggregate_unit_alignment_scores(
    alignment_scores: np.ndarray,
    *,
    unit_ids: Sequence[int],
    pooling: str = "max",
    top_k: int = 3,
) -> np.ndarray:
    """Aggregate token alignment scores into contiguous unit scores."""
    if alignment_scores.ndim != 1:
        raise ValueError("alignment_scores must be rank-1")
    token_count = alignment_scores.shape[0]
    if len(unit_ids) != token_count:
        raise ValueError("unit_ids must match the token count")
    if pooling not in {"max", "mean", "topk"}:
        raise ValueError(f"unknown sentence pooling: {pooling}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if token_count == 0:
        raise ValueError("alignment_scores must be non-empty")

    groups: list[list[float]] = []
    current_unit: int | None = None
    for local_position, unit_id in enumerate(unit_ids):
        if current_unit != unit_id:
            groups.append([])
            current_unit = unit_id
        groups[-1].append(float(alignment_scores[local_position]))

    aggregated: list[float] = []
    for values in groups:
        scores = np.asarray(values, dtype=np.float32)
        if pooling == "max":
            aggregated.append(float(scores.max()))
        elif pooling == "mean":
            aggregated.append(float(scores.mean()))
        else:
            ordered = np.sort(scores)[::-1]
            aggregated.append(float(ordered[: min(top_k, ordered.shape[0])].mean()))
    return np.asarray(aggregated, dtype=np.float32)


def peak_zscore(scores: np.ndarray) -> float:
    """Measure how strongly one peak stands out inside a score distribution."""
    if scores.ndim != 1:
        raise ValueError("scores must be rank-1")
    if scores.shape[0] == 0:
        raise ValueError("scores must be non-empty")
    mean = float(scores.mean())
    std = float(scores.std())
    if std <= 1e-8:
        return 0.0
    return float((float(scores.max()) - mean) / std)


def select_adaptive_retained_positions(
    alignment_scores: np.ndarray,
    *,
    unit_ids: Sequence[int],
    cache_budget_tokens: int,
    always_keep_prefix_tokens: int = 1,
    preserve_recent_window: int = 0,
    pooling: str = "max",
    top_k: int = 3,
) -> tuple[tuple[int, ...], str, float, float]:
    """Choose between token and sentence retention online from score shape."""
    token_signal = peak_zscore(alignment_scores)
    sentence_scores = aggregate_unit_alignment_scores(
        alignment_scores,
        unit_ids=unit_ids,
        pooling=pooling,
        top_k=top_k,
    )
    sentence_signal = peak_zscore(sentence_scores)

    if sentence_signal > token_signal:
        return (
            select_sentence_retained_positions(
                alignment_scores,
                unit_ids=unit_ids,
                cache_budget_tokens=cache_budget_tokens,
                always_keep_prefix_tokens=always_keep_prefix_tokens,
                preserve_recent_window=preserve_recent_window,
                pooling=pooling,
                top_k=top_k,
            ),
            "sentence",
            token_signal,
            sentence_signal,
        )

    return (
        select_scalar_score_retained_positions(
            alignment_scores,
            cache_budget_tokens=cache_budget_tokens,
            always_keep_prefix_tokens=always_keep_prefix_tokens,
            preserve_recent_window=preserve_recent_window,
        ),
        "token",
        token_signal,
        sentence_signal,
    )



def run_live_eviction(
    plan: ExecutionPlan,
    cases: tuple[BenchmarkCase, ...],
    generator: LiveEvictionTextGenerator,
    *,
    on_case: Callable[[CaseObservation], None] | None = None,
    progress_logger: ProgressLogger | None = None,
) -> tuple[RunResult, tuple[LiveEvictionPredictionTrace, ...]]:
    if plan.run.cache_budget_tokens is None:
        raise ValueError("live eviction baseline requires cache_budget_tokens")

    config = LiveEvictionConfig(
        canonical_layer=plan.run.canonical_layer,
        recent_token_window=plan.run.recent_token_window,
        cache_budget_tokens=plan.run.cache_budget_tokens,
        retention_policy=plan.run.retention_policy or "vorn",
        random_seed=plan.run.random_seed or 17,
        always_keep_prefix_tokens=plan.run.always_keep_prefix_tokens,
        preserve_recent_window=plan.run.preserve_recent_window,
        eviction_unit=plan.run.eviction_unit,
        sentence_pooling=plan.run.sentence_pooling or "max",
        sentence_top_k=plan.run.sentence_top_k or 3,
        eviction_trigger=plan.run.eviction_trigger,
        sentence_boundary_lookahead_tokens=(
            plan.run.sentence_boundary_lookahead_tokens
        ),
        force_eviction_overflow_ratio=plan.run.force_eviction_overflow_ratio,
    )

    predictions: list[str] = []
    traces: list[LiveEvictionPredictionTrace] = []
    mean_retention_total = 0.0
    total_eviction_steps = 0
    total_preprocessing_elapsed_seconds = 0.0
    summary_contract = SEMANTIC_SUMMARY_CONTRACT
    summary_fingerprint = ""
    total_adaptive_token_steps = 0
    total_adaptive_sentence_steps = 0
    adaptive_selector_contract = ""
    edge_kinds: set[str] = set()
    suite_ids: set[str] = set()

    n_cases = len(cases)
    if progress_logger is not None:
        progress_logger(format_dataset_loaded(n_cases))
    running_hits = 0

    for case_index, case in enumerate(cases, start=1):
        prediction, stats = generator.generate_with_live_eviction(
            case.prompt,
            config,
            case_id=case.case_id,
            expected_answer=case.expected_answer,
        )
        predictions.append(prediction)
        edge_kind = case.metadata.get("edge_kind", "")
        suite_id = case.metadata.get("suite_id", "")
        traces.append(
            LiveEvictionPredictionTrace(
                case_id=case.case_id,
                prediction=prediction,
                mean_retention_ratio=stats.mean_retention_ratio,
                eviction_steps=stats.eviction_steps,
                edge_kind=edge_kind,
            )
        )
        observation = build_case_observation(case, prediction)
        if on_case is not None:
            on_case(observation)
        if observation.correct:
            running_hits += 1
        if progress_logger is not None:
            progress_logger(
                format_case_progress(
                    case_index,
                    n_cases,
                    observation.correct,
                    running_hits / case_index,
                )
            )
        mean_retention_total += stats.mean_retention_ratio
        total_eviction_steps += stats.eviction_steps
        total_preprocessing_elapsed_seconds += stats.preprocessing_elapsed_seconds
        summary_contract = stats.summary_contract
        if not summary_fingerprint:
            summary_fingerprint = stats.summary_fingerprint
        total_adaptive_token_steps += stats.adaptive_token_steps
        total_adaptive_sentence_steps += stats.adaptive_sentence_steps
        if not adaptive_selector_contract:
            adaptive_selector_contract = stats.adaptive_selector_contract
        if edge_kind:
            edge_kinds.add(edge_kind)
        if suite_id:
            suite_ids.add(suite_id)

    suite_id_value = ""
    if len(suite_ids) == 1:
        suite_id_value = next(iter(suite_ids))
    elif len(suite_ids) > 1:
        suite_id_value = ",".join(sorted(suite_ids))

    metrics = score_predictions(plan.benchmark.name, cases, tuple(predictions))
    observations = build_case_observations(cases, tuple(predictions))
    if progress_logger is not None:
        progress_logger(
            format_complete(n_cases, running_hits / n_cases if n_cases else 0.0)
        )
    mean_retention_ratio = (
        mean_retention_total / len(cases) if cases else 0.0
    )
    result = RunResult(
        run_id=plan.run.run_id,
        benchmark=plan.benchmark.name,
        baseline=plan.baseline.name,
        metrics=metrics,
        metadata={
            "model": plan.run.model,
            "gpu": plan.run.gpu,
            "canonical_layer": str(plan.run.canonical_layer),
            "recent_token_window": str(plan.run.recent_token_window),
            "cache_budget_tokens": str(plan.run.cache_budget_tokens),
            "retention_policy": config.retention_policy,
            "random_seed": str(config.random_seed),
            "always_keep_prefix_tokens": str(config.always_keep_prefix_tokens),
            "preserve_recent_window": str(config.preserve_recent_window).lower(),
            "eviction_unit": config.eviction_unit,
            "sentence_pooling": config.sentence_pooling,
            "sentence_top_k": str(config.sentence_top_k),
            "eviction_trigger": config.eviction_trigger,
            "sentence_boundary_lookahead_tokens": str(
                config.sentence_boundary_lookahead_tokens
            ),
            "force_eviction_overflow_ratio": (
                f"{config.force_eviction_overflow_ratio:.2f}"
            ),
            "compression_mode": plan.run.compression_mode or "unknown",
            "summary_contract": summary_contract,
            "summary_fingerprint": summary_fingerprint,
            "mean_retention_ratio": f"{mean_retention_ratio:.4f}",
            "total_eviction_steps": str(total_eviction_steps),
            "adaptive_token_steps": str(total_adaptive_token_steps),
            "adaptive_sentence_steps": str(total_adaptive_sentence_steps),
            "adaptive_selector_contract": adaptive_selector_contract,
            "suite_id": suite_id_value,
            "case_count": str(len(cases)),
            "edge_kinds": ",".join(sorted(edge_kinds)),
        },
        preprocessing_elapsed_seconds=total_preprocessing_elapsed_seconds,
        observations=observations,
    )
    return result, tuple(traces)
