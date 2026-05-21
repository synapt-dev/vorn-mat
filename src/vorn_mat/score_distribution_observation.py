"""Score-distribution shape summaries for granularity observation runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .baselines.live_eviction import peak_zscore
from .results import CaseObservation


@dataclass(frozen=True)
class ScoreDistributionStats:
    position_count: int
    score_min: float
    score_max: float
    score_mean: float
    score_median: float
    score_std: float
    score_q10: float
    score_q25: float
    score_q75: float
    score_q90: float
    peak_zscore: float
    top10_mass_fraction: float
    top25_mass_fraction: float
    top50_mass_fraction: float
    entropy: float
    normalized_entropy: float
    kl_divergence_from_uniform: float
    q90_minus_q50: float
    q75_minus_q25: float
    above_median_plus_std_count: int
    above_median_plus_std_fraction: float
    spatial_coherence: float


@dataclass(frozen=True)
class ScoreDistributionObservationStep:
    step_index: int
    active_token_count: int
    granularity_stats: dict[str, ScoreDistributionStats]


@dataclass(frozen=True)
class ScoreDistributionObservationCase:
    case_id: str
    expected_answer: str
    prediction: str
    success: bool
    observations: tuple[CaseObservation, ...]
    steps: tuple[ScoreDistributionObservationStep, ...]


@dataclass(frozen=True)
class ScoreDistributionObservationReport:
    dataset_config: str
    split: str
    case_count: int
    cache_budget_tokens: int
    retention_policy: str
    always_keep_prefix_tokens: int
    preserve_recent_window: bool
    sentence_pooling: str
    sentence_top_k: int
    model_id: str
    elapsed_seconds: float
    estimated_cost_usd: float
    cases: tuple[ScoreDistributionObservationCase, ...]


def adjacency_fraction(indices: Sequence[int]) -> float:
    if len(indices) <= 1:
        return 1.0
    adjacent_pairs = sum(
        1
        for left, right in zip(indices, indices[1:], strict=False)
        if right - left == 1
    )
    return adjacent_pairs / (len(indices) - 1)


def score_distribution_stats(scores: np.ndarray) -> ScoreDistributionStats:
    if scores.ndim != 1:
        raise ValueError("scores must be rank-1")
    if scores.shape[0] == 0:
        raise ValueError("scores must be non-empty")

    values = np.asarray(scores, dtype=np.float32)
    probabilities = _softmax_probabilities(values)
    quantiles = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    median = float(quantiles[2])
    std = float(values.std())
    threshold = median + std
    above_threshold_count = int((values > threshold).sum())
    ranked_indices = sorted(
        range(values.shape[0]),
        key=lambda index: (float(values[index]), -index),
        reverse=True,
    )[: min(10, values.shape[0])]
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
    max_entropy = math.log(values.shape[0]) if values.shape[0] > 1 else 0.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return ScoreDistributionStats(
        position_count=int(values.shape[0]),
        score_min=float(values.min()),
        score_max=float(values.max()),
        score_mean=float(values.mean()),
        score_median=median,
        score_std=std,
        score_q10=float(quantiles[0]),
        score_q25=float(quantiles[1]),
        score_q75=float(quantiles[3]),
        score_q90=float(quantiles[4]),
        peak_zscore=peak_zscore(values),
        top10_mass_fraction=_top_k_mass_fraction(probabilities, k=10),
        top25_mass_fraction=_top_k_mass_fraction(probabilities, k=25),
        top50_mass_fraction=_top_k_mass_fraction(probabilities, k=50),
        entropy=entropy,
        normalized_entropy=normalized_entropy,
        kl_divergence_from_uniform=(
            max_entropy - entropy if max_entropy > 0 else 0.0
        ),
        q90_minus_q50=float(quantiles[4] - quantiles[2]),
        q75_minus_q25=float(quantiles[3] - quantiles[1]),
        above_median_plus_std_count=above_threshold_count,
        above_median_plus_std_fraction=(
            above_threshold_count / values.shape[0]
        ),
        spatial_coherence=adjacency_fraction(tuple(sorted(ranked_indices))),
    )


def _softmax_probabilities(scores: np.ndarray) -> np.ndarray:
    shifted = scores - float(scores.max())
    exp = np.exp(shifted, dtype=np.float64)
    total = float(exp.sum())
    if total <= 0.0:
        raise ValueError("softmax normalization must be positive")
    return np.asarray(exp / total, dtype=np.float64)


def _top_k_mass_fraction(probabilities: np.ndarray, *, k: int) -> float:
    if probabilities.ndim != 1:
        raise ValueError("probabilities must be rank-1")
    if k <= 0:
        raise ValueError("k must be positive")
    ordered = np.sort(probabilities)[::-1]
    return float(min(1.0, max(0.0, ordered[: min(k, ordered.shape[0])].sum())))


def write_score_distribution_observation_report(
    report: ScoreDistributionObservationReport,
    *,
    json_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
