from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat import (
    A100_80GB_PER_SECOND,
    DEFAULT_CANONICAL_LAYER,
    DEFAULT_MODEL,
    DEFAULT_RECENT_WINDOW,
    ExperimentDefaults,
    build_week1_run_matrix,
    estimate_week1_cost_window,
    middle_layer_index,
    per_hour_rate,
)


def test_middle_layer_index_uses_zero_based_middle():
    assert middle_layer_index(32) == 16
    assert middle_layer_index(1) == 0


def test_experiment_defaults_match_merged_research_spec():
    defaults = ExperimentDefaults()

    assert defaults.model == DEFAULT_MODEL
    assert defaults.canonical_layer == DEFAULT_CANONICAL_LAYER
    assert defaults.recent_token_window == DEFAULT_RECENT_WINDOW
    assert defaults.eviction_unit == "token_position"
    assert defaults.retrieval_space == "canonical_residual_layer"
    assert defaults.gpu == "A100-80GB"


def test_week1_run_matrix_covers_required_baselines_and_benchmarks():
    runs = build_week1_run_matrix()

    assert len(runs) == 6
    assert {run.baseline for run in runs} == {"vanilla", "h2o", "streamingllm"}
    assert {run.benchmark for run in runs} == {"ruler", "niah"}
    assert all(run.model == DEFAULT_MODEL for run in runs)
    assert all(run.canonical_layer == DEFAULT_CANONICAL_LAYER for run in runs)
    assert all(run.recent_token_window == DEFAULT_RECENT_WINDOW for run in runs)


def test_week1_run_ids_are_stable():
    runs = build_week1_run_matrix()

    assert [run.run_id for run in runs] == [
        "week1-ruler-vanilla",
        "week1-ruler-h2o",
        "week1-ruler-streamingllm",
        "week1-niah-vanilla",
        "week1-niah-h2o",
        "week1-niah-streamingllm",
    ]


def test_a100_cost_window_matches_spec():
    lower, upper = estimate_week1_cost_window()

    assert round(per_hour_rate(A100_80GB_PER_SECOND), 4) == 2.4984
    assert round(lower, 3) == 524.664
    assert round(upper, 3) == 874.440
