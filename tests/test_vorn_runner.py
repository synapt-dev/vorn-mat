from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat import (
    build_execution_plans,
    build_step1_run_matrix,
    compute_vorn_direction,
    run_vorn,
    select_retained_positions,
)
from vorn_mat.baselines.vorn import RetentionStats, VornTextGenerator
from vorn_mat.benchmarks.common import BenchmarkCase


class FakeVornGenerator(VornTextGenerator):
    def __init__(self, answers: dict[str, str], stats: RetentionStats):
        self.answers = answers
        self.stats = stats

    def generate_with_retention(self, prompt: str, config):
        return self.answers[prompt], self.stats


def test_compute_vorn_direction_pools_recent_window():
    summaries = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    vorn = compute_vorn_direction(summaries, recent_token_window=2)

    assert np.allclose(vorn, np.array([1.0, 0.0], dtype=np.float32))


def test_select_retained_positions_preserves_required_and_alignment_budget():
    summaries = np.array(
        [
            [0.0, 1.0],  # required prefix
            [1.0, 0.0],  # highest aligned discretionary token
            [0.1, 0.9],
            [1.0, 0.0],  # required recent token
        ],
        dtype=np.float32,
    )
    vorn = np.array([1.0, 0.0], dtype=np.float32)

    kept = select_retained_positions(
        summaries,
        vorn,
        cache_budget_tokens=3,
        always_keep_prefix_tokens=1,
        preserve_recent_window=1,
    )

    assert kept == (0, 1, 3)


def test_run_vorn_emits_result_and_retention_metadata():
    plan = build_execution_plans(build_step1_run_matrix())[1]
    cases = (
        BenchmarkCase("case-1", "prompt-1", "amber", {}),
        BenchmarkCase("case-2", "prompt-2", "cedar", {}),
    )
    generator = FakeVornGenerator(
        {"prompt-1": "amber", "prompt-2": "wrong"},
        RetentionStats(
            original_token_count=20,
            kept_token_count=8,
            kept_positions=(0, 4, 9, 10, 11, 12, 18, 19),
            dropped_positions=(1, 2, 3, 5, 6, 7, 8, 13, 14, 15, 16, 17),
        ),
    )

    result, traces = run_vorn(plan, cases, generator)

    assert result.run_id == "step1-niah-vorn"
    assert result.benchmark == "niah"
    assert result.baseline == "vorn"
    assert result.metrics == {"needle_hit_rate": 0.5}
    assert result.metadata["cache_budget_tokens"] == "256"
    assert result.metadata["compression_mode"] == "prompt_retention_proxy"
    assert result.metadata["mean_retention_ratio"] == "0.4000"
    assert len(traces) == 2
    assert traces[0].kept_token_count == 8
