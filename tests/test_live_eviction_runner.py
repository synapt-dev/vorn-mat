from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat import (
    build_execution_plans,
    build_live_eviction_run,
    extract_canonical_token_summaries,
    run_live_eviction,
    summary_fingerprint,
)
from vorn_mat.baselines.live_eviction import (
    ADAPTIVE_SELECTOR_CONTRACT,
    LiveEvictionStats,
    LiveEvictionTextGenerator,
    accumulate_attention_scores,
    extract_last_token_attention_scores,
    select_attention_score_retained_positions,
    select_prefix_suffix_retained_positions,
    select_random_retained_positions,
    select_sliding_window_retained_positions,
)
from vorn_mat.benchmarks.common import BenchmarkCase
from vorn_mat.plan import LiveEvictionDefaults


class FakeLiveEvictionGenerator(LiveEvictionTextGenerator):
    def __init__(self, answers: dict[str, str], stats: LiveEvictionStats):
        self.answers = answers
        self.stats = stats

    def generate_with_live_eviction(self, prompt: str, config):
        return self.answers[prompt], self.stats


def test_extract_canonical_token_summaries_is_float32_and_deterministic():
    hidden_states = (
        torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float16),
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float16),
        torch.tensor([[[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float16),
    )

    first = extract_canonical_token_summaries(hidden_states, canonical_layer=1)
    second = extract_canonical_token_summaries(hidden_states, canonical_layer=1)

    assert first.dtype == np.float32
    assert first.shape == (2, 2)
    assert np.array_equal(first, second)
    assert np.array_equal(first, np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))


def test_extract_canonical_token_summaries_accepts_bfloat16_boundary():
    hidden_states = (
        torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.bfloat16),
        torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.bfloat16),
    )

    summaries = extract_canonical_token_summaries(hidden_states, canonical_layer=0)

    assert summaries.dtype == np.float32
    assert np.array_equal(
        summaries,
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )


def test_summary_fingerprint_is_stable_for_same_input():
    summaries = np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32)

    first = summary_fingerprint(summaries)
    second = summary_fingerprint(summaries.copy())

    assert first == second


def test_select_random_retained_positions_is_seeded_and_constraint_aware():
    rng = np.random.default_rng(17)

    first = select_random_retained_positions(
        token_count=10,
        cache_budget_tokens=5,
        always_keep_prefix_tokens=1,
        preserve_recent_window=2,
        rng=rng,
    )
    second = select_random_retained_positions(
        token_count=10,
        cache_budget_tokens=5,
        always_keep_prefix_tokens=1,
        preserve_recent_window=2,
        rng=np.random.default_rng(17),
    )

    assert first == second
    assert first[0] == 0
    assert 8 in first and 9 in first
    assert len(first) == 5


def test_select_sliding_window_retained_positions_keeps_recent_budget_only():
    kept = select_sliding_window_retained_positions(
        token_count=10,
        cache_budget_tokens=4,
    )

    assert kept == (6, 7, 8, 9)


def test_select_prefix_suffix_retained_positions_drops_middle():
    kept = select_prefix_suffix_retained_positions(
        token_count=20,
        cache_budget_tokens=8,
        prefix_token_count=3,
    )

    assert kept == (0, 1, 2, 15, 16, 17, 18, 19)


def test_extract_last_token_attention_scores_averages_heads_from_final_layer():
    attentions = (
        torch.zeros((1, 2, 2, 3), dtype=torch.float32),
        torch.tensor(
            [
                [
                    [[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]],
                    [[0.0, 0.5, 0.5], [0.1, 0.6, 0.3]],
                ]
            ],
            dtype=torch.float32,
        ),
    )

    scores = extract_last_token_attention_scores(attentions)

    assert scores.dtype == np.float32
    assert np.allclose(scores, np.array([0.2, 0.45, 0.35], dtype=np.float32))


def test_select_attention_score_retained_positions_keeps_top_scores_with_constraints():
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.7], dtype=np.float32)

    kept = select_attention_score_retained_positions(
        scores,
        cache_budget_tokens=3,
        always_keep_prefix_tokens=1,
        preserve_recent_window=1,
    )

    assert kept == (0, 1, 4)


def test_accumulate_attention_scores_adds_stepwise_mass():
    running = np.array([0.2, 0.1, 0.0], dtype=np.float32)
    current = np.array([0.3, 0.4, 0.5], dtype=np.float32)

    updated = accumulate_attention_scores(running, current)

    assert np.allclose(updated, np.array([0.5, 0.5, 0.5], dtype=np.float32))


def test_run_live_eviction_emits_result_and_contract_metadata():
    plan = build_execution_plans((build_live_eviction_run(),))[0]
    cases = (
        BenchmarkCase("case-1", "prompt-1", "amber", {}),
        BenchmarkCase("case-2", "prompt-2", "cedar", {}),
    )
    generator = FakeLiveEvictionGenerator(
        {"prompt-1": "amber", "prompt-2": "wrong"},
        LiveEvictionStats(
            prompt_token_count=40,
            generated_token_count=3,
            mean_kept_token_count=18.0,
            final_kept_token_count=19,
            eviction_steps=2,
            mean_retention_ratio=0.45,
            retention_policy="vorn",
            summary_contract="canonical_hidden_state_float32_per_token_from_layer_L_star",
            summary_fingerprint="abc123",
            preprocessing_elapsed_seconds=1.5,
        ),
    )

    result, traces = run_live_eviction(plan, cases, generator)

    assert result.run_id == "step2-niah-vorn-live"
    assert result.benchmark == "niah"
    assert result.baseline == "vorn_live"
    assert result.metrics == {"needle_hit_rate": 0.5}
    assert result.metadata["cache_budget_tokens"] == "256"
    assert result.metadata["retention_policy"] == "vorn"
    assert result.metadata["random_seed"] == "17"
    assert result.metadata["always_keep_prefix_tokens"] == "1"
    assert result.metadata["preserve_recent_window"] == "true"
    assert result.metadata["eviction_unit"] == "token_position"
    assert result.metadata["sentence_pooling"] == "max"
    assert result.metadata["sentence_top_k"] == "3"
    assert result.metadata["compression_mode"] == "live_eviction_only"
    assert result.metadata["summary_contract"] == (
        "canonical_hidden_state_float32_per_token_from_layer_L_star"
    )
    assert result.metadata["summary_fingerprint"] == "abc123"
    assert result.metadata["mean_retention_ratio"] == "0.4500"
    assert result.metadata["total_eviction_steps"] == "4"
    assert result.preprocessing_elapsed_seconds == 3.0
    assert [observation.fixture_id for observation in result.observations] == [
        "case-1",
        "case-2",
    ]
    assert [observation.correct for observation in result.observations] == [True, False]
    assert len(traces) == 2
    assert traces[0].eviction_steps == 2


def test_run_live_eviction_emits_adaptive_selector_metadata():
    plan = build_execution_plans(
        (
            build_live_eviction_run(
                live=LiveEvictionDefaults(
                    cache_budget_tokens=1024,
                    baseline="adaptive_vorn_live",
                    retention_policy="adaptive_vorn",
                    eviction_unit="adaptive_token_or_sentence",
                    sentence_pooling="max",
                    sentence_top_k=3,
                    compression_mode="live_eviction_adaptive_vorn_max",
                )
            ),
        )
    )[0]
    cases = (
        BenchmarkCase("case-1", "prompt-1", "amber", {}),
        BenchmarkCase("case-2", "prompt-2", "cedar", {}),
    )
    generator = FakeLiveEvictionGenerator(
        {"prompt-1": "amber", "prompt-2": "wrong"},
        LiveEvictionStats(
            prompt_token_count=40,
            generated_token_count=3,
            mean_kept_token_count=18.0,
            final_kept_token_count=19,
            eviction_steps=2,
            mean_retention_ratio=0.45,
            retention_policy="adaptive_vorn",
            summary_contract="canonical_hidden_state_float32_per_token_from_layer_L_star",
            summary_fingerprint="abc123",
            adaptive_token_steps=1,
            adaptive_sentence_steps=1,
            adaptive_selector_contract=ADAPTIVE_SELECTOR_CONTRACT,
        ),
    )

    result, _traces = run_live_eviction(plan, cases, generator)

    assert result.baseline == "adaptive_vorn_live"
    assert result.metadata["retention_policy"] == "adaptive_vorn"
    assert result.metadata["adaptive_token_steps"] == "2"
    assert result.metadata["adaptive_sentence_steps"] == "2"
    assert result.metadata["adaptive_selector_contract"] == ADAPTIVE_SELECTOR_CONTRACT
