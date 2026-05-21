from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat import (
    DEFAULT_LIVE_EVICTION_CACHE_BUDGET,
    DEFAULT_LIVE_EVICTION_CASE_LIMIT,
    DEFAULT_RANDOM_LIVE_EVICTION_SEED,
    LiveEvictionDefaults,
    build_live_eviction_run,
)


def test_live_eviction_run_locks_single_arm_niah_mechanism_probe():
    run = build_live_eviction_run()

    assert run.run_id == "step2-niah-vorn-live"
    assert run.benchmark == "niah"
    assert run.baseline == "vorn_live"
    assert run.case_limit == DEFAULT_LIVE_EVICTION_CASE_LIMIT
    assert run.cache_budget_tokens == DEFAULT_LIVE_EVICTION_CACHE_BUDGET
    assert run.experiment_stage == "step2"
    assert run.compression_mode == "live_eviction_only"


def test_live_eviction_run_supports_budget_sweep_variants():
    run = build_live_eviction_run(
        live=LiveEvictionDefaults(
            cache_budget_tokens=512,
            baseline="vorn_live",
            retention_policy="vorn",
            random_seed=DEFAULT_RANDOM_LIVE_EVICTION_SEED,
            compression_mode="live_eviction_only_vorn",
        )
    )

    assert run.run_id == "step2-niah-vorn-live-b512"
    assert run.baseline == "vorn_live"
    assert run.cache_budget_tokens == 512
    assert run.retention_policy == "vorn"
    assert run.random_seed == DEFAULT_RANDOM_LIVE_EVICTION_SEED


def test_live_eviction_run_supports_random_control_arm():
    run = build_live_eviction_run(
        live=LiveEvictionDefaults(
            baseline="random_live",
            retention_policy="random",
            random_seed=23,
            compression_mode="live_eviction_only_random",
        )
    )

    assert run.run_id == "step2-niah-random-live-b256"
    assert run.baseline == "random_live"
    assert run.retention_policy == "random"
    assert run.random_seed == 23


def test_live_eviction_run_supports_tova_control_arm():
    run = build_live_eviction_run(
        live=LiveEvictionDefaults(
            cache_budget_tokens=1024,
            baseline="tova_live",
            retention_policy="tova",
            random_seed=17,
            compression_mode="live_eviction_only_tova",
        )
    )

    assert run.run_id == "step2-niah-tova-live-b1024"
    assert run.baseline == "tova_live"
    assert run.retention_policy == "tova"


def test_live_eviction_run_supports_h2o_control_arm():
    run = build_live_eviction_run(
        live=LiveEvictionDefaults(
            cache_budget_tokens=1024,
            baseline="h2o_live",
            retention_policy="h2o",
            random_seed=17,
            compression_mode="live_eviction_only_h2o",
        )
    )

    assert run.run_id == "step2-niah-h2o-live-b1024"
    assert run.baseline == "h2o_live"
    assert run.retention_policy == "h2o"


def test_live_eviction_run_supports_no_guardrails_vorn_variant():
    run = build_live_eviction_run(
        live=LiveEvictionDefaults(
            cache_budget_tokens=1024,
            baseline="vorn_live",
            retention_policy="vorn",
            random_seed=17,
            always_keep_prefix_tokens=0,
            preserve_recent_window=False,
            compression_mode="live_eviction_only_vorn_no_guardrails",
        )
    )

    assert run.run_id == "step2-niah-vorn-live-b1024-noguards"
    assert run.always_keep_prefix_tokens == 0
    assert run.preserve_recent_window is False


def test_live_eviction_run_supports_sentence_level_variants():
    run = build_live_eviction_run(
        live=LiveEvictionDefaults(
            cache_budget_tokens=1024,
            baseline="sentence_vorn_live",
            retention_policy="sentence_vorn",
            random_seed=17,
            eviction_unit="sentence",
            sentence_pooling="max",
            sentence_top_k=3,
            compression_mode="live_eviction_sentence_vorn_max",
        )
    )

    assert run.run_id == "step2-niah-sentence-vorn-live-b1024"
    assert run.baseline == "sentence_vorn_live"
    assert run.retention_policy == "sentence_vorn"
    assert run.eviction_unit == "sentence"
    assert run.sentence_pooling == "max"
    assert run.sentence_top_k == 3


def test_live_eviction_run_supports_sentence_level_no_guardrails_variant():
    run = build_live_eviction_run(
        live=LiveEvictionDefaults(
            cache_budget_tokens=1024,
            baseline="sentence_vorn_live",
            retention_policy="sentence_vorn",
            random_seed=17,
            always_keep_prefix_tokens=0,
            preserve_recent_window=False,
            eviction_unit="sentence",
            sentence_pooling="max",
            sentence_top_k=3,
            compression_mode="live_eviction_sentence_vorn_max_no_guardrails",
        )
    )

    assert run.run_id == "step2-niah-sentence-vorn-live-b1024-noguards"
    assert run.always_keep_prefix_tokens == 0
    assert run.preserve_recent_window is False
    assert run.eviction_unit == "sentence"

def test_live_eviction_run_supports_sentence_boundary_trigger_variant():
    run = build_live_eviction_run(
        live=LiveEvictionDefaults(
            cache_budget_tokens=1536,
            baseline="sentence_vorn_live",
            retention_policy="sentence_vorn",
            random_seed=17,
            eviction_unit="sentence",
            sentence_pooling="max",
            sentence_top_k=3,
            eviction_trigger="sentence_boundary",
            sentence_boundary_lookahead_tokens=25,
            force_eviction_overflow_ratio=1.2,
            compression_mode="live_eviction_sentence_vorn_max_sentence_boundary",
        )
    )

    assert run.run_id == "step2-niah-sentence-vorn-live-b1536-sentbound"
    assert run.eviction_trigger == "sentence_boundary"
    assert run.sentence_boundary_lookahead_tokens == 25
    assert run.force_eviction_overflow_ratio == 1.2


def test_live_eviction_run_supports_word_level_variants():
    run = build_live_eviction_run(
        live=LiveEvictionDefaults(
            cache_budget_tokens=1024,
            baseline="word_vorn_live",
            retention_policy="word_vorn",
            random_seed=17,
            eviction_unit="word",
            sentence_pooling="max",
            sentence_top_k=3,
            compression_mode="live_eviction_word_vorn_max",
        )
    )

    assert run.run_id == "step2-niah-word-vorn-live-b1024"
    assert run.baseline == "word_vorn_live"
    assert run.retention_policy == "word_vorn"
    assert run.eviction_unit == "word"
    assert run.sentence_pooling == "max"
    assert run.sentence_top_k == 3


def test_live_eviction_run_supports_word_level_no_guardrails_variant():
    run = build_live_eviction_run(
        live=LiveEvictionDefaults(
            cache_budget_tokens=1024,
            baseline="word_vorn_live",
            retention_policy="word_vorn",
            random_seed=17,
            always_keep_prefix_tokens=0,
            preserve_recent_window=False,
            eviction_unit="word",
            sentence_pooling="max",
            sentence_top_k=3,
            compression_mode="live_eviction_word_vorn_max_no_guardrails",
        )
    )

    assert run.run_id == "step2-niah-word-vorn-live-b1024-noguards"
    assert run.always_keep_prefix_tokens == 0
    assert run.preserve_recent_window is False
    assert run.eviction_unit == "word"


def test_live_eviction_run_supports_adaptive_vorn_variant():
    run = build_live_eviction_run(
        live=LiveEvictionDefaults(
            cache_budget_tokens=1536,
            baseline="adaptive_vorn_live",
            retention_policy="adaptive_vorn",
            random_seed=17,
            eviction_unit="adaptive_token_or_sentence",
            sentence_pooling="max",
            sentence_top_k=3,
            compression_mode="live_eviction_adaptive_vorn_max",
        )
    )

    assert run.run_id == "step2-niah-adaptive-vorn-live-b1536"
    assert run.baseline == "adaptive_vorn_live"
    assert run.retention_policy == "adaptive_vorn"
    assert run.eviction_unit == "adaptive_token_or_sentence"
    assert run.sentence_pooling == "max"
