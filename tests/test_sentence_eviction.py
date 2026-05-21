from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat.baselines.live_eviction import (
    aggregate_unit_alignment_scores,
    assign_sentence_ids_from_offsets,
    assign_word_ids_from_offsets,
    build_unit_ids_for_active_positions,
    build_sentence_unit_ids_for_active_positions,
    peak_zscore,
    select_adaptive_retained_positions,
    select_sentence_retained_positions,
    should_trigger_sentence_boundary_eviction,
    text_ends_at_sentence_boundary,
)
from vorn_mat.text_spans import sentence_char_spans, word_char_spans


def test_sentence_char_spans_splits_on_terminal_punctuation_and_newlines():
    text = "Alpha. Beta?\nGamma!"

    spans = sentence_char_spans(text)

    assert spans == ((0, 6), (7, 12), (13, 19))


def test_assign_sentence_ids_from_offsets_maps_tokens_into_sentence_order():
    text = "Alpha. Beta? Gamma!"
    offsets = (
        (0, 5),
        (5, 6),
        (7, 11),
        (11, 12),
        (13, 18),
        (18, 19),
    )

    sentence_ids = assign_sentence_ids_from_offsets(text, offsets)

    assert sentence_ids == (0, 0, 1, 1, 2, 2)


def test_word_char_spans_splits_on_whitespace_and_punctuation_boundaries():
    text = "Alpha beta, gamma-delta."

    spans = word_char_spans(text)

    assert spans == (
        (0, 5),
        (6, 10),
        (10, 11),
        (12, 17),
        (17, 18),
        (18, 23),
        (23, 24),
    )


def test_assign_word_ids_from_offsets_maps_tokens_into_word_order():
    text = "Alpha beta gamma"
    offsets = (
        (0, 5),
        (6, 10),
        (11, 16),
    )

    word_ids = assign_word_ids_from_offsets(text, offsets)

    assert word_ids == (0, 1, 2)


def test_build_sentence_unit_ids_uses_prompt_sentence_ids_and_singletons_for_generated():
    unit_ids = build_sentence_unit_ids_for_active_positions(
        prompt_sentence_ids=(0, 0, 1, 1),
        active_absolute_positions=(0, 1, 2, 3, 4, 5),
    )

    assert unit_ids == (0, 0, 1, 1, 6, 7)


def test_build_unit_ids_uses_prompt_units_and_singletons_for_generated():
    unit_ids = build_unit_ids_for_active_positions(
        prompt_unit_ids=(0, 1, 2),
        active_absolute_positions=(0, 1, 2, 3, 4),
    )

    assert unit_ids == (0, 1, 2, 6, 7)


def test_select_sentence_retained_positions_keeps_whole_required_units():
    scores = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2], dtype=np.float32)
    unit_ids = (0, 0, 0, 1, 1, 1)

    kept = select_sentence_retained_positions(
        scores,
        unit_ids=unit_ids,
        cache_budget_tokens=3,
        always_keep_prefix_tokens=1,
        preserve_recent_window=0,
        pooling="max",
    )

    assert kept == (0, 1, 2)


def test_select_sentence_retained_positions_promotes_recent_window_to_whole_sentence():
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    unit_ids = (0, 0, 1, 1, 2, 2)

    kept = select_sentence_retained_positions(
        scores,
        unit_ids=unit_ids,
        cache_budget_tokens=4,
        always_keep_prefix_tokens=0,
        preserve_recent_window=1,
        pooling="max",
    )

    assert kept == (2, 3, 4, 5)


def test_select_sentence_retained_positions_supports_topk_pooling():
    scores = np.array([0.9, 0.1, 0.4, 0.4, 0.5, 0.5], dtype=np.float32)
    unit_ids = (0, 0, 1, 1, 2, 2)

    kept = select_sentence_retained_positions(
        scores,
        unit_ids=unit_ids,
        cache_budget_tokens=4,
        always_keep_prefix_tokens=0,
        preserve_recent_window=0,
        pooling="topk",
        top_k=2,
    )

    assert kept == (0, 1, 4, 5)


def test_text_ends_at_sentence_boundary_requires_terminal_punctuation():
    assert text_ends_at_sentence_boundary("Alpha. Beta?")
    assert text_ends_at_sentence_boundary("Alpha.\nBeta!   ")
    assert not text_ends_at_sentence_boundary("Alpha Beta")


def test_sentence_boundary_trigger_waits_for_boundary_when_projection_overflows():
    assert not should_trigger_sentence_boundary_eviction(
        rendered_text="The answer is still unfolding",
        token_count=1540,
        cache_budget_tokens=1536,
        lookahead_tokens=25,
        force_overflow_ratio=1.2,
    )
    assert should_trigger_sentence_boundary_eviction(
        rendered_text="The answer is now complete.",
        token_count=1540,
        cache_budget_tokens=1536,
        lookahead_tokens=25,
        force_overflow_ratio=1.2,
    )


def test_sentence_boundary_trigger_forces_eviction_at_hard_cap():
    assert should_trigger_sentence_boundary_eviction(
        rendered_text="still mid thought",
        token_count=1900,
        cache_budget_tokens=1536,
        lookahead_tokens=25,
        force_overflow_ratio=1.2,
    )


def test_aggregate_unit_alignment_scores_respects_contiguous_units():
    scores = np.array([0.9, 0.1, 0.4, 0.4, 0.5, 0.5], dtype=np.float32)
    unit_ids = (0, 0, 1, 1, 2, 2)

    aggregated = aggregate_unit_alignment_scores(
        scores,
        unit_ids=unit_ids,
        pooling="topk",
        top_k=2,
    )

    assert np.allclose(aggregated, np.array([0.5, 0.4, 0.5], dtype=np.float32))


def test_peak_zscore_returns_zero_for_flat_distribution():
    scores = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    assert peak_zscore(scores) == 0.0


def test_select_adaptive_retained_positions_prefers_sentence_when_sentence_peak_is_stronger():
    scores = np.array([0.9, 0.89, 0.88, 0.87, 0.1, 0.1], dtype=np.float32)
    unit_ids = (0, 0, 0, 0, 1, 1)

    kept, selected_unit, token_signal, sentence_signal = (
        select_adaptive_retained_positions(
            scores,
            unit_ids=unit_ids,
            cache_budget_tokens=4,
            always_keep_prefix_tokens=0,
            preserve_recent_window=0,
            pooling="max",
        )
    )

    assert kept == (0, 1, 2, 3)
    assert selected_unit == "sentence"
    assert sentence_signal > token_signal


def test_select_adaptive_retained_positions_prefers_token_when_sentence_pooling_blurs_signal():
    scores = np.array([0.9, 0.1, 0.4, 0.4, 0.5, 0.5], dtype=np.float32)
    unit_ids = (0, 0, 1, 1, 2, 2)

    kept, selected_unit, token_signal, sentence_signal = (
        select_adaptive_retained_positions(
            scores,
            unit_ids=unit_ids,
            cache_budget_tokens=4,
            always_keep_prefix_tokens=0,
            preserve_recent_window=0,
            pooling="max",
        )
    )

    assert kept == (0, 2, 4, 5)
    assert selected_unit == "token"
    assert token_signal >= sentence_signal
