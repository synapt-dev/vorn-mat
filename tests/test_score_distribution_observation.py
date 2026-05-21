from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat.baselines.live_eviction import (
    assign_word_ids_from_offsets,
    build_word_unit_ids_for_active_positions,
)
from vorn_mat.score_distribution_observation import (
    adjacency_fraction,
    score_distribution_stats,
)
from vorn_mat.text_spans import word_char_spans


def test_word_char_spans_splits_words_and_punctuation() -> None:
    text = "Alpha, beta!"

    spans = word_char_spans(text)

    assert spans == ((0, 5), (5, 6), (7, 11), (11, 12))


def test_assign_word_ids_from_offsets_maps_tokens_into_word_order() -> None:
    text = "Alpha, beta!"
    offsets = (
        (0, 5),
        (5, 6),
        (7, 11),
        (11, 12),
    )

    word_ids = assign_word_ids_from_offsets(text, offsets)

    assert word_ids == (0, 1, 2, 3)


def test_build_word_unit_ids_uses_prompt_word_ids_and_singletons_for_generated() -> None:
    unit_ids = build_word_unit_ids_for_active_positions(
        prompt_word_ids=(0, 1, 2, 3),
        active_absolute_positions=(0, 1, 2, 3, 4, 5),
    )

    assert unit_ids == (0, 1, 2, 3, 8, 9)


def test_adjacency_fraction_measures_top_index_clustering() -> None:
    assert adjacency_fraction((2, 3, 4, 9)) == 2 / 3
    assert adjacency_fraction((1, 4, 8)) == 0.0


def test_score_distribution_stats_tracks_mass_entropy_and_threshold_shape() -> None:
    scores = np.asarray([5.0] + [0.0] * 19, dtype=np.float32)

    stats = score_distribution_stats(scores)

    assert stats.position_count == 20
    assert stats.peak_zscore > 4.0
    assert stats.top10_mass_fraction > 0.9
    assert stats.top25_mass_fraction == 1.0
    assert 0.0 <= stats.normalized_entropy < 0.3
    assert stats.kl_divergence_from_uniform > 0.0
    assert stats.q90_minus_q50 == 0.0
    assert stats.above_median_plus_std_count == 1
    assert stats.above_median_plus_std_fraction == 0.05
