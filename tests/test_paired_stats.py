from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from vorn_mat.paired_stats import build_paired_correctness_table, exact_mcnemar
from vorn_mat.results import CaseObservation


def test_build_paired_correctness_table_counts_discordant_pairs():
    lhs = (
        CaseObservation(fixture_id="a", correct=True, prediction="x"),
        CaseObservation(fixture_id="b", correct=True, prediction="x"),
        CaseObservation(fixture_id="c", correct=False, prediction="x"),
        CaseObservation(fixture_id="d", correct=False, prediction="x"),
    )
    rhs = (
        CaseObservation(fixture_id="a", correct=True, prediction="x"),
        CaseObservation(fixture_id="b", correct=False, prediction="x"),
        CaseObservation(fixture_id="c", correct=True, prediction="x"),
        CaseObservation(fixture_id="d", correct=False, prediction="x"),
    )

    table = build_paired_correctness_table(lhs, rhs)

    assert table == ((1, 1), (1, 1))


def test_build_paired_correctness_table_requires_matching_fixture_sets():
    lhs = (CaseObservation(fixture_id="a", correct=True, prediction="x"),)
    rhs = (CaseObservation(fixture_id="b", correct=True, prediction="x"),)

    with pytest.raises(ValueError, match="same fixture ids"):
        build_paired_correctness_table(lhs, rhs)


def test_exact_mcnemar_returns_exact_two_sided_p_value():
    result = exact_mcnemar(((0, 8), (0, 42)))

    assert result.discordant_pairs == 8
    assert result.p_value == 0.0078125
    assert result.lhs_discordant_share == 1.0


def test_exact_mcnemar_handles_zero_discordant_pairs():
    result = exact_mcnemar(((10, 0), (0, 40)))

    assert result.discordant_pairs == 0
    assert result.p_value == 1.0
    assert result.lhs_discordant_share is None
