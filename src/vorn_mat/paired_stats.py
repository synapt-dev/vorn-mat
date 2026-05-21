"""Paired-case significance helpers for same-slice benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

from .results import CaseObservation


@dataclass(frozen=True)
class McNemarExactResult:
    table: tuple[tuple[int, int], tuple[int, int]]
    both_correct: int
    lhs_correct_rhs_wrong: int
    lhs_wrong_rhs_correct: int
    both_wrong: int
    discordant_pairs: int
    p_value: float
    lhs_discordant_share: float | None


def build_paired_correctness_table(
    lhs: tuple[CaseObservation, ...],
    rhs: tuple[CaseObservation, ...],
) -> tuple[tuple[int, int], tuple[int, int]]:
    lhs_map = {observation.fixture_id: observation.correct for observation in lhs}
    rhs_map = {observation.fixture_id: observation.correct for observation in rhs}
    if set(lhs_map) != set(rhs_map):
        raise ValueError("paired observations must cover the same fixture ids")

    both_correct = 0
    lhs_correct_rhs_wrong = 0
    lhs_wrong_rhs_correct = 0
    both_wrong = 0
    for fixture_id in sorted(lhs_map):
        lhs_correct = lhs_map[fixture_id]
        rhs_correct = rhs_map[fixture_id]
        if lhs_correct and rhs_correct:
            both_correct += 1
        elif lhs_correct and not rhs_correct:
            lhs_correct_rhs_wrong += 1
        elif not lhs_correct and rhs_correct:
            lhs_wrong_rhs_correct += 1
        else:
            both_wrong += 1
    return (
        (both_correct, lhs_correct_rhs_wrong),
        (lhs_wrong_rhs_correct, both_wrong),
    )


def exact_mcnemar(table: tuple[tuple[int, int], tuple[int, int]]) -> McNemarExactResult:
    both_correct, lhs_correct_rhs_wrong = table[0]
    lhs_wrong_rhs_correct, both_wrong = table[1]
    discordant_pairs = lhs_correct_rhs_wrong + lhs_wrong_rhs_correct
    if discordant_pairs == 0:
        p_value = 1.0
        lhs_discordant_share = None
    else:
        tail = sum(
            comb(discordant_pairs, k)
            for k in range(min(lhs_correct_rhs_wrong, lhs_wrong_rhs_correct) + 1)
        ) / (2**discordant_pairs)
        p_value = min(1.0, 2.0 * tail)
        lhs_discordant_share = lhs_correct_rhs_wrong / discordant_pairs
    return McNemarExactResult(
        table=table,
        both_correct=both_correct,
        lhs_correct_rhs_wrong=lhs_correct_rhs_wrong,
        lhs_wrong_rhs_correct=lhs_wrong_rhs_correct,
        both_wrong=both_wrong,
        discordant_pairs=discordant_pairs,
        p_value=p_value,
        lhs_discordant_share=lhs_discordant_share,
    )

