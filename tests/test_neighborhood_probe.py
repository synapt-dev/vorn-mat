from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat.neighborhood_probe import (
    NeighborhoodSpan,
    analyze_neighborhood_probes,
    build_standard_neighborhoods,
    expand_token_span,
    sentence_char_span,
    token_span_from_offsets,
)
from vorn_mat.observation import ObservationCase, ObservationReport, ObservationStep


def test_sentence_char_span_finds_sentence_containing_needle():
    text = "Alpha. One of the special magic numbers is: 12345. Omega."
    start = text.index("12345")
    end = start + len("12345")

    span = sentence_char_span(text, start, end)

    assert text[span[0] : span[1]] == "One of the special magic numbers is: 12345."


def test_token_span_from_offsets_maps_char_range_to_tokens():
    offsets = ((0, 5), (6, 8), (9, 16), (17, 24), (25, 28))

    span = token_span_from_offsets(offsets, char_start=9, char_end=24)

    assert span == (2, 4)


def test_expand_token_span_clamps_to_token_limit():
    expanded = expand_token_span((5, 8), left_tokens=10, right_tokens=4, token_limit=12)

    assert expanded == (0, 12)


def test_build_standard_neighborhoods_marks_degenerate_line_and_paragraph():
    rendered_prompt = "User: Alpha. One of the special magic numbers is: 12345. Omega."
    offsets = tuple((idx, idx + 1) for idx in range(len(rendered_prompt)))
    answer_start = rendered_prompt.index("12345")
    answer_end = answer_start + len("12345")

    neighborhoods = build_standard_neighborhoods(
        rendered_prompt=rendered_prompt,
        offsets=offsets,
        answer_token_spans=((answer_start, answer_end),),
    )

    labels = {neighborhood.label: neighborhood for neighborhood in neighborhoods}
    assert "sentence" in labels
    assert labels["line"].degenerate is True
    assert labels["paragraph"].degenerate is True


def test_analyze_neighborhood_probes_shows_window_signal_above_exact_answer():
    report = ObservationReport(
        dataset_config="niah_multikey_1_4k",
        split="validation[:2]",
        case_count=2,
        elapsed_seconds=1.0,
        estimated_cost_usd=0.1,
        cases=(
            ObservationCase(
                case_id="success",
                expected_answer="needle",
                prediction="needle",
                success=True,
                prompt_token_count=6,
                answer_token_spans=((1, 2),),
                steps=(
                    ObservationStep(
                        step_index=0,
                        generated_token_id=1,
                        generated_token_text="a",
                        vorn_vector=[0.1],
                        alignment_scores=[0.9, 0.2, 0.8, 0.7, 0.1, 0.0],
                        residual_norms=[1.0] * 6,
                        attention_by_layer={},
                        top_alignment_positions=(0, 2, 3),
                        top_alignment_scores=(0.9, 0.8, 0.7),
                        ranking_stability_with_prev=None,
                    ),
                ),
            ),
            ObservationCase(
                case_id="failure",
                expected_answer="needle",
                prediction="wrong",
                success=False,
                prompt_token_count=6,
                answer_token_spans=((1, 2),),
                steps=(
                    ObservationStep(
                        step_index=0,
                        generated_token_id=2,
                        generated_token_text="b",
                        vorn_vector=[0.1],
                        alignment_scores=[0.9, 0.1, 0.2, 0.3, 0.4, 0.5],
                        residual_norms=[1.0] * 6,
                        attention_by_layer={},
                        top_alignment_positions=(0, 5, 4),
                        top_alignment_scores=(0.9, 0.5, 0.4),
                        ranking_stability_with_prev=None,
                    ),
                ),
            ),
        ),
    )

    summary = analyze_neighborhood_probes(
        report,
        case_neighborhoods={
            "success": (
                NeighborhoodSpan(label="exact_answer", token_spans=((1, 2),)),
                NeighborhoodSpan(label="sentence", token_spans=((1, 4),)),
            ),
            "failure": (
                NeighborhoodSpan(label="exact_answer", token_spans=((1, 2),)),
                NeighborhoodSpan(label="sentence", token_spans=((1, 4),)),
            ),
        },
        top_k=3,
    )

    assert summary["probes"]["exact_answer"]["success"]["cases_with_top_k_hit"] == 0
    assert summary["probes"]["sentence"]["success"]["cases_with_top_k_hit"] == 1
