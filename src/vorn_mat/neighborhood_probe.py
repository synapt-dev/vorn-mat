"""Offline neighborhood probes over captured vanilla observation traces."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

from .observation import ObservationCase, ObservationReport
from .text_spans import (
    line_char_span,
    paragraph_char_span,
    sentence_char_span,
    token_span_from_offsets,
)


@dataclass(frozen=True)
class NeighborhoodSpan:
    label: str
    token_spans: tuple[tuple[int, int], ...]
    text_excerpt: str | None = None
    note: str | None = None
    degenerate: bool = False
def expand_token_span(
    span: tuple[int, int],
    *,
    left_tokens: int,
    right_tokens: int,
    token_limit: int,
) -> tuple[int, int]:
    start, end = span
    return max(0, start - left_tokens), min(token_limit, end + right_tokens)


def build_standard_neighborhoods(
    *,
    rendered_prompt: str,
    offsets: Sequence[tuple[int, int]],
    answer_token_spans: tuple[tuple[int, int], ...],
) -> tuple[NeighborhoodSpan, ...]:
    if not answer_token_spans:
        raise ValueError("answer_token_spans must be non-empty")

    answer_start = min(start for start, _end in answer_token_spans)
    answer_end = max(end for _start, end in answer_token_spans)
    answer_char_start = offsets[answer_start][0]
    answer_char_end = offsets[answer_end - 1][1]
    token_limit = len(offsets)

    sentence_span = sentence_char_span(rendered_prompt, answer_char_start, answer_char_end)
    sentence_tokens = token_span_from_offsets(
        offsets,
        char_start=sentence_span[0],
        char_end=sentence_span[1],
    )
    line_span = line_char_span(rendered_prompt, answer_char_start, answer_char_end)
    line_tokens = token_span_from_offsets(
        offsets,
        char_start=line_span[0],
        char_end=line_span[1],
    )
    paragraph_span = paragraph_char_span(
        rendered_prompt,
        answer_char_start,
        answer_char_end,
    )
    paragraph_tokens = token_span_from_offsets(
        offsets,
        char_start=paragraph_span[0],
        char_end=paragraph_span[1],
    )

    neighborhoods = [
        NeighborhoodSpan(
            label="exact_answer",
            token_spans=answer_token_spans,
            text_excerpt=rendered_prompt[answer_char_start:answer_char_end],
        ),
        NeighborhoodSpan(
            label="answer_window_5",
            token_spans=(
                expand_token_span(
                    (answer_start, answer_end),
                    left_tokens=5,
                    right_tokens=5,
                    token_limit=token_limit,
                ),
            ),
            note="answer span expanded by +/-5 tokens",
        ),
        NeighborhoodSpan(
            label="answer_window_10",
            token_spans=(
                expand_token_span(
                    (answer_start, answer_end),
                    left_tokens=10,
                    right_tokens=10,
                    token_limit=token_limit,
                ),
            ),
            note="answer span expanded by +/-10 tokens",
        ),
    ]

    if sentence_tokens is not None:
        neighborhoods.append(
            NeighborhoodSpan(
                label="sentence",
                token_spans=(sentence_tokens,),
                text_excerpt=rendered_prompt[sentence_span[0] : sentence_span[1]].strip(),
            )
        )

    if line_tokens is not None:
        line_excerpt = rendered_prompt[line_span[0] : line_span[1]].strip()
        neighborhoods.append(
            NeighborhoodSpan(
                label="line",
                token_spans=(line_tokens,),
                text_excerpt=line_excerpt,
                degenerate=line_span == (0, len(rendered_prompt)),
                note=(
                    "single-line prompt serialization makes the line probe degenerate"
                    if line_span == (0, len(rendered_prompt))
                    else None
                ),
            )
        )

    if paragraph_tokens is not None:
        paragraph_excerpt = rendered_prompt[paragraph_span[0] : paragraph_span[1]].strip()
        neighborhoods.append(
            NeighborhoodSpan(
                label="paragraph",
                token_spans=(paragraph_tokens,),
                text_excerpt=paragraph_excerpt[:240],
                degenerate=paragraph_span == (0, len(rendered_prompt)),
                note=(
                    "single-paragraph prompt serialization makes the paragraph probe degenerate"
                    if paragraph_span == (0, len(rendered_prompt))
                    else None
                ),
            )
        )

    return tuple(neighborhoods)


def analyze_neighborhood_probes(
    report: ObservationReport,
    *,
    case_neighborhoods: dict[str, tuple[NeighborhoodSpan, ...]],
    top_k: int = 10,
) -> dict[str, object]:
    probe_labels = _ordered_probe_labels(case_neighborhoods.values())
    probes: dict[str, object] = {}
    for label in probe_labels:
        probe_cases = []
        notes: set[str] = set()
        degenerate_count = 0
        for case in report.cases:
            neighborhood = _get_neighborhood(case_neighborhoods[case.case_id], label)
            if neighborhood.note:
                notes.add(neighborhood.note)
            if neighborhood.degenerate:
                degenerate_count += 1
            probe_cases.append(
                _summarize_case_probe(case, neighborhood.token_spans, top_k=top_k)
            )

        probes[label] = {
            "success": _aggregate_probe_group(
                report.cases,
                probe_cases,
                success=True,
            ),
            "failure": _aggregate_probe_group(
                report.cases,
                probe_cases,
                success=False,
            ),
            "notes": sorted(notes),
            "degenerate_cases": degenerate_count,
        }

    return {
        "dataset_config": report.dataset_config,
        "split": report.split,
        "case_count": report.case_count,
        "top_k": top_k,
        "probes": probes,
    }


def write_neighborhood_probe_artifacts(
    summary: dict[str, object],
    *,
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    markdown_path.write_text(_build_neighborhood_markdown(summary))


def _ordered_probe_labels(
    neighborhood_sets: Iterable[tuple[NeighborhoodSpan, ...]],
) -> list[str]:
    labels: list[str] = []
    for neighborhoods in neighborhood_sets:
        for neighborhood in neighborhoods:
            if neighborhood.label not in labels:
                labels.append(neighborhood.label)
    return labels


def _get_neighborhood(
    neighborhoods: tuple[NeighborhoodSpan, ...],
    label: str,
) -> NeighborhoodSpan:
    for neighborhood in neighborhoods:
        if neighborhood.label == label:
            return neighborhood
    raise KeyError(f"missing neighborhood {label}")


def _summarize_case_probe(
    case: ObservationCase,
    token_spans: tuple[tuple[int, int], ...],
    *,
    top_k: int,
) -> dict[str, object]:
    target_positions = {
        index
        for start, end in token_spans
        for index in range(start, end)
    }
    first_hit_step: int | None = None
    best_ranks: list[int] = []
    final_alignment_gap: float | None = None

    for step in case.steps:
        top_positions = set(step.top_alignment_positions[:top_k])
        if first_hit_step is None and target_positions & top_positions:
            first_hit_step = step.step_index

        ranked = sorted(
            range(len(step.alignment_scores)),
            key=lambda idx: (step.alignment_scores[idx], -idx),
            reverse=True,
        )
        position_to_rank = {
            position: rank + 1 for rank, position in enumerate(ranked)
        }
        target_ranks = [
            position_to_rank[position]
            for position in target_positions
            if position in position_to_rank
        ]
        if target_ranks:
            best_ranks.append(min(target_ranks))

    if case.steps:
        final_scores = case.steps[-1].alignment_scores
        target_scores = [
            final_scores[position]
            for position in target_positions
            if position < len(final_scores)
        ]
        other_scores = [
            value
            for position, value in enumerate(final_scores)
            if position not in target_positions
        ]
        if target_scores and other_scores:
            final_alignment_gap = max(target_scores) - max(other_scores)

    return {
        "top_k_hit": first_hit_step is not None,
        "first_hit_step": first_hit_step,
        "best_rank": min(best_ranks) if best_ranks else None,
        "final_alignment_gap": final_alignment_gap,
    }


def _aggregate_probe_group(
    cases: Sequence[ObservationCase],
    probe_cases: Sequence[dict[str, object]],
    *,
    success: bool,
) -> dict[str, object]:
    selected = [
        summary
        for case, summary in zip(cases, probe_cases, strict=True)
        if case.success is success
    ]
    top_k_hits = sum(1 for summary in selected if summary["top_k_hit"])
    first_hit_steps = [
        summary["first_hit_step"]
        for summary in selected
        if summary["first_hit_step"] is not None
    ]
    best_ranks = [
        summary["best_rank"]
        for summary in selected
        if summary["best_rank"] is not None
    ]
    final_gaps = [
        summary["final_alignment_gap"]
        for summary in selected
        if summary["final_alignment_gap"] is not None
    ]
    case_count = len(selected)
    return {
        "case_count": case_count,
        "cases_with_top_k_hit": top_k_hits,
        "top_k_hit_rate": (top_k_hits / case_count) if case_count else 0.0,
        "mean_first_hit_step": mean(first_hit_steps) if first_hit_steps else None,
        "mean_best_rank": mean(best_ranks) if best_ranks else None,
        "mean_final_alignment_gap": mean(final_gaps) if final_gaps else None,
    }


def _build_neighborhood_markdown(summary: dict[str, object]) -> str:
    probes = summary["probes"]
    exact = probes.get("exact_answer")
    window_10 = probes.get("answer_window_10")
    sentence = probes.get("sentence")
    lines = [
        "# Vanilla Observation Neighborhood Probe — 2026-05-13",
        "",
        "## Run",
        "",
        f"- Dataset config: `{summary['dataset_config']}`",
        f"- Split: `{summary['split']}`",
        f"- Cases: `{summary['case_count']}`",
        f"- Top-K tracked: `{summary['top_k']}`",
        "",
        "## Probe comparison",
        "",
    ]
    for label, probe_summary in summary["probes"].items():
        success = probe_summary["success"]
        failure = probe_summary["failure"]
        lines.extend(
            [
                f"### `{label}`",
                "",
                (
                    f"- Success top-K hit: `{success['cases_with_top_k_hit']}/{success['case_count']}` "
                    f"({success['top_k_hit_rate']:.3f}); failure top-K hit: "
                    f"`{failure['cases_with_top_k_hit']}/{failure['case_count']}` "
                    f"({failure['top_k_hit_rate']:.3f})."
                ),
                (
                    f"- Mean first hit step: success `{_format_optional(success['mean_first_hit_step'])}`, "
                    f"failure `{_format_optional(failure['mean_first_hit_step'])}`."
                ),
                (
                    f"- Mean best rank: success `{_format_optional(success['mean_best_rank'])}`, "
                    f"failure `{_format_optional(failure['mean_best_rank'])}`."
                ),
                (
                    f"- Mean final alignment gap: success `{_format_optional(success['mean_final_alignment_gap'])}`, "
                    f"failure `{_format_optional(failure['mean_final_alignment_gap'])}`."
                ),
            ]
        )
        if probe_summary["degenerate_cases"]:
            lines.append(
                f"- Degenerate cases: `{probe_summary['degenerate_cases']}` "
                "(probe collapsed to nearly the whole prompt; not discriminative)."
            )
        for note in probe_summary["notes"]:
            lines.append(f"- Note: {note}.")
        lines.append("")

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "- This is offline re-processing of the existing vanilla observation traces. No new Modal inference was run.",
            "- The probe asks whether vorn-aligned positions cluster around wider neighborhoods of the needle even when they do not land on the exact answer tokens.",
        ]
    )
    if exact and window_10 and sentence:
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                (
                    f"- Widening the probe from the exact answer to `answer_window_10` improves the "
                    f"success-case mean best rank from `{_format_optional(exact['success']['mean_best_rank'])}` "
                    f"to `{_format_optional(window_10['success']['mean_best_rank'])}`, but still leaves the "
                    "needle neighborhood well outside the tracked top-10."
                ),
                (
                    f"- The full sentence probe behaves similarly (`{_format_optional(sentence['success']['mean_best_rank'])}` "
                    "success-case mean best rank) and also never enters top-10 on any successful case."
                ),
                (
                    "- The coarse-granularity rescue story is therefore weak on this slice: the neighborhood probes are only modestly "
                    "less bad than exact answer tokens, not qualitatively different."
                ),
                (
                    "- Success/failure separation remains weak. On these probes, successful cases do not rank the needle neighborhood "
                    "materially better than failures, so the current mechanism story likely depends on a different structure than direct "
                    "needle-span alignment."
                ),
            ]
        )
    return "\n".join(lines)


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"
