#!/usr/bin/env python3
"""Phase 0 SEMU trajectory mining for active vorn eviction.

This is an offline analysis over existing observation artifacts. It does not run
fresh model generations and does not estimate counterfactual SEMU contribution.
"""
# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import json
import statistics
import sys
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer

from vorn_mat import DEFAULT_MODEL
from vorn_mat import load_observation_report, load_ruler_hf_niah_slice
from vorn_mat.text_spans import sentence_char_spans, token_span_from_offsets


@dataclass(frozen=True)
class SemuSpan:
    semu_id: int
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    text_preview: str
    answer_overlap: bool


def _render_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.fmean(values) if values else None


def _median(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.median(values) if values else None


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    index = q * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def _top_k_mean(values: Sequence[float], *, k: int) -> float:
    ordered = sorted((float(value) for value in values), reverse=True)
    return statistics.fmean(ordered[: min(k, len(ordered))])


def _answer_positions(answer_token_spans: Sequence[Sequence[int]]) -> set[int]:
    positions: set[int] = set()
    for start, end in answer_token_spans:
        positions.update(range(int(start), int(end)))
    return positions


def _build_sentence_semus(
    *,
    rendered_prompt: str,
    offsets: Sequence[tuple[int, int]],
    answer_token_spans: Sequence[Sequence[int]],
) -> list[SemuSpan]:
    answer_positions = _answer_positions(answer_token_spans)
    semus: list[SemuSpan] = []
    for char_start, char_end in sentence_char_spans(rendered_prompt):
        token_span = token_span_from_offsets(
            offsets,
            char_start=char_start,
            char_end=char_end,
        )
        if token_span is None:
            continue
        token_start, token_end = token_span
        token_positions = set(range(token_start, token_end))
        text = " ".join(rendered_prompt[char_start:char_end].split())
        semus.append(
            SemuSpan(
                semu_id=len(semus),
                char_start=char_start,
                char_end=char_end,
                token_start=token_start,
                token_end=token_end,
                text_preview=text[:120],
                answer_overlap=bool(answer_positions & token_positions),
            )
        )
    return semus


def _rank_scores(scores: Sequence[float]) -> list[int]:
    ordered = sorted(
        enumerate(scores),
        key=lambda item: (float(item[1]), -item[0]),
        reverse=True,
    )
    ranks = [0] * len(scores)
    for rank, (index, _score) in enumerate(ordered, start=1):
        ranks[index] = rank
    return ranks


def _jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def _case_matrix(observation_case, rendered_prompt: str, offsets) -> dict[str, object]:
    semus = _build_sentence_semus(
        rendered_prompt=rendered_prompt,
        offsets=offsets,
        answer_token_spans=observation_case.answer_token_spans,
    )
    if not semus:
        raise ValueError(f"no sentence SEMUs for {observation_case.case_id}")

    step_scores: list[list[float]] = []
    step_ranks: list[list[int]] = []
    step_top5: list[list[int]] = []

    for step in observation_case.steps:
        semu_scores: list[float] = []
        for semu in semus:
            token_scores = [
                float(score)
                for score in step.alignment_scores[semu.token_start : semu.token_end]
            ]
            semu_scores.append(_top_k_mean(token_scores, k=3))
        ranks = _rank_scores(semu_scores)
        step_scores.append(semu_scores)
        step_ranks.append(ranks)
        step_top5.append(
            sorted(range(len(semu_scores)), key=lambda idx: ranks[idx])[:5]
        )

    semu_records: list[dict[str, object]] = []
    score_ranges: list[float] = []
    answer_rank_by_step: list[int | None] = []
    answer_score_by_step: list[float | None] = []

    answer_semu_ids = [semu.semu_id for semu in semus if semu.answer_overlap]
    for step_index in range(len(step_scores)):
        if answer_semu_ids:
            answer_rank_by_step.append(
                min(step_ranks[step_index][semu_id] for semu_id in answer_semu_ids)
            )
            answer_score_by_step.append(
                max(step_scores[step_index][semu_id] for semu_id in answer_semu_ids)
            )
        else:
            answer_rank_by_step.append(None)
            answer_score_by_step.append(None)

    for semu in semus:
        scores = [step[semu.semu_id] for step in step_scores]
        ranks = [step[semu.semu_id] for step in step_ranks]
        score_range = max(scores) - min(scores)
        score_ranges.append(score_range)
        semu_records.append(
            {
                "semu_id": semu.semu_id,
                "token_start": semu.token_start,
                "token_end": semu.token_end,
                "char_start": semu.char_start,
                "char_end": semu.char_end,
                "answer_overlap": semu.answer_overlap,
                "text_preview": semu.text_preview,
                "scores_by_step": [_round(score) for score in scores],
                "ranks_by_step": ranks,
                "score_range": _round(score_range),
                "score_std": _round(statistics.pstdev(scores) if len(scores) > 1 else 0.0),
                "first_score": _round(scores[0]),
                "final_score": _round(scores[-1]),
                "delta_final_minus_first": _round(scores[-1] - scores[0]),
                "best_rank": min(ranks),
                "final_rank": ranks[-1],
            }
        )

    top1_by_step = [top5[0] for top5 in step_top5]
    top1_switches = sum(
        int(left != right) for left, right in zip(top1_by_step, top1_by_step[1:])
    )
    top5_jaccards = [
        _jaccard(left, right) for left, right in zip(step_top5, step_top5[1:])
    ]

    return {
        "case_id": observation_case.case_id,
        "success": bool(observation_case.success),
        "prompt_hash": _sha256_text(rendered_prompt),
        "prompt_token_count": observation_case.prompt_token_count,
        "step_count": len(observation_case.steps),
        "semu_granularity": "sentence",
        "semu_count": len(semus),
        "answer_semu_ids": answer_semu_ids,
        "top1_by_step": top1_by_step,
        "top1_switches": top1_switches,
        "mean_top5_jaccard": _round(_mean(top5_jaccards)),
        "mean_semu_score_range": _round(_mean(score_ranges)),
        "median_semu_score_range": _round(_median(score_ranges)),
        "p90_semu_score_range": _round(_quantile(score_ranges, 0.9)),
        "answer_rank_by_step": answer_rank_by_step,
        "answer_score_by_step": [_round(score) for score in answer_score_by_step],
        "answer_first_rank": answer_rank_by_step[0] if answer_rank_by_step else None,
        "answer_final_rank": answer_rank_by_step[-1] if answer_rank_by_step else None,
        "answer_best_rank": (
            min(rank for rank in answer_rank_by_step if rank is not None)
            if any(rank is not None for rank in answer_rank_by_step)
            else None
        ),
        "semu_matrix": semu_records,
    }


def _summarize_cases(case_matrices: Sequence[dict[str, object]]) -> dict[str, object]:
    successes = [case for case in case_matrices if case["success"]]
    failures = [case for case in case_matrices if not case["success"]]
    with_answer = [case for case in case_matrices if case["answer_semu_ids"]]
    score_ranges = [float(case["mean_semu_score_range"]) for case in case_matrices]
    top1_switches = [int(case["top1_switches"]) for case in case_matrices]
    top5_jaccards = [float(case["mean_top5_jaccard"]) for case in case_matrices]
    answer_first = [
        int(case["answer_first_rank"])
        for case in with_answer
        if case["answer_first_rank"] is not None
    ]
    answer_final = [
        int(case["answer_final_rank"])
        for case in with_answer
        if case["answer_final_rank"] is not None
    ]
    answer_best = [
        int(case["answer_best_rank"])
        for case in with_answer
        if case["answer_best_rank"] is not None
    ]

    return {
        "case_count": len(case_matrices),
        "success_cases": len(successes),
        "failure_cases": len(failures),
        "cases_with_answer_semu": len(with_answer),
        "mean_sentence_semus_per_case": _round(
            _mean(float(case["semu_count"]) for case in case_matrices)
        ),
        "mean_steps_per_case": _round(
            _mean(float(case["step_count"]) for case in case_matrices)
        ),
        "trajectory_variation": {
            "mean_case_mean_semu_score_range": _round(_mean(score_ranges)),
            "median_case_mean_semu_score_range": _round(_median(score_ranges)),
            "p90_case_mean_semu_score_range": _round(_quantile(score_ranges, 0.9)),
            "mean_top1_switches": _round(_mean(float(value) for value in top1_switches)),
            "cases_with_top1_switch": sum(value > 0 for value in top1_switches),
            "mean_top5_jaccard": _round(_mean(top5_jaccards)),
        },
        "answer_semu_ranks": {
            "mean_first_rank": _round(_mean(float(value) for value in answer_first)),
            "mean_final_rank": _round(_mean(float(value) for value in answer_final)),
            "mean_best_rank": _round(_mean(float(value) for value in answer_best)),
            "median_best_rank": _round(_median(float(value) for value in answer_best)),
            "best_rank_top10_cases": sum(value <= 10 for value in answer_best),
            "best_rank_top20_cases": sum(value <= 20 for value in answer_best),
        },
    }


def _build_markdown(artifact: dict[str, object]) -> str:
    summary = artifact["summary"]
    variation = summary["trajectory_variation"]
    answer = summary["answer_semu_ranks"]
    matrix_path = artifact["outputs"]["json_path"]

    lines = [
        "# Vorn-Active Eviction Phase 0 SEMU Trajectory Findings",
        "",
        "Date: 2026-06-03",
        "",
        "## Scope",
        "",
        "This is an offline Phase 0 analysis over existing vorn-mat observation artifacts. "
        "It does not run fresh generations and does not estimate counterfactual SEMU contribution.",
        "",
        "Current trajectory-bearing coverage is narrower than the 21,600 method-level "
        "evaluation observations in the public corpus. The committed run analyzes the "
        "`vanilla-observation-2026-05-13` sharded observation report, which contains "
        "Mistral `niah_multikey_1_4k` generation traces with token-level vorn alignment scores.",
        "",
        "## Corpus",
        "",
        f"- Observation report: `{artifact['source_observation_report']}`",
        f"- Dataset config: `{artifact['dataset_config']}`",
        f"- Split: `{artifact['split']}`",
        f"- Model: `{artifact['model_id']}`",
        f"- Cases: {summary['case_count']} ({summary['success_cases']} success / "
        f"{summary['failure_cases']} failure)",
        f"- Sentence SEMUs per case, mean: {summary['mean_sentence_semus_per_case']}",
        f"- Steps per case, mean: {summary['mean_steps_per_case']}",
        f"- Matrix artifact: `{matrix_path}`",
        "",
        "## Findings",
        "",
        "### F1. Existing trajectory data is targeted, not corpus-wide",
        "",
        "The 21,600 public observations remain method-level fixture outcomes. They do not "
        "carry per-token or per-SEMU trajectory traces. The available SEMU trajectory "
        "mining path is through targeted observation artifacts such as the sharded "
        "vanilla observation report and the 8k score-distribution observation runs.",
        "",
        "### F2. Sentence-level vorn scores vary, but rankings are mostly stable",
        "",
        f"- Mean per-case mean SEMU score range: {variation['mean_case_mean_semu_score_range']}",
        f"- Median per-case mean SEMU score range: {variation['median_case_mean_semu_score_range']}",
        f"- P90 per-case mean SEMU score range: {variation['p90_case_mean_semu_score_range']}",
        f"- Mean top-1 SEMU switches per case: {variation['mean_top1_switches']}",
        f"- Cases with at least one top-1 SEMU switch: {variation['cases_with_top1_switch']}/{summary['case_count']}",
        f"- Mean top-5 Jaccard across adjacent steps: {variation['mean_top5_jaccard']}",
        "",
        "Interpretation: score values are not static snapshots, but rank order is "
        "mostly stable in this targeted Mistral NIAH observation corpus. That is a "
        "narrower finding than the strongest temporal/Jenga hypothesis. It supports "
        "recording trajectories, but suggests tool-result-before/after-decision "
        "fixtures are the better test surface for true temporal decay.",
        "",
        "### F3. Answer-bearing SEMUs are recoverable as a ranking probe",
        "",
        f"- Cases with answer-overlapping sentence SEMU: {summary['cases_with_answer_semu']}/{summary['case_count']}",
        f"- Mean answer SEMU first-step rank: {answer['mean_first_rank']}",
        f"- Mean answer SEMU final-step rank: {answer['mean_final_rank']}",
        f"- Mean answer SEMU best rank: {answer['mean_best_rank']}",
        f"- Median answer SEMU best rank: {answer['median_best_rank']}",
        f"- Answer SEMU reached top-10 at least once: {answer['best_rank_top10_cases']}/{summary['cases_with_answer_semu']}",
        f"- Answer SEMU reached top-20 at least once: {answer['best_rank_top20_cases']}/{summary['cases_with_answer_semu']}",
        "",
        "Interpretation: answer-span overlap gives a useful proxy probe for Phase 0 "
        "ranking analysis, but it is still a proxy. Counterfactual deletion is required "
        "to label a SEMU as load-bearing.",
        "",
        "### F4. Cross-family SEMU trajectory comparison is not available yet",
        "",
        "The current trajectory-bearing observation artifacts are not seven-family. "
        "Cross-family active-eviction claims therefore require fresh instrumentation "
        "or new observation runs on the same family panel. Existing seven-family rows "
        "can select strata and families, but cannot answer whether families have "
        "different SEMU trajectory shapes.",
        "",
        "## Implications for the design doc",
        "",
        "- Keep Phase 0 framed as proxy signal mining.",
        "- Keep Pilot A as signal-detection, not threshold calibration.",
        "- Preserve sentence-level SEMU as the first intervention granularity.",
        "- Require fresh counterfactual runs for causal contribution labels.",
        "- Require fresh cross-family observation instrumentation before claiming family-conditional SEMU trajectories.",
        "",
        "## Honest negatives",
        "",
        "- This analysis only sees generation-time vorn movement on targeted Mistral observation traces.",
        "- It does not include tool-result-before/after-decision workflows.",
        "- It does not establish that low-vorn SEMUs are safe to drop.",
        "- It does not compare seven families at trajectory level.",
        "- Sentence segmentation is a practical first granularity, not proven optimal.",
        "",
    ]
    return "\n".join(lines)


def analyze_phase0(
    *,
    observation_json: Path,
    output_json: Path,
    output_md: Path,
    model_id: str,
) -> None:
    report = load_observation_report(observation_json)
    benchmark_cases = {
        case.case_id: case
        for case in load_ruler_hf_niah_slice(
            report.dataset_config,
            split="validation",
            case_limit=report.case_count,
        )
    }
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    case_matrices: list[dict[str, object]] = []
    for observation_case in report.cases:
        benchmark_case = benchmark_cases[observation_case.case_id]
        rendered_prompt = _render_chat_prompt(tokenizer, benchmark_case.prompt)
        encoding = tokenizer(
            rendered_prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = tuple(tuple(offset) for offset in encoding["offset_mapping"])
        token_count = len(encoding["input_ids"])
        if token_count != observation_case.prompt_token_count:
            raise ValueError(
                f"token count mismatch for {observation_case.case_id}: "
                f"rendered={token_count} observed={observation_case.prompt_token_count}"
            )
        case_matrices.append(_case_matrix(observation_case, rendered_prompt, offsets))

    artifact = {
        "schema_version": "vorn-active-eviction-phase0/v1",
        "date": "2026-06-03",
        "source_observation_report": _display_path(observation_json),
        "dataset_config": report.dataset_config,
        "split": report.split,
        "model_id": model_id,
        "semu_granularity": "sentence",
        "summary": _summarize_cases(case_matrices),
        "outputs": {
            "json_path": _display_path(output_json),
            "markdown_path": _display_path(output_md),
        },
        "limits": [
            "method-level vorn-mat artifacts do not carry per-SEMU trajectories",
            "current committed trajectory extraction is targeted Mistral-only",
            "counterfactual quality deltas require fresh runs",
            "agentic tool-result integration requires new workflow fixtures",
        ],
        "cases": case_matrices,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    output_md.write_text(_build_markdown(artifact))

    print(f"output_json={output_json}")
    print(f"output_md={output_md}")
    summary = artifact["summary"]
    variation = summary["trajectory_variation"]
    print(f"cases={summary['case_count']}")
    print(f"mean_case_mean_semu_score_range={variation['mean_case_mean_semu_score_range']}")
    print(f"cases_with_top1_switch={variation['cases_with_top1_switch']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observation-json",
        type=Path,
        default=ROOT / "results" / "vanilla-observation-2026-05-13.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "results" / "vorn-active-eviction-phase0-2026-06-03.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "results" / "vorn-active-eviction-phase0-2026-06-03.md",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    args = parser.parse_args()

    analyze_phase0(
        observation_json=args.observation_json,
        output_json=args.output_json,
        output_md=args.output_md,
        model_id=args.model_id,
    )


if __name__ == "__main__":
    main()
