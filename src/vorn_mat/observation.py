"""Observation substrate for vorn dynamics under vanilla inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ObservationTopPosition:
    position: int
    alignment_score: float
    residual_norm: float
    is_answer_token: bool


@dataclass(frozen=True)
class ObservationStep:
    step_index: int
    generated_token_id: int
    generated_token_text: str
    vorn_vector: list[float]
    alignment_scores: list[float]
    residual_norms: list[float]
    attention_by_layer: dict[str, list[float]]
    top_alignment_positions: tuple[int, ...]
    top_alignment_scores: tuple[float, ...]
    ranking_stability_with_prev: float | None


@dataclass(frozen=True)
class ObservationCase:
    case_id: str
    expected_answer: str
    prediction: str
    success: bool
    prompt_token_count: int
    answer_token_spans: tuple[tuple[int, int], ...]
    steps: tuple[ObservationStep, ...]


@dataclass(frozen=True)
class ObservationReport:
    dataset_config: str
    split: str
    case_count: int
    elapsed_seconds: float
    estimated_cost_usd: float
    cases: tuple[ObservationCase, ...]


def find_subsequence_spans(
    haystack: Iterable[int],
    needle: Iterable[int],
) -> tuple[tuple[int, int], ...]:
    hay = tuple(haystack)
    sub = tuple(needle)
    if not sub or len(sub) > len(hay):
        return ()
    spans: list[tuple[int, int]] = []
    width = len(sub)
    for start in range(len(hay) - width + 1):
        if hay[start : start + width] == sub:
            spans.append((start, start + width))
    return tuple(spans)


def answer_token_indices(spans: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    indices: list[int] = []
    for start, end in spans:
        indices.extend(range(start, end))
    return tuple(indices)


def residual_l2_norms(token_summaries: np.ndarray) -> np.ndarray:
    if token_summaries.ndim != 2:
        raise ValueError("token_summaries must be rank-2")
    return np.linalg.norm(token_summaries, axis=1).astype(np.float32, copy=False)


def select_top_alignment_positions(
    *,
    alignment_scores: np.ndarray,
    residual_norms: np.ndarray,
    answer_token_spans: tuple[tuple[int, int], ...],
    top_k: int,
) -> tuple[ObservationTopPosition, ...]:
    if alignment_scores.ndim != 1 or residual_norms.ndim != 1:
        raise ValueError("alignment_scores and residual_norms must be rank-1")
    if alignment_scores.shape != residual_norms.shape:
        raise ValueError("alignment_scores and residual_norms must match in shape")
    answer_positions = set(answer_token_indices(answer_token_spans))
    ranked = sorted(
        range(alignment_scores.shape[0]),
        key=lambda idx: (float(alignment_scores[idx]), -idx),
        reverse=True,
    )[:top_k]
    return tuple(
        ObservationTopPosition(
            position=idx,
            alignment_score=float(alignment_scores[idx]),
            residual_norm=float(residual_norms[idx]),
            is_answer_token=idx in answer_positions,
        )
        for idx in ranked
    )


def jaccard_similarity(left: Iterable[int], right: Iterable[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def round_float_list(values: np.ndarray | Iterable[float], *, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def analyze_observation_report(
    report: ObservationReport,
    *,
    top_k: int = 10,
) -> dict[str, object]:
    success_cases = [case for case in report.cases if case.success]
    failure_cases = [case for case in report.cases if not case.success]

    return {
        "total_cases": len(report.cases),
        "success_cases": len(success_cases),
        "failure_cases": len(failure_cases),
        "success": _analyze_case_group(success_cases, top_k=top_k),
        "failure": _analyze_case_group(failure_cases, top_k=top_k),
    }


def _analyze_case_group(
    cases: list[ObservationCase],
    *,
    top_k: int,
) -> dict[str, object]:
    if not cases:
        return {
            "cases_with_answer_in_top_k": 0,
            "mean_first_answer_top_k_step": None,
            "answer_top_k_hit_rate_by_step": [],
            "alignment_gap_by_step": [],
            "residual_gap_by_step": [],
            "ranking_stability_by_step": [],
        }

    first_hit_steps: list[int] = []
    cases_with_hit = 0
    max_steps = max(len(case.steps) for case in cases)

    hit_rate_by_step: list[float] = []
    alignment_gap_by_step: list[float | None] = []
    residual_gap_by_step: list[float | None] = []
    stability_by_step: list[float | None] = []

    for case in cases:
        answer_positions = set(answer_token_indices(case.answer_token_spans))
        first_hit = None
        for step in case.steps:
            if answer_positions & set(step.top_alignment_positions[:top_k]):
                first_hit = step.step_index
                break
        if first_hit is not None:
            cases_with_hit += 1
            first_hit_steps.append(first_hit)

    for step_index in range(max_steps):
        hits: list[int] = []
        alignment_gaps: list[float] = []
        residual_gaps: list[float] = []
        stabilities: list[float] = []
        for case in cases:
            if step_index >= len(case.steps):
                continue
            step = case.steps[step_index]
            answer_positions = set(answer_token_indices(case.answer_token_spans))
            hits.append(int(bool(answer_positions & set(step.top_alignment_positions[:top_k]))))
            if answer_positions:
                answer_alignment = [
                    step.alignment_scores[idx]
                    for idx in answer_positions
                    if idx < len(step.alignment_scores)
                ]
                non_answer_alignment = [
                    value
                    for idx, value in enumerate(step.alignment_scores)
                    if idx not in answer_positions
                ]
                answer_residual = [
                    step.residual_norms[idx]
                    for idx in answer_positions
                    if idx < len(step.residual_norms)
                ]
                non_answer_residual = [
                    value
                    for idx, value in enumerate(step.residual_norms)
                    if idx not in answer_positions
                ]
                if answer_alignment and non_answer_alignment:
                    alignment_gaps.append(
                        max(answer_alignment) - max(non_answer_alignment)
                    )
                if answer_residual and non_answer_residual:
                    residual_gaps.append(
                        mean(answer_residual) - mean(non_answer_residual)
                    )
            if step.ranking_stability_with_prev is not None:
                stabilities.append(step.ranking_stability_with_prev)

        hit_rate_by_step.append(mean(hits) if hits else 0.0)
        alignment_gap_by_step.append(mean(alignment_gaps) if alignment_gaps else None)
        residual_gap_by_step.append(mean(residual_gaps) if residual_gaps else None)
        stability_by_step.append(mean(stabilities) if stabilities else None)

    return {
        "cases_with_answer_in_top_k": cases_with_hit,
        "mean_first_answer_top_k_step": (
            mean(first_hit_steps) if first_hit_steps else None
        ),
        "answer_top_k_hit_rate_by_step": hit_rate_by_step,
        "alignment_gap_by_step": alignment_gap_by_step,
        "residual_gap_by_step": residual_gap_by_step,
        "ranking_stability_by_step": stability_by_step,
    }


def write_observation_artifacts(
    report: ObservationReport,
    *,
    json_path: Path,
    markdown_path: Path,
    figures_dir: Path,
    top_k: int = 10,
    cases_per_shard: int = 5,
) -> dict[str, object]:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = analyze_observation_report(report, top_k=top_k)
    _write_observation_manifest(
        report,
        summary=summary,
        json_path=json_path,
        cases_per_shard=cases_per_shard,
    )
    _render_observation_plots(summary, figures_dir)
    markdown_path.write_text(
        _build_observation_markdown(
            report,
            summary,
            json_path=json_path,
            figures_dir=figures_dir,
            top_k=top_k,
        )
    )
    return summary


def load_observation_report(json_path: Path) -> ObservationReport:
    payload = json.loads(json_path.read_text())
    if payload.get("format") == "observation-report-sharded-v1":
        return _load_sharded_observation_report(json_path, payload)
    return _load_inline_observation_report(payload)


def _load_sharded_observation_report(
    json_path: Path,
    manifest: dict[str, object],
) -> ObservationReport:
    cases: list[ObservationCase] = []
    for shard in manifest["case_shards"]:
        shard_path = json_path.parent / shard["path"]
        with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
            shard_payload = json.load(handle)
        cases.extend(_case_from_dict(case) for case in shard_payload["cases"])
    return ObservationReport(
        dataset_config=manifest["dataset_config"],
        split=manifest["split"],
        case_count=manifest["case_count"],
        elapsed_seconds=manifest["elapsed_seconds"],
        estimated_cost_usd=manifest["estimated_cost_usd"],
        cases=tuple(cases),
    )


def _load_inline_observation_report(payload: dict[str, object]) -> ObservationReport:
    return ObservationReport(
        dataset_config=payload["dataset_config"],
        split=payload["split"],
        case_count=payload["case_count"],
        elapsed_seconds=payload["elapsed_seconds"],
        estimated_cost_usd=payload["estimated_cost_usd"],
        cases=tuple(_case_from_dict(case) for case in payload["cases"]),
    )


def _case_from_dict(payload: dict[str, object]) -> ObservationCase:
    return ObservationCase(
        case_id=payload["case_id"],
        expected_answer=payload["expected_answer"],
        prediction=payload["prediction"],
        success=payload["success"],
        prompt_token_count=payload["prompt_token_count"],
        answer_token_spans=tuple(tuple(span) for span in payload["answer_token_spans"]),
        steps=tuple(_step_from_dict(step) for step in payload["steps"]),
    )


def _step_from_dict(payload: dict[str, object]) -> ObservationStep:
    return ObservationStep(
        step_index=payload["step_index"],
        generated_token_id=payload["generated_token_id"],
        generated_token_text=payload["generated_token_text"],
        vorn_vector=list(payload["vorn_vector"]),
        alignment_scores=list(payload["alignment_scores"]),
        residual_norms=list(payload["residual_norms"]),
        attention_by_layer={
            key: list(values) for key, values in payload["attention_by_layer"].items()
        },
        top_alignment_positions=tuple(payload["top_alignment_positions"]),
        top_alignment_scores=tuple(payload["top_alignment_scores"]),
        ranking_stability_with_prev=payload["ranking_stability_with_prev"],
    )


def _write_observation_manifest(
    report: ObservationReport,
    *,
    summary: dict[str, object],
    json_path: Path,
    cases_per_shard: int,
) -> None:
    if cases_per_shard <= 0:
        raise ValueError("cases_per_shard must be positive")

    shards_dir = json_path.parent / f"{json_path.stem}-shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    case_dicts = [asdict(case) for case in report.cases]
    shard_entries: list[dict[str, object]] = []

    for shard_index, start in enumerate(range(0, len(case_dicts), cases_per_shard)):
        shard_cases = case_dicts[start : start + cases_per_shard]
        shard_name = f"part-{shard_index:03d}.json.gz"
        shard_path = shards_dir / shard_name
        with gzip.open(shard_path, "wt", encoding="utf-8") as handle:
            json.dump(
                {
                    "format": "observation-case-shard-v1",
                    "case_count": len(shard_cases),
                    "cases": shard_cases,
                },
                handle,
                separators=(",", ":"),
                sort_keys=True,
            )
        shard_entries.append(
            {
                "path": str(shard_path.relative_to(json_path.parent)),
                "case_count": len(shard_cases),
                "case_ids": [case["case_id"] for case in shard_cases],
                "compressed": True,
            }
        )

    manifest = {
        "format": "observation-report-sharded-v1",
        "dataset_config": report.dataset_config,
        "split": report.split,
        "case_count": report.case_count,
        "elapsed_seconds": report.elapsed_seconds,
        "estimated_cost_usd": report.estimated_cost_usd,
        "summary": summary,
        "case_shards": shard_entries,
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _render_observation_plots(summary: dict[str, object], figures_dir: Path) -> None:
    success = summary["success"]
    failure = summary["failure"]

    _line_plot(
        success["answer_top_k_hit_rate_by_step"],
        failure["answer_top_k_hit_rate_by_step"],
        title="Answer Token In Top-K by Decode Step",
        ylabel="Rate",
        output=figures_dir / "answer-topk-hit-rate-by-step.png",
    )
    _line_plot(
        success["alignment_gap_by_step"],
        failure["alignment_gap_by_step"],
        title="Answer Alignment Gap by Decode Step",
        ylabel="Max answer alignment - max non-answer alignment",
        output=figures_dir / "alignment-gap-by-step.png",
    )
    _line_plot(
        success["ranking_stability_by_step"],
        failure["ranking_stability_by_step"],
        title="Top-K Ranking Stability by Decode Step",
        ylabel="Mean Jaccard vs previous step",
        output=figures_dir / "ranking-stability-by-step.png",
    )
    _line_plot(
        success["residual_gap_by_step"],
        failure["residual_gap_by_step"],
        title="Residual Norm Gap by Decode Step",
        ylabel="Mean answer norm - mean non-answer norm",
        output=figures_dir / "residual-gap-by-step.png",
    )


def _line_plot(
    success_values: list[float | None],
    failure_values: list[float | None],
    *,
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.5))
    x_success = list(range(len(success_values)))
    x_failure = list(range(len(failure_values)))
    plt.plot(x_success, _nanify(success_values), label="success")
    plt.plot(x_failure, _nanify(failure_values), label="failure")
    plt.title(title)
    plt.xlabel("Decode step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=144)
    plt.close()


def _nanify(values: list[float | None]) -> list[float]:
    return [float("nan") if value is None else float(value) for value in values]


def _build_observation_markdown(
    report: ObservationReport,
    summary: dict[str, object],
    *,
    json_path: Path,
    figures_dir: Path,
    top_k: int,
) -> str:
    success = summary["success"]
    failure = summary["failure"]
    success_hit_rate = _safe_ratio(
        success["cases_with_answer_in_top_k"],
        summary["success_cases"],
    )
    failure_hit_rate = _safe_ratio(
        failure["cases_with_answer_in_top_k"],
        summary["failure_cases"],
    )
    success_alignment_final = _last_non_null(success["alignment_gap_by_step"])
    failure_alignment_final = _last_non_null(failure["alignment_gap_by_step"])
    success_residual_final = _last_non_null(success["residual_gap_by_step"])
    failure_residual_final = _last_non_null(failure["residual_gap_by_step"])

    return "\n".join(
        [
            "# Vanilla Observation — 2026-05-13",
            "",
            "## Run",
            "",
            f"- Dataset config: `{report.dataset_config}`",
            f"- Split: `{report.split}`",
            f"- Cases: `{report.case_count}`",
            f"- Wall-clock: `{report.elapsed_seconds:.2f}s`",
            f"- Estimated cost: `${report.estimated_cost_usd:.4f}`",
            f"- Top-K tracked: `{top_k}`",
            f"- Observation manifest: `{json_path}`",
            f"- Case shards: `{json_path.parent / f'{json_path.stem}-shards'}`",
            "",
            "## Initial findings",
            "",
            (
                f"- Success cases with any answer token entering top-{top_k}: "
                f"`{success['cases_with_answer_in_top_k']}/{summary['success_cases']}` "
                f"({success_hit_rate:.3f}); failure cases: "
                f"`{failure['cases_with_answer_in_top_k']}/{summary['failure_cases']}` "
                f"({failure_hit_rate:.3f})."
            ),
            (
                f"- Mean first answer-in-top-{top_k} step: success "
                f"`{_format_optional(success['mean_first_answer_top_k_step'])}`; failure "
                f"`{_format_optional(failure['mean_first_answer_top_k_step'])}`."
            ),
            (
                f"- Final-step alignment gap (answer vs strongest non-answer position): "
                f"success `{_format_optional(success_alignment_final)}`, failure "
                f"`{_format_optional(failure_alignment_final)}`."
            ),
            (
                f"- Final-step residual-norm gap (answer mean minus non-answer mean): "
                f"success `{_format_optional(success_residual_final)}`, failure "
                f"`{_format_optional(failure_residual_final)}`."
            ),
            "",
            "## Figures",
            "",
            f"- [answer-topk-hit-rate-by-step.png]({figures_dir / 'answer-topk-hit-rate-by-step.png'})",
            f"- [alignment-gap-by-step.png]({figures_dir / 'alignment-gap-by-step.png'})",
            f"- [ranking-stability-by-step.png]({figures_dir / 'ranking-stability-by-step.png'})",
            f"- [residual-gap-by-step.png]({figures_dir / 'residual-gap-by-step.png'})",
            "",
            "## Interpretation boundary",
            "",
            "- This run is pure observation under vanilla inference: no eviction, no replay, no cache intervention.",
            "- The patterns here describe what the unmodified model naturally does on this slice. They do not by themselves validate an eviction policy.",
        ]
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _last_non_null(values: list[float | None]) -> float | None:
    for value in reversed(values):
        if value is not None:
            return value
    return None


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"
