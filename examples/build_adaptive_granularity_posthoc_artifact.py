#!/usr/bin/env python3
"""Build a descriptive post-hoc adaptive-granularity artifact."""
# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vorn_mat.results import CaseObservation

RESULTS = ROOT / "results"
BENCHMARKS = ROOT / ".benchmarks"
CROSS_MODEL = BENCHMARKS / "cross-model"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _md_write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n")


def _wilson_ci(hits: int, cases: int, z: float = 1.96) -> tuple[float, float]:
    if cases <= 0:
        return (0.0, 0.0)
    p = hits / cases
    denom = 1.0 + (z**2 / cases)
    center = (p + (z**2 / (2 * cases))) / denom
    margin = (
        z
        * (((p * (1 - p)) / cases) + (z**2 / (4 * cases**2))) ** 0.5
        / denom
    )
    return (center - margin, center + margin)


def _unit_from_method(method: str) -> str:
    if method == "token_vorn":
        return "token"
    if method == "sentence_vorn":
        return "sentence"
    if method == "word_vorn":
        return "word"
    if method == "vanilla":
        return "full_context"
    return method


def _normalize_existing_row(
    row: dict[str, Any],
    *,
    model_id: str,
    context_length: str,
    source_artifact: str,
) -> dict[str, Any]:
    normalized = dict(row)
    normalized["model_id"] = model_id
    normalized["context_length"] = context_length
    normalized["unit"] = _unit_from_method(str(row["method"]))
    normalized["source_artifact"] = source_artifact
    normalized["source_type"] = "artifact"
    return normalized


def _normalize_existing_ceiling(
    ceiling: dict[str, Any],
    *,
    model_id: str,
    context_length: str,
    source_artifact: str,
) -> dict[str, Any]:
    normalized = dict(ceiling)
    normalized.setdefault("label", f"{context_length}_vanilla_ceiling")
    normalized.setdefault("method", "vanilla")
    normalized.setdefault("unit", "full_context")
    normalized.setdefault("metric", "needle_hit_rate")
    if "hits" not in normalized or "cases" not in normalized:
        cases = 50
        hits = int(round(float(normalized["hit_rate"]) * cases))
        normalized["hits"] = hits
        normalized["cases"] = cases
        normalized["wilson_ci_95"] = list(_wilson_ci(hits, cases))
    normalized["model_id"] = model_id
    normalized["context_length"] = context_length
    normalized["source_artifact"] = source_artifact
    normalized["source_type"] = "artifact"
    return normalized


def _observations_from_report(report: dict[str, Any]) -> tuple[CaseObservation, ...]:
    result = report["result"]
    assert isinstance(result, dict)
    payload = result.get("observations", [])
    assert isinstance(payload, list)
    return tuple(CaseObservation(**item) for item in payload)


def _primary_metric(report: dict[str, Any]) -> tuple[str, float]:
    result = report["result"]
    assert isinstance(result, dict)
    metrics = result.get("metrics", {})
    assert isinstance(metrics, dict)
    if len(metrics) != 1:
        raise ValueError(f"expected exactly one primary metric, got {metrics}")
    name, value = next(iter(metrics.items()))
    return str(name), float(value)


def _row_from_cross_model_report(
    filename: str,
    *,
    label: str,
    method: str,
    unit: str,
    budget: int,
    guardrails: str,
    context_length: str,
    model_id: str,
) -> dict[str, Any]:
    report = _load_json(CROSS_MODEL / filename)
    observations = _observations_from_report(report)
    metric_name, hit_rate = _primary_metric(report)
    hits = sum(1 for observation in observations if observation.correct)
    cases = len(observations)
    result = report["result"]
    assert isinstance(result, dict)
    metadata = result.get("metadata", {})
    assert isinstance(metadata, dict)
    return {
        "label": label,
        "method": method,
        "unit": unit,
        "metric": metric_name,
        "budget": budget,
        "guardrails": guardrails,
        "hits": hits,
        "cases": cases,
        "hit_rate": hit_rate,
        "wilson_ci_95": list(_wilson_ci(hits, cases)),
        "elapsed_seconds": float(report["elapsed_seconds"]),
        "estimated_cost_usd": float(report["estimated_cost_usd"]),
        "preprocessing_elapsed_seconds": float(
            result.get("preprocessing_elapsed_seconds", 0.0)
        ),
        "preprocessing_cost_usd": float(result.get("preprocessing_cost_usd", 0.0)),
        "mean_retention_ratio": float(metadata.get("mean_retention_ratio", 1.0)),
        "summary_contract": str(metadata.get("summary_contract", "")),
        "run_id": str(result["run_id"]),
        "model_id": model_id,
        "context_length": context_length,
        "source_artifact": f".benchmarks/cross-model/{filename}",
        "source_type": "raw_report",
        "observations": [
            {
                "fixture_id": observation.fixture_id,
                "correct": observation.correct,
                "prediction": observation.prediction,
            }
            for observation in observations
        ],
    }


def _ceiling_from_cross_model_report(
    filename: str,
    *,
    model_id: str,
    context_length: str,
) -> dict[str, Any]:
    report = _load_json(CROSS_MODEL / filename)
    observations = _observations_from_report(report)
    metric_name, hit_rate = _primary_metric(report)
    hits = sum(1 for observation in observations if observation.correct)
    cases = len(observations)
    return {
        "label": f"{context_length}_vanilla_ceiling",
        "method": "vanilla",
        "unit": "full_context",
        "metric": metric_name,
        "budget_regime": "full_context",
        "hits": hits,
        "cases": cases,
        "hit_rate": hit_rate,
        "wilson_ci_95": list(_wilson_ci(hits, cases)),
        "elapsed_seconds": float(report["elapsed_seconds"]),
        "estimated_cost_usd": float(report["estimated_cost_usd"]),
        "preprocessing_elapsed_seconds": 0.0,
        "preprocessing_cost_usd": 0.0,
        "mean_retention_ratio": 1.0,
        "run_id": str(report["result"]["run_id"]),
        "model_id": model_id,
        "context_length": context_length,
        "source_artifact": f".benchmarks/cross-model/{filename}",
        "source_type": "raw_report",
        "observations": [
            {
                "fixture_id": observation.fixture_id,
                "correct": observation.correct,
                "prediction": observation.prediction,
            }
            for observation in observations
        ],
    }


def _selection_sort_key(row: dict[str, Any]) -> tuple[float, float, int, str]:
    guardrail_rank = 0 if row["guardrails"] == "none" else 1
    return (
        float(row["elapsed_seconds"]),
        float(row["estimated_cost_usd"]),
        guardrail_rank,
        str(row["label"]),
    )


def _select_regime_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_hit_rate = max(float(row["hit_rate"]) for row in rows)
    if max_hit_rate <= 0.0:
        return {
            "selected_label": None,
            "selected_method": None,
            "selected_unit": None,
            "selected_guardrails": None,
            "selected_hit_rate": 0.0,
            "selection_status": "no_positive_recovery",
            "competitive_labels": [str(row["label"]) for row in rows],
        }

    competitive = [
        row for row in rows if float(row["hit_rate"]) == max_hit_rate
    ]
    selected = sorted(competitive, key=_selection_sort_key)[0]
    return {
        "selected_label": selected["label"],
        "selected_method": selected["method"],
        "selected_unit": selected["unit"],
        "selected_guardrails": selected["guardrails"],
        "selected_hit_rate": selected["hit_rate"],
        "selection_status": (
            "unique_max" if len(competitive) == 1 else "tie_broken_by_efficiency"
        ),
        "competitive_labels": [str(row["label"]) for row in competitive],
    }


def _candidate_hit_rate(
    rows: list[dict[str, Any]],
    *,
    unit: str,
    guardrails: str | None = None,
) -> float | None:
    for row in rows:
        if row["unit"] != unit:
            continue
        if guardrails is not None and row.get("guardrails") != guardrails:
            continue
        return float(row["hit_rate"])
    return None


def _regime_row(
    *,
    model_id: str,
    context_length: str,
    budget: int,
    ceiling_hit_rate: float,
    candidates: list[dict[str, Any]],
    scope: str,
) -> dict[str, Any]:
    selection = _select_regime_rows(candidates)
    selected_row = None
    if selection["selected_label"] is not None:
        selected_row = next(
            row for row in candidates if row["label"] == selection["selected_label"]
        )

    return {
        "regime": f"{model_id}:{context_length}:b{budget}",
        "model_id": model_id,
        "context_length": context_length,
        "budget": budget,
        "scope": scope,
        "ceiling_hit_rate": ceiling_hit_rate,
        "token_guarded_hit_rate": _candidate_hit_rate(
            candidates, unit="token", guardrails="prefix_plus_recent"
        ),
        "sentence_guarded_hit_rate": _candidate_hit_rate(
            candidates, unit="sentence", guardrails="prefix_plus_recent"
        ),
        "sentence_noguards_hit_rate": _candidate_hit_rate(
            candidates, unit="sentence", guardrails="none"
        ),
        "word_guarded_hit_rate": _candidate_hit_rate(
            candidates, unit="word", guardrails="prefix_plus_recent"
        ),
        "word_noguards_hit_rate": _candidate_hit_rate(
            candidates, unit="word", guardrails="none"
        ),
        "candidate_labels": [str(row["label"]) for row in candidates],
        "selected_label": selection["selected_label"],
        "selected_method": selection["selected_method"],
        "selected_unit": selection["selected_unit"],
        "selected_guardrails": selection["selected_guardrails"],
        "selected_hit_rate": selection["selected_hit_rate"],
        "selected_elapsed_seconds": (
            None if selected_row is None else float(selected_row["elapsed_seconds"])
        ),
        "selected_estimated_cost_usd": (
            None
            if selected_row is None
            else float(selected_row["estimated_cost_usd"])
        ),
        "selected_vs_ceiling_delta": (
            None
            if selected_row is None
            else float(selected_row["hit_rate"]) - ceiling_hit_rate
        ),
        "selected_vs_token_delta": (
            None
            if selected_row is None
            else float(selected_row["hit_rate"])
            - float(
                _candidate_hit_rate(
                    candidates, unit="token", guardrails="prefix_plus_recent"
                )
                or 0.0
            )
        ),
        "selection_status": selection["selection_status"],
        "competitive_labels": selection["competitive_labels"],
    }


def _winner_counts(selector_rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"token": 0, "sentence": 0, "word": 0, "none": 0}
    for row in selector_rows:
        unit = row["selected_unit"]
        if unit is None:
            counts["none"] += 1
        elif unit in counts:
            counts[unit] += 1
    return counts


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _format_selected(row: dict[str, Any]) -> str:
    if row["selected_label"] is None:
        return "none"
    unit = str(row["selected_unit"])
    guardrails = str(row["selected_guardrails"])
    suffix = "G" if guardrails == "prefix_plus_recent" else "NG"
    return f"{unit}:{suffix}"


def _table_row(row: dict[str, Any]) -> str:
    return (
        "| "
        + " | ".join(
            [
                str(row["model_id"]).replace("`", ""),
                str(row["context_length"]),
                str(row["budget"]),
                f"{float(row['ceiling_hit_rate']):.2f}",
                _format_rate(row["token_guarded_hit_rate"]),
                _format_rate(row["sentence_guarded_hit_rate"]),
                _format_rate(row["sentence_noguards_hit_rate"]),
                _format_rate(row["word_guarded_hit_rate"]),
                _format_rate(row["word_noguards_hit_rate"]),
                _format_selected(row),
                _format_rate(row["selected_hit_rate"]),
                row["selection_status"],
            ]
        )
        + " |"
    )


def main() -> None:
    source_artifacts = [
        "results/sentence-level-eviction-4k-budget-sweep-2026-05-13.json",
        "results/sentence-level-eviction-8k-budget-sweep-2026-05-13.json",
        ".benchmarks/cross-model/qwen-4k-vanilla.json",
        ".benchmarks/cross-model/qwen-4k-token-1024.json",
        ".benchmarks/cross-model/qwen-4k-token-b256.json",
        ".benchmarks/cross-model/qwen-4k-token-b512.json",
        ".benchmarks/cross-model/qwen-4k-token-b1536.json",
        ".benchmarks/cross-model/qwen-4k-token-b2048.json",
        ".benchmarks/cross-model/qwen-4k-sentence-1024-guarded.json",
        ".benchmarks/cross-model/qwen-4k-sentence-1024-noguards.json",
        ".benchmarks/cross-model/qwen-4k-sentence-b256.json",
        ".benchmarks/cross-model/qwen-4k-sentence-b256-noguards.json",
        ".benchmarks/cross-model/qwen-4k-sentence-b512.json",
        ".benchmarks/cross-model/qwen-4k-sentence-b512-noguards.json",
        ".benchmarks/cross-model/qwen-4k-sentence-b1536.json",
        ".benchmarks/cross-model/qwen-4k-sentence-b1536-noguards.json",
        ".benchmarks/cross-model/qwen-4k-sentence-b2048.json",
        ".benchmarks/cross-model/qwen-4k-sentence-b2048-noguards.json",
        ".benchmarks/cross-model/qwen-4k-word-b256.json",
        ".benchmarks/cross-model/qwen-4k-word-b256-noguards.json",
        ".benchmarks/cross-model/qwen-4k-word-b512.json",
        ".benchmarks/cross-model/qwen-4k-word-b512-noguards.json",
        ".benchmarks/cross-model/qwen-4k-word-b1024.json",
        ".benchmarks/cross-model/qwen-4k-word-b1024-noguards.json",
        ".benchmarks/cross-model/qwen-4k-word-b1536.json",
        ".benchmarks/cross-model/qwen-4k-word-b1536-noguards.json",
        ".benchmarks/cross-model/qwen-4k-word-b2048.json",
        ".benchmarks/cross-model/qwen-4k-word-b2048-noguards.json",
    ]

    mistral_4k = _load_json(
        RESULTS / "sentence-level-eviction-4k-budget-sweep-2026-05-13.json"
    )
    mistral_8k = _load_json(
        RESULTS / "sentence-level-eviction-8k-budget-sweep-2026-05-13.json"
    )

    ceilings = [
        _normalize_existing_ceiling(
            mistral_4k["ceiling"],
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            context_length="4k",
            source_artifact=source_artifacts[0],
        ),
        _normalize_existing_ceiling(
            mistral_8k["ceiling"],
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            context_length="8k",
            source_artifact=source_artifacts[1],
        ),
        _ceiling_from_cross_model_report(
            "qwen-4k-vanilla.json",
            model_id="Qwen/Qwen2.5-7B-Instruct",
            context_length="4k",
        ),
    ]

    candidate_rows: list[dict[str, Any]] = []

    for row in mistral_4k["token_rows"] + mistral_4k["sentence_rows"]:
        candidate_rows.append(
            _normalize_existing_row(
                row,
                model_id="mistralai/Mistral-7B-Instruct-v0.3",
                context_length="4k",
                source_artifact=source_artifacts[0],
            )
        )

    for row in mistral_8k["token_rows"] + mistral_8k["sentence_rows"]:
        candidate_rows.append(
            _normalize_existing_row(
                row,
                model_id="mistralai/Mistral-7B-Instruct-v0.3",
                context_length="8k",
                source_artifact=source_artifacts[1],
            )
        )

    qwen_files = [
        ("qwen-4k-token-b256.json", "qwen_token_256_guarded", "token_vorn", "token", 256, "prefix_plus_recent"),
        ("qwen-4k-token-b512.json", "qwen_token_512_guarded", "token_vorn", "token", 512, "prefix_plus_recent"),
        ("qwen-4k-token-1024.json", "qwen_token_1024_guarded", "token_vorn", "token", 1024, "prefix_plus_recent"),
        ("qwen-4k-token-b1536.json", "qwen_token_1536_guarded", "token_vorn", "token", 1536, "prefix_plus_recent"),
        ("qwen-4k-token-b2048.json", "qwen_token_2048_guarded", "token_vorn", "token", 2048, "prefix_plus_recent"),
        ("qwen-4k-sentence-b256.json", "qwen_sentence_256_guarded", "sentence_vorn", "sentence", 256, "prefix_plus_recent"),
        ("qwen-4k-sentence-b256-noguards.json", "qwen_sentence_256_noguards", "sentence_vorn", "sentence", 256, "none"),
        ("qwen-4k-sentence-b512.json", "qwen_sentence_512_guarded", "sentence_vorn", "sentence", 512, "prefix_plus_recent"),
        ("qwen-4k-sentence-b512-noguards.json", "qwen_sentence_512_noguards", "sentence_vorn", "sentence", 512, "none"),
        ("qwen-4k-sentence-1024-guarded.json", "qwen_sentence_1024_guarded", "sentence_vorn", "sentence", 1024, "prefix_plus_recent"),
        ("qwen-4k-sentence-1024-noguards.json", "qwen_sentence_1024_noguards", "sentence_vorn", "sentence", 1024, "none"),
        ("qwen-4k-sentence-b1536.json", "qwen_sentence_1536_guarded", "sentence_vorn", "sentence", 1536, "prefix_plus_recent"),
        ("qwen-4k-sentence-b1536-noguards.json", "qwen_sentence_1536_noguards", "sentence_vorn", "sentence", 1536, "none"),
        ("qwen-4k-sentence-b2048.json", "qwen_sentence_2048_guarded", "sentence_vorn", "sentence", 2048, "prefix_plus_recent"),
        ("qwen-4k-sentence-b2048-noguards.json", "qwen_sentence_2048_noguards", "sentence_vorn", "sentence", 2048, "none"),
        ("qwen-4k-word-b256.json", "qwen_word_256_guarded", "word_vorn", "word", 256, "prefix_plus_recent"),
        ("qwen-4k-word-b256-noguards.json", "qwen_word_256_noguards", "word_vorn", "word", 256, "none"),
        ("qwen-4k-word-b512.json", "qwen_word_512_guarded", "word_vorn", "word", 512, "prefix_plus_recent"),
        ("qwen-4k-word-b512-noguards.json", "qwen_word_512_noguards", "word_vorn", "word", 512, "none"),
        ("qwen-4k-word-b1024.json", "qwen_word_1024_guarded", "word_vorn", "word", 1024, "prefix_plus_recent"),
        ("qwen-4k-word-b1024-noguards.json", "qwen_word_1024_noguards", "word_vorn", "word", 1024, "none"),
        ("qwen-4k-word-b1536.json", "qwen_word_1536_guarded", "word_vorn", "word", 1536, "prefix_plus_recent"),
        ("qwen-4k-word-b1536-noguards.json", "qwen_word_1536_noguards", "word_vorn", "word", 1536, "none"),
        ("qwen-4k-word-b2048.json", "qwen_word_2048_guarded", "word_vorn", "word", 2048, "prefix_plus_recent"),
        ("qwen-4k-word-b2048-noguards.json", "qwen_word_2048_noguards", "word_vorn", "word", 2048, "none"),
    ]
    for filename, label, method, unit, budget, guardrails in qwen_files:
        candidate_rows.append(
            _row_from_cross_model_report(
                filename,
                label=label,
                method=method,
                unit=unit,
                budget=budget,
                guardrails=guardrails,
                context_length="4k",
                model_id="Qwen/Qwen2.5-7B-Instruct",
            )
        )

    ceiling_by_regime = {
        (row["model_id"], row["context_length"]): float(row["hit_rate"]) for row in ceilings
    }

    regimes = [
        ("mistralai/Mistral-7B-Instruct-v0.3", "4k", 512, "common_budget"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "4k", 1024, "common_budget"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "4k", 1536, "common_budget"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "4k", 2048, "common_budget"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "8k", 512, "common_budget"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "8k", 1024, "common_budget"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "8k", 1536, "common_budget"),
        ("mistralai/Mistral-7B-Instruct-v0.3", "8k", 2048, "common_budget"),
        ("Qwen/Qwen2.5-7B-Instruct", "4k", 512, "common_budget"),
        ("Qwen/Qwen2.5-7B-Instruct", "4k", 1024, "common_budget"),
        ("Qwen/Qwen2.5-7B-Instruct", "4k", 1536, "common_budget"),
        ("Qwen/Qwen2.5-7B-Instruct", "4k", 2048, "common_budget"),
        ("Qwen/Qwen2.5-7B-Instruct", "4k", 256, "model_specific_appendix"),
    ]

    selector_rows: list[dict[str, Any]] = []
    for model_id, context_length, budget, scope in regimes:
        candidates = [
            row
            for row in candidate_rows
            if row["model_id"] == model_id
            and row["context_length"] == context_length
            and int(row["budget"]) == budget
        ]
        selector_rows.append(
            _regime_row(
                model_id=model_id,
                context_length=context_length,
                budget=budget,
                ceiling_hit_rate=ceiling_by_regime[(model_id, context_length)],
                candidates=candidates,
                scope=scope,
            )
        )

    common_selector_rows = [
        row for row in selector_rows if row["scope"] == "common_budget"
    ]
    appendix_rows = [
        row for row in selector_rows if row["scope"] == "model_specific_appendix"
    ]

    winner_counts = _winner_counts(common_selector_rows)

    payload = {
        "schema_version": "result-envelope/v0.2",
        "artifact": "adaptive-granularity-posthoc",
        "date": "2026-05-14",
        "selection_policy": {
            "type": "posthoc_descriptive_only",
            "primary": "max_hit_rate",
            "tie_breakers": [
                "lower_elapsed_seconds",
                "lower_estimated_cost_usd",
                "guardrail_free_preferred_when_other_metrics_tie",
                "label_lexical_order",
            ],
            "all_zero_regime": "no_selection",
        },
        "source_artifacts": source_artifacts,
        "read": (
            "This is a descriptive post-hoc selector map over already-run benchmark rows. "
            "It does not add new inferential claims beyond the paired significance already "
            "reported in the source artifacts. Across the common 4k/8k budget regimes, "
            "sentence-level wins the Mistral 4k lane and the Mistral 8k mid-budget band, "
            "while token-level wins the Mistral 8k edge budgets and the Qwen 4k positive "
            "recovery band. Word-level never wins. That supports adaptive granularity "
            "selection as the honest architectural next step, while constraining the claim "
            "to per-model, per-context, and per-budget regimes."
        ),
        "notes": [
            "Mistral rows come from durable v0.2 result artifacts already on main.",
            "Qwen rows were reconstructed directly from raw cross-model reports because the main branch retains those reports but not a durable standalone Qwen sweep artifact.",
            "The Qwen 256 appendix is included for completeness, but the common selector map only compares budgets shared with the Mistral artifacts (512/1024/1536/2048).",
        ],
        "ceiling_rows": ceilings,
        "candidate_rows": candidate_rows,
        "selector_rows": selector_rows,
        "summary": {
            "common_budget_regimes": len(common_selector_rows),
            "winner_counts_by_unit": winner_counts,
            "appendix_regimes": len(appendix_rows),
        },
    }

    md_lines = [
        "# Adaptive Granularity Post-Hoc Analysis — 2026-05-14",
        "",
        "This artifact is descriptive and post-hoc. It adds no new Modal runs and no new inferential claims. "
        "It reuses the already-published paired-result artifacts for Mistral and the raw cross-model reports for Qwen "
        "to ask a narrower question: if an adaptive selector could choose the best granularity/unit per regime after the fact, "
        "what would it have chosen?",
        "",
        "Selection policy:",
        "- Primary criterion: highest hit rate within the regime",
        "- Tie-breakers: lower wall-clock, then lower inference cost, then guardrail-free variant when other metrics tie",
        "- All-zero regimes remain unselected (`none`) rather than inventing a winner",
        "",
        "Source surfaces:",
        "- [sentence-level-eviction-4k-budget-sweep-2026-05-13.md](sentence-level-eviction-4k-budget-sweep-2026-05-13.md)",
        "- [sentence-level-eviction-8k-budget-sweep-2026-05-13.md](sentence-level-eviction-8k-budget-sweep-2026-05-13.md)",
        "- raw Qwen cross-model reports under the benchmark cache directory",
        "",
        "## Common-Budget Selector Map",
        "",
        "| Model | Context | Budget | Ceiling | Token G | Sentence G | Sentence NG | Word G | Word NG | Selected | Selected hit | Status |",
        "|------|---------|--------|---------|---------|------------|-------------|--------|---------|----------|--------------|--------|",
    ]
    md_lines.extend(_table_row(row) for row in common_selector_rows)
    if appendix_rows:
        md_lines.extend(
            [
                "",
                "## Model-Specific Appendix",
                "",
                "| Model | Context | Budget | Ceiling | Token G | Sentence G | Sentence NG | Word G | Word NG | Selected | Selected hit | Status |",
                "|------|---------|--------|---------|---------|------------|-------------|--------|---------|----------|--------------|--------|",
            ]
        )
        md_lines.extend(_table_row(row) for row in appendix_rows)

    md_lines.extend(
        [
            "",
            "## Read",
            "",
            f"- Across the 12 common-budget regimes, the post-hoc selector chooses sentence-level in {winner_counts['sentence']} regimes, token-level in {winner_counts['token']} regimes, word-level in {winner_counts['word']} regimes, and leaves {winner_counts['none']} regime unselected.",
            "- Mistral 4k is sentence-dominant across the shared budget band. The peak stays at `1024`, which matches the earlier 4k sweet-spot result.",
            "- Mistral 8k is regime-shaped rather than monotonic: token wins at the edge budgets (`512`, `2048`), while sentence wins in the mid-budget band (`1024`, `1536`).",
            "- Qwen 4k does not reproduce the Mistral sentence law. Its only positive recovery band is token-level, peaking at `1536`, while word-level never wins and the `256` appendix is an all-zero regime.",
            "- Word-level never wins in any observed regime. That keeps the cross-model claim narrow: the architectural need is adaptive unit selection, not a universal move upward from token to larger semantic units.",
            "- The selector result should be read as evidence for where a forward adaptive policy is worth implementing, not as proof that an ex post oracle is deployable without additional online selection logic.",
        ]
    )

    _json_write(
        RESULTS / "adaptive-granularity-posthoc-2026-05-14.json",
        payload,
    )
    _md_write(
        RESULTS / "adaptive-granularity-posthoc-2026-05-14.md",
        "\n".join(md_lines),
    )


if __name__ == "__main__":
    main()
