#!/usr/bin/env python3
"""Build the sentence-trigger ablation artifact from Modal reports."""
# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vorn_mat.paired_stats import build_paired_correctness_table, exact_mcnemar
from vorn_mat.results import CaseObservation

RESULTS = ROOT / "results"
BENCHMARKS = ROOT / ".benchmarks"


def _load_report(name: str) -> dict[str, object]:
    return json.loads((BENCHMARKS / name).read_text())


def _observations(report: dict[str, object]) -> tuple[CaseObservation, ...]:
    result = report["result"]
    assert isinstance(result, dict)
    payload = result.get("observations", [])
    assert isinstance(payload, list)
    return tuple(CaseObservation(**item) for item in payload)


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


def _row(report: dict[str, object], *, label: str) -> dict[str, object]:
    observations = _observations(report)
    result = report["result"]
    assert isinstance(result, dict)
    metadata = result["metadata"]
    assert isinstance(metadata, dict)
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    hit_rate = float(metrics["needle_hit_rate"])
    hits = sum(1 for obs in observations if obs.correct)
    cases = len(observations)
    return {
        "label": label,
        "method": "sentence_vorn",
        "metric": "needle_hit_rate",
        "dataset_config": report["dataset_config"],
        "budget": int(report["cache_budget_tokens"]),
        "guardrails": (
            "prefix_plus_recent"
            if bool(report["always_keep_prefix_tokens"]) and bool(report["preserve_recent_window"])
            else "none"
        ),
        "eviction_trigger": report["eviction_trigger"],
        "sentence_pooling": report["sentence_pooling"],
        "sentence_top_k": int(report["sentence_top_k"]),
        "hits": hits,
        "cases": cases,
        "hit_rate": hit_rate,
        "wilson_ci_95": list(_wilson_ci(hits, cases)),
        "elapsed_seconds": float(report["elapsed_seconds"]),
        "estimated_cost_usd": float(report["estimated_cost_usd"]),
        "preprocessing_elapsed_seconds": float(
            result.get("preprocessing_elapsed_seconds", 0.0)
        ),
        "preprocessing_cost_usd": float(
            result.get("preprocessing_cost_usd", 0.0)
        ),
        "mean_retention_ratio": float(metadata["mean_retention_ratio"]),
        "run_id": str(result["run_id"]),
        "summary_contract": str(metadata["summary_contract"]),
        "observations": [
            {
                "fixture_id": obs.fixture_id,
                "correct": obs.correct,
                "prediction": obs.prediction,
            }
            for obs in observations
        ],
    }


def _paired_test(lhs: dict[str, object], rhs: dict[str, object]) -> dict[str, object]:
    lhs_obs = tuple(CaseObservation(**item) for item in lhs["observations"])  # type: ignore[arg-type]
    rhs_obs = tuple(CaseObservation(**item) for item in rhs["observations"])  # type: ignore[arg-type]
    table = build_paired_correctness_table(lhs_obs, rhs_obs)
    result = exact_mcnemar(table)
    return {
        "lhs": lhs["label"],
        "rhs": rhs["label"],
        "paired": True,
        "same_n": True,
        "test": "mcnemar_exact_two_sided",
        "table": [list(table[0]), list(table[1])],
        "p_value": result.p_value,
        "discordant_pairs": result.discordant_pairs,
        "lhs_discordant_share": result.lhs_discordant_share,
    }


def _json_write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _md_write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n")


def main() -> None:
    threshold = _row(
        _load_report("8k-sentence-vorn-b1536-threshold-trigger-report.json"),
        label="sentence_1536_threshold_trigger",
    )
    boundary = _row(
        _load_report("8k-sentence-vorn-b1536-sentence-boundary-trigger-report.json"),
        label="sentence_1536_sentence_boundary_trigger",
    )
    paired = _paired_test(boundary, threshold)

    payload = {
        "schema_version": "result-envelope/v0.2",
        "metadata": {
            "experiment_id": "sentence-trigger-ablation-8k-1536-2026-05-14",
            "date": "2026-05-14",
            "team": "synapt",
            "slice": {
                "dataset": "rbiswasfc/ruler",
                "name": "niah_multikey_1_8k",
                "split": "validation",
                "n": 50,
                "selector": "validation[:50]",
            },
            "model": {
                "name": "mistralai/Mistral-7B-Instruct-v0.3",
                "provider": "modal",
                "profile": "author",
                "deterministic": True,
            },
            "comparison_axis": "eviction_trigger",
            "budget": 1536,
            "retention_policy": "sentence_vorn",
            "sentence_pooling": "max",
            "sentence_top_k": 3,
        },
        "rows": [threshold, boundary],
        "pairwise_tests": [paired],
    }
    _json_write(
        RESULTS / "sentence-trigger-ablation-2026-05-14.json",
        payload,
    )

    direction = (
        "supports a real quality gain for sentence-boundary triggering"
        if paired["p_value"] < 0.05
        else "does not support a decisive quality difference at n=50"
    )
    better_row = boundary if boundary["hit_rate"] >= threshold["hit_rate"] else threshold
    md = f"""
# Sentence Trigger Ablation — 2026-05-14

Run conditions:
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_8k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Retention policy: `sentence_vorn`
- Budget: `1536`
- Pooling: `max`

| Variant | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | KV savings |
|--------|----------|---------------|------------|----------------|-----------|
| Threshold trigger | {threshold['hit_rate']:.2f} | [{threshold['wilson_ci_95'][0]:.4f}, {threshold['wilson_ci_95'][1]:.4f}] | {threshold['elapsed_seconds']:.2f}s | ${threshold['estimated_cost_usd']:.4f} | {(1.0 - float(threshold['mean_retention_ratio'])) * 100:.2f}% |
| Sentence-boundary trigger | {boundary['hit_rate']:.2f} | [{boundary['wilson_ci_95'][0]:.4f}, {boundary['wilson_ci_95'][1]:.4f}] | {boundary['elapsed_seconds']:.2f}s | ${boundary['estimated_cost_usd']:.4f} | {(1.0 - float(boundary['mean_retention_ratio'])) * 100:.2f}% |

## Paired test

- sentence_boundary_trigger vs threshold_trigger, exact McNemar on `{paired['table']}`: `p = {paired['p_value']:.6g}`

## Read

- This run {direction}.
- The higher point estimate in this pair is `{better_row['label']}` at `{better_row['hit_rate']:.2f}`.
- `observations[]` is preserved in the JSON artifact, so the paired claim is reconstructible downstream.
"""
    _md_write(RESULTS / "sentence-trigger-ablation-2026-05-14.md", md)


if __name__ == "__main__":
    main()
