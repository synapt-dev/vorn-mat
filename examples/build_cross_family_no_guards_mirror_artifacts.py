#!/usr/bin/env python3
"""Build the cross-family no-guards mirror and Llama threshold artifacts."""
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
BENCHMARKS = ROOT / ".benchmarks" / "cross-model"


def _load_report(name: str) -> dict[str, object]:
    return json.loads((BENCHMARKS / name).read_text())


def _observations(report: dict[str, object]) -> tuple[CaseObservation, ...]:
    result = report["result"]
    assert isinstance(result, dict)
    payload = result.get("observations", [])
    assert isinstance(payload, list)
    return tuple(CaseObservation(**item) for item in payload)


def _run_id(report: dict[str, object]) -> str:
    result = report["result"]
    assert isinstance(result, dict)
    return str(result["run_id"])


def _retention_ratio(report: dict[str, object], default: float = 1.0) -> float:
    result = report["result"]
    assert isinstance(result, dict)
    metadata = result.get("metadata", {})
    assert isinstance(metadata, dict)
    value = metadata.get("mean_retention_ratio")
    if value is None:
        return default
    return float(value)


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


def _primary_metric(report: dict[str, object]) -> tuple[str, float]:
    result = report["result"]
    assert isinstance(result, dict)
    metrics = result.get("metrics", {})
    assert isinstance(metrics, dict)
    if len(metrics) != 1:
        raise ValueError(f"expected exactly one primary metric, got {metrics}")
    name, value = next(iter(metrics.items()))
    return str(name), float(value)


def _row_from_report(
    filename: str,
    *,
    label: str,
    family: str,
    method: str,
    budget: int,
    guardrails: str,
) -> dict[str, object]:
    report = _load_report(filename)
    observations = _observations(report)
    metric_name, hit_rate = _primary_metric(report)
    hits = sum(1 for observation in observations if observation.correct)
    metadata = report["result"]["metadata"]
    assert isinstance(metadata, dict)
    return {
        "label": label,
        "family": family,
        "method": method,
        "metric": metric_name,
        "budget": budget,
        "guardrails": guardrails,
        "hits": hits,
        "cases": len(observations),
        "hit_rate": hit_rate,
        "wilson_ci_95": list(_wilson_ci(hits, len(observations))),
        "elapsed_seconds": float(report["elapsed_seconds"]),
        "estimated_cost_usd": float(report["estimated_cost_usd"]),
        "mean_retention_ratio": _retention_ratio(report),
        "run_id": _run_id(report),
        "model_id": metadata.get("model_id"),
        "retention_policy": metadata.get("retention_policy"),
        "source_report": str(Path(".benchmarks") / "cross-model" / filename),
        "observations": [
            {
                "fixture_id": observation.fixture_id,
                "correct": observation.correct,
                "prediction": observation.prediction,
            }
            for observation in observations
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


def _guardrails_label(value: str) -> str:
    return "prefix + recent" if value == "prefix_plus_recent" else "none"


def _mirror_row_cells(row: dict[str, object]) -> str:
    return (
        "| "
        + " | ".join(
            [
                str(row["family"]),
                str(row["method"]),
                _guardrails_label(str(row["guardrails"])),
                f"{row['hit_rate']:.2f}",
                f"[{row['wilson_ci_95'][0]:.4f}, {row['wilson_ci_95'][1]:.4f}]",
                f"{row['elapsed_seconds']:.2f}s",
                f"${row['estimated_cost_usd']:.4f}",
            ]
        )
        + " |"
    )


def _threshold_row_cells(row: dict[str, object]) -> str:
    return (
        "| "
        + " | ".join(
            [
                str(row["method"]),
                str(row["budget"]),
                _guardrails_label(str(row["guardrails"])),
                f"{row['hit_rate']:.2f}",
                f"[{row['wilson_ci_95'][0]:.4f}, {row['wilson_ci_95'][1]:.4f}]",
                f"{row['elapsed_seconds']:.2f}s",
                f"${row['estimated_cost_usd']:.4f}",
            ]
        )
        + " |"
    )


def main() -> None:
    mirror_rows = [
        _row_from_report(
            "gemma-4-4k-token-1024.json",
            label="gemma_token_guarded",
            family="Gemma 4",
            method="token_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-token-1024-noguards.json",
            label="gemma_token_noguards",
            family="Gemma 4",
            method="token_vorn",
            budget=1024,
            guardrails="none",
        ),
        _row_from_report(
            "gemma-4-4k-sentence-1024-guarded.json",
            label="gemma_sentence_guarded",
            family="Gemma 4",
            method="sentence_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-sentence-1024-noguards.json",
            label="gemma_sentence_noguards",
            family="Gemma 4",
            method="sentence_vorn",
            budget=1024,
            guardrails="none",
        ),
        _row_from_report(
            "gemma-4-4k-tova-1024.json",
            label="gemma_tova_guarded",
            family="Gemma 4",
            method="tova",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-tova-1024-noguards.json",
            label="gemma_tova_noguards",
            family="Gemma 4",
            method="tova",
            budget=1024,
            guardrails="none",
        ),
        _row_from_report(
            "gemma-4-4k-h2o-1024.json",
            label="gemma_h2o_guarded",
            family="Gemma 4",
            method="h2o",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-h2o-1024-noguards.json",
            label="gemma_h2o_noguards",
            family="Gemma 4",
            method="h2o",
            budget=1024,
            guardrails="none",
        ),
        _row_from_report(
            "llama31-4k-token-1024.json",
            label="llama_token_guarded",
            family="Llama 3.1",
            method="token_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "llama31-4k-token-1024-noguards.json",
            label="llama_token_noguards",
            family="Llama 3.1",
            method="token_vorn",
            budget=1024,
            guardrails="none",
        ),
        _row_from_report(
            "llama31-4k-sentence-1024-guarded.json",
            label="llama_sentence_guarded",
            family="Llama 3.1",
            method="sentence_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "llama31-4k-sentence-1024-noguards.json",
            label="llama_sentence_noguards",
            family="Llama 3.1",
            method="sentence_vorn",
            budget=1024,
            guardrails="none",
        ),
        _row_from_report(
            "llama31-4k-tova-1024.json",
            label="llama_tova_guarded",
            family="Llama 3.1",
            method="tova",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "llama31-4k-tova-1024-noguards.json",
            label="llama_tova_noguards",
            family="Llama 3.1",
            method="tova",
            budget=1024,
            guardrails="none",
        ),
        _row_from_report(
            "llama31-4k-h2o-1024.json",
            label="llama_h2o_guarded",
            family="Llama 3.1",
            method="h2o",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "llama31-4k-h2o-1024-noguards.json",
            label="llama_h2o_noguards",
            family="Llama 3.1",
            method="h2o",
            budget=1024,
            guardrails="none",
        ),
    ]
    mirror_by_label = {str(row["label"]): row for row in mirror_rows}
    mirror_tests = [
        _paired_test(mirror_by_label["gemma_token_noguards"], mirror_by_label["gemma_token_guarded"]),
        _paired_test(mirror_by_label["gemma_sentence_noguards"], mirror_by_label["gemma_sentence_guarded"]),
        _paired_test(mirror_by_label["gemma_tova_noguards"], mirror_by_label["gemma_tova_guarded"]),
        _paired_test(mirror_by_label["gemma_h2o_noguards"], mirror_by_label["gemma_h2o_guarded"]),
        _paired_test(mirror_by_label["llama_token_noguards"], mirror_by_label["llama_token_guarded"]),
        _paired_test(mirror_by_label["llama_sentence_noguards"], mirror_by_label["llama_sentence_guarded"]),
        _paired_test(mirror_by_label["llama_tova_noguards"], mirror_by_label["llama_tova_guarded"]),
        _paired_test(mirror_by_label["llama_h2o_noguards"], mirror_by_label["llama_h2o_guarded"]),
    ]

    mirror_payload = {
        "schema_version": "result-envelope/v0.2",
        "provenance": {
            "raw_predictions_embedded": True,
            "raw_report_home": ".benchmarks/cross-model/",
            "raw_report_files": [row["source_report"] for row in mirror_rows],
            "durable_prediction_home": "rows[].observations[]",
            "notes": (
                "Cross-family no-guards mirror artifact. Published rows embed per-fixture "
                "predictions directly and record raw-report filenames + run IDs."
            ),
        },
        "rows": mirror_rows,
        "pairwise_tests": mirror_tests,
        "run_conditions": {
            "profile": "author",
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "niah_multikey_1_4k",
            "split": "validation[:50]",
            "budgets": [1024],
            "families": ["google/gemma-4-E4B-it", "meta-llama/Llama-3.1-8B-Instruct"],
            "note": (
                "Closes the cross-family mirror-image guardrail question by adding token/TOVA/H2O "
                "no-guards rows at 1024 while reusing the earlier sentence no-guards anchors."
            ),
        },
    }
    _json_write(RESULTS / "cross-family-no-guards-mirror-2026-05-15.json", mirror_payload)

    mirror_row_lines = "\n".join(_mirror_row_cells(row) for row in mirror_rows)
    mirror_test_lines = "\n".join(
        "| " + " | ".join(
            [
                str(test["lhs"]),
                str(test["rhs"]),
                str(test["table"]),
                f"{float(test['p_value']):.6g}",
            ]
        ) + " |"
        for test in mirror_tests
    )
    _md_write(
        RESULTS / "cross-family-no-guards-mirror-2026-05-15.md",
        f"""
# Cross-Family No-Guards Mirror @ 1024

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Families: `google/gemma-4-E4B-it`, `meta-llama/Llama-3.1-8B-Instruct`

## Rows

| Family | Method | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost |
| --- | --- | --- | --- | --- | --- | --- |
{mirror_row_lines}

## Guarded vs No-Guards Paired Tests

| LHS | RHS | Table | Exact McNemar p |
| --- | --- | --- | --- |
{mirror_test_lines}

## Read

- **Gemma 4**: the attention-weight preservation story is not a guardrail artifact on this slice. `TOVA` and `H2O` remain at `0.94` with or without prefix/recent-window guardrails, while `token_vorn` stays collapsed (`0.02` guarded, `0.00` no-guards) and `sentence_vorn` remains low (`0.24` guarded, `0.22` no-guards).
- **Llama 3.1**: the residual-direction preservation story is not a guardrail artifact either. Both `token_vorn` and `sentence_vorn` remain at `1.00` with guardrails removed, while `TOVA` and `H2O` stay at `0.90` / `0.94`.
- The mirror-image claim therefore survives guardrail removal cleanly on both families. Gemma's strong rows are genuinely attention-weight-dominant, and Llama's strong rows are genuinely residual-direction-preserving at this budget.
""",
    )

    threshold_rows = [
        _row_from_report(
            "llama31-4k-token-1024.json",
            label="llama_token_1024_guarded",
            family="Llama 3.1",
            method="token_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "llama31-4k-sentence-1024-guarded.json",
            label="llama_sentence_1024_guarded",
            family="Llama 3.1",
            method="sentence_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "llama31-4k-tova-1024.json",
            label="llama_tova_1024_guarded",
            family="Llama 3.1",
            method="tova",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "llama31-4k-h2o-1024.json",
            label="llama_h2o_1024_guarded",
            family="Llama 3.1",
            method="h2o",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "llama31-4k-token-b512.json",
            label="llama_token_512_guarded",
            family="Llama 3.1",
            method="token_vorn",
            budget=512,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "llama31-4k-sentence-b512.json",
            label="llama_sentence_512_guarded",
            family="Llama 3.1",
            method="sentence_vorn",
            budget=512,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "llama31-4k-tova-b512.json",
            label="llama_tova_512_guarded",
            family="Llama 3.1",
            method="tova",
            budget=512,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "llama31-4k-h2o-b512.json",
            label="llama_h2o_512_guarded",
            family="Llama 3.1",
            method="h2o",
            budget=512,
            guardrails="prefix_plus_recent",
        ),
    ]
    threshold_by_label = {str(row["label"]): row for row in threshold_rows}
    threshold_tests = [
        _paired_test(threshold_by_label["llama_token_512_guarded"], threshold_by_label["llama_token_1024_guarded"]),
        _paired_test(threshold_by_label["llama_sentence_512_guarded"], threshold_by_label["llama_sentence_1024_guarded"]),
        _paired_test(threshold_by_label["llama_tova_512_guarded"], threshold_by_label["llama_tova_1024_guarded"]),
        _paired_test(threshold_by_label["llama_h2o_512_guarded"], threshold_by_label["llama_h2o_1024_guarded"]),
    ]

    threshold_payload = {
        "schema_version": "result-envelope/v0.2",
        "provenance": {
            "raw_predictions_embedded": True,
            "raw_report_home": ".benchmarks/cross-model/",
            "raw_report_files": [row["source_report"] for row in threshold_rows],
            "durable_prediction_home": "rows[].observations[]",
            "notes": (
                "Llama 3.1 threshold-cut artifact. Published rows embed per-fixture predictions "
                "directly and record raw-report filenames + run IDs."
            ),
        },
        "rows": threshold_rows,
        "pairwise_tests": threshold_tests,
        "run_conditions": {
            "profile": "author",
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "niah_multikey_1_4k",
            "split": "validation[:50]",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "budgets": [512, 1024],
            "note": (
                "First threshold-cut artifact below the earlier 1024 ceiling regime. Compares the "
                "same four methods at 512 vs 1024 on the identical slice."
            ),
        },
    }
    _json_write(RESULTS / "llama31-4k-threshold-cut-2026-05-15.json", threshold_payload)

    threshold_row_lines = "\n".join(_threshold_row_cells(row) for row in threshold_rows)
    threshold_test_lines = "\n".join(
        "| " + " | ".join(
            [
                str(test["lhs"]),
                str(test["rhs"]),
                str(test["table"]),
                f"{float(test['p_value']):.6g}",
            ]
        ) + " |"
        for test in threshold_tests
    )
    _md_write(
        RESULTS / "llama31-4k-threshold-cut-2026-05-15.md",
        f"""
# Llama 3.1 8B Threshold Cut @ 512

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `meta-llama/Llama-3.1-8B-Instruct`

## Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost |
| --- | --- | --- | --- | --- | --- | --- |
{threshold_row_lines}

## 512 vs 1024 Paired Tests

| LHS | RHS | Table | Exact McNemar p |
| --- | --- | --- | --- |
{threshold_test_lines}

## Read

- Llama's `1024` threshold regime does not extend cleanly down to `512`.
- `sentence_vorn` remains at ceiling (`1.00`), and `token_vorn` only drops slightly (`1.00 -> 0.96`).
- The attention-weight baselines crack much earlier: `TOVA` drops from `0.90` to `0.56`, and `H2O` drops from `0.94` to `0.56`.
- The immediate paper consequence is that the Llama family has a budget-sensitive break, but the break is **method-asymmetric**: at `512`, the attention-weight baselines lose far more quality than the residual-direction baselines.
""",
    )


if __name__ == "__main__":
    main()
