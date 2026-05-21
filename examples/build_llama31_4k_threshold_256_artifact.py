#!/usr/bin/env python3
"""Build the Llama 3.1 4k threshold-curve artifact including budget 256."""
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
    method: str,
    budget: int,
) -> dict[str, object]:
    report = _load_report(filename)
    observations = _observations(report)
    metric_name, hit_rate = _primary_metric(report)
    hits = sum(1 for observation in observations if observation.correct)
    result = report["result"]
    assert isinstance(result, dict)
    metadata = result.get("metadata", {})
    assert isinstance(metadata, dict)
    return {
        "label": label,
        "method": method,
        "metric": metric_name,
        "budget": budget,
        "guardrails": "prefix_plus_recent",
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


def _row_cells(row: dict[str, object]) -> str:
    return (
        "| "
        + " | ".join(
            [
                str(row["method"]),
                str(row["budget"]),
                f"{row['hit_rate']:.2f}",
                f"[{row['wilson_ci_95'][0]:.4f}, {row['wilson_ci_95'][1]:.4f}]",
                f"{row['elapsed_seconds']:.2f}s",
                f"${row['estimated_cost_usd']:.4f}",
                f"{(1.0 - float(row['mean_retention_ratio'])) * 100:.2f}%",
            ]
        )
        + " |"
    )


def main() -> None:
    rows = [
        _row_from_report(
            "llama31-4k-token-1024.json",
            label="token_1024",
            method="token_vorn",
            budget=1024,
        ),
        _row_from_report(
            "llama31-4k-token-b512.json",
            label="token_512",
            method="token_vorn",
            budget=512,
        ),
        _row_from_report(
            "llama31-4k-token-b256.json",
            label="token_256",
            method="token_vorn",
            budget=256,
        ),
        _row_from_report(
            "llama31-4k-sentence-1024-guarded.json",
            label="sentence_1024",
            method="sentence_vorn",
            budget=1024,
        ),
        _row_from_report(
            "llama31-4k-sentence-b512.json",
            label="sentence_512",
            method="sentence_vorn",
            budget=512,
        ),
        _row_from_report(
            "llama31-4k-sentence-b256.json",
            label="sentence_256",
            method="sentence_vorn",
            budget=256,
        ),
        _row_from_report(
            "llama31-4k-tova-1024.json",
            label="tova_1024",
            method="tova",
            budget=1024,
        ),
        _row_from_report(
            "llama31-4k-tova-b512.json",
            label="tova_512",
            method="tova",
            budget=512,
        ),
        _row_from_report(
            "llama31-4k-tova-b256.json",
            label="tova_256",
            method="tova",
            budget=256,
        ),
        _row_from_report(
            "llama31-4k-h2o-1024.json",
            label="h2o_1024",
            method="h2o",
            budget=1024,
        ),
        _row_from_report(
            "llama31-4k-h2o-b512.json",
            label="h2o_512",
            method="h2o",
            budget=512,
        ),
        _row_from_report(
            "llama31-4k-h2o-b256.json",
            label="h2o_256",
            method="h2o",
            budget=256,
        ),
    ]
    by_label = {str(row["label"]): row for row in rows}
    pairwise_tests = [
        _paired_test(by_label["token_512"], by_label["token_1024"]),
        _paired_test(by_label["token_256"], by_label["token_512"]),
        _paired_test(by_label["token_256"], by_label["token_1024"]),
        _paired_test(by_label["sentence_512"], by_label["sentence_1024"]),
        _paired_test(by_label["sentence_256"], by_label["sentence_512"]),
        _paired_test(by_label["sentence_256"], by_label["sentence_1024"]),
        _paired_test(by_label["tova_512"], by_label["tova_1024"]),
        _paired_test(by_label["tova_256"], by_label["tova_512"]),
        _paired_test(by_label["tova_256"], by_label["tova_1024"]),
        _paired_test(by_label["h2o_512"], by_label["h2o_1024"]),
        _paired_test(by_label["h2o_256"], by_label["h2o_512"]),
        _paired_test(by_label["h2o_256"], by_label["h2o_1024"]),
    ]

    payload = {
        "schema_version": "result-envelope/v0.2",
        "provenance": {
            "raw_predictions_embedded": True,
            "raw_report_home": ".benchmarks/cross-model/",
            "raw_report_files": [row["source_report"] for row in rows],
            "durable_prediction_home": "rows[].observations[]",
            "notes": (
                "Llama 3.1 threshold-curve artifact through budget 256. "
                "Published rows embed per-fixture predictions directly and record raw-report filenames + run IDs."
            ),
        },
        "rows": rows,
        "pairwise_tests": pairwise_tests,
        "run_conditions": {
            "profile": "author",
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "niah_multikey_1_4k",
            "split": "validation[:50]",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "budgets": [256, 512, 1024],
            "note": (
                "Extends the earlier 512-vs-1024 threshold-cut surface downward to budget 256 "
                "for token, sentence, TOVA, and H2O."
            ),
        },
    }
    _json_write(RESULTS / "llama31-4k-threshold-curve-2026-05-16.json", payload)

    row_lines = "\n".join(_row_cells(row) for row in rows)
    test_lines = "\n".join(
        "| "
        + " | ".join(
            [
                str(test["lhs"]),
                str(test["rhs"]),
                str(test["table"]),
                f"{float(test['p_value']):.6g}",
            ]
        )
        + " |"
        for test in pairwise_tests
    )

    _md_write(
        RESULTS / "llama31-4k-threshold-curve-2026-05-16.md",
        f"""
# Llama 3.1 8B Threshold Curve Through Budget 256 — 2026-05-16

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `meta-llama/Llama-3.1-8B-Instruct`

## Rows

| Method | Budget | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Mean evicted |
| --- | --- | --- | --- | --- | --- | --- |
{row_lines}

## Paired Tests

| LHS | RHS | Table | Exact McNemar p |
| --- | --- | --- | --- |
{test_lines}

## Read

- The residual-direction advantage on Llama does not merely survive to `256`; it sharpens into a hard lower-bound contrast.
- `sentence_vorn` remains at `1.00` at all three budgets (`1024`, `512`, `256`), with zero discordance against the higher-budget rows.
- The token baseline breaks substantially at `256`, falling from `1.00` at `1024` to `0.56`.
- The attention-weight baselines break much harder still: `TOVA` falls from `0.90` to `0.56` to `0.12`, and `H2O` falls from `0.94` to `0.56` to `0.08`.
- The practical consequence is that the Llama family is not merely threshold-shaped. It exhibits a strongly method-asymmetric floor: the sentence-level residual-direction policy remains ceiling-stable even at roughly six percent retention, while token and attention-weight policies degrade sharply.
""",
    )


if __name__ == "__main__":
    main()
