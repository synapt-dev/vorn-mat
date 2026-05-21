#!/usr/bin/env python3
"""Build the Gemma 4k degradation-curve artifact from Modal reports."""
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


def _retention_ratio(report: dict[str, object], default: float = 1.0) -> float:
    result = report["result"]
    assert isinstance(result, dict)
    metadata = result.get("metadata", {})
    assert isinstance(metadata, dict)
    value = metadata.get("mean_retention_ratio")
    if value is None:
        return default
    return float(value)


def _run_id(report: dict[str, object]) -> str:
    result = report["result"]
    assert isinstance(result, dict)
    return str(result["run_id"])


def _metadata(report: dict[str, object]) -> dict[str, object]:
    result = report["result"]
    assert isinstance(result, dict)
    metadata = result.get("metadata", {})
    assert isinstance(metadata, dict)
    return metadata


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
    budget: int | None,
    guardrails: str,
) -> dict[str, object]:
    report = _load_report(filename)
    observations = _observations(report)
    metric_name, hit_rate = _primary_metric(report)
    hits = sum(1 for observation in observations if observation.correct)
    cases = len(observations)
    metadata = _metadata(report)
    return {
        "label": label,
        "method": method,
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
            report["result"].get("preprocessing_elapsed_seconds", 0.0)  # type: ignore[index]
        ),
        "preprocessing_cost_usd": float(
            report["result"].get("preprocessing_cost_usd", 0.0)  # type: ignore[index]
        ),
        "mean_retention_ratio": _retention_ratio(report),
        "run_id": _run_id(report),
        "source_report": str(Path(".benchmarks") / "cross-model" / filename),
        "observations": [
            {
                "fixture_id": observation.fixture_id,
                "correct": observation.correct,
                "prediction": observation.prediction,
            }
            for observation in observations
        ],
        "model_id": metadata.get("model_id"),
        "retention_policy": metadata.get("retention_policy"),
        "gpu": metadata.get("gpu"),
        "random_seed": metadata.get("random_seed"),
        "suite_id": metadata.get("suite_id"),
        "eviction_unit": metadata.get("eviction_unit"),
        "sentence_pooling": metadata.get("sentence_pooling"),
        "sentence_top_k": metadata.get("sentence_top_k"),
        "always_keep_prefix_tokens": metadata.get("always_keep_prefix_tokens"),
        "preserve_recent_window": metadata.get("preserve_recent_window"),
        "total_eviction_steps": metadata.get("total_eviction_steps"),
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


def _row_cells(row: dict[str, object]) -> str:
    budget = "full context" if row["budget"] is None else str(row["budget"])
    return (
        "| "
        + " | ".join(
            [
                str(row["method"]),
                budget,
                _guardrails_label(str(row["guardrails"])),
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
            "gemma-4-4k-vanilla.json",
            label="vanilla_full_context",
            method="vanilla",
            budget=None,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-token-b512.json",
            label="token_512_guarded",
            method="token_vorn",
            budget=512,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-sentence-b512.json",
            label="sentence_512_guarded",
            method="sentence_vorn",
            budget=512,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-tova-b512.json",
            label="tova_512_guarded",
            method="tova",
            budget=512,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-token-1024.json",
            label="token_1024_guarded",
            method="token_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-sentence-1024-guarded.json",
            label="sentence_1024_guarded",
            method="sentence_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-tova-1024.json",
            label="tova_1024_guarded",
            method="tova",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-h2o-1024.json",
            label="h2o_1024_guarded",
            method="h2o",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-token-b1536.json",
            label="token_1536_guarded",
            method="token_vorn",
            budget=1536,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-sentence-b1536.json",
            label="sentence_1536_guarded",
            method="sentence_vorn",
            budget=1536,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-tova-b1536.json",
            label="tova_1536_guarded",
            method="tova",
            budget=1536,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-token-b2048.json",
            label="token_2048_guarded",
            method="token_vorn",
            budget=2048,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-sentence-b2048.json",
            label="sentence_2048_guarded",
            method="sentence_vorn",
            budget=2048,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-4k-tova-b2048.json",
            label="tova_2048_guarded",
            method="tova",
            budget=2048,
            guardrails="prefix_plus_recent",
        ),
    ]
    by_label = {str(row["label"]): row for row in rows}
    pairwise_tests = [
        _paired_test(by_label["sentence_512_guarded"], by_label["token_512_guarded"]),
        _paired_test(by_label["tova_512_guarded"], by_label["sentence_512_guarded"]),
        _paired_test(by_label["tova_512_guarded"], by_label["token_512_guarded"]),
        _paired_test(by_label["sentence_1024_guarded"], by_label["token_1024_guarded"]),
        _paired_test(by_label["tova_1024_guarded"], by_label["sentence_1024_guarded"]),
        _paired_test(by_label["h2o_1024_guarded"], by_label["sentence_1024_guarded"]),
        _paired_test(by_label["tova_1024_guarded"], by_label["token_1024_guarded"]),
        _paired_test(by_label["h2o_1024_guarded"], by_label["token_1024_guarded"]),
        _paired_test(by_label["tova_1024_guarded"], by_label["h2o_1024_guarded"]),
        _paired_test(by_label["sentence_1536_guarded"], by_label["token_1536_guarded"]),
        _paired_test(by_label["tova_1536_guarded"], by_label["sentence_1536_guarded"]),
        _paired_test(by_label["tova_1536_guarded"], by_label["token_1536_guarded"]),
        _paired_test(by_label["sentence_2048_guarded"], by_label["token_2048_guarded"]),
        _paired_test(by_label["tova_2048_guarded"], by_label["sentence_2048_guarded"]),
        _paired_test(by_label["tova_2048_guarded"], by_label["token_2048_guarded"]),
    ]

    payload = {
        "schema_version": "result-envelope/v0.2",
        "provenance": {
            "raw_predictions_embedded": True,
            "raw_report_home": ".benchmarks/cross-model/",
            "raw_report_files": [row["source_report"] for row in rows],
            "durable_prediction_home": "rows[].observations[]",
            "notes": (
                "Published artifact embeds per-fixture predictions directly in each row and "
                "records the source raw-report filenames + run_ids. The raw Modal report layer "
                "remains in .benchmarks/cross-model/ for replay/audit."
            ),
        },
        "rows": rows,
        "pairwise_tests": pairwise_tests,
        "run_conditions": {
            "profile": "author",
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "niah_multikey_1_4k",
            "split": "validation[:50]",
            "model": "google/gemma-4-E4B-it",
            "random_seed": 17,
            "canonical_layer": 16,
            "budgets": [512, 1024, 1536, 2048],
            "methods": ["token_vorn", "sentence_vorn", "tova", "h2o@1024 only", "vanilla"],
            "note": (
                "First-wave Gemma-only degradation curve. 1024 rows are reused from the earlier "
                "cross-baseline extension; 512/1536/2048 localize the bend before spending on "
                "256/768/3072 or additional H2O rows."
            ),
        },
    }
    _json_write(RESULTS / "gemma-4-4k-degradation-curve-2026-05-15.json", payload)

    row_lines = "\n".join(_row_cells(row) for row in rows)

    tests_of_interest = [
        ("tova_512_guarded", "sentence_512_guarded"),
        ("sentence_512_guarded", "token_512_guarded"),
        ("tova_1024_guarded", "sentence_1024_guarded"),
        ("tova_1536_guarded", "sentence_1536_guarded"),
        ("sentence_1536_guarded", "token_1536_guarded"),
        ("tova_2048_guarded", "sentence_2048_guarded"),
        ("sentence_2048_guarded", "token_2048_guarded"),
    ]
    test_lookup = {(test["lhs"], test["rhs"]): test for test in pairwise_tests}
    pairwise_lines = "\n".join(
        [
            "| "
            + " | ".join(
                [
                    lhs,
                    rhs,
                    str(test_lookup[(lhs, rhs)]["table"]),
                    f"{float(test_lookup[(lhs, rhs)]['p_value']):.6g}",
                ]
            )
            + " |"
            for lhs, rhs in tests_of_interest
        ]
    )

    markdown = f"""
# Gemma 4 E4B-it Degradation Curve @ NIAH Multikey 4k

## Setup
- Model: `google/gemma-4-E4B-it`
- Fixture: `rbiswasfc/ruler` `niah_multikey_1_4k` `validation[:50]`
- Profile: `author`
- Seed: `17`
- Budgets: `512`, `1024`, `1536`, `2048`
- Methods: vanilla, token-vorn, sentence-vorn, TOVA, sparse H2O anchor at `1024`

## Read
- Gemma's degradation curve has a sharp bend rather than a smooth decline.
- At `512`, the residual-direction methods are near floor: `sentence_vorn=0.04`, `token_vorn=0.00`, while `TOVA=0.34`.
- At `1024`, the earlier cross-baseline split remains the center anchor: `sentence_vorn=0.24`, `token_vorn=0.02`, `TOVA=0.94`, `H2O=0.94`.
- By `1536`, sentence-level retention becomes workable (`0.68`) but still trails the attention-weight baseline (`0.98`).
- By `2048`, sentence-level vorn nearly recovers the ceiling (`0.96`), while token-level is still materially degraded (`0.52`) and TOVA saturates (`1.00`).
- The honest local claim is now sharper: on Gemma 4, method ranking is budget-dependent inside the residual-direction family, but the attention-weight advantage persists across the whole observed bend.

## Rows
| Method | Budget | Guardrails | Hit rate | Wilson 95% CI | Elapsed | Cost | Positions evicted |
| --- | --- | --- | --- | --- | --- | --- | --- |
{row_lines}

## Pairwise Tests
| LHS | RHS | Table | Exact McNemar p |
| --- | --- | --- | --- |
{pairwise_lines}

## Provenance
- Raw predictions are embedded directly in the JSON artifact under `rows[].observations[]`.
- Source raw-report filenames and run IDs are recorded per row.
- Raw Modal report home for replay/audit: `.benchmarks/cross-model/`.
"""
    _md_write(RESULTS / "gemma-4-4k-degradation-curve-2026-05-15.md", markdown)


if __name__ == "__main__":
    main()
