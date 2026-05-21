#!/usr/bin/env python3
"""Build the Ministral 8B 4k fast-read artifact from Modal reports."""
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
    budget: int | None,
    guardrails: str,
) -> dict[str, object]:
    report = _load_report(filename)
    observations = _observations(report)
    metric_name, hit_rate = _primary_metric(report)
    hits = sum(1 for observation in observations if observation.correct)
    cases = len(observations)
    result = report["result"]
    assert isinstance(result, dict)
    metadata = result.get("metadata", {})
    assert isinstance(metadata, dict)
    row: dict[str, object] = {
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
    if "sentence_pooling" in metadata:
        row["sentence_pooling"] = metadata["sentence_pooling"]
    if "sentence_top_k" in metadata:
        row["sentence_top_k"] = int(str(metadata["sentence_top_k"]))
    return row


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


def _row_cells(row: dict[str, object], method_label: str) -> str:
    budget = "full context" if row["budget"] is None else str(row["budget"])
    return (
        "| "
        + " | ".join(
            [
                method_label,
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
    vanilla = _row_from_report(
        "ministral-8b-4k-vanilla.json",
        label="vanilla_floor",
        method="vanilla",
        budget=None,
        guardrails="prefix_plus_recent",
    )
    tova = _row_from_report(
        "ministral-8b-4k-tova-1024.json",
        label="tova_1024_guarded",
        method="tova",
        budget=1024,
        guardrails="prefix_plus_recent",
    )
    sentence = _row_from_report(
        "ministral-8b-4k-sentence-1024-guarded.json",
        label="sentence_1024_guarded",
        method="sentence_vorn",
        budget=1024,
        guardrails="prefix_plus_recent",
    )

    rows = [vanilla, tova, sentence]
    pairwise_tests = [
        _paired_test(sentence, tova),
        _paired_test(vanilla, tova),
        _paired_test(sentence, vanilla),
    ]

    payload = {
        "schema_version": "result-envelope/v0.2",
        "provenance": {
            "raw_predictions_embedded": True,
            "raw_report_home": ".benchmarks/cross-model/",
            "raw_report_files": [row["source_report"] for row in rows],
            "durable_prediction_home": "rows[].observations[]",
            "notes": (
                "Ministral 8B fast-read artifact. Published rows embed per-fixture "
                "predictions directly and record raw-report filenames + run IDs."
            ),
        },
        "rows": rows,
        "pairwise_tests": pairwise_tests,
        "run_conditions": {
            "profile": "author",
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "niah_multikey_1_4k",
            "split": "validation[:50]",
            "model": "mistralai/Ministral-8B-Instruct-2410",
            "random_seed": 17,
            "canonical_layer": 16,
            "recent_token_window": 16,
            "sentence_pooling": "max",
            "sentence_top_k": 3,
            "note": (
                "Three-row fast-read for the fifth-family architectural prediction test: "
                "vanilla, TOVA@1024, and sentence_vorn@1024 on the standard 4k NIAH slice."
            ),
        },
    }
    _json_write(RESULTS / "ministral-8b-4k-fast-read-2026-05-15.json", payload)

    row_lines = "\n".join(
        [
            _row_cells(vanilla, "Vanilla"),
            _row_cells(tova, "TOVA"),
            _row_cells(sentence, "Sentence vorn"),
        ]
    )
    pairwise_lines = "\n".join(
        f"- {pairwise['lhs']} vs {pairwise['rhs']}, exact McNemar on `{pairwise['table']}`: "
        f"`p = {pairwise['p_value']:.6g}`"
        for pairwise in pairwise_tests
    )

    _md_write(
        RESULTS / "ministral-8b-4k-fast-read-2026-05-15.md",
        f"""
# Ministral 8B 4k Fast-Read — 2026-05-15

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Ministral-8B-Instruct-2410`

## Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Mean evicted |
| --- | --- | --- | --- | --- | --- | --- | --- |
{row_lines}

## Pairwise Tests

{pairwise_lines}

## Read

- Ministral 8B does **not** look Gemma-like on this slice. Its full-context ceiling is `1.00`, but `TOVA@1024` drops sharply to `0.44` while `sentence_vorn@1024` stays at `1.00`.
- That makes the fifth-family fast-read a strong architectural prediction test in the residual-direction direction, not in the attention-weight direction. The interleaved-attention family here aligns with the sentence-preserving channel rather than the Gemma-style attention-weight dominance.
- The immediate next question is whether the full cross-baseline matrix on Ministral keeps this separation once token_vorn and H2O are added, or whether the fast-read is only exposing the most extreme channel contrast.
""",
    )


if __name__ == "__main__":
    main()
