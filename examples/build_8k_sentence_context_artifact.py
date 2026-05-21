#!/usr/bin/env python3
"""Build the 8k sentence-level eviction comparison artifact from Modal reports."""
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


def _retention_ratio(report: dict[str, object], default: float = 1.0) -> float:
    result = report["result"]
    assert isinstance(result, dict)
    metadata = result.get("metadata", {})
    assert isinstance(metadata, dict)
    value = metadata.get("mean_retention_ratio")
    if value is None:
        return default
    return float(value)


def _summary_contract(report: dict[str, object], default: str = "") -> str:
    result = report["result"]
    assert isinstance(result, dict)
    metadata = result.get("metadata", {})
    assert isinstance(metadata, dict)
    return str(metadata.get("summary_contract", default))


def _run_id(report: dict[str, object]) -> str:
    result = report["result"]
    assert isinstance(result, dict)
    return str(result["run_id"])


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


def _row_from_live_report(
    report: dict[str, object],
    *,
    label: str,
    method: str,
    budget: int,
    guardrails: str,
) -> dict[str, object]:
    observations = _observations(report)
    metric_name, hit_rate = _primary_metric(report)
    hits = sum(1 for observation in observations if observation.correct)
    cases = len(observations)
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
        "preprocessing_elapsed_seconds": float(
            report["result"].get("preprocessing_elapsed_seconds", 0.0)  # type: ignore[index]
        ),
        "preprocessing_cost_usd": float(
            report["result"].get("preprocessing_cost_usd", 0.0)  # type: ignore[index]
        ),
        "mean_retention_ratio": _retention_ratio(report),
        "summary_contract": _summary_contract(report),
        "run_id": _run_id(report),
        "observations": [
            {
                "fixture_id": observation.fixture_id,
                "correct": observation.correct,
                "prediction": observation.prediction,
            }
            for observation in observations
        ],
    }
    result = report["result"]
    assert isinstance(result, dict)
    metadata = result.get("metadata", {})
    assert isinstance(metadata, dict)
    if "sentence_pooling" in metadata:
        row["sentence_pooling"] = metadata["sentence_pooling"]
    if "sentence_top_k" in metadata:
        row["sentence_top_k"] = int(str(metadata["sentence_top_k"]))
    if "retention_policy" in metadata:
        row["retention_policy"] = metadata["retention_policy"]
    return row


def _row_from_vanilla_report(report: dict[str, object]) -> dict[str, object]:
    observations = _observations(report)
    metric_name, hit_rate = _primary_metric(report)
    hits = sum(1 for observation in observations if observation.correct)
    cases = len(observations)
    return {
        "label": "vanilla_ceiling",
        "method": "vanilla",
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
        "summary_contract": "full_context",
        "run_id": _run_id(report),
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


def main() -> None:
    vanilla = _row_from_vanilla_report(_load_report("8k-vanilla-report.json"))
    token_1024_guarded = _row_from_live_report(
        _load_report("8k-token-vorn-b1024-report.json"),
        label="token_1024_guarded",
        method="token_vorn",
        budget=1024,
        guardrails="prefix_plus_recent",
    )
    token_2048_guarded = _row_from_live_report(
        _load_report("8k-token-vorn-b2048-report.json"),
        label="token_2048_guarded",
        method="token_vorn",
        budget=2048,
        guardrails="prefix_plus_recent",
    )
    sentence_1024_guarded = _row_from_live_report(
        _load_report("8k-sentence-vorn-b1024-report.json"),
        label="sentence_1024_guarded",
        method="sentence_vorn",
        budget=1024,
        guardrails="prefix_plus_recent",
    )
    sentence_1024_noguards = _row_from_live_report(
        _load_report("8k-sentence-vorn-b1024-noguards-report.json"),
        label="sentence_1024_noguards",
        method="sentence_vorn",
        budget=1024,
        guardrails="none",
    )
    sentence_2048_guarded = _row_from_live_report(
        _load_report("8k-sentence-vorn-b2048-report.json"),
        label="sentence_2048_guarded",
        method="sentence_vorn",
        budget=2048,
        guardrails="prefix_plus_recent",
    )
    sentence_2048_noguards = _row_from_live_report(
        _load_report("8k-sentence-vorn-b2048-noguards-report.json"),
        label="sentence_2048_noguards",
        method="sentence_vorn",
        budget=2048,
        guardrails="none",
    )

    comparisons = [
        sentence_1024_guarded,
        sentence_1024_noguards,
        sentence_2048_guarded,
        sentence_2048_noguards,
    ]
    token_references = [token_1024_guarded, token_2048_guarded]
    pairwise_tests = [
        _paired_test(sentence_1024_guarded, token_1024_guarded),
        _paired_test(sentence_1024_noguards, sentence_1024_guarded),
        _paired_test(sentence_2048_guarded, token_2048_guarded),
        _paired_test(sentence_2048_noguards, sentence_2048_guarded),
        _paired_test(sentence_2048_guarded, sentence_1024_guarded),
        _paired_test(sentence_2048_noguards, sentence_1024_noguards),
    ]

    payload = {
        "schema_version": "result-envelope/v0.2",
        "ceiling": vanilla,
        "comparisons": comparisons,
        "token_references": token_references,
        "pairwise_tests": pairwise_tests,
        "run_conditions": {
            "profile": "author",
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "niah_multikey_1_8k",
            "split": "validation[:50]",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "random_seed": 17,
            "canonical_layer": 16,
            "recent_token_window": 16,
            "sentence_pooling": "max",
            "sentence_top_k": 3,
        },
    }
    _json_write(RESULTS / "sentence-level-eviction-8k-2026-05-13.json", payload)

    pairwise_lines = []
    for pairwise in pairwise_tests:
        pairwise_lines.append(
            f"- {pairwise['lhs']} vs {pairwise['rhs']}, exact McNemar on "
            f"`{pairwise['table']}`: `p = {pairwise['p_value']:.6g}`"
        )

    comparison_rows = "\n".join(
        "| "
        + " | ".join(
            [
                "Sentence vorn",
                str(row["budget"]),
                "prefix + recent" if row["guardrails"] == "prefix_plus_recent" else "none",
                f"{row['hit_rate']:.2f}",
                f"[{row['wilson_ci_95'][0]:.4f}, {row['wilson_ci_95'][1]:.4f}]",
                f"{row['elapsed_seconds']:.2f}s",
                f"${row['estimated_cost_usd']:.4f}",
                "0",
                f"{(1.0 - float(row['mean_retention_ratio'])) * 100:.2f}%",
            ]
        )
        + " |"
        for row in comparisons
    )
    token_rows = "\n".join(
        "| "
        + " | ".join(
            [
                "Token vorn",
                str(row["budget"]),
                "prefix + recent",
                f"{row['hit_rate']:.2f}",
                f"[{row['wilson_ci_95'][0]:.4f}, {row['wilson_ci_95'][1]:.4f}]",
                "same slice",
            ]
        )
        + " |"
        for row in token_references
    )

    _md_write(
        RESULTS / "sentence-level-eviction-8k-2026-05-13.md",
        f"""
# Sentence-Level Eviction @ 8k — 2026-05-13

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_8k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Pooling: `max`

## Ceiling Context

- Vanilla @ full context: `{vanilla['hit_rate']:.2f}` hit rate, `{vanilla['elapsed_seconds']:.2f}s`, `${vanilla['estimated_cost_usd']:.4f}`. This is a ceiling/context row, not a constrained-budget competitor.

## Comparison Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
{comparison_rows}

## Token References

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Reference |
|--------|--------|------------|----------|---------------|-----------|
{token_rows}

## Paired Tests

{chr(10).join(pairwise_lines)}

## Read

- At `1024`, sentence-level vorn extends the 4k coherence result to the 8k slice: `0.62` vs token-level `0.38`, with paired exact McNemar `p = {pairwise_tests[0]['p_value']:.6g}`.
- At both `1024` and `2048`, sentence-level rows are guardrail-independent on this slice: guarded and no-guardrails outcomes are identical (`p = 1` in both paired comparisons).
- At `2048`, token-level vorn catches up and slightly exceeds the sentence row on point estimate (`0.68` vs `0.62`), but the paired same-slice comparison is not decisive at `n=50` (`p = {pairwise_tests[2]['p_value']:.6g}`).
""",
    )


if __name__ == "__main__":
    main()
