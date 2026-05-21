#!/usr/bin/env python3
"""Build the 8k token-vs-sentence budget sweep artifact from Modal reports."""
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


def _guardrails_label(value: str) -> str:
    return "prefix + recent" if value == "prefix_plus_recent" else "none"


def _row_cells(row: dict[str, object], method_label: str) -> str:
    return (
        "| "
        + " | ".join(
            [
                method_label,
                str(row["budget"]),
                _guardrails_label(str(row["guardrails"])),
                f"{row['hit_rate']:.2f}",
                f"[{row['wilson_ci_95'][0]:.4f}, {row['wilson_ci_95'][1]:.4f}]",
                f"{row['elapsed_seconds']:.2f}s",
                f"${row['estimated_cost_usd']:.4f}",
                "0",
                f"{(1.0 - float(row['mean_retention_ratio'])) * 100:.2f}%",
            ]
        )
        + " |"
    )


def main() -> None:
    vanilla = _row_from_vanilla_report(_load_report("8k-vanilla-report.json"))

    token_rows = [
        _row_from_live_report(
            _load_report("8k-token-vorn-b512-report.json"),
            label="token_512_guarded",
            method="token_vorn",
            budget=512,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("8k-token-vorn-b1024-report.json"),
            label="token_1024_guarded",
            method="token_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("8k-token-vorn-b1536-report.json"),
            label="token_1536_guarded",
            method="token_vorn",
            budget=1536,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("8k-token-vorn-b2048-report.json"),
            label="token_2048_guarded",
            method="token_vorn",
            budget=2048,
            guardrails="prefix_plus_recent",
        ),
    ]
    sentence_rows = [
        _row_from_live_report(
            _load_report("8k-sentence-vorn-b512-report.json"),
            label="sentence_512_guarded",
            method="sentence_vorn",
            budget=512,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("8k-sentence-vorn-b512-noguards-report.json"),
            label="sentence_512_noguards",
            method="sentence_vorn",
            budget=512,
            guardrails="none",
        ),
        _row_from_live_report(
            _load_report("8k-sentence-vorn-b1024-report.json"),
            label="sentence_1024_guarded",
            method="sentence_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("8k-sentence-vorn-b1024-noguards-report.json"),
            label="sentence_1024_noguards",
            method="sentence_vorn",
            budget=1024,
            guardrails="none",
        ),
        _row_from_live_report(
            _load_report("8k-sentence-vorn-b1536-report.json"),
            label="sentence_1536_guarded",
            method="sentence_vorn",
            budget=1536,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("8k-sentence-vorn-b1536-noguards-report.json"),
            label="sentence_1536_noguards",
            method="sentence_vorn",
            budget=1536,
            guardrails="none",
        ),
        _row_from_live_report(
            _load_report("8k-sentence-vorn-b2048-report.json"),
            label="sentence_2048_guarded",
            method="sentence_vorn",
            budget=2048,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("8k-sentence-vorn-b2048-noguards-report.json"),
            label="sentence_2048_noguards",
            method="sentence_vorn",
            budget=2048,
            guardrails="none",
        ),
    ]

    token_by_budget = {int(row["budget"]): row for row in token_rows}
    sentence_guarded = {
        int(row["budget"]): row
        for row in sentence_rows
        if row["guardrails"] == "prefix_plus_recent"
    }
    sentence_noguards = {
        int(row["budget"]): row for row in sentence_rows if row["guardrails"] == "none"
    }

    pairwise_tests = []
    for budget in (512, 1024, 1536, 2048):
        pairwise_tests.append(_paired_test(sentence_guarded[budget], token_by_budget[budget]))
        pairwise_tests.append(_paired_test(sentence_noguards[budget], sentence_guarded[budget]))

    payload = {
        "schema_version": "result-envelope/v0.2",
        "ceiling": vanilla,
        "sentence_rows": sentence_rows,
        "token_rows": token_rows,
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
    _json_write(RESULTS / "sentence-level-eviction-8k-budget-sweep-2026-05-13.json", payload)

    sentence_lines = "\n".join(_row_cells(row, "Sentence vorn") for row in sentence_rows)
    token_lines = "\n".join(_row_cells(row, "Token vorn") for row in token_rows)
    pairwise_lines = "\n".join(
        f"- {pairwise['lhs']} vs {pairwise['rhs']}, exact McNemar on `{pairwise['table']}`: "
        f"`p = {pairwise['p_value']:.6g}`"
        for pairwise in pairwise_tests
    )

    _md_write(
        RESULTS / "sentence-level-eviction-8k-budget-sweep-2026-05-13.md",
        f"""
# Sentence-Level 8k Budget Sweep — 2026-05-13

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_8k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Pooling: `max`

## Ceiling Context

- Vanilla @ full context: `{vanilla['hit_rate']:.2f}` hit rate, `{vanilla['elapsed_seconds']:.2f}s`, `${vanilla['estimated_cost_usd']:.4f}`.

## Sentence Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
{sentence_lines}

## Token Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
{token_lines}

## Paired Tests

{pairwise_lines}

## Read

- The 8k crossover is regime-shaped, not monotonic. Sentence-level loses at `512` (`0.18` vs token `0.38`, paired `p = {pairwise_tests[0]['p_value']:.6g}`), wins clearly at `1024` and `1536` (`0.62` vs `0.38`, `0.74` vs `0.52`), then stops separating at `2048` (`0.62` vs `0.68`, paired `p = {pairwise_tests[6]['p_value']:.6g}`).
- Guardrail-independence is budget-dependent. At `512`, removing guardrails collapses sentence-level to `0.00` (`p = {pairwise_tests[1]['p_value']:.6g}`); at `1024`, `1536`, and `2048`, guarded and no-guardrails sentence rows are identical (`p = 1`).
- The strongest sentence-level zone on this slice is `1536`, where it reaches `0.74` at `81.39%` KV savings and still beats token-level on the same questions.
""",
    )


if __name__ == "__main__":
    main()
