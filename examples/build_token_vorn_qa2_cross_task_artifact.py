#!/usr/bin/env python3
"""Build the Phase 2E qa_2_4k token-vorn cross-task artifact."""
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
    family: str,
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
        "family": family,
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
    return "prefix + recent" if value == "prefix_plus_recent" else value


def _row_cells(row: dict[str, object]) -> str:
    budget = "full context" if row["budget"] is None else str(row["budget"])
    return (
        "| "
        + " | ".join(
            [
                str(row["family"]),
                str(row["method"]),
                budget,
                _guardrails_label(str(row["guardrails"])),
                f"{row['hit_rate']:.3f}",
                f"[{row['wilson_ci_95'][0]:.4f}, {row['wilson_ci_95'][1]:.4f}]",
                f"{row['elapsed_seconds']:.2f}s",
                f"${row['estimated_cost_usd']:.4f}",
                f"{(1.0 - float(row['mean_retention_ratio'])) * 100:.2f}%",
            ]
        )
        + " |"
    )


def _cost_per_correct(row: dict[str, object]) -> float:
    return float(row["hits"]) / float(row["estimated_cost_usd"])


def _family_read(
    *,
    family: str,
    sentence: dict[str, object],
    token: dict[str, object],
    vanilla: dict[str, object],
    pairwise: dict[tuple[str, str], dict[str, object]],
) -> str:
    sentence_vs_token = pairwise[(str(sentence["label"]), str(token["label"]))]
    sentence_vs_vanilla = pairwise[(str(sentence["label"]), str(vanilla["label"]))]
    token_vs_vanilla = pairwise[(str(token["label"]), str(vanilla["label"]))]
    return (
        f"- {family}: sentence-vorn `{float(sentence['hit_rate']):.3f}` "
        f"({int(sentence['hits'])}/{int(sentence['cases'])}) at "
        f"`{_cost_per_correct(sentence):.0f}` correct/$1 versus token-vorn "
        f"`{float(token['hit_rate']):.3f}` ({int(token['hits'])}/{int(token['cases'])}) "
        f"at `{_cost_per_correct(token):.0f}` correct/$1. "
        f"Sentence-vorn vs token-vorn McNemar `p = {sentence_vs_token['p_value']:.6g}`. "
        f"Sentence-vorn vs vanilla `p = {sentence_vs_vanilla['p_value']:.6g}`. "
        f"Token-vorn vs vanilla `p = {token_vs_vanilla['p_value']:.6g}`."
    )


def main() -> None:
    rows: list[dict[str, object]] = [
        _row_from_report(
            "gemma-4-qa2-vanilla-n200.json",
            family="Gemma 4 E4B-it",
            label="gemma4_vanilla",
            method="vanilla",
            budget=None,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-qa2-token-1024-guarded-n200.json",
            family="Gemma 4 E4B-it",
            label="gemma4_token_1024",
            method="vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-qa2-sentence-1024-guarded-n200.json",
            family="Gemma 4 E4B-it",
            label="gemma4_sentence_1024",
            method="sentence_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "llama31-qa2-vanilla-n200.json",
            family="Llama 3.1 8B",
            label="llama31_vanilla",
            method="vanilla",
            budget=None,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "llama31-qa2-token-1024-guarded-n200.json",
            family="Llama 3.1 8B",
            label="llama31_token_1024",
            method="vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "llama31-qa2-sentence-1024-guarded-n200.json",
            family="Llama 3.1 8B",
            label="llama31_sentence_1024",
            method="sentence_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
    ]

    by_label = {str(row["label"]): row for row in rows}
    pairwise_tests = [
        _paired_test(by_label["gemma4_sentence_1024"], by_label["gemma4_token_1024"]),
        _paired_test(by_label["gemma4_sentence_1024"], by_label["gemma4_vanilla"]),
        _paired_test(by_label["gemma4_token_1024"], by_label["gemma4_vanilla"]),
        _paired_test(by_label["llama31_sentence_1024"], by_label["llama31_token_1024"]),
        _paired_test(by_label["llama31_sentence_1024"], by_label["llama31_vanilla"]),
        _paired_test(by_label["llama31_token_1024"], by_label["llama31_vanilla"]),
    ]
    pairwise_lookup = {
        (str(item["lhs"]), str(item["rhs"])): item for item in pairwise_tests
    }

    payload = {
        "schema_version": "result-envelope/v0.2",
        "provenance": {
            "raw_predictions_embedded": True,
            "raw_report_home": ".benchmarks/cross-model/",
            "raw_report_files": [row["source_report"] for row in rows],
            "durable_prediction_home": "rows[].observations[]",
            "notes": (
                "Phase 2E qa_2_4k token-vorn addendum. This artifact fills the missing "
                "within-vorn token-vs-sentence cross-task comparison for Gemma 4 and "
                "Llama 3.1 at b=1024, n=200."
            ),
        },
        "rows": rows,
        "pairwise_tests": pairwise_tests,
        "run_conditions": {
            "profile": "author",
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "qa_2_4k",
            "split": "validation[:200]",
            "random_seed": 17,
            "canonical_layer": 16,
            "recent_token_window": 16,
            "budget_tokens": 1024,
            "sentence_pooling": "max",
            "sentence_top_k": 3,
            "note": (
                "Cross-task within-vorn comparison surface: token-vorn and sentence-vorn "
                "rows on Gemma 4 E4B-it and Llama 3.1 8B."
            ),
        },
    }
    _json_write(RESULTS / "token-vorn-qa2-cross-task-2026-05-20.json", payload)

    row_lines = "\n".join(_row_cells(row) for row in rows)
    pairwise_lines = "\n".join(
        f"- {pairwise['lhs']} vs {pairwise['rhs']}, exact McNemar on `{pairwise['table']}`: "
        f"`p = {pairwise['p_value']:.6g}`"
        for pairwise in pairwise_tests
    )
    read_lines = "\n".join(
        [
            _family_read(
                family="Gemma 4 E4B-it",
                sentence=by_label["gemma4_sentence_1024"],
                token=by_label["gemma4_token_1024"],
                vanilla=by_label["gemma4_vanilla"],
                pairwise=pairwise_lookup,
            ),
            _family_read(
                family="Llama 3.1 8B",
                sentence=by_label["llama31_sentence_1024"],
                token=by_label["llama31_token_1024"],
                vanilla=by_label["llama31_vanilla"],
                pairwise=pairwise_lookup,
            ),
        ]
    )

    _md_write(
        RESULTS / "token-vorn-qa2-cross-task-2026-05-20.md",
        f"""
# Phase 2E — Token-vorn `qa_2_4k` Cross-Task Addendum (2026-05-20)

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `qa_2_4k`
- Slice: `validation[:200]`
- Budget: `1024`
- Pooling: `max`
- Sentence top-k: `3`

## Rows

| Family | Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | KV savings |
|--------|--------|--------|------------|----------|---------------|------------|----------------|-----------|
{row_lines}

## Paired Tests

{pairwise_lines}

## Read

{read_lines}
        """,
    )


if __name__ == "__main__":
    main()
