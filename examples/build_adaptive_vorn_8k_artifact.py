#!/usr/bin/env python3
"""Build the 8k online adaptive-vorn comparison artifact."""
# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vorn_mat.paired_stats import build_paired_correctness_table, exact_mcnemar
from vorn_mat.results import CaseObservation

RESULTS = ROOT / "results"
BENCHMARKS = ROOT / ".benchmarks"


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


def _row_from_raw_report(path: Path, *, label: str) -> dict[str, Any]:
    report = _load_json(path)
    result = report["result"]
    assert isinstance(result, dict)
    observations_payload = result.get("observations", [])
    assert isinstance(observations_payload, list)
    observations = tuple(CaseObservation(**item) for item in observations_payload)
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    metric_name, hit_rate = next(iter(metrics.items()))
    metadata = result.get("metadata", {})
    assert isinstance(metadata, dict)
    hits = sum(1 for observation in observations if observation.correct)
    cases = len(observations)
    return {
        "label": label,
        "method": "adaptive_vorn",
        "metric": str(metric_name),
        "budget": int(metadata["cache_budget_tokens"]),
        "guardrails": (
            "prefix_plus_recent"
            if metadata.get("preserve_recent_window", "true") == "true"
            or metadata.get("always_keep_prefix_tokens", "1") != "0"
            else "none"
        ),
        "hits": hits,
        "cases": cases,
        "hit_rate": float(hit_rate),
        "wilson_ci_95": list(_wilson_ci(hits, cases)),
        "elapsed_seconds": float(report["elapsed_seconds"]),
        "estimated_cost_usd": float(report["estimated_cost_usd"]),
        "preprocessing_elapsed_seconds": float(
            result.get("preprocessing_elapsed_seconds", 0.0)
        ),
        "preprocessing_cost_usd": float(result.get("preprocessing_cost_usd", 0.0)),
        "mean_retention_ratio": float(metadata.get("mean_retention_ratio", 1.0)),
        "summary_contract": str(metadata.get("summary_contract", "")),
        "summary_fingerprint": str(metadata.get("summary_fingerprint", "")),
        "run_id": str(result["run_id"]),
        "adaptive_token_steps": int(metadata.get("adaptive_token_steps", 0)),
        "adaptive_sentence_steps": int(metadata.get("adaptive_sentence_steps", 0)),
        "adaptive_selector_contract": str(
            metadata.get("adaptive_selector_contract", "")
        ),
        "observations": [
            {
                "fixture_id": observation.fixture_id,
                "correct": observation.correct,
                "prediction": observation.prediction,
            }
            for observation in observations
        ],
        "source_report": str(path.relative_to(ROOT)),
    }


def _paired_test(lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, Any]:
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


def _guardrails_label(value: str) -> str:
    return "prefix + recent" if value == "prefix_plus_recent" else "none"


def main() -> None:
    eight_k = _load_json(RESULTS / "sentence-level-eviction-8k-budget-sweep-2026-05-13.json")
    token_rows = {int(row["budget"]): row for row in eight_k["token_rows"]}
    sentence_rows = {
        int(row["budget"]): row
        for row in eight_k["sentence_rows"]
        if row["guardrails"] == "prefix_plus_recent"
    }

    adaptive_rows = [
        _row_from_raw_report(BENCHMARKS / "adaptive-8k-b512.json", label="adaptive_512_guarded"),
        _row_from_raw_report(BENCHMARKS / "adaptive-8k-b1024.json", label="adaptive_1024_guarded"),
        _row_from_raw_report(BENCHMARKS / "adaptive-8k-b1536.json", label="adaptive_1536_guarded"),
        _row_from_raw_report(BENCHMARKS / "adaptive-8k-b2048.json", label="adaptive_2048_guarded"),
    ]

    pairwise_tests: list[dict[str, Any]] = []
    for row in adaptive_rows:
        budget = int(row["budget"])
        pairwise_tests.append(_paired_test(row, token_rows[budget]))
        pairwise_tests.append(_paired_test(row, sentence_rows[budget]))

    payload = {
        "schema_version": "result-envelope/v0.2",
        "artifact": "adaptive-vorn-online-8k",
        "date": "2026-05-14",
        "run_conditions": eight_k["run_conditions"],
        "ceiling": eight_k["ceiling"],
        "adaptive_rows": adaptive_rows,
        "token_reference_rows": list(token_rows.values()),
        "sentence_reference_rows": list(sentence_rows.values()),
        "pairwise_tests": pairwise_tests,
        "read": (
            "This artifact asks whether the online adaptive selector reproduces the "
            "previously observed 8k regime crossover without post-hoc hindsight. "
            "Adaptive rows are compared against the already-shipped token and guarded "
            "sentence rows on the same slice and budgets."
        ),
    }

    md_lines = [
        "# Adaptive Vorn Online 8k Comparison — 2026-05-14",
        "",
        "Run conditions:",
        "- Profile: `author`",
        "- Dataset: `rbiswasfc/ruler`",
        "- Config: `niah_multikey_1_8k`",
        "- Slice: `validation[:50]`",
        "- Model: `mistralai/Mistral-7B-Instruct-v0.3`",
        "- Adaptive selector: `choose_token_or_sentence_by_peak_zscore_over_current_alignment_scores`",
        "",
        "| Budget | Adaptive hit | Token ref | Sentence ref | Adaptive token steps | Adaptive sentence steps | Wall-clock | Cost |",
        "|--------|--------------|-----------|--------------|----------------------|-------------------------|------------|------|",
    ]
    for row in adaptive_rows:
        budget = int(row["budget"])
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(budget),
                    f"{row['hit_rate']:.2f}",
                    f"{token_rows[budget]['hit_rate']:.2f}",
                    f"{sentence_rows[budget]['hit_rate']:.2f}",
                    str(row["adaptive_token_steps"]),
                    str(row["adaptive_sentence_steps"]),
                    f"{row['elapsed_seconds']:.2f}s",
                    f"${row['estimated_cost_usd']:.4f}",
                ]
            )
            + " |"
        )

    md_lines.extend(["", "## Paired Tests", ""])
    for test in pairwise_tests:
        md_lines.append(
            f"- {test['lhs']} vs {test['rhs']}, exact McNemar on `{test['table']}`: `p = {test['p_value']:.6g}`"
        )

    _json_write(RESULTS / "adaptive-vorn-online-8k-2026-05-14.json", payload)
    _md_write(RESULTS / "adaptive-vorn-online-8k-2026-05-14.md", "\n".join(md_lines))


if __name__ == "__main__":
    main()
