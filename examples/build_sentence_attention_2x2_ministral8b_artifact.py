#!/usr/bin/env python3
"""Build the Ministral 8B sentence-attention extension artifact."""
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
HISTORICAL = ROOT / ".benchmarks" / "cross-model"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _load_report(name: str) -> dict[str, object]:
    return _load_json(BENCHMARKS / name)


def _load_historical(name: str) -> dict[str, object]:
    return _load_json(HISTORICAL / name)


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


def _summary_contract(report: dict[str, object], default: str = "") -> str:
    result = report["result"]
    assert isinstance(result, dict)
    metadata = result.get("metadata", {})
    assert isinstance(metadata, dict)
    return str(metadata.get("summary_contract", default))


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
    report: dict[str, object],
    *,
    label: str,
    method: str,
    budget: int | None,
    guardrails: str,
    ceiling_status: str,
    source_report: str,
) -> dict[str, object]:
    observations = _observations(report)
    metric_name, hit_rate = _primary_metric(report)
    hits = sum(1 for observation in observations if observation.correct)
    row: dict[str, object] = {
        "label": label,
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
        "summary_contract": _summary_contract(report),
        "run_id": _run_id(report),
        "ceiling_status": ceiling_status,
        "source_report": source_report,
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
    for key in (
        "sentence_pooling",
        "sentence_top_k",
        "retention_policy",
        "model_id",
        "summary_contract",
    ):
        if key in metadata:
            row[key] = metadata[key]
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
                str(row["ceiling_status"]),
                f"{row['elapsed_seconds']:.2f}s",
                f"${row['estimated_cost_usd']:.4f}",
                f"{(1.0 - float(row['mean_retention_ratio'])) * 100:.2f}%",
            ]
        )
        + " |"
    )


def _classify_gate(
    sentence_tova: dict[str, object],
    historical_sentence_vorn: dict[str, object],
    historical_tova: dict[str, object],
) -> tuple[str, str]:
    sentence_rate = float(sentence_tova["hit_rate"])
    sentence_vorn_rate = float(historical_sentence_vorn["hit_rate"])
    tova_rate = float(historical_tova["hit_rate"])
    if abs(sentence_rate - sentence_vorn_rate) <= 0.10:
        return (
            "mistral_like_full_rescue",
            "Sentence-TOVA-style reaches the sentence-vorn regime at the gate budget.",
        )
    if sentence_rate <= tova_rate + 0.10:
        return (
            "gemma_like_channel_persistence",
            "Sentence-TOVA-style stays close to token TOVA-style, so the attention channel does not rescue under sentence grouping.",
        )
    return (
        "intermediate_recovery",
        "Sentence-TOVA-style improves materially over token TOVA-style but does not fully reach the sentence-vorn regime.",
    )


def main() -> None:
    historical_vanilla = _row_from_report(
        _load_historical("ministral-8b-4k-vanilla.json"),
        label="vanilla_full_context_historical",
        method="vanilla",
        budget=None,
        guardrails="prefix_plus_recent",
        ceiling_status="historical_stale_reference",
        source_report=".benchmarks/cross-model/ministral-8b-4k-vanilla.json",
    )
    historical_sentence = _row_from_report(
        _load_historical("ministral-8b-4k-sentence-1024-guarded.json"),
        label="sentence_1024_guarded_historical",
        method="sentence_vorn",
        budget=1024,
        guardrails="prefix_plus_recent",
        ceiling_status="historical_stale_reference",
        source_report=".benchmarks/cross-model/ministral-8b-4k-sentence-1024-guarded.json",
    )
    historical_tova = _row_from_report(
        _load_historical("ministral-8b-4k-tova-1024.json"),
        label="tova_1024_guarded_historical",
        method="tova_style",
        budget=1024,
        guardrails="prefix_plus_recent",
        ceiling_status="historical_stale_reference",
        source_report=".benchmarks/cross-model/ministral-8b-4k-tova-1024.json",
    )

    sentence_tova_rows = []
    for budget in (256, 512, 1024):
        filename = f"ministral-8b-4k-sentence-tova-b{budget}-n50-report.json"
        sentence_tova_rows.append(
            _row_from_report(
                _load_report(filename),
                label=f"sentence_tova_{budget}_guarded",
                method="sentence_tova_style",
                budget=budget,
                guardrails="prefix_plus_recent",
                ceiling_status="current_n50_override",
                source_report=f".benchmarks/{filename}",
            )
        )

    gate_mode, gate_interpretation = _classify_gate(
        sentence_tova_rows[-1],
        historical_sentence,
        historical_tova,
    )

    pairwise_tests = [
        _paired_test(sentence_tova_rows[-1], historical_tova),
        _paired_test(sentence_tova_rows[-1], historical_sentence),
        _paired_test(historical_sentence, historical_tova),
    ]

    budget_reads = [
        {
            "budget": int(row["budget"]),
            "sentence_tova_style": float(row["hit_rate"]),
            "historical_gate_context": (
                {
                    "sentence_vorn_1024": float(historical_sentence["hit_rate"]),
                    "tova_style_1024": float(historical_tova["hit_rate"]),
                }
                if int(row["budget"]) == 1024
                else None
            ),
        }
        for row in sentence_tova_rows
    ]

    payload = {
        "schema_version": "result-envelope/v0.2",
        "ceiling": historical_vanilla,
        "historical_reference_rows": {
            "sentence_vorn": [historical_sentence],
            "tova_style": [historical_tova],
        },
        "sentence_attention_rows": sentence_tova_rows,
        "coverage_notes": [
            "This extension follows the narrowed 2026-05-20 Phase 2B dispatch: it adds sentence-TOVA-style rows across 256, 512, and 1024 for Ministral 8B.",
            "Historical fast-read controls at 1024 are reused as the paired gate comparison surface. No fresh token/H2O reruns were added in this pass.",
        ],
        "pairwise_tests": pairwise_tests,
        "budget_reads": budget_reads,
        "overall_read": gate_mode,
        "gate_interpretation": gate_interpretation,
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
        },
    }
    _json_write(RESULTS / "sentence-attention-2x2-ministral8b-2026-05-20.json", payload)

    row_lines = "\n".join(
        [
            _row_cells(historical_vanilla, "Vanilla"),
            _row_cells(historical_sentence, "Sentence vorn"),
            _row_cells(historical_tova, "TOVA-style"),
            *[
                _row_cells(row, f"Sentence TOVA-style @ {row['budget']}")
                for row in sentence_tova_rows
            ],
        ]
    )
    pairwise_lines = "\n".join(
        f"- {pairwise['lhs']} vs {pairwise['rhs']}, exact McNemar on `{pairwise['table']}`: "
        f"`p = {pairwise['p_value']:.6g}`"
        for pairwise in pairwise_tests
    )
    _md_write(
        RESULTS / "sentence-attention-2x2-ministral8b-2026-05-20.md",
        f"""
# Ministral 8B Sentence-Attention Extension — 2026-05-20

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Ministral-8B-Instruct-2410`

## Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost | Mean evicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{row_lines}

## Pairwise Tests

{pairwise_lines}

## Read

- Gate verdict: `{gate_mode}`.
- {gate_interpretation}
- The 256 and 512 rows extend the same surface below the original 1024 fast-read gate without claiming same-runner six-method controls.
""",
    )


if __name__ == "__main__":
    main()
