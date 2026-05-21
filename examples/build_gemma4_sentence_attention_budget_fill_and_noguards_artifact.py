#!/usr/bin/env python3
"""Build the Gemma 4 budget-fill + no-guards sentence-attention artifact."""
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
SOURCE_NO_GUARDS = RESULTS / "cross-family-no-guards-mirror-2026-05-15.json"


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


def _row_from_result_artifact(
    row: dict[str, object],
    *,
    source_report: str,
) -> dict[str, object]:
    cloned = dict(row)
    cloned["source_report"] = source_report
    cloned.setdefault("ceiling_status", "historical_stale_reference")
    return cloned


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


def _load_historical_gemma_rows() -> dict[str, dict[str, object]]:
    rows = {
        "sentence_512_guarded_historical": _row_from_report(
            _load_historical("gemma-4-4k-sentence-b512.json"),
            label="sentence_512_guarded_historical",
            method="sentence_vorn",
            budget=512,
            guardrails="prefix_plus_recent",
            ceiling_status="historical_stale_reference",
            source_report=".benchmarks/cross-model/gemma-4-4k-sentence-b512.json",
        ),
        "sentence_1536_guarded_historical": _row_from_report(
            _load_historical("gemma-4-4k-sentence-b1536.json"),
            label="sentence_1536_guarded_historical",
            method="sentence_vorn",
            budget=1536,
            guardrails="prefix_plus_recent",
            ceiling_status="historical_stale_reference",
            source_report=".benchmarks/cross-model/gemma-4-4k-sentence-b1536.json",
        ),
        "tova_512_guarded_historical": _row_from_report(
            _load_historical("gemma-4-4k-tova-b512.json"),
            label="tova_512_guarded_historical",
            method="tova_style",
            budget=512,
            guardrails="prefix_plus_recent",
            ceiling_status="historical_stale_reference",
            source_report=".benchmarks/cross-model/gemma-4-4k-tova-b512.json",
        ),
        "tova_1536_guarded_historical": _row_from_report(
            _load_historical("gemma-4-4k-tova-b1536.json"),
            label="tova_1536_guarded_historical",
            method="tova_style",
            budget=1536,
            guardrails="prefix_plus_recent",
            ceiling_status="historical_stale_reference",
            source_report=".benchmarks/cross-model/gemma-4-4k-tova-b1536.json",
        ),
        "h2o_1024_noguards_historical": None,
        "sentence_1024_noguards_historical": None,
        "tova_1024_noguards_historical": None,
    }
    no_guards = _load_json(SOURCE_NO_GUARDS)
    payload_rows = no_guards["rows"]
    assert isinstance(payload_rows, list)
    for raw in payload_rows:
        row = dict(raw)
        if row.get("label") == "gemma_sentence_noguards":
            rows["sentence_1024_noguards_historical"] = _row_from_result_artifact(
                row,
                source_report=str(SOURCE_NO_GUARDS.relative_to(ROOT)),
            )
        if row.get("label") == "gemma_tova_noguards":
            rows["tova_1024_noguards_historical"] = _row_from_result_artifact(
                row,
                source_report=str(SOURCE_NO_GUARDS.relative_to(ROOT)),
            )
        if row.get("label") == "gemma_h2o_noguards":
            rows["h2o_1024_noguards_historical"] = _row_from_result_artifact(
                row,
                source_report=str(SOURCE_NO_GUARDS.relative_to(ROOT)),
            )
    return rows


def main() -> None:
    historical = _load_historical_gemma_rows()

    fresh_rows = {
        "sentence_tova_512_guarded": _row_from_report(
            _load_report("gemma-4k-sentence-tova-b512-n50-report.json"),
            label="sentence_tova_512_guarded",
            method="sentence_tova_style",
            budget=512,
            guardrails="prefix_plus_recent",
            ceiling_status="current_n50_override",
            source_report=".benchmarks/gemma-4k-sentence-tova-b512-n50-report.json",
        ),
        "sentence_h2o_512_guarded": _row_from_report(
            _load_report("gemma-4k-sentence-h2o-b512-n50-report.json"),
            label="sentence_h2o_512_guarded",
            method="sentence_h2o_style",
            budget=512,
            guardrails="prefix_plus_recent",
            ceiling_status="current_n50_override",
            source_report=".benchmarks/gemma-4k-sentence-h2o-b512-n50-report.json",
        ),
        "sentence_tova_1536_guarded": _row_from_report(
            _load_report("gemma-4k-sentence-tova-b1536-n50-report.json"),
            label="sentence_tova_1536_guarded",
            method="sentence_tova_style",
            budget=1536,
            guardrails="prefix_plus_recent",
            ceiling_status="current_n50_override",
            source_report=".benchmarks/gemma-4k-sentence-tova-b1536-n50-report.json",
        ),
        "sentence_h2o_1536_guarded": _row_from_report(
            _load_report("gemma-4k-sentence-h2o-b1536-n50-report.json"),
            label="sentence_h2o_1536_guarded",
            method="sentence_h2o_style",
            budget=1536,
            guardrails="prefix_plus_recent",
            ceiling_status="current_n50_override",
            source_report=".benchmarks/gemma-4k-sentence-h2o-b1536-n50-report.json",
        ),
        "sentence_1024_noguards": _row_from_report(
            _load_report("gemma-4k-sentence-vorn-b1024-noguards-n50-current-report.json"),
            label="sentence_1024_noguards",
            method="sentence_vorn",
            budget=1024,
            guardrails="none",
            ceiling_status="current_n50_override",
            source_report=".benchmarks/gemma-4k-sentence-vorn-b1024-noguards-n50-current-report.json",
        ),
        "sentence_tova_1024_noguards": _row_from_report(
            _load_report("gemma-4k-sentence-tova-b1024-noguards-n50-report.json"),
            label="sentence_tova_1024_noguards",
            method="sentence_tova_style",
            budget=1024,
            guardrails="none",
            ceiling_status="current_n50_override",
            source_report=".benchmarks/gemma-4k-sentence-tova-b1024-noguards-n50-report.json",
        ),
    }

    pairwise_tests = [
        _paired_test(
            fresh_rows["sentence_tova_512_guarded"],
            historical["tova_512_guarded_historical"],
        ),
        _paired_test(
            fresh_rows["sentence_tova_512_guarded"],
            historical["sentence_512_guarded_historical"],
        ),
        _paired_test(
            fresh_rows["sentence_h2o_512_guarded"],
            historical["sentence_512_guarded_historical"],
        ),
        _paired_test(
            fresh_rows["sentence_tova_1536_guarded"],
            historical["tova_1536_guarded_historical"],
        ),
        _paired_test(
            fresh_rows["sentence_tova_1536_guarded"],
            historical["sentence_1536_guarded_historical"],
        ),
        _paired_test(
            fresh_rows["sentence_h2o_1536_guarded"],
            historical["sentence_1536_guarded_historical"],
        ),
        _paired_test(
            fresh_rows["sentence_1024_noguards"],
            historical["sentence_1024_noguards_historical"],
        ),
        _paired_test(
            fresh_rows["sentence_tova_1024_noguards"],
            historical["tova_1024_noguards_historical"],
        ),
        _paired_test(
            fresh_rows["sentence_tova_1024_noguards"],
            fresh_rows["sentence_1024_noguards"],
        ),
    ]

    payload = {
        "schema_version": "result-envelope/v0.2",
        "historical_reference_rows": historical,
        "fresh_rows": fresh_rows,
        "coverage_notes": [
            "This artifact follows the narrowed Tier 2 2026-05-20 dispatch: guarded sentence-attention fill at 512 and 1536, plus fresh sentence-vorn and sentence-TOVA-style no-guards at 1024.",
            "The Gemma sentence-vorn no-guards historical row already existed. It is rerun here to provide current-run confirmation against the same surface.",
        ],
        "pairwise_tests": pairwise_tests,
        "budget_reads": [
            {
                "budget": 512,
                "historical_sentence_vorn": float(historical["sentence_512_guarded_historical"]["hit_rate"]),
                "fresh_sentence_tova_style": float(fresh_rows["sentence_tova_512_guarded"]["hit_rate"]),
                "fresh_sentence_h2o_style": float(fresh_rows["sentence_h2o_512_guarded"]["hit_rate"]),
            },
            {
                "budget": 1536,
                "historical_sentence_vorn": float(historical["sentence_1536_guarded_historical"]["hit_rate"]),
                "fresh_sentence_tova_style": float(fresh_rows["sentence_tova_1536_guarded"]["hit_rate"]),
                "fresh_sentence_h2o_style": float(fresh_rows["sentence_h2o_1536_guarded"]["hit_rate"]),
            },
            {
                "budget": 1024,
                "historical_sentence_noguards": float(historical["sentence_1024_noguards_historical"]["hit_rate"]),
                "fresh_sentence_noguards": float(fresh_rows["sentence_1024_noguards"]["hit_rate"]),
                "historical_tova_noguards": float(historical["tova_1024_noguards_historical"]["hit_rate"]),
                "fresh_sentence_tova_noguards": float(fresh_rows["sentence_tova_1024_noguards"]["hit_rate"]),
            },
        ],
        "overall_read": "gemma_budget_fill_and_noguards_extension",
        "run_conditions": {
            "profile": "author",
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "niah_multikey_1_4k",
            "split": "validation[:50]",
            "model": "google/gemma-4-E4B-it",
            "random_seed": 17,
            "canonical_layer": 16,
            "recent_token_window": 16,
            "sentence_pooling": "max",
            "sentence_top_k": 3,
        },
    }
    _json_write(
        RESULTS / "sentence-attention-2x2-gemma4-budget-fill-and-no-guards-2026-05-20.json",
        payload,
    )

    guarded_rows = "\n".join(
        [
            _row_cells(historical["sentence_512_guarded_historical"], "Sentence vorn @ 512"),
            _row_cells(historical["tova_512_guarded_historical"], "TOVA-style @ 512"),
            _row_cells(fresh_rows["sentence_tova_512_guarded"], "Sentence TOVA-style @ 512"),
            _row_cells(fresh_rows["sentence_h2o_512_guarded"], "Sentence H2O-style @ 512"),
            _row_cells(historical["sentence_1536_guarded_historical"], "Sentence vorn @ 1536"),
            _row_cells(historical["tova_1536_guarded_historical"], "TOVA-style @ 1536"),
            _row_cells(fresh_rows["sentence_tova_1536_guarded"], "Sentence TOVA-style @ 1536"),
            _row_cells(fresh_rows["sentence_h2o_1536_guarded"], "Sentence H2O-style @ 1536"),
        ]
    )
    noguards_rows = "\n".join(
        [
            _row_cells(historical["sentence_1024_noguards_historical"], "Sentence vorn @ 1024 no-guards"),
            _row_cells(historical["tova_1024_noguards_historical"], "TOVA-style @ 1024 no-guards"),
            _row_cells(fresh_rows["sentence_1024_noguards"], "Sentence vorn @ 1024 no-guards fresh"),
            _row_cells(fresh_rows["sentence_tova_1024_noguards"], "Sentence TOVA-style @ 1024 no-guards"),
        ]
    )
    pairwise_lines = "\n".join(
        f"- {pairwise['lhs']} vs {pairwise['rhs']}, exact McNemar on `{pairwise['table']}`: "
        f"`p = {pairwise['p_value']:.6g}`"
        for pairwise in pairwise_tests
    )
    _md_write(
        RESULTS / "sentence-attention-2x2-gemma4-budget-fill-and-no-guards-2026-05-20.md",
        f"""
# Gemma 4 Sentence-Attention Budget Fill + No-Guards — 2026-05-20

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `google/gemma-4-E4B-it`

## Guarded budget fill

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost | Mean evicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{guarded_rows}

## No-guards extension

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost | Mean evicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{noguards_rows}

## Pairwise Tests

{pairwise_lines}

## Read

- This artifact fills the missing Gemma 4 sentence-attention budgets at 512 and 1536 and reruns the 1024 no-guards sentence surface for current-run confirmation.
- The guarded 512 and 1536 rows should be interpreted against historical token/sentence/TOVA controls, not as a fresh six-method rerun.
""",
    )


if __name__ == "__main__":
    main()
