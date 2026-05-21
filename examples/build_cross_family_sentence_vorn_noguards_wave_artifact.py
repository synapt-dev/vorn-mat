#!/usr/bin/env python3
"""Build the 2026-05-20 cross-family sentence-vorn no-guards wave artifact."""
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

FAMILY_SPECS = [
    {
        "family_id": "mistral",
        "display_name": "Mistral 7B v0.3",
        "budget": 512,
        "vanilla_path": BENCHMARKS / "mistral-4k-vanilla-n50-current-report.json",
        "guarded_path": BENCHMARKS / "mistral-4k-sentence-vorn-b512-n50-current-report.json",
        "guarded_source": ".benchmarks/mistral-4k-sentence-vorn-b512-n50-current-report.json",
        "guarded_status": "current_n50_override",
        "noguards_path": BENCHMARKS / "mistral-4k-sentence-vorn-b512-noguards-n50-current-report.json",
        "noguards_source": ".benchmarks/mistral-4k-sentence-vorn-b512-noguards-n50-current-report.json",
    },
    {
        "family_id": "ministral8b",
        "display_name": "Ministral 8B",
        "budget": 1024,
        "vanilla_path": HISTORICAL / "ministral-8b-4k-vanilla.json",
        "guarded_path": HISTORICAL / "ministral-8b-4k-sentence-1024-guarded.json",
        "guarded_source": ".benchmarks/cross-model/ministral-8b-4k-sentence-1024-guarded.json",
        "guarded_status": "historical_stale_reference",
        "noguards_path": BENCHMARKS / "ministral-8b-4k-sentence-vorn-b1024-noguards-n50-current-report.json",
        "noguards_source": ".benchmarks/ministral-8b-4k-sentence-vorn-b1024-noguards-n50-current-report.json",
    },
    {
        "family_id": "gemma2_9b",
        "display_name": "Gemma 2 9B",
        "budget": 1024,
        "vanilla_path": HISTORICAL / "gemma2-9b-4k-vanilla.json",
        "guarded_path": HISTORICAL / "gemma2-9b-4k-sentence-1024-guarded.json",
        "guarded_source": ".benchmarks/cross-model/gemma2-9b-4k-sentence-1024-guarded.json",
        "guarded_status": "historical_stale_reference",
        "noguards_path": BENCHMARKS / "gemma2-9b-4k-sentence-vorn-b1024-noguards-n50-current-report.json",
        "noguards_source": ".benchmarks/gemma2-9b-4k-sentence-vorn-b1024-noguards-n50-current-report.json",
    },
    {
        "family_id": "qwen3_8b",
        "display_name": "Qwen 3 8B",
        "budget": 1024,
        "vanilla_path": HISTORICAL / "qwen3-4k-vanilla.json",
        "guarded_path": HISTORICAL / "qwen3-4k-sentence-1024-guarded.json",
        "guarded_source": ".benchmarks/cross-model/qwen3-4k-sentence-1024-guarded.json",
        "guarded_status": "historical_stale_reference",
        "noguards_path": BENCHMARKS / "qwen3-8b-4k-sentence-vorn-b1024-noguards-n50-current-report.json",
        "noguards_source": ".benchmarks/qwen3-8b-4k-sentence-vorn-b1024-noguards-n50-current-report.json",
    },
    {
        "family_id": "qwen25_7b",
        "display_name": "Qwen 2.5 7B",
        "budget": 1024,
        "vanilla_path": BENCHMARKS / "qwen25-7b-4k-vanilla-n50-current-report.json",
        "guarded_path": BENCHMARKS / "qwen25-7b-4k-sentence-vorn-b1024-n50-current-report.json",
        "guarded_source": ".benchmarks/qwen25-7b-4k-sentence-vorn-b1024-n50-current-report.json",
        "guarded_status": "current_n50_override",
        "noguards_path": BENCHMARKS / "qwen25-7b-4k-sentence-vorn-b1024-noguards-n50-current-report.json",
        "noguards_source": ".benchmarks/qwen25-7b-4k-sentence-vorn-b1024-noguards-n50-current-report.json",
    },
]


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


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
    family: str,
    label: str,
    method: str,
    budget: int,
    guardrails: str,
    ceiling_status: str,
    source_report: str,
) -> dict[str, object]:
    observations = _observations(report)
    metric_name, hit_rate = _primary_metric(report)
    hits = sum(1 for observation in observations if observation.correct)
    row: dict[str, object] = {
        "family": family,
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


def _classify_family(vanilla: dict[str, object], guarded: dict[str, object], noguards: dict[str, object]) -> tuple[str, str]:
    vanilla_rate = float(vanilla["hit_rate"])
    guarded_rate = float(guarded["hit_rate"])
    noguards_rate = float(noguards["hit_rate"])
    if vanilla_rate <= 0.0:
        return (
            "observational_floor",
            "Vanilla is non-discriminative on this slice, so the no-guards row is observational rather than a valid guarded-vs-no-guards contrast.",
        )
    if abs(noguards_rate - guarded_rate) <= 0.10:
        return (
            "guardrail_robust",
            "Sentence-vorn stays in the same regime with or without prefix/recent guardrails.",
        )
    if noguards_rate < guarded_rate:
        return (
            "guardrail_sensitive",
            "Removing prefix/recent guardrails degrades sentence-vorn on this family-primary surface.",
        )
    return (
        "guardrail_improved",
        "Sentence-vorn improves when the guardrails are removed on this surface.",
    )


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
                str(row["ceiling_status"]),
                f"{row['elapsed_seconds']:.2f}s",
                f"${row['estimated_cost_usd']:.4f}",
            ]
        )
        + " |"
    )


def _family_payload(spec: dict[str, object]) -> dict[str, object]:
    vanilla = _row_from_report(
        _load_json(spec["vanilla_path"]),
        family=str(spec["display_name"]),
        label="vanilla_full_context_reference",
        method="vanilla",
        budget=int(spec["budget"]),
        guardrails="prefix_plus_recent",
        ceiling_status="current_n50_override" if "current-report" in str(spec["vanilla_path"]) else "historical_stale_reference",
        source_report=str(spec["vanilla_path"]).split("/")[-1],
    )
    guarded = _row_from_report(
        _load_json(spec["guarded_path"]),
        family=str(spec["display_name"]),
        label=f"sentence_{spec['budget']}_guarded",
        method="sentence_vorn",
        budget=int(spec["budget"]),
        guardrails="prefix_plus_recent",
        ceiling_status=str(spec["guarded_status"]),
        source_report=str(spec["guarded_source"]),
    )
    noguards = _row_from_report(
        _load_json(spec["noguards_path"]),
        family=str(spec["display_name"]),
        label=f"sentence_{spec['budget']}_noguards",
        method="sentence_vorn",
        budget=int(spec["budget"]),
        guardrails="none",
        ceiling_status="current_n50_override",
        source_report=str(spec["noguards_source"]),
    )
    overall_read, interpretation = _classify_family(vanilla, guarded, noguards)
    return {
        "family_id": spec["family_id"],
        "display_name": spec["display_name"],
        "budget": spec["budget"],
        "rows": [vanilla, guarded, noguards],
        "pairwise_tests": [_paired_test(noguards, guarded)],
        "delta_vs_guarded": float(noguards["hit_rate"]) - float(guarded["hit_rate"]),
        "overall_read": overall_read,
        "interpretation": interpretation,
    }


def main() -> None:
    families = [_family_payload(spec) for spec in FAMILY_SPECS if Path(spec["noguards_path"]).exists()]
    total_cost = sum(
        float(row["estimated_cost_usd"])
        for family in families
        for row in family["rows"]
        if row["ceiling_status"] == "current_n50_override"
    )
    payload = {
        "schema_version": "result-envelope/v0.2",
        "wave": "cross_family_sentence_vorn_noguards_2026_05_20",
        "families": families,
        "total_estimated_cost_usd": total_cost,
        "coverage_notes": [
            "Fresh no-guards rows are current_n50_override by construction.",
            "Guarded references are reused from the freshest available artifact for each family-primary contrast.",
            "Qwen 3 8B is retained as an observational boundary because the guarded vanilla row is non-discriminative on this slice.",
        ],
    }
    _json_write(RESULTS / "cross-family-sentence-vorn-noguards-wave-2026-05-20.json", payload)

    blocks = [
        "# Cross-Family Sentence-Vorn No-Guards Wave — 2026-05-20",
        "",
        f"Fresh wave cost recorded so far: `${total_cost:.4f}`.",
        "",
    ]
    for family in families:
        pairwise = family["pairwise_tests"][0]
        blocks.extend(
            [
                f"## {family['display_name']}",
                "",
                f"- Read: `{family['overall_read']}`",
                f"- Interpretation: {family['interpretation']}",
                f"- Delta vs guarded sentence-vorn: `{family['delta_vs_guarded']:+.2f}`",
                "",
                "| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
                _row_cells(family["rows"][0], "Vanilla reference"),
                _row_cells(family["rows"][1], "Sentence vorn guarded"),
                _row_cells(family["rows"][2], "Sentence vorn no-guards"),
                "",
                "Pairwise tests:",
                f"- {pairwise['lhs']} vs {pairwise['rhs']}, exact McNemar on `{pairwise['table']}`: `p = {pairwise['p_value']:.6g}`",
                "",
            ]
        )

    _md_write(RESULTS / "cross-family-sentence-vorn-noguards-wave-2026-05-20.md", "\n".join(blocks))


if __name__ == "__main__":
    main()
