#!/usr/bin/env python3
"""Build the 2026-05-20 cross-family extension wave artifact."""
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

MODEL_SPECS = [
    {
        "family_id": "gemma3",
        "display_name": "Gemma 3",
        "model_candidates": [
            {
                "prefix": "gemma3-12b",
                "model_id": "google/gemma-3-12b-pt",
                "label": "12B primary",
            },
            {
                "prefix": "gemma3-4b",
                "model_id": "google/gemma-3-4b-pt",
                "label": "4B fallback",
            },
        ],
    },
    {
        "family_id": "qwen25",
        "display_name": "Qwen 2.5 7B",
        "model_candidates": [
            {
                "prefix": "qwen25-7b",
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "label": "7B primary",
            }
        ],
    },
    {
        "family_id": "qwen3a3b",
        "display_name": "Qwen 3 30B-A3B",
        "model_candidates": [
            {
                "prefix": "qwen3-30b-a3b",
                "model_id": "Qwen/Qwen3-30B-A3B",
                "label": "30B-A3B primary",
            }
        ],
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
    label: str,
    method: str,
    budget: int | None,
    guardrails: str,
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
        "ceiling_status": "current_n50_override",
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


def _load_row(path: Path, *, label: str, method: str, budget: int | None) -> dict[str, object]:
    report = _load_json(path)
    return _row_from_report(
        report,
        label=label,
        method=method,
        budget=budget,
        guardrails="prefix_plus_recent",
        source_report=f".benchmarks/{path.name}",
    )


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


def _classify_panel(rows_by_label: dict[str, dict[str, object]]) -> tuple[str, str]:
    vanilla = float(rows_by_label["vanilla_full_context"]["hit_rate"])
    if vanilla <= 0.0:
        return (
            "drowning_floor_non_discriminative",
            "Vanilla is non-discriminative on this slice, so the family remains an observational boundary rather than a valid five-method comparison surface.",
        )
    if "sentence_tova_1024_guarded" not in rows_by_label:
        return (
            "probe_only",
            "Only the discriminative probe has landed so far. The five-method 1024 panel is still incomplete.",
        )
    sentence_tova = float(rows_by_label["sentence_tova_1024_guarded"]["hit_rate"])
    sentence_vorn = float(rows_by_label["sentence_1024_guarded"]["hit_rate"])
    token_tova = float(rows_by_label["tova_1024_guarded"]["hit_rate"])
    if abs(sentence_tova - sentence_vorn) <= 0.10:
        return (
            "mistral_like_full_rescue",
            "Sentence-TOVA-style reaches the sentence-vorn regime at the 1024 gate.",
        )
    if sentence_tova <= token_tova + 0.10:
        return (
            "gemma_like_channel_persistence",
            "Sentence-TOVA-style stays close to token TOVA-style at the 1024 gate.",
        )
    return (
        "intermediate_recovery",
        "Sentence-TOVA-style improves materially over token TOVA-style but does not fully reach the sentence-vorn regime at the 1024 gate.",
    )


def _model_payload(spec: dict[str, object]) -> dict[str, object]:
    chosen = None
    for candidate in spec["model_candidates"]:  # type: ignore[index]
        prefix = candidate["prefix"]
        vanilla_path = BENCHMARKS / f"{prefix}-4k-vanilla-n50-current-report.json"
        if vanilla_path.exists():
            chosen = candidate
            break
    if chosen is None:
        chosen = spec["model_candidates"][0]  # type: ignore[index]

    prefix = chosen["prefix"]
    rows: list[dict[str, object]] = []
    rows_by_label: dict[str, dict[str, object]] = {}

    vanilla_path = BENCHMARKS / f"{prefix}-4k-vanilla-n50-current-report.json"
    if vanilla_path.exists():
        row = _load_row(
            vanilla_path,
            label="vanilla_full_context",
            method="vanilla",
            budget=None,
        )
        rows.append(row)
        rows_by_label[row["label"]] = row

    for budget in (1024, 512, 256):
        for label, filename, method in (
            (
                f"token_{budget}_guarded",
                f"{prefix}-4k-token-vorn-b{budget}-n50-current-report.json",
                "token_vorn",
            ),
            (
                f"sentence_{budget}_guarded",
                f"{prefix}-4k-sentence-vorn-b{budget}-n50-current-report.json",
                "sentence_vorn",
            ),
            (
                f"tova_{budget}_guarded",
                f"{prefix}-4k-tova-b{budget}-n50-current-report.json",
                "tova_style",
            ),
            (
                f"sentence_tova_{budget}_guarded",
                f"{prefix}-4k-sentence-tova-b{budget}-n50-report.json",
                "sentence_tova_style",
            ),
        ):
            path = BENCHMARKS / filename
            if path.exists():
                row = _load_row(path, label=label, method=method, budget=budget)
                rows.append(row)
                rows_by_label[label] = row

    pairwise_tests = []
    if all(key in rows_by_label for key in ("sentence_tova_1024_guarded", "tova_1024_guarded")):
        pairwise_tests.append(
            _paired_test(rows_by_label["sentence_tova_1024_guarded"], rows_by_label["tova_1024_guarded"])
        )
    if all(key in rows_by_label for key in ("sentence_tova_1024_guarded", "sentence_1024_guarded")):
        pairwise_tests.append(
            _paired_test(rows_by_label["sentence_tova_1024_guarded"], rows_by_label["sentence_1024_guarded"])
        )
    if all(key in rows_by_label for key in ("sentence_1024_guarded", "tova_1024_guarded")):
        pairwise_tests.append(
            _paired_test(rows_by_label["sentence_1024_guarded"], rows_by_label["tova_1024_guarded"])
        )

    overall_read, interpretation = _classify_panel(rows_by_label) if rows else ("not_run", "No rows have landed for this model yet.")
    return {
        "family_id": spec["family_id"],
        "display_name": spec["display_name"],
        "chosen_model": chosen,
        "rows": rows,
        "pairwise_tests": pairwise_tests,
        "overall_read": overall_read,
        "interpretation": interpretation,
    }


def main() -> None:
    models = [_model_payload(spec) for spec in MODEL_SPECS]
    total_cost = sum(
        float(row["estimated_cost_usd"])
        for model in models
        for row in model["rows"]
    )
    payload = {
        "schema_version": "result-envelope/v0.2",
        "wave": "cross_family_extension_2026_05_20",
        "models": models,
        "total_estimated_cost_usd": total_cost,
        "coverage_notes": [
            "This artifact is partial-friendly by design. It can be rebuilt after the vanilla probes, after the 1024 panels, and again after any threshold extensions.",
            "Freshness status is always explicit per row via current_n50_override.",
        ],
    }
    _json_write(RESULTS / "cross-family-extension-wave-2026-05-20.json", payload)

    blocks = [
        "# Cross-Family Extension Wave — 2026-05-20",
        "",
        f"Fresh wave cost recorded so far: `${total_cost:.4f}`.",
        "",
    ]
    for model in models:
        blocks.extend(
            [
                f"## {model['display_name']}",
                "",
                f"- Model choice: `{model['chosen_model']['model_id']}` ({model['chosen_model']['label']})",
                f"- Read: `{model['overall_read']}`",
                f"- Interpretation: {model['interpretation']}",
                "",
            ]
        )
        if model["rows"]:
            blocks.extend(
                [
                    "| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost | Mean evicted |",
                    "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
                ]
            )
            labels = {
                "vanilla": "Vanilla",
                "token_vorn": "Token vorn",
                "sentence_vorn": "Sentence vorn",
                "tova_style": "TOVA-style",
                "sentence_tova_style": "Sentence TOVA-style",
            }
            for row in model["rows"]:
                blocks.append(_row_cells(row, labels[str(row["method"])]))
            if model["pairwise_tests"]:
                blocks.extend(["", "Pairwise tests:"])
                for pairwise in model["pairwise_tests"]:
                    blocks.append(
                        f"- {pairwise['lhs']} vs {pairwise['rhs']}, exact McNemar on `{pairwise['table']}`: `p = {pairwise['p_value']:.6g}`"
                    )
        else:
            blocks.append("- No rows landed yet.")
        blocks.append("")

    _md_write(RESULTS / "cross-family-extension-wave-2026-05-20.md", "\n".join(blocks))


if __name__ == "__main__":
    main()
