#!/usr/bin/env python3
"""Build the Llama 3.1 8B 4k cross-baseline extension artifact."""
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
    report: dict[str, object],
    *,
    label: str,
    method: str,
    budget: int | None,
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
    if "retention_policy" in metadata:
        row["retention_policy"] = metadata["retention_policy"]
    if "model_id" in metadata:
        row["model_id"] = metadata["model_id"]
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
                "0",
                f"{(1.0 - float(row['mean_retention_ratio'])) * 100:.2f}%",
            ]
        )
        + " |"
    )


def main() -> None:
    floor = _row_from_report(
        _load_report("llama31-4k-vanilla.json"),
        label="vanilla_floor",
        method="vanilla",
        budget=None,
        guardrails="prefix_plus_recent",
    )
    token = _row_from_report(
        _load_report("llama31-4k-token-1024.json"),
        label="token_1024_guarded",
        method="token_vorn",
        budget=1024,
        guardrails="prefix_plus_recent",
    )
    sentence_guarded = _row_from_report(
        _load_report("llama31-4k-sentence-1024-guarded.json"),
        label="sentence_1024_guarded",
        method="sentence_vorn",
        budget=1024,
        guardrails="prefix_plus_recent",
    )
    sentence_noguards = _row_from_report(
        _load_report("llama31-4k-sentence-1024-noguards.json"),
        label="sentence_1024_noguards",
        method="sentence_vorn",
        budget=1024,
        guardrails="none",
    )
    tova = _row_from_report(
        _load_report("llama31-4k-tova-1024.json"),
        label="tova_1024_guarded",
        method="tova",
        budget=1024,
        guardrails="prefix_plus_recent",
    )
    h2o = _row_from_report(
        _load_report("llama31-4k-h2o-1024.json"),
        label="h2o_1024_guarded",
        method="h2o",
        budget=1024,
        guardrails="prefix_plus_recent",
    )

    rows = [floor, token, sentence_guarded, sentence_noguards, tova, h2o]
    pairwise_tests = [
        _paired_test(tova, token),
        _paired_test(h2o, token),
        _paired_test(tova, sentence_guarded),
        _paired_test(h2o, sentence_guarded),
        _paired_test(tova, h2o),
        _paired_test(tova, floor),
        _paired_test(h2o, floor),
    ]

    payload = {
        "schema_version": "result-envelope/v0.2",
        "rows": rows,
        "pairwise_tests": pairwise_tests,
        "run_conditions": {
            "profile": "author",
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "niah_multikey_1_4k",
            "split": "validation[:50]",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "random_seed": 17,
            "canonical_layer": 16,
            "cache_budget_tokens": 1024,
            "gate_answer_budget_tokens": 32,
            "note": (
                "Extends the original Llama 3.1 4k gate with TOVA- and H2O-style attention baselines "
                "on the same slice, budget, and paired-observation surface."
            ),
        },
    }
    _json_write(RESULTS / "llama31-4k-cross-baseline-extension-2026-05-15.json", payload)

    row_lines = "\n".join(
        [
            _row_cells(floor, "Vanilla"),
            _row_cells(token, "Token vorn"),
            _row_cells(sentence_guarded, "Sentence vorn"),
            _row_cells(sentence_noguards, "Sentence vorn"),
            _row_cells(tova, "TOVA"),
            _row_cells(h2o, "H2O"),
        ]
    )
    pairwise_lines = "\n".join(
        f"- {pairwise['lhs']} vs {pairwise['rhs']}, exact McNemar on `{pairwise['table']}`: "
        f"`p = {pairwise['p_value']:.6g}`"
        for pairwise in pairwise_tests
    )

    _md_write(
        RESULTS / "llama31-4k-cross-baseline-extension-2026-05-15.md",
        f"""
# Llama 3.1 8B 4k Cross-Baseline Extension — 2026-05-15

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Budget: `1024`

## Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
{row_lines}

## Paired Tests

{pairwise_lines}

## Read

- Llama 3.1 does **not** stay in the all-methods threshold class once the attention-weight baselines are included. At `1024`, the vorn rows remain at `1.00`, while `TOVA = 0.90` and `H2O = 0.94`.
- Unlike Qwen 3, Llama emits direct numeric or short declarative answers inside the existing `32`-token budget, so this extension remains a clean competence-comparison surface rather than a runner-surface mismatch case.
- The paper consequence is structural rather than headline-competitive: Llama is not just "Gemma with higher scores." Gemma preserves the ceiling under attention-weight baselines while residual-direction baselines collapse; Llama preserves the ceiling under residual-direction baselines while attention-weight baselines lose a small amount of quality.
- The paired rows are suggestive rather than decisive at `n = 50` (`TOVA` vs token/sentence `p = 0.0625`; `H2O` vs token/sentence `p = 0.25`). The honest cross-family claim is therefore about **effect-class shape**, not about a settled superiority result for any one baseline family on Llama.
""",
    )


if __name__ == "__main__":
    main()
