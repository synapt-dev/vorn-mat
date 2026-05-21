#!/usr/bin/env python3
"""Build the qa_2_4k n=200 multi-family cross-task artifact from Modal reports."""
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


def _row_if_present(
    filename: str,
    *,
    family: str,
    label: str,
    method: str,
    budget: int | None,
    guardrails: str,
) -> dict[str, object] | None:
    path = BENCHMARKS / filename
    if not path.exists():
        return None
    return _row_from_report(
        filename,
        family=family,
        label=label,
        method=method,
        budget=budget,
        guardrails=guardrails,
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


def _pair_key(lhs: str, rhs: str) -> tuple[str, str]:
    return (lhs, rhs)


def _read_line(
    family: str,
    vanilla: dict[str, object],
    sentence: dict[str, object],
    tova: dict[str, object] | None,
    h2o: dict[str, object] | None,
    pairwise: dict[tuple[str, str], dict[str, object]],
) -> str:
    compressed = [sentence]
    if tova is not None:
        compressed.append(tova)
    if h2o is not None:
        compressed.append(h2o)
    best = max(compressed, key=lambda row: float(row["hit_rate"]))
    parts = [
        f"- {family}: vanilla `{float(vanilla['hit_rate']):.3f}`",
        f"sentence `{float(sentence['hit_rate']):.3f}`",
    ]
    if tova is not None:
        parts.append(f"TOVA `{float(tova['hit_rate']):.3f}`")
    else:
        parts.append("TOVA `OOM at n=200 on current stack`")
    if h2o is not None:
        parts.append(f"H2O `{float(h2o['hit_rate']):.3f}`")
    else:
        parts.append("H2O `OOM at n=200 on current stack`")
    tail = [f"Best completed compressed method is `{best['method']}` at `{float(best['hit_rate']):.3f}`."]
    if tova is not None:
        sentence_vs_tova = pairwise[_pair_key(str(sentence["label"]), str(tova["label"]))]
        tail.append(f"Sentence vs TOVA McNemar `p = {sentence_vs_tova['p_value']:.6g}`.")
    if h2o is not None:
        sentence_vs_h2o = pairwise[_pair_key(str(sentence["label"]), str(h2o["label"]))]
        tail.append(f"Sentence vs H2O McNemar `p = {sentence_vs_h2o['p_value']:.6g}`.")
    return "; ".join(parts) + ". " + " ".join(tail)


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
            "gemma-4-qa2-sentence-1024-guarded-n200.json",
            family="Gemma 4 E4B-it",
            label="gemma4_sentence_1024",
            method="sentence_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-qa2-tova-1024-n200.json",
            family="Gemma 4 E4B-it",
            label="gemma4_tova_1024",
            method="tova",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "gemma-4-qa2-h2o-1024-n200.json",
            family="Gemma 4 E4B-it",
            label="gemma4_h2o_1024",
            method="h2o",
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
            "llama31-qa2-sentence-1024-guarded-n200.json",
            family="Llama 3.1 8B",
            label="llama31_sentence_1024",
            method="sentence_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "llama31-qa2-tova-1024-n200.json",
            family="Llama 3.1 8B",
            label="llama31_tova_1024",
            method="tova",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "llama31-qa2-h2o-1024-n200.json",
            family="Llama 3.1 8B",
            label="llama31_h2o_1024",
            method="h2o",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "ministral-qa2-vanilla-n200.json",
            family="Ministral 8B",
            label="ministral_vanilla",
            method="vanilla",
            budget=None,
            guardrails="prefix_plus_recent",
        ),
        _row_from_report(
            "ministral-qa2-sentence-1024-guarded-n200.json",
            family="Ministral 8B",
            label="ministral_sentence_1024",
            method="sentence_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
    ]
    ministral_tova = _row_if_present(
            "ministral-qa2-tova-1024-n200.json",
            family="Ministral 8B",
            label="ministral_tova_1024",
            method="tova",
            budget=1024,
            guardrails="prefix_plus_recent",
        )
    ministral_h2o = _row_if_present(
            "ministral-qa2-h2o-1024-n200.json",
            family="Ministral 8B",
            label="ministral_h2o_1024",
            method="h2o",
            budget=1024,
            guardrails="prefix_plus_recent",
        )
    if ministral_tova is not None:
        rows.append(ministral_tova)
    if ministral_h2o is not None:
        rows.append(ministral_h2o)
    by_label = {str(row["label"]): row for row in rows}
    pairwise_tests = []
    for family_prefix in ("gemma4", "llama31", "ministral"):
        vanilla = by_label[f"{family_prefix}_vanilla"]
        sentence = by_label[f"{family_prefix}_sentence_1024"]
        tova = by_label.get(f"{family_prefix}_tova_1024")
        h2o = by_label.get(f"{family_prefix}_h2o_1024")
        pairwise_tests.append(_paired_test(sentence, vanilla))
        if tova is not None:
            pairwise_tests.append(_paired_test(sentence, tova))
            pairwise_tests.append(_paired_test(tova, vanilla))
        if h2o is not None:
            pairwise_tests.append(_paired_test(sentence, h2o))
            pairwise_tests.append(_paired_test(h2o, vanilla))
    pairwise_lookup = {
        _pair_key(str(item["lhs"]), str(item["rhs"])): item for item in pairwise_tests
    }
    failed_rows: list[dict[str, object]] = []
    if ministral_tova is None:
        failed_rows.append(
            {
                "family": "Ministral 8B",
                "label": "ministral_tova_1024",
                "method": "tova",
                "budget": 1024,
                "guardrails": "prefix_plus_recent",
                "status": "failed_oom",
                "reason": (
                    "Repeated CUDA OOM in Ministral attention-weight path on qa_2_4k "
                    "validation[:200], including one retry with "
                    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."
                ),
            }
        )
    if ministral_h2o is None:
        failed_rows.append(
            {
                "family": "Ministral 8B",
                "label": "ministral_h2o_1024",
                "method": "h2o",
                "budget": 1024,
                "guardrails": "prefix_plus_recent",
                "status": "failed_oom",
                "reason": (
                    "Repeated CUDA OOM in Ministral attention-weight path on qa_2_4k "
                    "validation[:200], including one retry with "
                    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."
                ),
            }
        )

    payload = {
        "schema_version": "result-envelope/v0.2",
        "provenance": {
            "raw_predictions_embedded": True,
            "raw_report_home": ".benchmarks/cross-model/",
            "raw_report_files": [row["source_report"] for row in rows],
            "durable_prediction_home": "rows[].observations[]",
            "notes": (
                "qa_2_4k n=200 cross-task generalization artifact. Published rows embed "
                "per-fixture predictions directly and record raw-report filenames + run IDs."
            ),
        },
        "rows": rows,
        "failed_rows": failed_rows,
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
                "Cross-task generalization surface for the family-conditional channel story: "
                "Gemma 4, Llama 3.1, and Ministral 8B on qa_2_4k with vanilla, sentence_vorn, "
                "TOVA, and H2O."
            ),
        },
    }
    _json_write(RESULTS / "qa2-cross-task-multifamily-2026-05-16.json", payload)

    row_lines = "\n".join(_row_cells(row) for row in rows)
    pairwise_lines = "\n".join(
        f"- {pairwise['lhs']} vs {pairwise['rhs']}, exact McNemar on `{pairwise['table']}`: "
        f"`p = {pairwise['p_value']:.6g}`"
        for pairwise in pairwise_tests
    )
    failed_lines = "\n".join(
        f"- {item['label']}: {item['reason']}" for item in failed_rows
    )
    read_lines = "\n".join(
        [
            _read_line(
                "Gemma 4 E4B-it",
                by_label["gemma4_vanilla"],
                by_label["gemma4_sentence_1024"],
                by_label["gemma4_tova_1024"],
                by_label["gemma4_h2o_1024"],
                pairwise_lookup,
            ),
            _read_line(
                "Llama 3.1 8B",
                by_label["llama31_vanilla"],
                by_label["llama31_sentence_1024"],
                by_label["llama31_tova_1024"],
                by_label["llama31_h2o_1024"],
                pairwise_lookup,
            ),
            _read_line(
                "Ministral 8B",
                by_label["ministral_vanilla"],
                by_label["ministral_sentence_1024"],
                by_label.get("ministral_tova_1024"),
                by_label.get("ministral_h2o_1024"),
                pairwise_lookup,
            ),
        ]
    )
    if failed_rows:
        read_lines += (
            "\n- Execution boundary: Ministral attention-weight rows at `qa_2_4k`, "
            "`n=200`, `budget=1024` repeatedly OOMed on the current Modal/A100 stack, "
            "including one retry with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. "
            "Treat the missing TOVA/H2O cells as runtime-unsupported on this surface, not "
            "as scored negative rows."
        )

    _md_write(
        RESULTS / "qa2-cross-task-multifamily-2026-05-16.md",
        f"""
# Cross-Task Generalization (`qa_2_4k`, `n=200`) — 2026-05-16

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `qa_2_4k`
- Slice: `validation[:200]`
- Pooling: `max`

## Rows

| Family | Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | KV savings |
|--------|--------|--------|------------|----------|---------------|------------|----------------|-----------|
{row_lines}

## Paired Tests

{pairwise_lines}

## Execution Boundaries

{failed_lines if failed_lines else "- None."}

## Read

{read_lines}
        """,
    )


if __name__ == "__main__":
    main()
