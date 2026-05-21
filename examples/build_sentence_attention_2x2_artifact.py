#!/usr/bin/env python3
"""Build the Mistral 4k sentence-attention 2x2 artifact."""
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
SOURCE_SWEEP = RESULTS / "sentence-level-eviction-4k-budget-sweep-2026-05-13.json"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _load_report(name: str) -> dict[str, object]:
    return _load_json(BENCHMARKS / name)


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
                str(row["budget"]),
                method_label,
                f"{row['hit_rate']:.2f}",
                f"[{row['wilson_ci_95'][0]:.4f}, {row['wilson_ci_95'][1]:.4f}]",
                _guardrails_label(str(row["guardrails"])),
                f"{row['elapsed_seconds']:.2f}s",
                f"${row['estimated_cost_usd']:.4f}",
                f"{(1.0 - float(row['mean_retention_ratio'])) * 100:.2f}%",
            ]
        )
        + " |"
    )


def _source_rows() -> tuple[dict[str, object], dict[int, dict[str, object]], dict[int, dict[str, object]]]:
    payload = _load_json(SOURCE_SWEEP)
    ceiling = payload["ceiling"]
    assert isinstance(ceiling, dict)
    token_rows = payload["token_rows"]
    sentence_rows = payload["sentence_rows"]
    assert isinstance(token_rows, list)
    assert isinstance(sentence_rows, list)
    token_by_budget = {
        int(row["budget"]): row
        for row in token_rows
        if row["guardrails"] == "prefix_plus_recent"
    }
    sentence_by_budget = {
        int(row["budget"]): row
        for row in sentence_rows
        if row["guardrails"] == "prefix_plus_recent"
    }
    return ceiling, token_by_budget, sentence_by_budget


def _override_with_current_ceiling(ceiling: dict[str, object]) -> dict[str, object]:
    path = BENCHMARKS / "mistral-4k-vanilla-n50-current-report.json"
    if not path.exists():
        return ceiling
    payload = _load_json(path)
    metric_name, hit_rate = _primary_metric(payload)
    return {
        "label": "vanilla_full_context_n50_current",
        "method": "vanilla",
        "metric": metric_name,
        "hit_rate": hit_rate,
        "elapsed_seconds": float(payload["elapsed_seconds"]),
        "estimated_cost_usd": float(payload["estimated_cost_usd"]),
        "run_id": _run_id(payload),
        "cases": int(payload["case_count"]),
    }


def _ceiling_status(ceiling: dict[str, object]) -> str:
    if ceiling.get("label") == "vanilla_full_context_n50_current":
        return "current_n50_override"
    return "historical_stale_reference"


def _override_with_current_rows(
    *,
    token_by_budget: dict[int, dict[str, object]],
    sentence_by_budget: dict[int, dict[str, object]],
) -> tuple[dict[int, dict[str, object]], dict[int, dict[str, object]]]:
    current_specs = [
        (
            token_by_budget,
            512,
            "mistral-4k-token-vorn-b512-n50-current-report.json",
            "token_512_guarded",
            "token_vorn",
        ),
        (
            token_by_budget,
            1024,
            "mistral-4k-token-vorn-b1024-n50-current-report.json",
            "token_1024_guarded",
            "token_vorn",
        ),
        (
            sentence_by_budget,
            512,
            "mistral-4k-sentence-vorn-b512-n50-current-report.json",
            "sentence_512_guarded",
            "sentence_vorn",
        ),
        (
            sentence_by_budget,
            1024,
            "mistral-4k-sentence-vorn-b1024-n50-current-report.json",
            "sentence_1024_guarded",
            "sentence_vorn",
        ),
    ]
    for target, budget, filename, label, method in current_specs:
        path = BENCHMARKS / filename
        if path.exists():
            target[budget] = _row_from_live_report(
                _load_json(path),
                label=label,
                method=method,
                budget=budget,
                guardrails="prefix_plus_recent",
            )
    return token_by_budget, sentence_by_budget


def _classify_budget(
    *,
    sentence_vorn: dict[str, object],
    token_tova: dict[str, object],
    token_h2o: dict[str, object],
    sentence_tova: dict[str, object],
    sentence_h2o: dict[str, object],
) -> tuple[str, str]:
    sentence_attention_mean = (
        float(sentence_tova["hit_rate"]) + float(sentence_h2o["hit_rate"])
    ) / 2.0
    token_attention_mean = (
        float(token_tova["hit_rate"]) + float(token_h2o["hit_rate"])
    ) / 2.0
    sentence_vorn_rate = float(sentence_vorn["hit_rate"])
    sentence_gain = sentence_attention_mean - token_attention_mean
    residual_gap = sentence_vorn_rate - sentence_attention_mean

    if abs(sentence_gain) <= 0.05 and residual_gap >= 0.10:
        return (
            "scoring_channel_primary",
            (
                "Sentence-level attention-weight baselines stay close to their "
                "token-level counterparts while remaining clearly below sentence_vorn."
            ),
        )
    if sentence_gain >= 0.10 and abs(residual_gap) <= 0.05:
        return (
            "granularity_primary",
            (
                "Sentence grouping alone recovers most of the sentence_vorn gain, "
                "with sentence-level attention-weight rows close to sentence_vorn."
            ),
        )
    return (
        "additive_contribution",
        (
            "Sentence-level attention-weight rows improve over token attention, "
            "but they still trail sentence_vorn, which points to additive effects "
            "from granularity and scoring channel."
        ),
    )


def main() -> None:
    ceiling, token_vorn, sentence_vorn = _source_rows()
    ceiling = _override_with_current_ceiling(ceiling)
    token_vorn, sentence_vorn = _override_with_current_rows(
        token_by_budget=token_vorn,
        sentence_by_budget=sentence_vorn,
    )

    token_attention_rows = [
        _row_from_live_report(
            _load_report("mistral-4k-tova-b512-n50-report.json"),
            label="tova_512_guarded",
            method="tova_style",
            budget=512,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("mistral-4k-tova-b1024-n50-report.json"),
            label="tova_1024_guarded",
            method="tova_style",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("mistral-4k-h2o-b512-n50-report.json"),
            label="h2o_512_guarded",
            method="h2o_style",
            budget=512,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("mistral-4k-h2o-b1024-n50-report.json"),
            label="h2o_1024_guarded",
            method="h2o_style",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
    ]
    sentence_attention_rows = [
        _row_from_live_report(
            _load_report("mistral-4k-sentence-tova-b512-n50-report.json"),
            label="sentence_tova_512_guarded",
            method="sentence_tova_style",
            budget=512,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("mistral-4k-sentence-tova-b1024-n50-report.json"),
            label="sentence_tova_1024_guarded",
            method="sentence_tova_style",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("mistral-4k-sentence-h2o-b512-n50-report.json"),
            label="sentence_h2o_512_guarded",
            method="sentence_h2o_style",
            budget=512,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("mistral-4k-sentence-h2o-b1024-n50-report.json"),
            label="sentence_h2o_1024_guarded",
            method="sentence_h2o_style",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
    ]

    token_attention_by_label = {str(row["label"]): row for row in token_attention_rows}
    sentence_attention_by_label = {
        str(row["label"]): row for row in sentence_attention_rows
    }

    pairwise_tests = [
        _paired_test(
            sentence_attention_by_label["sentence_tova_512_guarded"],
            token_attention_by_label["tova_512_guarded"],
        ),
        _paired_test(
            sentence_attention_by_label["sentence_h2o_512_guarded"],
            token_attention_by_label["h2o_512_guarded"],
        ),
        _paired_test(
            sentence_attention_by_label["sentence_tova_512_guarded"],
            sentence_vorn[512],
        ),
        _paired_test(
            sentence_attention_by_label["sentence_h2o_512_guarded"],
            sentence_vorn[512],
        ),
        _paired_test(token_attention_by_label["tova_512_guarded"], token_vorn[512]),
        _paired_test(token_attention_by_label["h2o_512_guarded"], token_vorn[512]),
        _paired_test(
            sentence_attention_by_label["sentence_tova_1024_guarded"],
            token_attention_by_label["tova_1024_guarded"],
        ),
        _paired_test(
            sentence_attention_by_label["sentence_h2o_1024_guarded"],
            token_attention_by_label["h2o_1024_guarded"],
        ),
        _paired_test(
            sentence_attention_by_label["sentence_tova_1024_guarded"],
            sentence_vorn[1024],
        ),
        _paired_test(
            sentence_attention_by_label["sentence_h2o_1024_guarded"],
            sentence_vorn[1024],
        ),
        _paired_test(token_attention_by_label["tova_1024_guarded"], token_vorn[1024]),
        _paired_test(token_attention_by_label["h2o_1024_guarded"], token_vorn[1024]),
    ]

    budget_reads: list[dict[str, object]] = []
    overall_modes: list[str] = []
    for budget in (512, 1024):
        mode, interpretation = _classify_budget(
            sentence_vorn=sentence_vorn[budget],
            token_tova=token_attention_by_label[f"tova_{budget}_guarded"],
            token_h2o=token_attention_by_label[f"h2o_{budget}_guarded"],
            sentence_tova=sentence_attention_by_label[f"sentence_tova_{budget}_guarded"],
            sentence_h2o=sentence_attention_by_label[f"sentence_h2o_{budget}_guarded"],
        )
        overall_modes.append(mode)
        budget_reads.append(
            {
                "budget": budget,
                "mode": mode,
                "interpretation": interpretation,
                "token_vorn": float(token_vorn[budget]["hit_rate"]),
                "sentence_vorn": float(sentence_vorn[budget]["hit_rate"]),
                "tova_style": float(
                    token_attention_by_label[f"tova_{budget}_guarded"]["hit_rate"]
                ),
                "sentence_tova_style": float(
                    sentence_attention_by_label[f"sentence_tova_{budget}_guarded"][
                        "hit_rate"
                    ]
                ),
                "h2o_style": float(
                    token_attention_by_label[f"h2o_{budget}_guarded"]["hit_rate"]
                ),
                "sentence_h2o_style": float(
                    sentence_attention_by_label[f"sentence_h2o_{budget}_guarded"][
                        "hit_rate"
                    ]
                ),
            }
        )

    overall_read = (
        "scoring_channel_primary"
        if overall_modes == ["scoring_channel_primary", "scoring_channel_primary"]
        else "granularity_primary"
        if overall_modes == ["granularity_primary", "granularity_primary"]
        else "additive_or_mixed"
    )

    payload = {
        "schema_version": "result-envelope/v0.2",
        "ceiling": ceiling,
        "ceiling_status": _ceiling_status(ceiling),
        "token_vorn_rows": [token_vorn[512], token_vorn[1024]],
        "sentence_vorn_rows": [sentence_vorn[512], sentence_vorn[1024]],
        "token_attention_rows": token_attention_rows,
        "sentence_attention_rows": sentence_attention_rows,
        "pairwise_tests": pairwise_tests,
        "budget_reads": budget_reads,
        "overall_read": overall_read,
        "run_conditions": {
            "profile": "author",
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "niah_multikey_1_4k",
            "split": "validation[:50]",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "random_seed": 17,
            "canonical_layer": 16,
            "recent_token_window": 16,
            "sentence_pooling": "max",
            "sentence_top_k": 3,
        },
    }
    _json_write(RESULTS / "sentence-attention-2x2-mistral-2026-05-19.json", payload)

    row_order = [
        token_vorn[512],
        sentence_vorn[512],
        token_attention_by_label["tova_512_guarded"],
        sentence_attention_by_label["sentence_tova_512_guarded"],
        token_attention_by_label["h2o_512_guarded"],
        sentence_attention_by_label["sentence_h2o_512_guarded"],
        token_vorn[1024],
        sentence_vorn[1024],
        token_attention_by_label["tova_1024_guarded"],
        sentence_attention_by_label["sentence_tova_1024_guarded"],
        token_attention_by_label["h2o_1024_guarded"],
        sentence_attention_by_label["sentence_h2o_1024_guarded"],
    ]
    labels = {
        "token_vorn": "Token vorn",
        "sentence_vorn": "Sentence vorn",
        "tova_style": "TOVA-style",
        "sentence_tova_style": "Sentence TOVA-style",
        "h2o_style": "H2O-style",
        "sentence_h2o_style": "Sentence H2O-style",
    }
    table_lines = "\n".join(
        _row_cells(row, labels[str(row["method"])]) for row in row_order
    )
    pairwise_lines = "\n".join(
        f"- {pairwise['lhs']} vs {pairwise['rhs']}, exact McNemar on `{pairwise['table']}`: "
        f"`p = {pairwise['p_value']:.6g}`"
        for pairwise in pairwise_tests
    )
    read_lines = "\n".join(
        (
            f"- Budget `{item['budget']}`: token_vorn `{item['token_vorn']:.2f}`, "
            f"sentence_vorn `{item['sentence_vorn']:.2f}`, TOVA-style "
            f"`{item['tova_style']:.2f}`, sentence-TOVA-style "
            f"`{item['sentence_tova_style']:.2f}`, H2O-style "
            f"`{item['h2o_style']:.2f}`, sentence-H2O-style "
            f"`{item['sentence_h2o_style']:.2f}`. "
            f"{item['interpretation']}"
        )
        for item in budget_reads
    )

    _md_write(
        RESULTS / "sentence-attention-2x2-mistral-2026-05-19.md",
        f"""
# Sentence-Attention 2×2 on Mistral 4k — 2026-05-19

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Pooling: `max`

## Ceiling Context

- Vanilla @ full context: `{ceiling['hit_rate']:.2f}` hit rate, `{ceiling['elapsed_seconds']:.2f}s`, `${ceiling['estimated_cost_usd']:.4f}`.
- Ceiling status: `{_ceiling_status(ceiling)}`.
- The interpretation below is driven by the fresh paired `n=50` rows in this artifact. If the ceiling status is `historical_stale_reference`, treat the ceiling row as provenance context only, not as part of the mechanism read.

## Rows

| Budget | Method | Hit rate | 95% Wilson CI | Guardrails | Wall-clock | Inference cost | KV savings |
|--------|--------|----------|---------------|------------|------------|----------------|-----------|
{table_lines}

## Paired Tests

{pairwise_lines}

## Read

{read_lines}

- Overall outcome classification: `{overall_read}`.
""",
    )


if __name__ == "__main__":
    main()
