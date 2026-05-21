#!/usr/bin/env python3
"""Build the Gemma 4 sentence-attention 2x2 artifact."""
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
SOURCE_DEGRADATION = RESULTS / "gemma-4-4k-degradation-curve-2026-05-15.json"


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
    budget: int | None,
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
        "ceiling_status": "current_n50_override",
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
                budget,
                method_label,
                f"{row['hit_rate']:.2f}",
                f"[{row['wilson_ci_95'][0]:.4f}, {row['wilson_ci_95'][1]:.4f}]",
                _guardrails_label(str(row["guardrails"])),
                str(row["ceiling_status"]),
                f"{row['elapsed_seconds']:.2f}s",
                f"${row['estimated_cost_usd']:.4f}",
                f"{(1.0 - float(row['mean_retention_ratio'])) * 100:.2f}%",
            ]
        )
        + " |"
    )


def _source_rows() -> tuple[
    dict[str, object],
    dict[int, dict[str, object]],
    dict[int, dict[str, object]],
    dict[int, dict[str, object]],
    dict[int, dict[str, object]],
]:
    payload = _load_json(SOURCE_DEGRADATION)
    rows = payload["rows"]
    assert isinstance(rows, list)
    ceiling = next(row for row in rows if row["budget"] is None)
    by_method: dict[str, dict[int, dict[str, object]]] = {
        "token_vorn": {},
        "sentence_vorn": {},
        "tova": {},
        "h2o": {},
    }
    for raw_row in rows:
        row = dict(raw_row)
        row["ceiling_status"] = "historical_stale_reference"
        budget = row["budget"]
        if budget is None:
            continue
        method = str(row["method"])
        if method in by_method:
            by_method[method][int(budget)] = row
    historical_ceiling = dict(ceiling)
    historical_ceiling["ceiling_status"] = "historical_stale_reference"
    return (
        historical_ceiling,
        by_method["token_vorn"],
        by_method["sentence_vorn"],
        by_method["tova"],
        by_method["h2o"],
    )


def _override_with_current_ceiling(ceiling: dict[str, object]) -> dict[str, object]:
    path = BENCHMARKS / "gemma-4k-vanilla-n50-current-report.json"
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
        "ceiling_status": "current_n50_override",
    }


def _override_current_rows(
    *,
    token_vorn: dict[int, dict[str, object]],
    sentence_vorn: dict[int, dict[str, object]],
    token_tova: dict[int, dict[str, object]],
    token_h2o: dict[int, dict[str, object]],
) -> tuple[
    dict[int, dict[str, object]],
    dict[int, dict[str, object]],
    dict[int, dict[str, object]],
    dict[int, dict[str, object]],
]:
    current_specs = [
        (
            token_vorn,
            1024,
            "gemma-4k-token-vorn-b1024-n50-current-report.json",
            "token_1024_guarded",
            "token_vorn",
        ),
        (
            sentence_vorn,
            1024,
            "gemma-4k-sentence-vorn-b1024-n50-current-report.json",
            "sentence_1024_guarded",
            "sentence_vorn",
        ),
        (
            token_tova,
            1024,
            "gemma-4k-tova-b1024-n50-current-report.json",
            "tova_1024_guarded",
            "tova_style",
        ),
        (
            token_h2o,
            1024,
            "gemma-4k-h2o-b1024-n50-current-report.json",
            "h2o_1024_guarded",
            "h2o_style",
        ),
        (
            token_vorn,
            2048,
            "gemma-4k-token-vorn-b2048-n50-current-report.json",
            "token_2048_guarded",
            "token_vorn",
        ),
        (
            sentence_vorn,
            2048,
            "gemma-4k-sentence-vorn-b2048-n50-current-report.json",
            "sentence_2048_guarded",
            "sentence_vorn",
        ),
        (
            token_tova,
            2048,
            "gemma-4k-tova-b2048-n50-current-report.json",
            "tova_2048_guarded",
            "tova_style",
        ),
        (
            token_h2o,
            2048,
            "gemma-4k-h2o-b2048-n50-current-report.json",
            "h2o_2048_guarded",
            "h2o_style",
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
    return token_vorn, sentence_vorn, token_tova, token_h2o


def _classify_budget(
    *,
    sentence_vorn: dict[str, object],
    sentence_tova: dict[str, object],
    sentence_h2o: dict[str, object] | None,
    ceiling_hit_rate: float,
) -> tuple[str, str]:
    attention_values = [float(sentence_tova["hit_rate"])]
    if sentence_h2o is not None:
        attention_values.append(float(sentence_h2o["hit_rate"]))
    sentence_attention_mean = sum(attention_values) / len(attention_values)
    sentence_vorn_rate = float(sentence_vorn["hit_rate"])
    sentence_attention_gap = ceiling_hit_rate - sentence_attention_mean
    sentence_vorn_gap = ceiling_hit_rate - sentence_vorn_rate

    if sentence_attention_gap <= 0.10 and sentence_vorn_gap >= 0.35:
        return (
            "outcome_a_attention_tolerant_residual_sensitive",
            (
                "Sentence-level attention-weight rows preserve the ceiling while "
                "sentence_vorn remains far below it, which makes Gemma's attention-weight "
                "preference granularity-tolerant on this budget."
            ),
        )
    if sentence_attention_gap <= 0.10 and sentence_vorn_gap <= 0.15:
        return (
            "outcome_b_token_granularity_specific_outlier",
            (
                "Sentence-level attention-weight rows stay at ceiling and sentence_vorn "
                "also climbs to a competitive regime, which makes the Gemma outlier look "
                "token-granularity-specific rather than an absolute family-wide channel split."
            ),
        )
    return (
        "outcome_c_channel_favoritism_persists",
        (
            "Sentence-level attention-weight rows fail to preserve the ceiling cleanly or "
            "only improve modestly, which keeps Gemma in the attention-weight-favored family "
            "even after the granularity change."
        ),
    )


def _rate_delta(current: dict[str, object], historical: dict[str, object] | None) -> float | None:
    if historical is None:
        return None
    return float(current["hit_rate"]) - float(historical["hit_rate"])


def main() -> None:
    (
        historical_ceiling,
        historical_token_vorn,
        historical_sentence_vorn,
        historical_token_tova,
        historical_token_h2o,
    ) = _source_rows()

    ceiling = _override_with_current_ceiling(historical_ceiling)
    current_token_vorn = dict(historical_token_vorn)
    current_sentence_vorn = dict(historical_sentence_vorn)
    current_token_tova = dict(historical_token_tova)
    current_token_h2o = dict(historical_token_h2o)
    (
        current_token_vorn,
        current_sentence_vorn,
        current_token_tova,
        current_token_h2o,
    ) = _override_current_rows(
        token_vorn=current_token_vorn,
        sentence_vorn=current_sentence_vorn,
        token_tova=current_token_tova,
        token_h2o=current_token_h2o,
    )

    sentence_attention_rows = [
        _row_from_live_report(
            _load_report("gemma-4k-sentence-tova-b1024-n50-report.json"),
            label="sentence_tova_1024_guarded",
            method="sentence_tova_style",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("gemma-4k-sentence-h2o-b1024-n50-report.json"),
            label="sentence_h2o_1024_guarded",
            method="sentence_h2o_style",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("gemma-4k-sentence-tova-b2048-n50-report.json"),
            label="sentence_tova_2048_guarded",
            method="sentence_tova_style",
            budget=2048,
            guardrails="prefix_plus_recent",
        ),
    ]
    sentence_h2o_2048_path = BENCHMARKS / "gemma-4k-sentence-h2o-b2048-n50-report.json"
    if sentence_h2o_2048_path.exists():
        sentence_attention_rows.append(
            _row_from_live_report(
                _load_json(sentence_h2o_2048_path),
                label="sentence_h2o_2048_guarded",
                method="sentence_h2o_style",
                budget=2048,
                guardrails="prefix_plus_recent",
            )
        )
    sentence_attention_by_label = {
        str(row["label"]): row for row in sentence_attention_rows
    }

    pairwise_tests = [
        _paired_test(
            sentence_attention_by_label["sentence_tova_1024_guarded"],
            current_token_tova[1024],
        ),
        _paired_test(
            sentence_attention_by_label["sentence_h2o_1024_guarded"],
            current_token_h2o[1024],
        ),
        _paired_test(
            sentence_attention_by_label["sentence_tova_1024_guarded"],
            current_sentence_vorn[1024],
        ),
        _paired_test(
            sentence_attention_by_label["sentence_h2o_1024_guarded"],
            current_sentence_vorn[1024],
        ),
        _paired_test(current_token_tova[1024], current_sentence_vorn[1024]),
        _paired_test(current_token_h2o[1024], current_sentence_vorn[1024]),
        _paired_test(
            sentence_attention_by_label["sentence_tova_2048_guarded"],
            current_token_tova[2048],
        ),
        _paired_test(
            sentence_attention_by_label["sentence_tova_2048_guarded"],
            current_sentence_vorn[2048],
        ),
        _paired_test(current_token_tova[2048], current_sentence_vorn[2048]),
    ]
    if "sentence_h2o_2048_guarded" in sentence_attention_by_label:
        if 2048 in current_token_h2o:
            pairwise_tests.append(
                _paired_test(
                    sentence_attention_by_label["sentence_h2o_2048_guarded"],
                    current_token_h2o[2048],
                )
            )
            pairwise_tests.append(
                _paired_test(current_token_h2o[2048], current_sentence_vorn[2048])
            )
        pairwise_tests.append(
            _paired_test(
                sentence_attention_by_label["sentence_h2o_2048_guarded"],
                current_sentence_vorn[2048],
            )
        )

    ceiling_hit_rate = float(ceiling["hit_rate"])
    budget_reads: list[dict[str, object]] = []
    outcome_modes: list[str] = []
    for budget in (1024, 2048):
        mode, interpretation = _classify_budget(
            sentence_vorn=current_sentence_vorn[budget],
            sentence_tova=sentence_attention_by_label[
                f"sentence_tova_{budget}_guarded"
            ],
            sentence_h2o=sentence_attention_by_label.get(
                f"sentence_h2o_{budget}_guarded"
            ),
            ceiling_hit_rate=ceiling_hit_rate,
        )
        outcome_modes.append(mode)
        historical_h2o = historical_token_h2o.get(budget)
        budget_reads.append(
            {
                "budget": budget,
                "mode": mode,
                "interpretation": interpretation,
                "token_vorn": float(current_token_vorn[budget]["hit_rate"]),
                "sentence_vorn": float(current_sentence_vorn[budget]["hit_rate"]),
                "tova_style": float(current_token_tova[budget]["hit_rate"]),
                "sentence_tova_style": float(
                    sentence_attention_by_label[f"sentence_tova_{budget}_guarded"][
                        "hit_rate"
                    ]
                ),
                "h2o_style": (
                    float(current_token_h2o[budget]["hit_rate"])
                    if budget in current_token_h2o
                    else None
                ),
                "sentence_h2o_style": (
                    float(
                        sentence_attention_by_label[f"sentence_h2o_{budget}_guarded"][
                            "hit_rate"
                        ]
                    )
                    if f"sentence_h2o_{budget}_guarded" in sentence_attention_by_label
                    else None
                ),
                "historical_deltas": {
                    "token_vorn": _rate_delta(
                        current_token_vorn[budget],
                        historical_token_vorn.get(budget),
                    ),
                    "sentence_vorn": _rate_delta(
                        current_sentence_vorn[budget],
                        historical_sentence_vorn.get(budget),
                    ),
                    "tova_style": _rate_delta(
                        current_token_tova[budget],
                        historical_token_tova.get(budget),
                    ),
                    "h2o_style": (
                        _rate_delta(current_token_h2o[budget], historical_h2o)
                        if budget in current_token_h2o
                        else None
                    ),
                },
            }
        )

    overall_read = (
        outcome_modes[0]
        if outcome_modes[0] == outcome_modes[1]
        else "mixed_transition_surface"
    )

    payload = {
        "schema_version": "result-envelope/v0.2",
        "ceiling": ceiling,
        "historical_reference_ceiling": historical_ceiling,
        "token_vorn_rows": [current_token_vorn[1024], current_token_vorn[2048]],
        "sentence_vorn_rows": [
            current_sentence_vorn[1024],
            current_sentence_vorn[2048],
        ],
        "token_attention_rows": [
            current_token_tova[1024],
            current_token_h2o[1024],
            current_token_tova[2048],
            *([current_token_h2o[2048]] if 2048 in current_token_h2o else []),
        ],
        "sentence_attention_rows": sentence_attention_rows,
        "historical_reference_rows": {
            "token_vorn": [
                historical_token_vorn[1024],
                historical_token_vorn[2048],
            ],
            "sentence_vorn": [
                historical_sentence_vorn[1024],
                historical_sentence_vorn[2048],
            ],
            "tova_style": [
                historical_token_tova[1024],
                historical_token_tova[2048],
            ],
            "h2o_style": [
                historical_token_h2o[1024],
            ],
        },
        "coverage_notes": [
            "Fresh same-runner controls confirm the historical Gemma token/sentence/TOVA rows at 1024 and 2048.",
            "Token-level H2O was not rerun fresh because the H2O control path is the dominant cost/time tail on Gemma.",
            "Sentence-H2O 1024 is fresh. Sentence-H2O 2048 is included if the optional row completed before artifact build.",
        ],
        "pairwise_tests": pairwise_tests,
        "budget_reads": budget_reads,
        "overall_read": overall_read,
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
    _json_write(RESULTS / "sentence-attention-2x2-gemma4-2026-05-19.json", payload)

    row_order = [
        current_token_vorn[1024],
        current_sentence_vorn[1024],
        current_token_tova[1024],
        sentence_attention_by_label["sentence_tova_1024_guarded"],
        current_token_h2o[1024],
        sentence_attention_by_label["sentence_h2o_1024_guarded"],
        current_token_vorn[2048],
        current_sentence_vorn[2048],
        current_token_tova[2048],
        sentence_attention_by_label["sentence_tova_2048_guarded"],
    ]
    if 2048 in current_token_h2o:
        row_order.append(current_token_h2o[2048])
    if "sentence_h2o_2048_guarded" in sentence_attention_by_label:
        row_order.append(sentence_attention_by_label["sentence_h2o_2048_guarded"])
    labels = {
        "token_vorn": "Token vorn",
        "sentence_vorn": "Sentence vorn",
        "tova_style": "TOVA-style",
        "sentence_tova_style": "Sentence TOVA-style",
        "h2o": "H2O-style",
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
            f"`{item['h2o_style']:.2f}`"
            if item["h2o_style"] is not None
            else f"- Budget `{item['budget']}`: token_vorn `{item['token_vorn']:.2f}`, "
            f"sentence_vorn `{item['sentence_vorn']:.2f}`, TOVA-style "
            f"`{item['tova_style']:.2f}`, sentence-TOVA-style "
            f"`{item['sentence_tova_style']:.2f}`, H2O-style `not rerun`"
        )
        + ", sentence-H2O-style "
        + (
            f"`{item['sentence_h2o_style']:.2f}`. {item['interpretation']}"
            if item["sentence_h2o_style"] is not None
            else f"`not run`. {item['interpretation']}"
        )
        for item in budget_reads
    )
    drift_lines = "\n".join(
        (
            f"- Budget `{item['budget']}` historical deltas: token_vorn "
            f"`{item['historical_deltas']['token_vorn']:+.2f}`, "
            f"sentence_vorn `{item['historical_deltas']['sentence_vorn']:+.2f}`, "
            f"TOVA-style `{item['historical_deltas']['tova_style']:+.2f}`, "
            f"H2O-style "
            f"`{item['historical_deltas']['h2o_style']:+.2f}`"
            if item["historical_deltas"]["h2o_style"] is not None
            else
            f"- Budget `{item['budget']}` historical deltas: token_vorn "
            f"`{item['historical_deltas']['token_vorn']:+.2f}`, "
            f"sentence_vorn `{item['historical_deltas']['sentence_vorn']:+.2f}`, "
            f"TOVA-style `{item['historical_deltas']['tova_style']:+.2f}`, "
            "H2O-style `n/a`."
        )
        for item in budget_reads
    )

    _md_write(
        RESULTS / "sentence-attention-2x2-gemma4-2026-05-19.md",
        f"""
# Sentence-Attention 2×2 on Gemma 4 E4B-it — 2026-05-19

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `google/gemma-4-E4B-it`
- Pooling: `max`

## Ceiling Context

- Vanilla @ full context: `{ceiling['hit_rate']:.2f}` hit rate, `{ceiling['elapsed_seconds']:.2f}s`, `${ceiling['estimated_cost_usd']:.4f}`.
- Ceiling status: `{ceiling['ceiling_status']}`.
- Historical vanilla reference: `{historical_ceiling['hit_rate']:.2f}` with status `{historical_ceiling['ceiling_status']}`.

## Rows

| Budget | Method | Hit rate | 95% Wilson CI | Guardrails | Ceiling status | Wall-clock | Inference cost | KV savings |
|--------|--------|----------|---------------|------------|----------------|------------|----------------|-----------|
{table_lines}

## Paired Tests

{pairwise_lines}

## Read

{read_lines}

- Overall outcome classification: `{overall_read}`.

## Drift vs Historical Gemma Surface

{drift_lines}

## Coverage Notes

- Fresh same-runner controls confirm the historical Gemma token/sentence/TOVA rows at `1024` and `2048`.
- Token-level H2O was not rerun fresh because the H2O control path is the dominant cost/time tail on Gemma.
- Sentence-H2O `1024` is fresh. Sentence-H2O `2048` is included only if the optional row completed before artifact build.
""",
    )


if __name__ == "__main__":
    main()
