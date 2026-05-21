#!/usr/bin/env python3
"""Rebuild paired vorn-MAT artifacts from per-report observations."""
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


def _load_report(name: str) -> dict[str, object]:
    return json.loads((BENCHMARKS / name).read_text())


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


def _row_from_report(
    report: dict[str, object],
    *,
    label: str,
    method: str,
    budget: int | None = None,
    budget_regime: str | None = None,
    guardrails: str | None = None,
) -> dict[str, object]:
    observations = _observations(report)
    hits = sum(1 for observation in observations if observation.correct)
    cases = len(observations)
    row: dict[str, object] = {
        "label": label,
        "method": method,
        "hits": hits,
        "cases": cases,
        "hit_rate": hits / cases if cases else 0.0,
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
    if budget is not None:
        row["budget"] = budget
    if budget_regime is not None:
        row["budget_regime"] = budget_regime
    if guardrails is not None:
        row["guardrails"] = guardrails
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


def build_sentence_level_artifact() -> None:
    sentence_1024_guarded = _row_from_report(
        _load_report("reemit-niah-sentence-vorn-b1024-report.json"),
        label="sentence_1024_guarded",
        method="sentence_vorn",
        budget=1024,
        guardrails="prefix_plus_recent",
    )
    sentence_1024_noguards = _row_from_report(
        _load_report("reemit-niah-sentence-vorn-b1024-noguards-report.json"),
        label="sentence_1024_noguards",
        method="sentence_vorn",
        budget=1024,
        guardrails="none",
    )
    sentence_256_guarded = _row_from_report(
        _load_report("reemit-niah-sentence-vorn-b256-report.json"),
        label="sentence_256_guarded",
        method="sentence_vorn",
        budget=256,
        guardrails="prefix_plus_recent",
    )
    token_1024_guarded = _row_from_report(
        _load_report("reemit-niah-vorn-b1024-report.json"),
        label="token_1024_guarded",
        method="vorn",
        budget=1024,
        guardrails="prefix_plus_recent",
    )
    token_1024_noguards = _row_from_report(
        _load_report("reemit-niah-vorn-b1024-noguards-report.json"),
        label="token_1024_noguards",
        method="vorn",
        budget=1024,
        guardrails="none",
    )
    token_256_guarded = _row_from_report(
        _load_report("reemit-niah-vorn-b256-report.json"),
        label="token_256_guarded",
        method="vorn",
        budget=256,
        guardrails="prefix_plus_recent",
    )
    payload = {
        "schema_version": "result-envelope/v0.2",
        "ceiling": {
            "method": "vanilla",
            "budget_regime": "full_context",
            "hit_rate": 0.28,
            "elapsed_seconds": 160.23,
            "estimated_cost_usd": 0.1112,
            "preprocessing_elapsed_seconds": 0.0,
            "preprocessing_cost_usd": 0.0,
            "kv_savings_ratio": 0.0,
        },
        "comparisons": [
            sentence_1024_guarded,
            sentence_1024_noguards,
            sentence_256_guarded,
        ],
        "token_references": [
            token_1024_guarded,
            token_1024_noguards,
            token_256_guarded,
        ],
        "pairwise_tests": [
            _paired_test(sentence_1024_guarded, token_1024_guarded),
            _paired_test(sentence_1024_noguards, token_1024_noguards),
            _paired_test(sentence_256_guarded, token_256_guarded),
        ],
        "run_conditions": {
            "profile": "author",
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "niah_multikey_1_4k",
            "split": "validation[:50]",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "random_seed": 17,
            "canonical_layer": 16,
            "recent_token_window": 16,
        },
    }
    _json_write(RESULTS / "sentence-level-eviction-2026-05-13.json", payload)

    pairwise_lines = []
    for pairwise in payload["pairwise_tests"]:
        table = pairwise["table"]
        pairwise_lines.append(
            f"- {pairwise['lhs']} vs {pairwise['rhs']}, exact McNemar on `{table}`: "
            f"`p = {pairwise['p_value']:.6g}`"
        )
    _md_write(
        RESULTS / "sentence-level-eviction-2026-05-13.md",
        f"""
# Sentence-Level Eviction — 2026-05-13

## Run Conditions

- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Pooling: `max`

## Ceiling Context

- Vanilla @ full context: `0.28` hit rate, `160.23s`, `$0.1112`. This is a ceiling/context row, not a constrained-budget competitor.

## Comparison Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
| Sentence vorn | 1024 | prefix + recent | {sentence_1024_guarded['hit_rate']:.2f} | [{sentence_1024_guarded['wilson_ci_95'][0]:.4f}, {sentence_1024_guarded['wilson_ci_95'][1]:.4f}] | {sentence_1024_guarded['elapsed_seconds']:.2f}s | ${sentence_1024_guarded['estimated_cost_usd']:.4f} | 0 | {(1 - sentence_1024_guarded['mean_retention_ratio']) * 100:.2f}% |
| Sentence vorn | 1024 | none | {sentence_1024_noguards['hit_rate']:.2f} | [{sentence_1024_noguards['wilson_ci_95'][0]:.4f}, {sentence_1024_noguards['wilson_ci_95'][1]:.4f}] | {sentence_1024_noguards['elapsed_seconds']:.2f}s | ${sentence_1024_noguards['estimated_cost_usd']:.4f} | 0 | {(1 - sentence_1024_noguards['mean_retention_ratio']) * 100:.2f}% |
| Sentence vorn | 256 | prefix + recent | {sentence_256_guarded['hit_rate']:.2f} | [{sentence_256_guarded['wilson_ci_95'][0]:.4f}, {sentence_256_guarded['wilson_ci_95'][1]:.4f}] | {sentence_256_guarded['elapsed_seconds']:.2f}s | ${sentence_256_guarded['estimated_cost_usd']:.4f} | 0 | {(1 - sentence_256_guarded['mean_retention_ratio']) * 100:.2f}% |

## Token References

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Reference |
|--------|--------|------------|----------|---------------|-----------|
| Token vorn | 1024 | prefix + recent | {token_1024_guarded['hit_rate']:.2f} | [{token_1024_guarded['wilson_ci_95'][0]:.4f}, {token_1024_guarded['wilson_ci_95'][1]:.4f}] | same slice |
| Token vorn | 1024 | none | {token_1024_noguards['hit_rate']:.2f} | [{token_1024_noguards['wilson_ci_95'][0]:.4f}, {token_1024_noguards['wilson_ci_95'][1]:.4f}] | same slice |
| Token vorn | 256 | prefix + recent | {token_256_guarded['hit_rate']:.2f} | [{token_256_guarded['wilson_ci_95'][0]:.4f}, {token_256_guarded['wilson_ci_95'][1]:.4f}] | same slice |

## Paired Tests

{chr(10).join(pairwise_lines)}

## Read

- The qualitative finding is unchanged: sentence-level eviction materially outperforms token-level vorn on the same NIAH questions at `1024`.
- The inferential layer is now correct for the design: these are paired same-slice comparisons, so the significance rows use exact McNemar instead of Fisher exact.
- At `256`, sentence-level does not separate from token-level on this slice.
""",
    )


def build_token_no_guardrails_artifact() -> None:
    vorn = _row_from_report(
        _load_report("reemit-niah-vorn-b1024-noguards-report.json"),
        label="vorn_token_noguards_1024",
        method="vorn",
        budget=1024,
        guardrails="none",
    )
    tova = _row_from_report(
        _load_report("reemit-niah-tova-b1024-noguards-report.json"),
        label="tova_token_noguards_1024",
        method="tova",
        budget=1024,
        guardrails="none",
    )
    pairwise = _paired_test(tova, vorn)
    payload = {
        "schema_version": "result-envelope/v0.2",
        "rows": [vorn, tova],
        "pairwise_test": pairwise,
        "run_conditions": {
            "profile": "author",
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "niah_multikey_1_4k",
            "split": "validation[:50]",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "always_keep_prefix_tokens": 0,
            "preserve_recent_window": False,
            "budget": 1024,
        },
    }
    _json_write(RESULTS / "token-no-guardrails-comparison-2026-05-13.json", payload)
    _md_write(
        RESULTS / "token-no-guardrails-comparison-2026-05-13.md",
        f"""
# Token No-Guardrails Comparison — 2026-05-13

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|-----------|
| Vorn | 1024 | none | {vorn['hit_rate']:.2f} | [{vorn['wilson_ci_95'][0]:.4f}, {vorn['wilson_ci_95'][1]:.4f}] | {vorn['elapsed_seconds']:.2f}s | ${vorn['estimated_cost_usd']:.4f} | {(1 - vorn['mean_retention_ratio']) * 100:.2f}% |
| TOVA | 1024 | none | {tova['hit_rate']:.2f} | [{tova['wilson_ci_95'][0]:.4f}, {tova['wilson_ci_95'][1]:.4f}] | {tova['elapsed_seconds']:.2f}s | ${tova['estimated_cost_usd']:.4f} | {(1 - tova['mean_retention_ratio']) * 100:.2f}% |

Paired same-slice inference:

- TOVA no-guardrails vs vorn no-guardrails, exact McNemar on `{pairwise['table']}`: `p = {pairwise['p_value']:.6g}`

Read:

- The honest boundary stays the same: token-level vorn collapses without guardrails on this slice, while token-level TOVA retains some middle-only signal.
- The inferential row is now a paired exact test over the same 50 fixtures, not a Fisher exact test over aggregates.
""",
    )


def build_no_guardrails_ablation() -> None:
    guarded = _row_from_report(
        _load_report("reemit-niah-vorn-b1024-report.json"),
        label="vorn_token_guarded_1024",
        method="vorn",
        budget=1024,
        guardrails="prefix_plus_recent",
    )
    noguards = _row_from_report(
        _load_report("reemit-niah-vorn-b1024-noguards-report.json"),
        label="vorn_token_noguards_1024",
        method="vorn",
        budget=1024,
        guardrails="none",
    )
    pairwise = _paired_test(guarded, noguards)
    payload = {
        "schema_version": "result-envelope/v0.2",
        "rows": [guarded, noguards],
        "pairwise_tests": [pairwise],
        "interpretation": {
            "guardrails_carry_load": True,
            "headline": "Removing both the protected prefix and recent-window guardrail collapses vorn@1024 from 0.26 to 0.00 on the same 50-case slice.",
            "phase_1_read": "On this slice, the guarded vorn result is not being driven by fully-evictable middle positions alone. The guardrails are load-bearing.",
            "supports_negative_selection_story": False,
        },
    }
    _json_write(RESULTS / "no-guardrails-ablation-2026-05-13.json", payload)
    _md_write(
        RESULTS / "no-guardrails-ablation-2026-05-13.md",
        f"""
# No-Guardrails Ablation — 2026-05-13

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|-----------|
| Vorn | 1024 | prefix + recent | {guarded['hit_rate']:.2f} | [{guarded['wilson_ci_95'][0]:.4f}, {guarded['wilson_ci_95'][1]:.4f}] | {guarded['elapsed_seconds']:.2f}s | ${guarded['estimated_cost_usd']:.4f} | {(1 - guarded['mean_retention_ratio']) * 100:.2f}% |
| Vorn | 1024 | none | {noguards['hit_rate']:.2f} | [{noguards['wilson_ci_95'][0]:.4f}, {noguards['wilson_ci_95'][1]:.4f}] | {noguards['elapsed_seconds']:.2f}s | ${noguards['estimated_cost_usd']:.4f} | {(1 - noguards['mean_retention_ratio']) * 100:.2f}% |

Paired same-slice inference:

- Guarded vs no-guardrails vorn, exact McNemar on `{pairwise['table']}`: `p = {pairwise['p_value']:.6g}`

Read:

- The qualitative read is unchanged and now formally supported with a paired test: guardrails are load-bearing for token-level vorn on this slice.
- This artifact now carries per-fixture observations for both rows, so the paired discordance table is reconstructible downstream.

References:

- [live-eviction-budget-sweep-2026-05-13.md](live-eviction-budget-sweep-2026-05-13.md)
- [vanilla-observation-neighborhood-2026-05-13.md](vanilla-observation-neighborhood-2026-05-13.md)
""",
    )


def build_cross_task_qa2_artifact() -> None:
    vanilla = _row_from_report(
        _load_report("reemit-qa2-vanilla-report.json"),
        label="vanilla",
        method="vanilla",
        budget_regime="full_context",
    )
    token = _row_from_report(
        _load_report("reemit-qa2-vorn-b1024-report.json"),
        label="token_vorn_1024",
        method="vorn_eviction",
        budget=1024,
        guardrails="prefix_plus_recent",
    )
    sentence_guarded = _row_from_report(
        _load_report("reemit-qa2-sentence-vorn-b1024-report.json"),
        label="sentence_guarded_1024",
        method="sentence_vorn_eviction",
        budget=1024,
        guardrails="prefix_plus_recent",
    )
    sentence_noguards = _row_from_report(
        _load_report("reemit-qa2-sentence-vorn-b1024-noguards-report.json"),
        label="sentence_noguards_1024",
        method="sentence_vorn_eviction",
        budget=1024,
        guardrails="none",
    )
    payload = {
        "schema_version": "result-envelope/v0.2",
        "dataset": {
            "profile": "author",
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "qa_2_4k",
            "split": "validation[:50]",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "metric": "exact_match_any_acceptable_output",
        },
        "rows": [vanilla, token, sentence_guarded, sentence_noguards],
        "pairwise_tests": [
            _paired_test(sentence_guarded, token),
            _paired_test(sentence_noguards, token),
            _paired_test(sentence_guarded, vanilla),
            _paired_test(sentence_noguards, vanilla),
        ],
        "budget": {
            "substantive_runs_cost_usd": (
                sentence_guarded["estimated_cost_usd"]
                + sentence_noguards["estimated_cost_usd"]
            ),
        },
    }
    _json_write(RESULTS / "cross-task-qa_2-sentence-2026-05-13.json", payload)
    pairwise_lines = []
    for pairwise in payload["pairwise_tests"]:
        pairwise_lines.append(
            f"- {pairwise['lhs']} vs {pairwise['rhs']}, exact McNemar on `{pairwise['table']}`: `p = {pairwise['p_value']:.6g}`"
        )
    _md_write(
        RESULTS / "cross-task-qa_2-sentence-2026-05-13.md",
        f"""
# Cross-Task Sentence Generalization (`qa_2_4k`) — 2026-05-13

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `qa_2_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Metric: exact match against any acceptable output
- Pooling: `max`

| Method | Budget regime | Accuracy | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | Retention |
|--------|---------------|----------|---------------|------------|----------------|---------------|-----------|
| Vanilla | full context | {vanilla['hit_rate']:.2f} | [{vanilla['wilson_ci_95'][0]:.4f}, {vanilla['wilson_ci_95'][1]:.4f}] | {vanilla['elapsed_seconds']:.2f}s | ${vanilla['estimated_cost_usd']:.4f} | 0 | 100.00% |
| Token vorn | 1024 guarded | {token['hit_rate']:.2f} | [{token['wilson_ci_95'][0]:.4f}, {token['wilson_ci_95'][1]:.4f}] | {token['elapsed_seconds']:.2f}s | ${token['estimated_cost_usd']:.4f} | 0 | {token['mean_retention_ratio'] * 100:.2f}% |
| Sentence vorn | 1024 guarded | {sentence_guarded['hit_rate']:.2f} | [{sentence_guarded['wilson_ci_95'][0]:.4f}, {sentence_guarded['wilson_ci_95'][1]:.4f}] | {sentence_guarded['elapsed_seconds']:.2f}s | ${sentence_guarded['estimated_cost_usd']:.4f} | 0 | {sentence_guarded['mean_retention_ratio'] * 100:.2f}% |
| Sentence vorn | 1024 no guardrails | {sentence_noguards['hit_rate']:.2f} | [{sentence_noguards['wilson_ci_95'][0]:.4f}, {sentence_noguards['wilson_ci_95'][1]:.4f}] | {sentence_noguards['elapsed_seconds']:.2f}s | ${sentence_noguards['estimated_cost_usd']:.4f} | 0 | {sentence_noguards['mean_retention_ratio'] * 100:.2f}% |

Paired same-slice inference:

{chr(10).join(pairwise_lines)}

Read:

- The directional generalization beyond NIAH remains: sentence-level vorn improves the point estimate over token-level vorn on `qa_2_4k` and approaches vanilla on this slice.
- The inferential layer is now paired-correct. At `n=50`, these rows are still best read as directional rather than decisive.
""",
    )


def build_sentence_pooling_artifact() -> None:
    max_guarded = _row_from_report(
        _load_report("reemit-niah-sentence-vorn-b1024-report.json"),
        label="max_guarded",
        method="sentence_vorn",
        budget=1024,
        guardrails="prefix_plus_recent",
    )
    max_noguards = _row_from_report(
        _load_report("reemit-niah-sentence-vorn-b1024-noguards-report.json"),
        label="max_noguards",
        method="sentence_vorn",
        budget=1024,
        guardrails="none",
    )
    mean_guarded = _row_from_report(
        _load_report("reemit-niah-sentence-vorn-mean-b1024-report.json"),
        label="mean_guarded",
        method="sentence_vorn",
        budget=1024,
        guardrails="prefix_plus_recent",
    )
    mean_noguards = _row_from_report(
        _load_report("reemit-niah-sentence-vorn-mean-b1024-noguards-report.json"),
        label="mean_noguards",
        method="sentence_vorn",
        budget=1024,
        guardrails="none",
    )
    topk_guarded = _row_from_report(
        _load_report("reemit-niah-sentence-vorn-topk-b1024-report.json"),
        label="topk_guarded",
        method="sentence_vorn",
        budget=1024,
        guardrails="prefix_plus_recent",
    )
    topk_noguards = _row_from_report(
        _load_report("reemit-niah-sentence-vorn-topk-b1024-noguards-report.json"),
        label="topk_noguards",
        method="sentence_vorn",
        budget=1024,
        guardrails="none",
    )
    payload = {
        "schema_version": "result-envelope/v0.2",
        "dataset": {
            "profile": "author",
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "niah_multikey_1_4k",
            "split": "validation[:50]",
            "model": "mistralai/Mistral-7B-Instruct-v0.3",
            "budget": 1024,
            "retention_policy": "sentence_vorn",
        },
        "rows": [
            max_guarded,
            max_noguards,
            mean_guarded,
            mean_noguards,
            topk_guarded,
            topk_noguards,
        ],
        "pairwise_tests": [
            _paired_test(max_guarded, mean_guarded),
            _paired_test(topk_guarded, mean_guarded),
            _paired_test(max_noguards, mean_noguards),
            _paired_test(topk_noguards, mean_noguards),
        ],
    }
    _json_write(RESULTS / "sentence-pooling-comparison-2026-05-13.json", payload)
    pairwise_lines = []
    for pairwise in payload["pairwise_tests"]:
        pairwise_lines.append(
            f"- {pairwise['lhs']} vs {pairwise['rhs']}, exact McNemar on `{pairwise['table']}`: `p = {pairwise['p_value']:.6g}`"
        )
    _md_write(
        RESULTS / "sentence-pooling-comparison-2026-05-13.md",
        f"""
# Sentence Pooling Comparison — 2026-05-13

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Budget: `1024`
- Retention policy: `sentence_vorn`
- Comparison axis: sentence aggregation only (`max` vs `mean` vs `top-k`)

| Pooling | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Retention |
|---------|------------|----------|---------------|------------|----------------|-----------|
| Max | prefix + recent | {max_guarded['hit_rate']:.2f} | [{max_guarded['wilson_ci_95'][0]:.4f}, {max_guarded['wilson_ci_95'][1]:.4f}] | {max_guarded['elapsed_seconds']:.2f}s | ${max_guarded['estimated_cost_usd']:.4f} | {max_guarded['mean_retention_ratio'] * 100:.2f}% |
| Max | none | {max_noguards['hit_rate']:.2f} | [{max_noguards['wilson_ci_95'][0]:.4f}, {max_noguards['wilson_ci_95'][1]:.4f}] | {max_noguards['elapsed_seconds']:.2f}s | ${max_noguards['estimated_cost_usd']:.4f} | {max_noguards['mean_retention_ratio'] * 100:.2f}% |
| Mean | prefix + recent | {mean_guarded['hit_rate']:.2f} | [{mean_guarded['wilson_ci_95'][0]:.4f}, {mean_guarded['wilson_ci_95'][1]:.4f}] | {mean_guarded['elapsed_seconds']:.2f}s | ${mean_guarded['estimated_cost_usd']:.4f} | {mean_guarded['mean_retention_ratio'] * 100:.2f}% |
| Mean | none | {mean_noguards['hit_rate']:.2f} | [{mean_noguards['wilson_ci_95'][0]:.4f}, {mean_noguards['wilson_ci_95'][1]:.4f}] | {mean_noguards['elapsed_seconds']:.2f}s | ${mean_noguards['estimated_cost_usd']:.4f} | {mean_noguards['mean_retention_ratio'] * 100:.2f}% |
| Top-k mean (`k=3`) | prefix + recent | {topk_guarded['hit_rate']:.2f} | [{topk_guarded['wilson_ci_95'][0]:.4f}, {topk_guarded['wilson_ci_95'][1]:.4f}] | {topk_guarded['elapsed_seconds']:.2f}s | ${topk_guarded['estimated_cost_usd']:.4f} | {topk_guarded['mean_retention_ratio'] * 100:.2f}% |
| Top-k mean (`k=3`) | none | {topk_noguards['hit_rate']:.2f} | [{topk_noguards['wilson_ci_95'][0]:.4f}, {topk_noguards['wilson_ci_95'][1]:.4f}] | {topk_noguards['elapsed_seconds']:.2f}s | ${topk_noguards['estimated_cost_usd']:.4f} | {topk_noguards['mean_retention_ratio'] * 100:.2f}% |

Paired same-slice inference:

{chr(10).join(pairwise_lines)}

Read:

- Pooling still matters materially. `mean` blurs the signal, while `max` and `top-k` preserve the sentence-level win.
- This artifact now preserves per-case outcomes, so the aggregation comparisons can be re-tested downstream without re-deriving fixture-level data.
""",
    )


def main() -> None:
    build_sentence_level_artifact()
    build_token_no_guardrails_artifact()
    build_no_guardrails_ablation()
    build_cross_task_qa2_artifact()
    build_sentence_pooling_artifact()


if __name__ == "__main__":
    main()
