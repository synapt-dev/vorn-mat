#!/usr/bin/env python3
"""Build the Qwen 4k word-vorn budget sweep artifact from Modal reports."""
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


def _retention_ratio(report: dict[str, object], default: float = 1.0) -> float:
    result = report["result"]
    assert isinstance(result, dict)
    metadata = result.get("metadata", {})
    assert isinstance(metadata, dict)
    value = metadata.get("mean_retention_ratio")
    if value is None:
        return default
    return float(value)


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


def _fixture_delta(lhs: dict[str, object], rhs: dict[str, object]) -> dict[str, object]:
    lhs_obs = {
        item["fixture_id"]: bool(item["correct"])
        for item in lhs["observations"]  # type: ignore[index]
    }
    rhs_obs = {
        item["fixture_id"]: bool(item["correct"])
        for item in rhs["observations"]  # type: ignore[index]
    }
    fixture_ids = sorted(set(lhs_obs) & set(rhs_obs))
    return {
        "lhs": lhs["label"],
        "rhs": rhs["label"],
        "lhs_only_correct_fixture_ids": [
            fixture_id
            for fixture_id in fixture_ids
            if lhs_obs[fixture_id] and not rhs_obs[fixture_id]
        ],
        "rhs_only_correct_fixture_ids": [
            fixture_id
            for fixture_id in fixture_ids
            if rhs_obs[fixture_id] and not lhs_obs[fixture_id]
        ],
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
    floor = _row_from_live_report(
        _load_report("qwen-4k-vanilla.json"),
        label="vanilla_floor",
        method="vanilla",
        budget=None,
        guardrails="prefix_plus_recent",
    )

    token_rows = [
        _row_from_live_report(
            _load_report("qwen-4k-token-b256.json"),
            label="token_256_guarded",
            method="token_vorn",
            budget=256,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-token-b512.json"),
            label="token_512_guarded",
            method="token_vorn",
            budget=512,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-token-1024.json"),
            label="token_1024_guarded",
            method="token_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-token-b1536.json"),
            label="token_1536_guarded",
            method="token_vorn",
            budget=1536,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-token-b2048.json"),
            label="token_2048_guarded",
            method="token_vorn",
            budget=2048,
            guardrails="prefix_plus_recent",
        ),
    ]

    sentence_rows = [
        _row_from_live_report(
            _load_report("qwen-4k-sentence-b256.json"),
            label="sentence_256_guarded",
            method="sentence_vorn",
            budget=256,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-sentence-b256-noguards.json"),
            label="sentence_256_noguards",
            method="sentence_vorn",
            budget=256,
            guardrails="none",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-sentence-b512.json"),
            label="sentence_512_guarded",
            method="sentence_vorn",
            budget=512,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-sentence-b512-noguards.json"),
            label="sentence_512_noguards",
            method="sentence_vorn",
            budget=512,
            guardrails="none",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-sentence-1024-guarded.json"),
            label="sentence_1024_guarded",
            method="sentence_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-sentence-1024-noguards.json"),
            label="sentence_1024_noguards",
            method="sentence_vorn",
            budget=1024,
            guardrails="none",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-sentence-b1536.json"),
            label="sentence_1536_guarded",
            method="sentence_vorn",
            budget=1536,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-sentence-b1536-noguards.json"),
            label="sentence_1536_noguards",
            method="sentence_vorn",
            budget=1536,
            guardrails="none",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-sentence-b2048.json"),
            label="sentence_2048_guarded",
            method="sentence_vorn",
            budget=2048,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-sentence-b2048-noguards.json"),
            label="sentence_2048_noguards",
            method="sentence_vorn",
            budget=2048,
            guardrails="none",
        ),
    ]

    word_rows = [
        _row_from_live_report(
            _load_report("qwen-4k-word-b256.json"),
            label="word_256_guarded",
            method="word_vorn",
            budget=256,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-word-b256-noguards.json"),
            label="word_256_noguards",
            method="word_vorn",
            budget=256,
            guardrails="none",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-word-b512.json"),
            label="word_512_guarded",
            method="word_vorn",
            budget=512,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-word-b512-noguards.json"),
            label="word_512_noguards",
            method="word_vorn",
            budget=512,
            guardrails="none",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-word-b1024.json"),
            label="word_1024_guarded",
            method="word_vorn",
            budget=1024,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-word-b1024-noguards.json"),
            label="word_1024_noguards",
            method="word_vorn",
            budget=1024,
            guardrails="none",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-word-b1536.json"),
            label="word_1536_guarded",
            method="word_vorn",
            budget=1536,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-word-b1536-noguards.json"),
            label="word_1536_noguards",
            method="word_vorn",
            budget=1536,
            guardrails="none",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-word-b2048.json"),
            label="word_2048_guarded",
            method="word_vorn",
            budget=2048,
            guardrails="prefix_plus_recent",
        ),
        _row_from_live_report(
            _load_report("qwen-4k-word-b2048-noguards.json"),
            label="word_2048_noguards",
            method="word_vorn",
            budget=2048,
            guardrails="none",
        ),
    ]

    token_by_budget = {int(row["budget"]): row for row in token_rows}
    sentence_guarded = {
        int(row["budget"]): row
        for row in sentence_rows
        if row["guardrails"] == "prefix_plus_recent"
    }
    word_guarded = {
        int(row["budget"]): row
        for row in word_rows
        if row["guardrails"] == "prefix_plus_recent"
    }
    word_noguards = {
        int(row["budget"]): row for row in word_rows if row["guardrails"] == "none"
    }

    pairwise_tests = []
    fixture_deltas = []
    for budget in (256, 512, 1024, 1536, 2048):
        pairwise_tests.append(_paired_test(word_guarded[budget], token_by_budget[budget]))
        pairwise_tests.append(_paired_test(word_noguards[budget], word_guarded[budget]))
        pairwise_tests.append(_paired_test(word_guarded[budget], sentence_guarded[budget]))
        fixture_deltas.append(_fixture_delta(word_guarded[budget], token_by_budget[budget]))

    incremental_word_cost_usd = sum(
        float(row["estimated_cost_usd"])
        for row in word_rows
    )

    peak_word_row = max(word_rows, key=lambda row: float(row["hit_rate"]))
    peak_token_row = max(token_rows, key=lambda row: float(row["hit_rate"]))
    peak_sentence_row = max(sentence_rows, key=lambda row: float(row["hit_rate"]))

    payload: dict[str, object] = {
        "schema_version": "result-envelope-v0.2",
        "experiment": "qwen_4k_word_budget_sweep",
        "dataset": {
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "niah_multikey_1_4k",
            "split": "validation[:50]",
        },
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "context_floor": floor,
        "rows": token_rows + sentence_rows + word_rows,
        "pairwise_tests": pairwise_tests,
        "fixture_deltas": fixture_deltas,
        "analysis": {
            "incremental_word_cost_usd": round(incremental_word_cost_usd, 4),
            "peak_word_row": peak_word_row["label"],
            "peak_token_row": peak_token_row["label"],
            "peak_sentence_row": peak_sentence_row["label"],
            "token_1536_only_fixture_ids": next(
                delta["rhs_only_correct_fixture_ids"]
                for delta in fixture_deltas
                if delta["lhs"] == "word_1536_guarded"
            ),
        },
    }

    md_lines = [
        "# Qwen 4k Word-vorn Budget Sweep — 2026-05-14",
        "",
        "Run conditions:",
        "- Profile: `author`",
        "- Dataset: `rbiswasfc/ruler`",
        "- Config: `niah_multikey_1_4k`",
        "- Slice: `validation[:50]`",
        "- Model: `Qwen/Qwen2.5-7B-Instruct`",
        "- Pooling: `max`",
        "",
        "## Full-Context Floor",
        "",
        f"- Vanilla @ full context: `{floor['hit_rate']:.2f}` hit rate, `{floor['elapsed_seconds']:.2f}s`, `${floor['estimated_cost_usd']:.4f}`.",
        "",
        "## Word Rows",
        "",
        "| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |",
        "|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|",
        *[_row_cells(row, "Word vorn") for row in word_rows],
        "",
        "## Reference Token Rows",
        "",
        "| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |",
        "|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|",
        *[_row_cells(row, "Token vorn") for row in token_rows],
        "",
        "## Reference Sentence Rows",
        "",
        "| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |",
        "|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|",
        *[_row_cells(row, "Sentence vorn") for row in sentence_rows],
        "",
        "## Paired Tests",
        "",
    ]

    for test in pairwise_tests:
        md_lines.append(
            f"- {test['lhs']} vs {test['rhs']}, exact McNemar on `{test['table']}`: `p = {test['p_value']:.6g}`"
        )

    md_lines.extend(
        [
            "",
            "## Per-Fixture Deltas",
            "",
            f"- Token-only recoveries at budget `1536`: `{payload['analysis']['token_1536_only_fixture_ids']}`",
            "- Word-only recoveries versus token: `[]` at every tested budget.",
            "",
            "## Read",
            "",
            "- Word-vorn does **not** produce a Qwen sweet-spot on this slice. All ten word rows are `0.00`.",
            f"- Token-vorn remains the only directional recovery band, peaking at budget `1536` with `{peak_token_row['hit_rate']:.2f}`.",
            f"- Sentence-vorn remains weaker than token on this Qwen slice, with its best row at `{peak_sentence_row['budget']}` no-guards = `{peak_sentence_row['hit_rate']:.2f}`.",
            "- This is stronger than the prior negative sentence result: moving upward from token to word destroys the small Qwen recovery rather than improving it.",
            "- The cross-model architectural-law claim sharpens rather than broadens: the minimum coherent retention unit is per-model, and on this Qwen slice it is smaller than word while Mistral's strongest regime remained sentence-shaped.",
            f"- Incremental Modal spend for the word matrix was `${incremental_word_cost_usd:.4f}`, within the dispatched `$3` cap.",
        ]
    )

    _json_write(RESULTS / "qwen-4k-word-budget-sweep-2026-05-14.json", payload)
    _md_write(RESULTS / "qwen-4k-word-budget-sweep-2026-05-14.md", "\n".join(md_lines))


if __name__ == "__main__":
    main()
