#!/usr/bin/env python3
"""Build the 8k score-distribution observation artifact."""
# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "results"
BENCHMARKS = ROOT / ".benchmarks"


RUN_SPECS = (
    {
        "budget": 512,
        "oracle_granularity": "token",
        "retention_policy": "vorn",
        "path": BENCHMARKS / "score-dist-8k-b512-token.json",
    },
    {
        "budget": 1024,
        "oracle_granularity": "sentence",
        "retention_policy": "sentence_vorn",
        "path": BENCHMARKS / "score-dist-8k-b1024-sentence.json",
    },
    {
        "budget": 1536,
        "oracle_granularity": "sentence",
        "retention_policy": "sentence_vorn",
        "path": BENCHMARKS / "score-dist-8k-b1536-sentence.json",
    },
    {
        "budget": 2048,
        "oracle_granularity": "token",
        "retention_policy": "vorn",
        "path": BENCHMARKS / "score-dist-8k-b2048-token.json",
    },
)

SUMMARY_KEYS = (
    "position_count",
    "peak_zscore",
    "top10_mass_fraction",
    "top25_mass_fraction",
    "top50_mass_fraction",
    "normalized_entropy",
    "kl_divergence_from_uniform",
    "q90_minus_q50",
    "q75_minus_q25",
    "above_median_plus_std_fraction",
    "spatial_coherence",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _md_write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n")


def _hit_rate(report: dict[str, Any]) -> float:
    cases = report["cases"]
    if not cases:
        return 0.0
    hits = sum(
        1
        for case in cases
        if case["observations"] and case["observations"][0]["correct"]
    )
    return hits / len(cases)


def _aggregate_granularity(report: dict[str, Any], *, granularity: str) -> dict[str, float]:
    samples: list[dict[str, Any]] = []
    for case in report["cases"]:
        for step in case["steps"]:
            samples.append(step["granularity_stats"][granularity])
    if not samples:
        raise ValueError(f"no samples for granularity {granularity}")
    return {
        "step_count": len(samples),
        **{
            f"mean_{key}": float(
                statistics.fmean(float(sample[key]) for sample in samples)
            )
            for key in SUMMARY_KEYS
        },
    }


def main() -> None:
    budget_runs: list[dict[str, Any]] = []
    for spec in RUN_SPECS:
        report = _load_json(spec["path"])
        budget_runs.append(
            {
                "budget": spec["budget"],
                "oracle_granularity": spec["oracle_granularity"],
                "retention_policy": spec["retention_policy"],
                "hit_rate": _hit_rate(report),
                "elapsed_seconds": float(report["elapsed_seconds"]),
                "estimated_cost_usd": float(report["estimated_cost_usd"]),
                "always_keep_prefix_tokens": int(report["always_keep_prefix_tokens"]),
                "preserve_recent_window": bool(report["preserve_recent_window"]),
                "sentence_pooling": str(report["sentence_pooling"]),
                "sentence_top_k": int(report["sentence_top_k"]),
                "model_id": str(report["model_id"]),
                "observations": [
                    observation
                    for case in report["cases"]
                    for observation in case["observations"]
                ],
                "aggregates": {
                    granularity: _aggregate_granularity(report, granularity=granularity)
                    for granularity in ("token", "word", "sentence")
                },
                "cases": report["cases"],
                "source_report": str(spec["path"].relative_to(ROOT)),
            }
        )

    payload = {
        "schema_version": "result-envelope/v0.2+distribution-v1",
        "artifact": "score-distribution-observation-8k",
        "date": "2026-05-14",
        "run_conditions": {
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "niah_multikey_1_8k",
            "split": "validation[:50]",
            "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
            "canonical_layer": 16,
            "recent_token_window": 16,
            "word_and_sentence_pooling": "max",
            "word_and_sentence_top_k": 3,
            "measurement_mode": (
                "live budgeted observation on current retained state; "
                "metrics logged before each eviction decision and not used for selection"
            ),
        },
        "budget_runs": budget_runs,
        "initial_findings": [
            "Peak-zscore does not track the oracle winner. Token or word has the highest mean peak-zscore at every budget, including the sentence-oracle mid-band.",
            "Sentence-oracle budgets 1024 and 1536 are the regimes where sentence-level mass concentration and tail spread separate most strongly from token-level, while the token-edge budgets 512 and 2048 show materially smaller sentence-vs-token gaps on those same metrics.",
            "Word-level score shape shadows token-level score shape throughout this sweep and does not surface a distinct winning regime.",
        ],
        "read": (
            "This artifact characterizes how token, word, and sentence alignment-score "
            "distributions differ on the same live retained state at the 8k oracle "
            "budget boundaries. It is observational only: no selector uses these "
            "metrics yet."
        ),
    }

    lines = [
        "# Score Distribution Observation — 8k — 2026-05-14",
        "",
        "Run conditions:",
        "- Dataset: `rbiswasfc/ruler`",
        "- Config: `niah_multikey_1_8k`",
        "- Slice: `validation[:50]`",
        "- Model: `mistralai/Mistral-7B-Instruct-v0.3`",
        "- Vorn measurement: canonical layer `L*=16`, recent window `W=16`",
        "- Word/sentence aggregation: `max`",
        "- Observation mode: score-shape metrics logged before each live eviction decision; metrics did not drive selection",
        "",
        "## Initial findings",
        "",
        "- Peak-zscore does not track the oracle winner. Token or word has the highest mean peak-zscore at every budget, including the sentence-oracle mid-band.",
        "- Sentence-oracle budgets `1024` and `1536` are the regimes where sentence-level mass concentration and tail spread separate most strongly from token-level. The token-edge budgets `512` and `2048` show materially smaller sentence-vs-token gaps on those same metrics.",
        "- Word-level score shape shadows token-level score shape throughout this sweep and does not surface a distinct winning regime.",
        "",
        "## Oracle Runs",
        "",
        "| Budget | Oracle granularity | Retention policy | Hit rate | Wall-clock | Cost |",
        "|--------|--------------------|------------------|----------|------------|------|",
    ]
    for run in budget_runs:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run["budget"]),
                    str(run["oracle_granularity"]),
                    str(run["retention_policy"]),
                    f"{run['hit_rate']:.2f}",
                    f"{run['elapsed_seconds']:.2f}s",
                    f"${run['estimated_cost_usd']:.4f}",
                ]
            )
            + " |"
        )

    for run in budget_runs:
        lines.extend(
            [
                "",
                f"## Budget {run['budget']} ({run['oracle_granularity']} oracle)",
                "",
                "| Granularity | Step count | Mean positions | Mean peak-z | Mean top10 mass | Mean entropy(norm) | Mean KL-from-uniform | Mean q90-q50 | Mean above-threshold frac | Mean spatial coherence |",
                "|-------------|------------|----------------|-------------|-----------------|--------------------|----------------------|--------------|---------------------------|------------------------|",
            ]
        )
        for granularity in ("token", "word", "sentence"):
            aggregate = run["aggregates"][granularity]
            lines.append(
                "| "
                + " | ".join(
                    [
                        granularity,
                        str(int(aggregate["step_count"])),
                        f"{aggregate['mean_position_count']:.2f}",
                        f"{aggregate['mean_peak_zscore']:.3f}",
                        f"{aggregate['mean_top10_mass_fraction']:.3f}",
                        f"{aggregate['mean_normalized_entropy']:.3f}",
                        f"{aggregate['mean_kl_divergence_from_uniform']:.3f}",
                        f"{aggregate['mean_q90_minus_q50']:.3f}",
                        f"{aggregate['mean_above_median_plus_std_fraction']:.3f}",
                        f"{aggregate['mean_spatial_coherence']:.3f}",
                    ]
                )
                + " |"
            )

    _json_write(RESULTS / "score-distribution-observation-8k-2026-05-14.json", payload)
    _md_write(RESULTS / "score-distribution-observation-8k-2026-05-14.md", "\n".join(lines))


if __name__ == "__main__":
    main()
