#!/usr/bin/env python3
"""Build the cross-model gate artifact for the first non-Mistral probe."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / ".benchmarks" / "cross-model"
RESULTS = ROOT / "results"

QWEN_ROWS = (
    ("Vanilla", "qwen-4k-vanilla.json", "full context"),
    ("Token vorn", "qwen-4k-token-1024.json", "1024 guarded"),
    ("Sentence vorn", "qwen-4k-sentence-1024-guarded.json", "1024 guarded"),
    ("Sentence vorn", "qwen-4k-sentence-1024-noguards.json", "1024 no guardrails"),
)


def load_report(name: str) -> dict:
    return json.loads((BENCHMARKS / name).read_text())


def summarize_row(label: str, filename: str, budget_regime: str) -> dict:
    report = load_report(filename)
    observations = report["result"].get("observations", [])
    bad_predictions = [
        observation["prediction"]
        for observation in observations
        if not observation["correct"]
    ][:3]
    return {
        "method": label,
        "budget_regime": budget_regime,
        "run_id": report["result"]["run_id"],
        "hit_rate": report["result"]["metrics"]["needle_hit_rate"],
        "elapsed_seconds": report["elapsed_seconds"],
        "estimated_cost_usd": report["estimated_cost_usd"],
        "model_id": report["result"]["metadata"]["model_id"],
        "sample_incorrect_predictions": bad_predictions,
    }


def write_artifacts() -> tuple[Path, Path]:
    qwen_rows = [
        summarize_row(label, filename, budget_regime)
        for label, filename, budget_regime in QWEN_ROWS
    ]

    artifact = {
        "artifact": "cross-model-gate",
        "date": "2026-05-13",
        "dataset_id": "rbiswasfc/ruler",
        "dataset_config": "niah_multikey_1_4k",
        "split": "validation[:50]",
        "models": {
            "qwen": {
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "status": "completed",
                "read": (
                    "Non-discriminative on this slice. Vanilla ceiling is 0.00, "
                    "so the sweet-spot comparison cannot support a cross-model "
                    "generalization claim."
                ),
                "rows": qwen_rows,
            },
            "llama": {
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "status": "blocked",
                "failure": {
                    "type": "gated_repo",
                    "summary": (
                        "Modal Hugging Face secret lacks access to the Meta "
                        "Llama 3.1 8B Instruct gated repository (403)."
                    ),
                },
            },
        },
        "read": (
            "This lane delivered the model-selection seam and a first external-model "
            "gate, but it did not produce a publishable architecture-generalization "
            "result. Qwen 2.5 7B is non-discriminative on the NIAH 4k sweet-spot "
            "slice, and Llama 3.1 8B is blocked by model access."
        ),
        "next_options": (
            "Either add a model-specific prompt/eval compatibility pass for Qwen, "
            "obtain Llama access, or switch the cross-model validity lane to a "
            "different open-weight instruct model with demonstrated vanilla headroom "
            "on the target slice."
        ),
    }

    markdown_lines = [
        "# Cross-Model Gate (`Qwen 2.5 7B`, `Llama 3.1 8B`) — 2026-05-13",
        "",
        "Run conditions:",
        "- Profile: `author`",
        "- Dataset: `rbiswasfc/ruler`",
        "- Config: `niah_multikey_1_4k`",
        "- Slice: `validation[:50]`",
        "- Budget target: `1024` sweet-spot gate on the 4k slice",
        "",
        "| Model | Method | Budget regime | Hit rate | Wall-clock | Inference cost |",
        "|------|--------|---------------|----------|------------|----------------|",
    ]
    for row in qwen_rows:
        markdown_lines.append(
            "| "
            f"`{row['model_id']}` | {row['method']} | {row['budget_regime']} | "
            f"{row['hit_rate']:.2f} | {row['elapsed_seconds']:.2f}s | "
            f"${row['estimated_cost_usd']:.4f} |"
        )
    markdown_lines.extend(
        [
            "",
            "Qwen read:",
            "",
            "- This is not a valid cross-model architecture comparison on this slice because the full-context ceiling is already `0.00`.",
            "- The first incorrect predictions are visibly garbled rather than merely wrong answers, which points to a model-specific prompt/runtime compatibility problem in the current runner surface:",
        ]
    )
    for row in qwen_rows[:1]:
        for prediction in row["sample_incorrect_predictions"]:
            markdown_lines.append(f"  - `{prediction}`")
    markdown_lines.extend(
        [
            "",
            "Llama gate:",
            "",
            "- `meta-llama/Llama-3.1-8B-Instruct` could not be run because the Modal Hugging Face secret does not have access to the gated repository (`403`).",
            "",
            "Read:",
            "",
            "- The lane still paid off technically: the Modal runner now accepts `model_id` explicitly, tags remote outputs by model, and can support future cross-model sweeps without ad hoc code edits.",
            "- The empirical result is narrower: cross-model validity remains open. Qwen 2.5 7B is non-discriminative on the target NIAH slice, and Llama 3.1 8B is operationally blocked.",
            "- The honest next step is not to overclaim generalization. It is to either fix Qwen compatibility, obtain Llama access, or select a different open model with demonstrated vanilla headroom first.",
        ]
    )

    md_path = RESULTS / "cross-model-gate-2026-05-13.md"
    json_path = RESULTS / "cross-model-gate-2026-05-13.json"
    md_path.write_text("\n".join(markdown_lines) + "\n")
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return md_path, json_path


def main() -> None:
    md_path, json_path = write_artifacts()
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
