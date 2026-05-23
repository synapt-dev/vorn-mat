#!/usr/bin/env python3
"""Build per-family canonical matrix-backfill artifacts.

Walks .benchmarks/matrix-backfill-2026-05-23/{family}/ (produced by
examples/run_modal_cells_parallel.py or any matrix-shaped entrypoint that
emits per-cell JSON reports) and assembles per-family artifacts at
results/{family}-4k-matrix-backfill-2026-05-23.json in the
result-envelope/v0.2 schema, plus a companion .md summary.

OOM markers (matrix-backfill-cell-marker/v1) are surfaced under
`excluded_cells` with reason.

Method labels carry the historical `_style` suffix (tova_style,
sentence_tova_style, h2o_style, sentence_h2o_style) for downstream extractor
compatibility with cross-family canonical artifacts; the unsuffixed
retention_policy value (tova / sentence_tova / h2o / sentence_h2o) is also
persisted in row metadata so the harness contract round-trips.

Premium boundary: pure OSS. Artifact assembly from canonical Modal reports
is a local primitive with no identity or org semantics.

Usage:
  python build_matrix_backfill_artifact.py                  # all families
  python build_matrix_backfill_artifact.py --family mistral
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BENCH_ROOT = ROOT / ".benchmarks" / "matrix-backfill-2026-05-23"
RESULTS = ROOT / "results"

# Maps retention_policy to display-label method name used by downstream
# extractors (which prefer the historical _style suffix where applicable).
DISPLAY_METHOD = {
    "vorn": "token_vorn",
    "sentence_vorn": "sentence_vorn",
    "tova": "tova_style",
    "sentence_tova": "sentence_tova_style",
    "h2o": "h2o_style",
    "sentence_h2o": "sentence_h2o_style",
    "vanilla": "vanilla",
}

FAMILY_DISPLAY = {
    "mistral": "Mistral 7B v0.3",
    "llama31": "Llama 3.1 8B",
    "ministral": "Ministral 8B",
    "gemma2": "Gemma 2 9B",
    "gemma4": "Gemma 4 E4B-it",
    "qwen25": "Qwen 2.5 7B Instruct",
    "gemma3": "Gemma 3 12B-it",
}


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


@dataclass(frozen=True)
class CellArtifact:
    cell_key: str
    family: str
    retention_policy: str
    cache_budget_tokens: int | None
    report: dict


def _load_cell_reports(family: str) -> tuple[list[CellArtifact], list[dict]]:
    family_dir = BENCH_ROOT / family
    cells: list[CellArtifact] = []
    markers: list[dict] = []
    if not family_dir.exists():
        return cells, markers
    for path in sorted(family_dir.iterdir()):
        if path.suffix != ".json":
            continue
        if path.name.endswith(".oom-marker.json"):
            markers.append(json.loads(path.read_text()))
            continue
        report = json.loads(path.read_text())
        stem = path.stem
        rest = stem.removeprefix(f"{family}-")
        if rest == "vanilla":
            retention_policy = "vanilla"
            budget: int | None = None
        else:
            parts = rest.rsplit("-", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                raise SystemExit(f"unrecognized cell filename: {path}")
            retention_policy = parts[0]
            budget = int(parts[1])
        cells.append(
            CellArtifact(
                cell_key=stem,
                family=family,
                retention_policy=retention_policy,
                cache_budget_tokens=budget,
                report=report,
            )
        )
    return cells, markers


def _build_row(cell: CellArtifact) -> dict:
    report = cell.report
    result = report["result"]
    observations = result.get("observations", [])
    hits = sum(1 for obs in observations if obs.get("correct"))
    cases = len(observations)
    metrics = result.get("metrics", {})
    if len(metrics) != 1:
        raise SystemExit(
            f"{cell.cell_key}: expected exactly one metric, got {metrics!r}"
        )
    metric_name, hit_rate = next(iter(metrics.items()))
    metadata = result.get("metadata", {}) or {}
    mean_retention_ratio = float(metadata.get("mean_retention_ratio", 1.0))
    label = DISPLAY_METHOD.get(cell.retention_policy, cell.retention_policy)
    if cell.cache_budget_tokens is not None:
        label = f"{label}_{cell.cache_budget_tokens}_guarded"
    return {
        "label": label,
        "method": DISPLAY_METHOD.get(cell.retention_policy, cell.retention_policy),
        "metric": metric_name,
        "budget": cell.cache_budget_tokens,
        "guardrails": "prefix_plus_recent",
        "hits": hits,
        "cases": cases,
        "hit_rate": float(hit_rate),
        "wilson_ci_95": list(_wilson_ci(hits, cases)),
        "elapsed_seconds": float(report.get("elapsed_seconds", 0.0)),
        "estimated_cost_usd": float(report.get("estimated_cost_usd", 0.0)),
        "preprocessing_elapsed_seconds": float(
            result.get("preprocessing_elapsed_seconds", 0.0)
        ),
        "preprocessing_cost_usd": float(
            result.get("preprocessing_cost_usd", 0.0)
        ),
        "mean_retention_ratio": mean_retention_ratio,
        "run_id": str(result.get("run_id", "")),
        "observations": list(observations),
        "retention_policy": cell.retention_policy,
        "model_id": str(metadata.get("model_id", "")),
    }


def _build_family_artifact(family: str) -> dict | None:
    cells, markers = _load_cell_reports(family)
    if not cells and not markers:
        return None
    floor = None
    sentence_rows: list[dict] = []
    token_rows: list[dict] = []
    for cell in cells:
        row = _build_row(cell)
        if cell.retention_policy == "vanilla":
            floor = row
            continue
        if cell.retention_policy.startswith("sentence_"):
            sentence_rows.append(row)
        else:
            token_rows.append(row)
    sentence_rows.sort(key=lambda r: (r["method"], r["budget"] or 0))
    token_rows.sort(key=lambda r: (r["method"], r["budget"] or 0))
    excluded = [
        {
            "cell_key": m.get("cell_key"),
            "method": DISPLAY_METHOD.get(
                m.get("retention_policy", ""), m.get("retention_policy", "")
            ),
            "retention_policy": m.get("retention_policy"),
            "budget": m.get("cache_budget_tokens"),
            "error_type": m.get("error_type"),
            "error_text": m.get("error_text"),
            "model_id": m.get("model_id"),
        }
        for m in markers
    ]
    excluded.sort(key=lambda e: (e.get("method") or "", e.get("budget") or 0))
    return {
        "schema_version": "result-envelope/v0.2",
        "provenance": {
            "build_script": "examples/build_matrix_backfill_artifact.py",
            "orchestrator": "examples/run_modal_cells_parallel.py",
            "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "family_id": family,
            "family_display": FAMILY_DISPLAY.get(family, family),
        },
        "floor": floor,
        "sentence_rows": sentence_rows,
        "token_rows": token_rows,
        "excluded_cells": excluded,
        "run_conditions": {
            "profile": "layne",
            "dataset_id": "rbiswasfc/ruler",
            "dataset_config": "niah_multikey_1_4k",
            "split": "validation",
            "case_limit": 50,
            "random_seed": 17,
            "guardrails": "prefix_plus_recent",
            "max_new_tokens": 32,
        },
    }


def _format_md(family: str, artifact: dict) -> str:
    display = FAMILY_DISPLAY.get(family, family)
    lines = [
        f"# {display} 4k matrix backfill",
        "",
        f"Family: `{family}`",
        f"Built: {artifact['provenance']['built_at']}",
        "",
        "## Floor",
        "",
    ]
    if artifact.get("floor") is None:
        lines.append("_no vanilla cell in this family's backfill scope_")
    else:
        floor = artifact["floor"]
        lines.append(
            f"`{floor['method']}` ({floor.get('hits', 0)}/{floor.get('cases', 0)})"
            f" = {floor['hit_rate']:.4f}"
        )
    lines.append("")
    lines.append("## Cells")
    lines.append("")
    lines.append("| method | budget | hit_rate | hits/cases | ci_low | ci_high |")
    lines.append("|---|---|---|---|---|---|")
    for row in artifact.get("sentence_rows", []) + artifact.get("token_rows", []):
        ci = row.get("wilson_ci_95", [0.0, 0.0])
        lines.append(
            f"| `{row['method']}` | {row['budget']} | {row['hit_rate']:.4f} "
            f"| {row['hits']}/{row['cases']} | {ci[0]:.4f} | {ci[1]:.4f} |"
        )
    excluded = artifact.get("excluded_cells", [])
    if excluded:
        lines.append("")
        lines.append("## Excluded cells (OOM / timeout / harness-error)")
        lines.append("")
        lines.append("| cell_key | method | budget | error_type |")
        lines.append("|---|---|---|---|")
        for ex in excluded:
            lines.append(
                f"| `{ex['cell_key']}` | `{ex.get('method')}` "
                f"| {ex.get('budget')} | `{ex.get('error_type')}` |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        choices=sorted(FAMILY_DISPLAY.keys()),
    )
    args = parser.parse_args()
    families = [args.family] if args.family else sorted(FAMILY_DISPLAY.keys())
    RESULTS.mkdir(parents=True, exist_ok=True)
    built = 0
    for family in families:
        artifact = _build_family_artifact(family)
        if artifact is None:
            print(f"# {family}: no cell reports, skipping")
            continue
        json_path = RESULTS / f"{family}-4k-matrix-backfill-2026-05-23.json"
        md_path = RESULTS / f"{family}-4k-matrix-backfill-2026-05-23.md"
        json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
        md_path.write_text(_format_md(family, artifact))
        n_rows = len(artifact.get("sentence_rows", [])) + len(
            artifact.get("token_rows", [])
        )
        n_excl = len(artifact.get("excluded_cells", []))
        floor_str = "ok" if artifact.get("floor") else "."
        print(
            f"# {family}: floor={floor_str} rows={n_rows} excluded={n_excl} "
            f"-> {json_path.relative_to(ROOT)}"
        )
        built += 1
    return 0 if built else 1


if __name__ == "__main__":
    sys.exit(main())
