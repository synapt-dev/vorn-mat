#!/usr/bin/env python3
"""Modal-native parallel cell execution entrypoint.

ONE local lifecycle (single `modal run --detach`). N server-side function
invocations via binding.remote_fn.spawn(spec). Per-cell exception isolation
via per-handle .get() with capture. Modal scheduler handles GPU acquisition
up to max_containers on the function binding (default 10 per Layne directive
2026-05-23).

Replaces user-side parallelism patterns (ThreadPoolExecutor wrapping per-cell
`modal run`) which created N independent local-client lifecycles = N
independent disconnect-class failure points. The new pattern collapses that
to one.

Per-case persistence: each cell's run_modal_live_eviction_niah call appends
every completed CaseObservation to output_path.with_suffix(".observations.jsonl")
BEFORE the next case runs, so mid-cell kills preserve completed cases on the
Modal Volume mount.

Usage:
  modal run --detach examples/run_modal_cells_parallel.py \\
    --cell-spec-path .benchmarks/cell-specs.json

The cell spec file is a JSON list of dicts; each dict is the kwargs for one
ModalLiveEvictionRunRequest (dataset_config, retention_policy, cache_budget_tokens,
model_id, output_path, etc.). One element per cell.
"""
# ruff: noqa: E402

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import modal

from vorn_mat import (
    ModalLiveEvictionRunRequest,
    build_vorn_entrypoint,
    collect_cells_parallel,
    run_modal_live_eviction_niah,
)


binding = build_vorn_entrypoint(run_modal_live_eviction_niah, modal_module=modal)
app = binding.app


@app.local_entrypoint()
def main(
    cell_spec_path: str,
    output_dir: str = str(ROOT / ".benchmarks" / "parallel-cells"),
) -> None:
    cell_specs = json.loads(Path(cell_spec_path).read_text())
    requests = [ModalLiveEvictionRunRequest(**spec) for spec in cell_specs]

    report = collect_cells_parallel(binding, requests)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "reports.json").write_text(
        json.dumps([asdict(r) for r in report.reports], indent=2, sort_keys=True)
    )
    if report.failures:
        (output_path / "failures.json").write_text(
            json.dumps(
                [
                    {"request": asdict(f.request), "error": f.error}
                    for f in report.failures
                ],
                indent=2,
                sort_keys=True,
            )
        )

    print(f"cells_fired={len(requests)}")
    print(f"cells_succeeded={len(report.reports)}")
    print(f"cells_failed={len(report.failures)}")
    print(f"output_dir={output_path}")
