#!/usr/bin/env python3
"""Modal-native parallel cell execution entrypoint (Layer 3: cloud-side
orchestrator).

Layer 2 shipped server-side fanout via binding.remote_fn.spawn() called from
a local entrypoint. Modal's launch warning surfaced a narrowed protection
story under --detach: "running a local entrypoint in detached mode only
keeps the last triggered Modal function alive after the parent process has
been killed or disconnected."

Layer 3 (this entrypoint) moves the .spawn() loop one altitude up: the local
entrypoint makes ONE binding.orchestrate_wave.remote(specs) call. Under
--detach, that single .remote() call is unambiguously protected; the spawned
cells become children of that protected call and inherit its protection.
Per-case persistence on the Modal Volume mount still applies inside each
cell, so even partial-wave failures preserve completed cases on disk.

ONE local lifecycle. ONE protected .remote() call. N server-side .spawn()
invocations under max_containers=10. Per-cell exception isolation via
collect_cells_parallel inside orchestrate_wave.

Per Layne directive 2026-05-23 + Opus Layer 3 ratification 2026-05-23.

Usage:
  modal run --detach examples/run_modal_cells_parallel.py \\
    --cell-spec-path .benchmarks/cell-specs.json

The cell spec file is a JSON list of dicts; each dict is the kwargs for one
ModalLiveEvictionRunRequest. One element per cell.
"""
# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import modal

from vorn_mat import (
    ModalLiveEvictionRunRequest,
    build_vorn_entrypoint,
    run_modal_live_eviction_niah,
    run_wave_serialized,
)


binding = build_vorn_entrypoint(run_modal_live_eviction_niah, modal_module=modal)
app = binding.app


@app.function(
    image=binding.image,
    timeout=86400,
    volumes={binding.spec.volume_path: binding.volume},
    secrets=[modal.Secret.from_name(binding.spec.hf_secret_name)],
    max_containers=1,
)
def orchestrate_wave(cell_specs: list[dict]) -> dict:
    """Cloud-side orchestrator. --detach protects this single .remote() call;
    spawned cells inside this function are children of it and inherit the
    protection. Returns a JSON-safe wave-report dict for cross-container
    transport.
    """
    return run_wave_serialized(binding, cell_specs, ModalLiveEvictionRunRequest)


@app.local_entrypoint()
def main(
    cell_spec_path: str,
    output_dir: str = str(ROOT / ".benchmarks" / "parallel-cells"),
) -> None:
    cell_specs = json.loads(Path(cell_spec_path).read_text())

    # ONE .remote() call from local; --detach protects this; orchestrate_wave
    # runs on cloud and is what spawns the per-cell calls under server-side
    # max_containers=10 scheduling.
    wave_report = orchestrate_wave.remote(cell_specs)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "reports.json").write_text(
        json.dumps(wave_report["reports"], indent=2, sort_keys=True)
    )
    if wave_report["failures"]:
        (output_path / "failures.json").write_text(
            json.dumps(wave_report["failures"], indent=2, sort_keys=True)
        )

    print(f"cells_fired={len(cell_specs)}")
    print(f"cells_succeeded={len(wave_report['reports'])}")
    print(f"cells_failed={len(wave_report['failures'])}")
    print(f"output_dir={output_path}")
