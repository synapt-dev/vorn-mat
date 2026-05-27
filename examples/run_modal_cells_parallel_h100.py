#!/usr/bin/env python3
"""H100 variant of run_modal_cells_parallel.py for OOM-prone cells.

Same Layer 3 orchestrator pattern (one protected .remote() call from local,
server-side .spawn() fanout under max_containers=10), but bound to H100 at
app construction time for cells that don't fit on A100-80GB (notably the
Qwen 3 30B-A3B attention-weight methods which need attention-state
tracking on top of ~60GB bf16 weights, exceeding the A100 envelope).

Usage:
  modal run --detach examples/run_modal_cells_parallel_h100.py \\
    --cell-spec-path .benchmarks/cell-specs-h100.json
"""
# ruff: noqa: E402

from __future__ import annotations

from dataclasses import replace
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
from vorn_mat.modal_app import default_modal_app_spec


binding = build_vorn_entrypoint(
    run_modal_live_eviction_niah,
    modal_module=modal,
    app_spec=replace(default_modal_app_spec(), gpu="H100"),
)
app = binding.app


@app.function(
    image=binding.image,
    timeout=86400,
    volumes={binding.spec.volume_path: binding.volume},
    secrets=[modal.Secret.from_name(binding.spec.hf_secret_name)],
    max_containers=1,
)
def orchestrate_wave(cell_specs: list[dict]) -> dict:
    return run_wave_serialized(binding, cell_specs, ModalLiveEvictionRunRequest)


@app.local_entrypoint()
def main(
    cell_spec_path: str,
    output_dir: str = str(ROOT / ".benchmarks" / "parallel-cells-h100"),
) -> None:
    cell_specs = json.loads(Path(cell_spec_path).read_text())

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
