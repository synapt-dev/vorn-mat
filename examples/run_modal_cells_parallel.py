#!/usr/bin/env python3
"""Modal-native parallel cell execution entrypoint with fire-and-forget dispatch.

CANONICAL FLOW (fire-and-forget, default — agents dispatch + walk away):

  1. Fire (returns within seconds, prints call_id):
     modal run --detach examples/run_modal_cells_parallel.py \\
       --cell-spec-path .benchmarks/cell-specs.json

  2. (work continues on Modal cloud, local entrypoint exits)

  3. Collect (blocks until the cloud call completes; idempotent re-runs OK):
     python examples/collect_modal_wave.py \\
       --wave-state-path .benchmarks/parallel-cells/wave-state.json

  4. Build per-family canonical artifacts:
     python examples/build_matrix_backfill_artifact.py --family mistral

Why this shape (substrate-fix 2026-05-26, second occurrence of
Modal-client-disconnect class-of-failure): Layer 3's ONE-`.remote()` call
correctly protects the cloud function under `--detach`, but `.remote()` BLOCKS
the local entrypoint synchronously waiting for the wave_report return. If
local disconnects mid-wait, `--detach` keeps the function running but the
local artifact-writing step (reports.json/failures.json) is lost. Atlas's
2026-05-26 B=1536/2048 wave required attended-mode protection because of
this exact concern. Switching to `.spawn()` + on-disk `wave-state.json` + a
separate collect_modal_wave.py retrieves the canonical Modal fire-and-forget
pattern: local exits within seconds carrying only the call_id; cloud function
runs to completion independent of spawning client lifecycle; collect step
materializes reports.json + failures.json from the FunctionCall by id.

LEGACY SYNC MODE (--wait flag for short waves / interactive dev):
  modal run examples/run_modal_cells_parallel.py \\
    --cell-spec-path .benchmarks/cell-specs.json --wait

  This preserves the prior behavior: local entrypoint blocks on
  orchestrate_wave.remote(specs) and writes artifacts before exit.

Per Layne directive 2026-05-23 (Layer 3 ratification) + Opus directive
2026-05-26 (fire-and-forget refinement).

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
    dispatch_wave_async,
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
    """Cloud-side orchestrator. Under `--detach`, this single .remote() (or
    .spawn()) call is unambiguously protected; the spawned cells inside this
    function are children of it and inherit the protection. Returns a JSON-
    safe wave-report dict for cross-container transport.
    """
    return run_wave_serialized(binding, cell_specs, ModalLiveEvictionRunRequest)


@app.local_entrypoint()
def main(
    cell_spec_path: str,
    output_dir: str = str(ROOT / ".benchmarks" / "parallel-cells"),
    wait: bool = False,
) -> None:
    """Default: fire-and-forget dispatch (recommended for any long-running wave).

    With `--wait`: blocks on .remote() and writes reports.json + failures.json
    before exit (legacy sync mode for short waves / interactive dev).
    """
    cell_specs = json.loads(Path(cell_spec_path).read_text())
    output_path = Path(output_dir)

    if wait:
        # Legacy sync mode: blocks until orchestrate_wave returns.
        wave_report = orchestrate_wave.remote(cell_specs)
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
        return

    # Fire-and-forget: .spawn() + persist wave-state.json + exit immediately.
    # Local lifecycle ends within seconds; cloud function runs to completion
    # independent of this client. collect_modal_wave.py retrieves the
    # wave_report later via modal.FunctionCall.from_id(call_id).get().
    dispatch = dispatch_wave_async(
        orchestrate_wave,
        cell_specs,
        output_dir=output_path,
        modal_app_name=app.name or "",
    )
    print(f"cells_dispatched={dispatch['spec_count']}")
    print(f"call_id={dispatch['call_id']}")
    print(f"wave_state_path={dispatch['state_path']}")
    print()
    print("Collect with:")
    print(
        f"  python examples/collect_modal_wave.py "
        f"--wave-state-path {dispatch['state_path']}"
    )
