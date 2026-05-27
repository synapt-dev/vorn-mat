"""Fire-and-forget Modal dispatch + separate result collection.

Substrate-fix for the Modal client-disconnect class-of-failure (2nd occurrence
2026-05-26; first 2026-05-23 cost ~$3-4 + 1.5h on Gemma 3). Layer 3's
ONE-`.remote()`-call architecture correctly protects the cloud function
under `--detach`, but the synchronous `.remote()` blocks the local
entrypoint waiting on the wave_report return. If local disconnects
mid-wait, `--detach` keeps the function running but the local
artifact-writing step (reports.json/failures.json) is lost.

This module provides the canonical Modal fire-and-forget pattern:

  dispatch (local exits within seconds):
    orchestrate_fn.spawn(cell_specs) → FunctionCall(object_id="fc_...")
    persist {call_id, modal_app_name, cell_specs_hash, ...} → wave-state.json
    print call_id; exit

  collect (local re-attaches to retrieve, idempotent):
    read wave-state.json → call_id
    modal.FunctionCall.from_id(call_id).get() → wave_report
    write reports.json + (failures.json if any) → output_dir
    summary returned for downstream artifact builders

Tests use fake spawn/FunctionCall surfaces so the dispatch + collection
shape is unit-testable without Modal-actual.

Reference: config/process/modal-async-patterns.md
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class WaveStateFile:
    """Typed wrapper for the wave-state.json serialization contract.

    Both dispatch (write) and collect (read) routes share this dataclass so
    schema drift on either side fails loudly at load() time instead of
    silently using stale defaults.
    """

    call_id: str
    spec_count: int
    dispatched_at: str
    output_dir: str
    modal_app_name: str
    cell_specs_hash: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> WaveStateFile:
        data = json.loads(Path(path).read_text())
        return cls(**{f: data[f] for f in cls.__dataclass_fields__})


def _hash_cell_specs(cell_specs: list[dict]) -> str:
    """SHA256 over the canonical JSON serialization of the spec list.

    Order-sensitive: spawn order determines cell-position mapping in reports,
    so a reordered spec list produces a different hash. Sorted keys within
    each spec dict ensure the hash is whitespace/key-order-stable.
    """
    payload = json.dumps(cell_specs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dispatch_wave_async(
    orchestrate_fn: Any,
    cell_specs: list[dict],
    output_dir: Path,
    modal_app_name: str = "",
) -> dict:
    """Spawn the cloud-side orchestrator and persist call_id to disk.

    Returns immediately after `orchestrate_fn.spawn(cell_specs)` returns the
    FunctionCall handle. Local entrypoint can exit; the cloud-side function
    runs to completion independent of the spawning client's lifecycle.
    `collect_wave_results` retrieves the result later via call_id.

    Args:
        orchestrate_fn: the @app.function-decorated orchestrator with a
            .spawn(cell_specs) method that returns an object with
            .object_id (a string call_id usable with FunctionCall.from_id).
        cell_specs: list of dicts, each one the kwargs for one cell request.
        output_dir: where to write wave-state.json (created if missing).
        modal_app_name: name of the Modal app for debugging context. Optional
            but recommended (lets collect verify it's looking at the right
            call_id later if state files get mixed up).

    Returns:
        Dict with state_path (str), call_id (str), spec_count (int) for the
        caller to print/log/forward to collect.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    function_call = orchestrate_fn.spawn(cell_specs)
    call_id = function_call.object_id

    state = WaveStateFile(
        call_id=call_id,
        spec_count=len(cell_specs),
        dispatched_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        output_dir=str(output_dir),
        modal_app_name=modal_app_name,
        cell_specs_hash=_hash_cell_specs(cell_specs),
    )

    state_path = output_dir / "wave-state.json"
    state.save(state_path)

    return {
        "state_path": str(state_path),
        "call_id": call_id,
        "spec_count": len(cell_specs),
    }


def collect_wave_results(
    wave_state_path: Path,
    output_dir: Path,
    modal_from_id: Callable[[str], Any] | None = None,
    force_recollect: bool = False,
) -> dict:
    """Retrieve the wave_report for a dispatched call_id and write artifacts.

    Idempotent: if reports.json already exists in output_dir, return the
    summary based on existing artifacts without re-fetching from Modal.
    Pass `force_recollect=True` to override (e.g., to overwrite stale
    artifacts after re-running an updated cloud-side orchestrator).

    Blocks on `modal.FunctionCall.from_id(call_id).get()` if Modal's
    FunctionCall is not yet complete; the canonical wait semantics live in
    Modal's client.

    Args:
        wave_state_path: path to the wave-state.json written by dispatch.
        output_dir: where to write reports.json + failures.json (created if
            missing).
        modal_from_id: callable that takes a call_id string and returns a
            FunctionCall-like object with .get() → wave_report dict. In
            production this is `modal.FunctionCall.from_id`. Tests inject a
            fake.
        force_recollect: if True, ignore existing artifacts and re-fetch.

    Returns:
        Summary dict {cells_succeeded, cells_failed, output_dir, call_id}.
    """
    state = WaveStateFile.load(Path(wave_state_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports_path = output_dir / "reports.json"
    failures_path = output_dir / "failures.json"

    if reports_path.exists() and not force_recollect:
        existing_reports = json.loads(reports_path.read_text())
        existing_failures = (
            json.loads(failures_path.read_text()) if failures_path.exists() else []
        )
        return {
            "call_id": state.call_id,
            "cells_succeeded": len(existing_reports),
            "cells_failed": len(existing_failures),
            "output_dir": str(output_dir),
            "from_cache": True,
        }

    if modal_from_id is None:
        import modal

        modal_from_id = modal.FunctionCall.from_id

    function_call = modal_from_id(state.call_id)
    wave_report = function_call.get()

    reports = wave_report.get("reports", [])
    failures = wave_report.get("failures", [])

    reports_path.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    if failures:
        failures_path.write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n")

    return {
        "call_id": state.call_id,
        "cells_succeeded": len(reports),
        "cells_failed": len(failures),
        "output_dir": str(output_dir),
        "from_cache": False,
    }
