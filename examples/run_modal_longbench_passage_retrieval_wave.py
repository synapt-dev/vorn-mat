#!/usr/bin/env python3
"""Run the preregistered LongBench PassageRetrieval-en 8-cell wave on Modal.

The default cell specs are generated from the config#316 lock: four families
times two methods, first 50 PassageRetrieval-en rows, B=1024, max_new_tokens=32.
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
    LONGBENCH_REVISION,
    ModalLongBenchLiveEvictionRunRequest,
    build_passage_retrieval_en_cell_specs,
    build_vorn_entrypoint,
    run_modal_live_eviction_longbench_passage_retrieval,
    run_wave_serialized,
)


binding = build_vorn_entrypoint(
    run_modal_live_eviction_longbench_passage_retrieval,
    modal_module=modal,
)
app = binding.app


def _modal_profile_from_specs(cell_specs: list[dict]) -> str:
    profiles = {str(spec.get("modal_profile", "")) for spec in cell_specs}
    if len(profiles) != 1 or "" in profiles:
        raise ValueError(
            "LongBench cell specs must carry exactly one non-empty modal_profile"
        )
    return next(iter(profiles))


@app.function(
    image=binding.image,
    timeout=86400,
    volumes={binding.spec.volume_path: binding.volume},
    secrets=[modal.Secret.from_name(binding.spec.hf_secret_name)],
    max_containers=1,
)
def orchestrate_wave(cell_specs: list[dict]) -> dict:
    """Run the whole LongBench wave from one protected cloud-side call."""

    return run_wave_serialized(
        binding,
        cell_specs,
        ModalLongBenchLiveEvictionRunRequest,
    )


@app.local_entrypoint()
def main(
    cell_spec_path: str = "",
    output_dir: str = str(ROOT / ".benchmarks" / "longbench-passage-wave"),
    dry_run: bool = False,
    dataset_revision: str = LONGBENCH_REVISION,
) -> None:
    if cell_spec_path:
        cell_specs = json.loads(Path(cell_spec_path).read_text())
    else:
        cell_specs = list(
            build_passage_retrieval_en_cell_specs(
                results_root=binding.spec.results_root,
                dataset_revision=dataset_revision,
            )
        )
    modal_profile = _modal_profile_from_specs(cell_specs)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "cell-specs.json").write_text(
        json.dumps(cell_specs, indent=2, sort_keys=True)
    )

    if dry_run:
        print("dry_run=true")
        print(f"profile={modal_profile}")
        print(f"cells_prepared={len(cell_specs)}")
        print(f"output_dir={output_path}")
        return

    wave_report = orchestrate_wave.remote(cell_specs)

    (output_path / "reports.json").write_text(
        json.dumps(wave_report["reports"], indent=2, sort_keys=True)
    )
    if wave_report["failures"]:
        (output_path / "failures.json").write_text(
            json.dumps(wave_report["failures"], indent=2, sort_keys=True)
        )

    print(f"profile={modal_profile}")
    print(f"cells_fired={len(cell_specs)}")
    print(f"cells_succeeded={len(wave_report['reports'])}")
    print(f"cells_failed={len(wave_report['failures'])}")
    print(f"output_dir={output_path}")
