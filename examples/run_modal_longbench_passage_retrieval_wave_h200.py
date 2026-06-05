#!/usr/bin/env python3
"""H200 variant of the config#316 LongBench PassageRetrieval-en wave runner.

This keeps the locked task, slice, models, methods, budget, prompt, scorer,
bf16 precision, and greedy decoding. Only the Modal GPU substrate changes
after A100 and H100 exposed long-context-forward OOM.
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
    LONGBENCH_REVISION,
    ModalLongBenchLiveEvictionRunRequest,
    build_passage_retrieval_en_cell_specs,
    build_vorn_entrypoint,
    run_modal_live_eviction_longbench_passage_retrieval,
    run_wave_serialized,
)
from vorn_mat.modal_app import default_modal_app_spec


binding = build_vorn_entrypoint(
    run_modal_live_eviction_longbench_passage_retrieval,
    modal_module=modal,
    app_spec=replace(default_modal_app_spec(), gpu="H200"),
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
    return run_wave_serialized(
        binding,
        cell_specs,
        ModalLongBenchLiveEvictionRunRequest,
    )


@app.local_entrypoint()
def main(
    cell_spec_path: str = "",
    output_dir: str = str(ROOT / ".benchmarks" / "longbench-passage-wave-h200"),
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
                attempt_label="config316-h200",
                gpu="H200",
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
        print("gpu=H200")
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
    print("gpu=H200")
    print(f"cells_fired={len(cell_specs)}")
    print(f"cells_succeeded={len(wave_report['reports'])}")
    print(f"cells_failed={len(wave_report['failures'])}")
    print(f"output_dir={output_path}")
