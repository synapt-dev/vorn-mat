#!/usr/bin/env python3
"""Run a real vanilla-only NIAH baseline slice on Modal H100."""
# ruff: noqa: E402

from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import modal

from vorn_mat import ModalVanillaRunRequest, build_vanilla_entrypoint, run_modal_vanilla_niah
from vorn_mat.modal_app import default_modal_app_spec


binding = build_vanilla_entrypoint(
    run_modal_vanilla_niah,
    modal_module=modal,
    app_spec=replace(default_modal_app_spec(), gpu="H100"),
)
app = binding.app


@app.local_entrypoint()
def main(
    dataset_config: str = "niah_multikey_1_4k",
    case_limit: int = 50,
    split: str = "validation",
    max_new_tokens: int = 32,
    model_id: str = "Qwen/Qwen3-30B-A3B",
    output: str = str(ROOT / ".benchmarks" / "modal-niah-vanilla-h100-report.json"),
) -> None:
    model_slug = model_id.replace("/", "--")
    request = ModalVanillaRunRequest(
        dataset_config=dataset_config,
        split=split,
        case_limit=case_limit,
        benchmark="niah",
        output_path=(
            f"{binding.spec.results_root}/modal-{model_slug}-"
            f"{dataset_config}-{case_limit}-vanilla-h100.jsonl"
        ),
        max_new_tokens=max_new_tokens,
        model_id=model_id,
        gpu="H100",
    )
    report = binding.remote_fn.remote(request)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))

    print("profile=author")
    print(f"dataset_config={report.dataset_config}")
    print(f"split={report.split}")
    print(f"case_count={report.case_count}")
    print(f"model_id={model_id}")
    print("gpu=H100")
    print(f"run_id={report.result.run_id}")
    print(f"metrics={report.result.metrics}")
    print(f"elapsed_seconds={report.elapsed_seconds:.3f}")
    print(f"estimated_cost_usd={report.estimated_cost_usd:.4f}")
    print(f"output={output_path}")
