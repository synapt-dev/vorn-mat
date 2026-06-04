#!/usr/bin/env python3
"""Run the active-eviction consumer-validation smoke cell on Modal."""
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
    ModalConsumerValidationRunRequest,
    build_consumer_validation_entrypoint,
    default_modal_app_spec,
    load_phase0_case,
    run_modal_consumer_validation_niah,
)


binding = build_consumer_validation_entrypoint(
    run_modal_consumer_validation_niah,
    modal_module=modal,
    app_spec=default_modal_app_spec(),
)
app = binding.app


@app.local_entrypoint()
def main(
    dataset_config: str = "niah_multikey_1_4k",
    split: str = "validation",
    case_offset_start: int = 1,
    case_id: str = "niah_multikey_1_4k-1",
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
    phase0_artifact: str = str(
        ROOT / "results" / "vorn-active-eviction-phase0-2026-06-03.json"
    ),
    output: str = str(
        ROOT
        / ".benchmarks"
        / "modal-active-eviction-consumer-validation-report.json"
    ),
) -> None:
    phase0_case = load_phase0_case(Path(phase0_artifact), case_id)
    request = ModalConsumerValidationRunRequest(
        dataset_config=dataset_config,
        split=split,
        case_offset_start=case_offset_start,
        phase0_case=phase0_case,
        model_id=model_id,
        output_jsonl_path=(
            f"{binding.spec.results_root}/"
            "vorn-active-eviction-pilot-a-consumer-validation-2026-06-04.jsonl"
        ),
        output_summary_path=(
            f"{binding.spec.results_root}/"
            "vorn-active-eviction-pilot-a-consumer-validation-2026-06-04.md"
        ),
    )
    report = binding.remote_fn.remote(request)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))

    print("profile=layne1penney")
    print(f"dataset_config={report.dataset_config}")
    print(f"split={report.split}")
    print(f"case_offset_start={report.case_offset_start}")
    print(f"case_id={report.case_id}")
    print(f"model_id={model_id}")
    print(f"elapsed_seconds={report.elapsed_seconds:.3f}")
    print(f"estimated_cost_usd={report.estimated_cost_usd:.4f}")
    print(f"output_jsonl_path={report.output_jsonl_path}")
    print(f"output_summary_path={report.output_summary_path}")
    print(f"local_report={output_path}")
