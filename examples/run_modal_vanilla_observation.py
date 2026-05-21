#!/usr/bin/env python3
"""Run vanilla Mistral observation on a real NIAH slice via Modal."""
# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import modal

from vorn_mat import (
    ModalVanillaObservationRequest,
    build_vanilla_entrypoint,
    run_modal_vanilla_observation_niah,
    write_observation_artifacts,
)


binding = build_vanilla_entrypoint(
    run_modal_vanilla_observation_niah,
    modal_module=modal,
)
app = binding.app


@app.local_entrypoint()
def main(
    dataset_config: str = "niah_multikey_1_4k",
    case_limit: int = 50,
    split: str = "validation",
    max_new_tokens: int = 32,
    canonical_layer: int = 16,
    recent_token_window: int = 16,
    top_k: int = 10,
    attention_last_n_layers: int = 4,
    output_json: str = str(
        ROOT / "results" / "vanilla-observation-2026-05-13.json"
    ),
    output_md: str = str(
        ROOT / "results" / "vanilla-observation-2026-05-13.md"
    ),
    figures_dir: str = str(ROOT / "results" / "figures"),
) -> None:
    request = ModalVanillaObservationRequest(
        dataset_config=dataset_config,
        split=split,
        case_limit=case_limit,
        max_new_tokens=max_new_tokens,
        canonical_layer=canonical_layer,
        recent_token_window=recent_token_window,
        top_k=top_k,
        attention_last_n_layers=attention_last_n_layers,
    )
    report = binding.remote_fn.remote(request)

    output_json_path = Path(output_json)
    summary = write_observation_artifacts(
        report,
        json_path=output_json_path,
        markdown_path=Path(output_md),
        figures_dir=Path(figures_dir),
        top_k=top_k,
    )

    print("profile=author")
    print(f"dataset_config={report.dataset_config}")
    print(f"split={report.split}")
    print(f"case_count={report.case_count}")
    print(f"elapsed_seconds={report.elapsed_seconds:.3f}")
    print(f"estimated_cost_usd={report.estimated_cost_usd:.4f}")
    print(f"success_cases={summary['success_cases']}")
    print(f"failure_cases={summary['failure_cases']}")
    print(f"output_json={output_json_path}")
    print(f"output_md={Path(output_md)}")
