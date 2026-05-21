#!/usr/bin/env python3
"""Run live score-distribution observation on Modal."""
# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import modal

from vorn_mat import (
    ModalScoreDistributionObservationRequest,
    build_vorn_entrypoint,
    run_modal_score_distribution_observation_niah,
    write_score_distribution_observation_report,
)


binding = build_vorn_entrypoint(
    run_modal_score_distribution_observation_niah,
    modal_module=modal,
)
app = binding.app


@app.local_entrypoint()
def main(
    dataset_config: str = "niah_multikey_1_8k",
    case_limit: int = 50,
    split: str = "validation",
    max_new_tokens: int = 32,
    canonical_layer: int = 16,
    recent_token_window: int = 16,
    cache_budget_tokens: int = 1024,
    retention_policy: str = "sentence_vorn",
    always_keep_prefix_tokens: int = 1,
    preserve_recent_window: bool = True,
    sentence_pooling: str = "max",
    sentence_top_k: int = 3,
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
    output: str = str(
        ROOT / ".benchmarks" / "score-dist-8k-b1024-sentence.json"
    ),
) -> None:
    request = ModalScoreDistributionObservationRequest(
        dataset_config=dataset_config,
        split=split,
        case_limit=case_limit,
        max_new_tokens=max_new_tokens,
        canonical_layer=canonical_layer,
        recent_token_window=recent_token_window,
        cache_budget_tokens=cache_budget_tokens,
        retention_policy=retention_policy,
        always_keep_prefix_tokens=always_keep_prefix_tokens,
        preserve_recent_window=preserve_recent_window,
        sentence_pooling=sentence_pooling,
        sentence_top_k=sentence_top_k,
        model_id=model_id,
    )
    report = binding.remote_fn.remote(request)

    output_path = Path(output)
    write_score_distribution_observation_report(
        report,
        json_path=output_path,
    )

    hit_count = sum(
        1
        for case in report.cases
        if case.observations and case.observations[0].correct
    )
    hit_rate = hit_count / report.case_count if report.case_count else 0.0

    print("profile=author")
    print(f"dataset_config={report.dataset_config}")
    print(f"split={report.split}")
    print(f"case_count={report.case_count}")
    print(f"model_id={report.model_id}")
    print(f"cache_budget_tokens={report.cache_budget_tokens}")
    print(f"retention_policy={report.retention_policy}")
    print(f"always_keep_prefix_tokens={report.always_keep_prefix_tokens}")
    print(f"preserve_recent_window={report.preserve_recent_window}")
    print(f"sentence_pooling={report.sentence_pooling}")
    print(f"sentence_top_k={report.sentence_top_k}")
    print(f"hit_rate={hit_rate:.2f}")
    print(f"elapsed_seconds={report.elapsed_seconds:.3f}")
    print(f"estimated_cost_usd={report.estimated_cost_usd:.4f}")
    print(f"output={output_path}")
