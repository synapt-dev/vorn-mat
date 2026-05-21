#!/usr/bin/env python3
"""Run the live eviction-only NIAH arm on Modal."""
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
    DEFAULT_MODEL,
    ModalLiveEvictionRunRequest,
    build_vorn_entrypoint,
    run_modal_live_eviction_niah,
)


binding = build_vorn_entrypoint(run_modal_live_eviction_niah, modal_module=modal)
app = binding.app


@app.local_entrypoint()
def main(
    dataset_config: str = "niah_multikey_1_4k",
    case_limit: int = 50,
    split: str = "validation",
    max_new_tokens: int = 32,
    cache_budget_tokens: int = 256,
    retention_policy: str = "vorn",
    random_seed: int = 17,
    always_keep_prefix_tokens: int = 1,
    preserve_recent_window: bool = True,
    sentence_pooling: str = "max",
    sentence_top_k: int = 3,
    eviction_trigger: str = "budget_threshold",
    sentence_boundary_lookahead_tokens: int = 25,
    force_eviction_overflow_ratio: float = 1.2,
    model_id: str = DEFAULT_MODEL,
    output: str = str(ROOT / ".benchmarks" / "modal-niah-live-eviction-report.json"),
) -> None:
    model_slug = model_id.replace("/", "--")
    request = ModalLiveEvictionRunRequest(
        dataset_config=dataset_config,
        split=split,
        case_limit=case_limit,
        benchmark="niah",
        output_path=(
            f"{binding.spec.results_root}/"
            f"modal-{model_slug}-{dataset_config}-{case_limit}-{retention_policy}-live-"
            f"{cache_budget_tokens}-{eviction_trigger}.jsonl"
        ),
        max_new_tokens=max_new_tokens,
        cache_budget_tokens=cache_budget_tokens,
        retention_policy=retention_policy,
        random_seed=random_seed,
        always_keep_prefix_tokens=always_keep_prefix_tokens,
        preserve_recent_window=preserve_recent_window,
        sentence_pooling=sentence_pooling,
        sentence_top_k=sentence_top_k,
        eviction_trigger=eviction_trigger,
        sentence_boundary_lookahead_tokens=sentence_boundary_lookahead_tokens,
        force_eviction_overflow_ratio=force_eviction_overflow_ratio,
        model_id=model_id,
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
    print(f"cache_budget_tokens={report.cache_budget_tokens}")
    print(f"retention_policy={report.retention_policy}")
    print(f"random_seed={request.random_seed}")
    print(f"always_keep_prefix_tokens={report.always_keep_prefix_tokens}")
    print(f"preserve_recent_window={report.preserve_recent_window}")
    print(f"sentence_pooling={report.sentence_pooling}")
    print(f"sentence_top_k={report.sentence_top_k}")
    print(f"eviction_trigger={report.eviction_trigger}")
    print(
        "sentence_boundary_lookahead_tokens="
        f"{report.sentence_boundary_lookahead_tokens}"
    )
    print(
        "force_eviction_overflow_ratio="
        f"{report.force_eviction_overflow_ratio:.2f}"
    )
    print(f"run_id={report.result.run_id}")
    print(f"metrics={report.result.metrics}")
    print(f"elapsed_seconds={report.elapsed_seconds:.3f}")
    print(f"estimated_cost_usd={report.estimated_cost_usd:.4f}")
    print(f"output={output_path}")
