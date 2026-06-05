#!/usr/bin/env python3
"""Run the preregistered LongBench PassageRetrieval-en compressed cell on Modal.

No LongBench cell is authorized until config#316 is ratified. This entrypoint is
scaffolding only until that gate clears.
"""
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
    LONGBENCH_REVISION,
    ModalLongBenchLiveEvictionRunRequest,
    PASSAGE_RETRIEVAL_EN_MAX_NEW_TOKENS,
    build_vorn_entrypoint,
    run_modal_live_eviction_longbench_passage_retrieval,
)


binding = build_vorn_entrypoint(
    run_modal_live_eviction_longbench_passage_retrieval,
    modal_module=modal,
)
app = binding.app


@app.local_entrypoint()
def main(
    case_limit: int = 50,
    case_offset_start: int = 0,
    max_new_tokens: int = PASSAGE_RETRIEVAL_EN_MAX_NEW_TOKENS,
    cache_budget_tokens: int = 1024,
    retention_policy: str = "sentence_vorn",
    random_seed: int = 17,
    always_keep_prefix_tokens: int = 1,
    preserve_recent_window: bool = True,
    sentence_pooling: str = "max",
    sentence_top_k: int = 3,
    eviction_trigger: str = "budget_threshold",
    sentence_boundary_lookahead_tokens: int = 25,
    force_eviction_overflow_ratio: float = 1.2,
    model_id: str = DEFAULT_MODEL,
    dataset_revision: str = LONGBENCH_REVISION,
    modal_profile: str = "layne1penney",
    output: str = str(ROOT / ".benchmarks" / "modal-longbench-passage-report.json"),
) -> None:
    model_slug = model_id.replace("/", "--")
    request = ModalLongBenchLiveEvictionRunRequest(
        dataset_revision=dataset_revision,
        case_limit=case_limit,
        case_offset_start=case_offset_start,
        output_path=(
            f"{binding.spec.results_root}/"
            f"modal-{model_slug}-longbench-passage-retrieval-en-{case_limit}-"
            f"{retention_policy}-b{cache_budget_tokens}.jsonl"
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
        modal_profile=modal_profile,
        preregistration="config#316",
    )
    report = binding.remote_fn.remote(request)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True))

    print(f"profile={request.modal_profile}")
    print(f"dataset_id={report.dataset_id}")
    print(f"dataset_revision={report.dataset_revision}")
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
    print(f"run_id={report.result.run_id}")
    print(f"metrics={report.result.metrics}")
    print(f"elapsed_seconds={report.elapsed_seconds:.3f}")
    print(f"estimated_cost_usd={report.estimated_cost_usd:.4f}")
    print(f"output={output_path}")
