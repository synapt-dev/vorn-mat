"""Execution planning for the Week 1 Modal baseline runner layer."""

from __future__ import annotations

from dataclasses import dataclass

from .baselines import BaselineSpec, get_baseline
from .benchmarks import BenchmarkSpec, get_benchmark
from .modal_app import ModalAppSpec, default_modal_app_spec
from .plan import Week1Run, build_week1_run_matrix


@dataclass(frozen=True)
class ExecutionPlan:
    run: Week1Run
    benchmark: BenchmarkSpec
    baseline: BaselineSpec
    results_path: str
    benchmark_cache_path: str
    model_cache_path: str


def build_execution_plans(
    runs: tuple[Week1Run, ...] | None = None,
    modal_spec: ModalAppSpec | None = None,
) -> tuple[ExecutionPlan, ...]:
    if runs is None:
        runs = build_week1_run_matrix()
    if modal_spec is None:
        modal_spec = default_modal_app_spec()

    plans: list[ExecutionPlan] = []
    for run in runs:
        plans.append(
            ExecutionPlan(
                run=run,
                benchmark=get_benchmark(run.benchmark),
                baseline=get_baseline(run.baseline),
                results_path=f"{modal_spec.results_root}/{run.run_id}.jsonl",
                benchmark_cache_path=modal_spec.benchmark_cache,
                model_cache_path=modal_spec.model_cache,
            )
        )
    return tuple(plans)
