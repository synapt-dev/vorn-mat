from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat import (
    DEFAULT_STEP1_CACHE_BUDGET,
    DEFAULT_STEP1_CASE_LIMIT,
    build_step1_run_matrix,
)


def test_step1_run_matrix_locks_small_two_arm_niah_comparison():
    vanilla, vorn = build_step1_run_matrix()

    assert vanilla.run_id == "step1-niah-vanilla"
    assert vorn.run_id == "step1-niah-vorn"
    assert vanilla.benchmark == "niah"
    assert vorn.benchmark == "niah"
    assert vanilla.baseline == "vanilla"
    assert vorn.baseline == "vorn"
    assert vanilla.case_limit == DEFAULT_STEP1_CASE_LIMIT
    assert vorn.case_limit == DEFAULT_STEP1_CASE_LIMIT
    assert vanilla.cache_budget_tokens is None
    assert vorn.cache_budget_tokens == DEFAULT_STEP1_CACHE_BUDGET
    assert vanilla.experiment_stage == "step1"
    assert vorn.experiment_stage == "step1"
    assert vorn.compression_mode == "prompt_retention_proxy"
