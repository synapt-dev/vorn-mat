"""Layer 4 Modal-visible progress logging.

Long-running cells (30-60min inference on A100) go silent after the
HuggingFace model-load phase. Modal captures stdout to the dashboard and
`modal app logs` CLI, but the cells emit nothing during the silent inference
phase, so from outside the local terminal there is no mid-cell progress
visibility.

Layer 4 adds:
- A ProgressLogger type alias (Callable[[str], None]) + default_progress_logger
  that print()s with flush=True so Modal sees output immediately, not buffered.
- A new progress_logger kwarg on run_vanilla / run_vorn / run_live_eviction.
- Wrappers default progress_logger to default_progress_logger so Modal
  entry-points emit by default; tests use None or a fake-collector to assert
  emissions.

Emission shape:
- dataset_loaded line (once at start): "vorn-mat: dataset_loaded n_cases=N"
- per-case lines: "vorn-mat: case I/N correct=true running_accuracy=0.500"
- inference_complete line (once at end): "vorn-mat: complete n_cases=N hit_rate=X"

Per Layne directive 2026-05-24 + Opus design ratification.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat import (
    build_execution_plans,
    build_step1_run_matrix,
    build_week1_run_matrix,
    default_progress_logger,
    run_vanilla,
)
from vorn_mat.baselines.live_eviction import (
    LiveEvictionStats,
    LiveEvictionTextGenerator,
    SEMANTIC_SUMMARY_CONTRACT,
    run_live_eviction,
)
from vorn_mat.baselines.vanilla import TextGenerator
from vorn_mat.baselines.vorn import (
    RetentionStats,
    VornTextGenerator,
    run_vorn,
)
from vorn_mat.benchmarks.common import BenchmarkCase
from vorn_mat.local_exec import select_live_eviction_plan


class _FakeVanillaGenerator(TextGenerator):
    def __init__(self, answers: dict[str, str]):
        self.answers = answers

    def generate(self, prompt: str) -> str:
        return self.answers[prompt]


class _FakeVornGenerator(VornTextGenerator):
    def __init__(self, answers: dict[str, str], stats: RetentionStats):
        self.answers = answers
        self.stats = stats

    def generate_with_retention(self, prompt: str, config):
        return self.answers[prompt], self.stats


class _FakeLiveGenerator(LiveEvictionTextGenerator):
    def __init__(self, answers: dict[str, str], stats: LiveEvictionStats):
        self.answers = answers
        self.stats = stats

    def generate_with_live_eviction(self, prompt: str, config):
        return self.answers[prompt], self.stats


def _three_cases_with_known_outcomes() -> tuple[BenchmarkCase, ...]:
    # Two correct, one wrong: running_accuracy should be 1/1, 1/2, 2/3
    return (
        BenchmarkCase("case-1", "prompt-1", "Paris", {}),
        BenchmarkCase("case-2", "prompt-2", "5", {}),
        BenchmarkCase("case-3", "prompt-3", "Tokyo", {}),
    )


def test_default_progress_logger_prints_to_stdout_with_flush(capsys):
    default_progress_logger("vorn-mat: dataset_loaded n_cases=50")

    captured = capsys.readouterr()
    assert captured.out == "vorn-mat: dataset_loaded n_cases=50\n"


def test_run_vanilla_with_no_logger_emits_nothing(capsys):
    plan = build_execution_plans(build_week1_run_matrix())[0]
    cases = _three_cases_with_known_outcomes()
    generator = _FakeVanillaGenerator(
        {"prompt-1": "Paris", "prompt-2": "5", "prompt-3": "Tokyo"}
    )

    run_vanilla(plan, cases, generator)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_run_vanilla_emits_dataset_loaded_milestone():
    plan = build_execution_plans(build_week1_run_matrix())[0]
    cases = _three_cases_with_known_outcomes()
    generator = _FakeVanillaGenerator(
        {"prompt-1": "Paris", "prompt-2": "5", "prompt-3": "Tokyo"}
    )
    emissions: list[str] = []

    run_vanilla(plan, cases, generator, progress_logger=emissions.append)

    assert emissions[0] == "vorn-mat: dataset_loaded n_cases=3"


def test_run_vanilla_emits_per_case_progress_with_running_accuracy():
    plan = build_execution_plans(build_week1_run_matrix())[0]
    cases = _three_cases_with_known_outcomes()
    # case-1 correct, case-2 wrong (answer "4" vs expected "5"), case-3 correct
    generator = _FakeVanillaGenerator(
        {"prompt-1": "Paris", "prompt-2": "4", "prompt-3": "Tokyo"}
    )
    emissions: list[str] = []

    run_vanilla(plan, cases, generator, progress_logger=emissions.append)

    # First emission is dataset_loaded; per-case emissions follow
    case_lines = [e for e in emissions if e.startswith("vorn-mat: case ")]
    assert len(case_lines) == 3
    assert case_lines[0] == "vorn-mat: case 1/3 correct=true running_accuracy=1.000"
    assert case_lines[1] == "vorn-mat: case 2/3 correct=false running_accuracy=0.500"
    assert case_lines[2] == "vorn-mat: case 3/3 correct=true running_accuracy=0.667"


def test_run_vanilla_emits_complete_milestone_with_final_hit_rate():
    plan = build_execution_plans(build_week1_run_matrix())[0]
    cases = _three_cases_with_known_outcomes()
    generator = _FakeVanillaGenerator(
        {"prompt-1": "Paris", "prompt-2": "4", "prompt-3": "Tokyo"}
    )
    emissions: list[str] = []

    run_vanilla(plan, cases, generator, progress_logger=emissions.append)

    assert emissions[-1] == "vorn-mat: complete n_cases=3 hit_rate=0.667"


def test_run_vorn_emits_progress_in_same_shape():
    _, vorn_plan = build_execution_plans(build_step1_run_matrix())
    cases = _three_cases_with_known_outcomes()
    generator = _FakeVornGenerator(
        {"prompt-1": "Paris", "prompt-2": "5", "prompt-3": "Tokyo"},
        stats=RetentionStats(
            original_token_count=100,
            kept_token_count=50,
            kept_positions=(0, 1),
            dropped_positions=(2, 3),
        ),
    )
    emissions: list[str] = []

    run_vorn(vorn_plan, cases, generator, progress_logger=emissions.append)

    assert emissions[0].startswith("vorn-mat: dataset_loaded n_cases=3")
    case_lines = [e for e in emissions if e.startswith("vorn-mat: case ")]
    assert len(case_lines) == 3
    assert "running_accuracy=" in case_lines[-1]
    assert emissions[-1].startswith("vorn-mat: complete n_cases=3")


def test_run_live_eviction_emits_progress_in_same_shape():
    plan = select_live_eviction_plan(cache_budget_tokens=256, retention_policy="vorn")
    cases = _three_cases_with_known_outcomes()
    generator = _FakeLiveGenerator(
        {"prompt-1": "Paris", "prompt-2": "5", "prompt-3": "Tokyo"},
        stats=LiveEvictionStats(
            prompt_token_count=100,
            generated_token_count=8,
            mean_kept_token_count=64.0,
            final_kept_token_count=64,
            eviction_steps=2,
            mean_retention_ratio=0.64,
            retention_policy="vorn",
            summary_contract=SEMANTIC_SUMMARY_CONTRACT,
            summary_fingerprint="abc",
        ),
    )
    emissions: list[str] = []

    run_live_eviction(plan, cases, generator, progress_logger=emissions.append)

    assert emissions[0] == "vorn-mat: dataset_loaded n_cases=3"
    case_lines = [e for e in emissions if e.startswith("vorn-mat: case ")]
    assert len(case_lines) == 3
    assert emissions[-1].startswith("vorn-mat: complete n_cases=3")


def test_progress_logger_lines_are_modal_friendly():
    """No multi-line emissions; each emission is a single line without newlines.

    Modal captures stdout line-by-line for the dashboard; multi-line emissions
    would fragment timestamps and make grep-based debugging painful.
    """
    plan = build_execution_plans(build_week1_run_matrix())[0]
    cases = _three_cases_with_known_outcomes()
    generator = _FakeVanillaGenerator(
        {"prompt-1": "Paris", "prompt-2": "5", "prompt-3": "Tokyo"}
    )
    emissions: list[str] = []

    run_vanilla(plan, cases, generator, progress_logger=emissions.append)

    for line in emissions:
        assert "\n" not in line, f"multi-line emission: {line!r}"
        assert line.startswith("vorn-mat: ")
