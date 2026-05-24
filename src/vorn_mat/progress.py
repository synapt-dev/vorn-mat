"""Modal-visible progress logging primitive for cell observability.

Cells run on Modal A100s for 30-60 minutes during inference. Modal captures
stdout to the dashboard + `modal app logs` CLI, but the cells go silent
after the HuggingFace model-load phase, so from outside the local terminal
there is no mid-cell visibility into per-case progress.

The progress logger emits phase milestones + per-case progress to stdout in
a Modal-friendly format (line-based, no multi-line):

- "vorn-mat: dataset_loaded n_cases=N"             (once at start)
- "vorn-mat: case I/N correct=<bool> running_accuracy=0.XXX"  (per case)
- "vorn-mat: complete n_cases=N hit_rate=0.XXX"    (once at end)

Modal auto-timestamps each line in the dashboard, so emissions do not need
local timestamps. flush=True ensures Modal sees output immediately rather
than buffered until container exit.

Per Layne directive 2026-05-24 + Opus design ratification.
"""

from __future__ import annotations

from typing import Callable

ProgressLogger = Callable[[str], None]


def default_progress_logger(message: str) -> None:
    print(message, flush=True)


def format_dataset_loaded(n_cases: int) -> str:
    return f"vorn-mat: dataset_loaded n_cases={n_cases}"


def format_case_progress(case_index: int, n_cases: int, correct: bool, running_accuracy: float) -> str:
    correct_str = "true" if correct else "false"
    return (
        f"vorn-mat: case {case_index}/{n_cases} "
        f"correct={correct_str} running_accuracy={running_accuracy:.3f}"
    )


def format_complete(n_cases: int, hit_rate: float) -> str:
    return f"vorn-mat: complete n_cases={n_cases} hit_rate={hit_rate:.3f}"
