#!/usr/bin/env python3
"""Run the Step 0 local vanilla smoke path."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vorn_mat.local_exec import run_local_vanilla_smoke  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "examples" / "niah_smoke.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".benchmarks" / "local-niah-vanilla-smoke.jsonl",
    )
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    result, traces = run_local_vanilla_smoke(
        benchmark="niah",
        dataset_path=args.dataset,
        output_path=args.output,
        case_limit=args.limit,
    )

    print(f"run_id={result.run_id}")
    print(f"metrics={result.metrics}")
    print(f"output={args.output}")
    for trace in traces:
        print(f"{trace.case_id}: {trace.prediction}")


if __name__ == "__main__":
    main()
