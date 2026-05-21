#!/usr/bin/env python3
"""Run the local Step 1 comparison pair on a NIAH slice."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vorn_mat.local_exec import run_local_step1_pair  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "examples" / "niah_smoke.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".benchmarks" / "step1-local",
    )
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    (vanilla_result, _), (vorn_result, _) = run_local_step1_pair(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        case_limit=args.limit,
    )

    print(f"vanilla_run_id={vanilla_result.run_id}")
    print(f"vanilla_metrics={vanilla_result.metrics}")
    print(f"vorn_run_id={vorn_result.run_id}")
    print(f"vorn_metrics={vorn_result.metrics}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
