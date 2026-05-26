#!/usr/bin/env python3
"""Collect results from a fire-and-forget Modal wave dispatched by
run_modal_cells_parallel.py.

USAGE:
  python examples/collect_modal_wave.py \\
    --wave-state-path .benchmarks/parallel-cells/wave-state.json

The command blocks on `modal.FunctionCall.from_id(call_id).get()` until the
cloud-side wave completes; on completion, writes reports.json (always) and
failures.json (only if non-empty) to the output_dir recorded in
wave-state.json.

IDEMPOTENT: re-running with the same wave-state.json returns the same
artifacts (from disk) without re-fetching from Modal. Use --force to override
and re-fetch (e.g., to pull a fresher copy if the cloud run was extended or
mutated).

Pair with run_modal_cells_parallel.py — the dispatch command prints the
exact collect invocation including the wave-state-path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vorn_mat import collect_wave_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--wave-state-path",
        type=Path,
        required=True,
        help="Path to wave-state.json written by run_modal_cells_parallel.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Override the output_dir recorded in wave-state.json. Default uses "
            "the output_dir from the state file (same dir as wave-state.json)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-fetch from Modal even if reports.json already exists in "
            "output_dir (default: idempotent — read from disk if present)."
        ),
    )
    args = parser.parse_args()

    if not args.wave_state_path.exists():
        print(
            f"ERROR: wave-state-path does not exist: {args.wave_state_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        # Derive output_dir from wave-state.json's recorded path
        import json as _json

        state_data = _json.loads(args.wave_state_path.read_text())
        output_dir = Path(state_data["output_dir"])

    summary = collect_wave_results(
        wave_state_path=args.wave_state_path,
        output_dir=output_dir,
        force_recollect=args.force,
    )

    print(f"call_id={summary['call_id']}")
    print(f"cells_succeeded={summary['cells_succeeded']}")
    print(f"cells_failed={summary['cells_failed']}")
    print(f"output_dir={summary['output_dir']}")
    print(f"from_cache={summary['from_cache']}")


if __name__ == "__main__":
    main()
