#!/usr/bin/env python3
"""Build the config#316 LongBench PassageRetrieval-en cell-spec file.

This script performs no Modal calls. It only writes the exact 8 request specs
that the preregistration authorizes after lock timestamp.
"""
# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vorn_mat import LONGBENCH_REVISION, build_passage_retrieval_en_cell_specs
from vorn_mat.modal_app import default_modal_app_spec


def main(
    output: str = str(
        ROOT / ".benchmarks" / "longbench-passage-retrieval-cell-specs.json"
    ),
    results_root: str = default_modal_app_spec().results_root,
    dataset_revision: str = LONGBENCH_REVISION,
) -> None:
    specs = build_passage_retrieval_en_cell_specs(
        results_root=results_root,
        dataset_revision=dataset_revision,
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(list(specs), indent=2, sort_keys=True))

    print(f"cell_specs={len(specs)}")
    print(f"dataset_revision={dataset_revision}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
