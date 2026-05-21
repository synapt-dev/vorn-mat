#!/usr/bin/env python3
"""Recompute Appendix A artifact accounting from the on-disk JSON artifacts.

This script is the authoritative source-of-truth for the numbers reported in
Appendix A.5 (artifact / row / observation / cost / GPU-time totals) and the
schema-bucket breakdown in Appendix A.2. Appendix A.5 wording explicitly
defers to this script's output via "computed via this script".

Counting rules (deliberate, explicit, narrow):

1. Artifacts counted: every `*.json` file in `prototypes/vorn-mat/results/`.

2. Rows counted: dict-valued entries inside any of the following row-array
   fields:
       rows, sentence_rows, token_rows, ceiling_rows, adaptive_rows,
       comparisons, token_vorn_rows, sentence_vorn_rows, token_attention_rows,
       sentence_attention_rows
   Excluded by design:
       - failed_rows (OOM / runtime_unsupported boundary cells; documented as
         scope-of-coverage boundary, not as result rows)
       - historical_reference_rows (aggregate-status records, not per-experiment)
       - fresh_rows (paired-status records used for freshness audit)

3. Observations: per-fixture entries in observations[] of each counted row.

4. Cost (USD): estimated_cost_usd of each counted row, summed.

5. GPU time (hours): elapsed_seconds of each counted row, summed, → hours.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
COUNTED_ROW_ARRAYS = (
    "rows", "sentence_rows", "token_rows",
    "ceiling_rows", "adaptive_rows", "comparisons",
    "token_vorn_rows", "sentence_vorn_rows",
    "token_attention_rows", "sentence_attention_rows",
)
EXCLUDED_ROW_ARRAYS = ("failed_rows", "historical_reference_rows", "fresh_rows")


def classify_bucket(d, top_keys):
    schema = d.get("schema_version", "NO_SCHEMA")
    if schema == "NO_SCHEMA":
        return "legacy_no_schema"
    if schema == "result-envelope-v0.2":
        return "v0.2_hyphenated"
    if "distribution-v1" in schema:
        return "v0.2_distribution_extended"
    if schema.startswith("vorn-mat/"):
        return "phase2c_2d_new_schema"
    if "sentence_rows" in top_keys or "token_rows" in top_keys:
        return "v0.2_sentence_token_split"
    if any(k in top_keys for k in ("ceiling_rows", "adaptive_rows", "comparisons")):
        return "v0.2_domain_specific"
    if any(k in top_keys for k in ("token_vorn_rows", "sentence_vorn_rows", "token_attention_rows", "sentence_attention_rows")):
        return "v0.2_2x2_multi_array"
    if "rows" in top_keys:
        return "v0.2_canonical_flat"
    return "v0.2_extension_wave"


def main():
    files = sorted(RESULTS_DIR.glob("*.json"))
    n_rows = rows_with_obs = total_obs = 0
    total_cost = total_elapsed_s = 0.0
    buckets = defaultdict(list)

    for fp in files:
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue
        top_keys = set(d.keys())
        buckets[classify_bucket(d, top_keys)].append(fp.name)
        for arr_key in COUNTED_ROW_ARRAYS:
            arr = d.get(arr_key)
            if not isinstance(arr, list):
                continue
            for row in arr:
                if not isinstance(row, dict):
                    continue
                n_rows += 1
                obs = row.get("observations")
                if isinstance(obs, list) and obs:
                    rows_with_obs += 1
                    total_obs += len(obs)
                cost = row.get("estimated_cost_usd")
                if isinstance(cost, (int, float)) and cost > 0:
                    total_cost += float(cost)
                el = row.get("elapsed_seconds")
                if isinstance(el, (int, float)) and el > 0:
                    total_elapsed_s += float(el)

    print(f"Total artifacts:        {len(files)}")
    print(f"Total counted rows:     {n_rows}")
    print(f"Rows with observations: {rows_with_obs}")
    print(f"Total observations:     {total_obs:,}")
    print(f"Total cost (USD):       ${total_cost:.2f}")
    print(f"Total GPU time (hours): {total_elapsed_s/3600:.2f}")
    print()
    print("Bucket breakdown:")
    total = 0
    for b in sorted(buckets):
        print(f"  {b:>30}: {len(buckets[b]):>3}")
        total += len(buckets[b])
    print(f"  {'TOTAL':>30}: {total:>3}")


if __name__ == "__main__":
    main()
