#!/usr/bin/env python3
"""Recompute Appendix A artifact accounting from the on-disk JSON artifacts.

This script is the authoritative source-of-truth for the numbers reported in
Appendix A.5 (artifact / row / observation / cost / GPU-time totals) and the
schema-bucket breakdown in Appendix A.2. Appendix A.5 wording explicitly
defers to this script's output via "computed via this script".

Counting rules (deliberate, explicit, narrow):

1. Artifacts counted: every `*.json` file in `results/`.

2. Rows counted: dict-valued entries inside any of the following row-array
   fields:
       rows, sentence_rows, token_rows, ceiling_rows, adaptive_rows,
       comparisons, token_vorn_rows, sentence_vorn_rows, token_attention_rows,
       sentence_attention_rows,
       phase1_cells, phase3_a100_cells  (Phase 3 eviction-gate composed artifact)

   Special row shapes also counted:
       - Top-level list artifacts where the file itself is a list of result
         row-dicts (eviction-gate dispatches that bypass the row-envelope).
       - Top-level single-cell artifacts where the file has a `result` dict
         with `observations[]` at top-level (vanilla diagnostic single-cell
         artifacts); treated as ONE row with estimated_cost_usd +
         elapsed_seconds at top-level and observations under result.

   Excluded by design:
       - failed_rows (OOM / runtime_unsupported boundary cells; documented as
         scope-of-coverage boundary, not as result rows)
       - historical_reference_rows (aggregate-status records, not per-experiment)
       - fresh_rows (paired-status records used for freshness audit)
       - cells_by_family (merged-view summary records in multistep-fillout;
         the underlying full-schema rows live in their source artifacts and
         are counted there to avoid double-counting)
       - phase3_h100_oom_skipped (skipped-by-policy records, not result rows)
       - failure-list artifacts where items have {error, request} shape
         (these mirror the failed_rows exclusion at the artifact level)

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
    "phase1_cells", "phase3_a100_cells",
)
EXCLUDED_ROW_ARRAYS = (
    "failed_rows", "historical_reference_rows", "fresh_rows",
    "phase3_h100_oom_skipped",
)


def classify_bucket(d, top_keys):
    if not isinstance(d, dict):
        # Top-level list artifacts: classify by inspection of first item.
        if isinstance(d, list) and d and isinstance(d[0], dict):
            if "error" in d[0] and "request" in d[0]:
                return "failure_list_envelope"
            return "top_level_list_envelope"
        return "unrecognized_top_level"
    schema = d.get("schema_version", "NO_SCHEMA")
    if schema == "NO_SCHEMA":
        return "legacy_no_schema"
    if schema == "result-envelope-v0.2":
        return "v0.2_hyphenated"
    if "distribution-v1" in schema:
        return "v0.2_distribution_extended"
    if schema.startswith("vorn-mat/"):
        return "phase2c_2d_new_schema"
    if "phase1_cells" in top_keys or "phase3_a100_cells" in top_keys:
        return "phase3_eviction_gate_composed"
    if "cells_by_family" in top_keys:
        return "multistep_fillout_merged_view"
    if "result" in top_keys and isinstance(d.get("result"), dict) and "observations" in d["result"]:
        return "top_level_single_cell_diagnostic"
    if "sentence_rows" in top_keys or "token_rows" in top_keys:
        return "v0.2_sentence_token_split"
    if any(k in top_keys for k in ("ceiling_rows", "adaptive_rows", "comparisons")):
        return "v0.2_domain_specific"
    if any(k in top_keys for k in ("token_vorn_rows", "sentence_vorn_rows", "token_attention_rows", "sentence_attention_rows")):
        return "v0.2_2x2_multi_array"
    if "rows" in top_keys:
        return "v0.2_canonical_flat"
    return "v0.2_extension_wave"


def _accumulate_row(row, counts):
    """Add a single row dict's contributions to the running counts.

    Observations may live at one of two row-relative paths depending on the
    artifact schema:
      - `row.observations[]` (canonical result-envelope rows)
      - `row.result.observations[]` (Phase 3 composed cells + top-level-list
        envelopes where the row wraps a `result` sub-dict)

    Prefer the direct path; fall back to the nested path if the direct slot
    is empty or absent. Never double-count.
    """
    if not isinstance(row, dict):
        return
    counts["n_rows"] += 1
    obs = row.get("observations")
    if not (isinstance(obs, list) and obs):
        inner = row.get("result")
        if isinstance(inner, dict):
            obs = inner.get("observations")
    if isinstance(obs, list) and obs:
        counts["rows_with_obs"] += 1
        counts["total_obs"] += len(obs)
    cost = row.get("estimated_cost_usd")
    if isinstance(cost, (int, float)) and cost > 0:
        counts["total_cost"] += float(cost)
    el = row.get("elapsed_seconds")
    if isinstance(el, (int, float)) and el > 0:
        counts["total_elapsed_s"] += float(el)


def _accumulate_top_level_single_cell(d, counts):
    """For diagnostic single-cell artifacts where the whole file IS one row.

    These artifacts have estimated_cost_usd + elapsed_seconds at top level and
    observations under result.observations[]. We treat the artifact as one
    row and count its top-level cost/elapsed + result.observations count.
    """
    result = d.get("result")
    if not isinstance(result, dict):
        return
    counts["n_rows"] += 1
    obs = result.get("observations")
    if isinstance(obs, list) and obs:
        counts["rows_with_obs"] += 1
        counts["total_obs"] += len(obs)
    cost = d.get("estimated_cost_usd")
    if isinstance(cost, (int, float)) and cost > 0:
        counts["total_cost"] += float(cost)
    el = d.get("elapsed_seconds")
    if isinstance(el, (int, float)) and el > 0:
        counts["total_elapsed_s"] += float(el)


def main():
    files = sorted(RESULTS_DIR.glob("*.json"))
    counts = {
        "n_rows": 0, "rows_with_obs": 0, "total_obs": 0,
        "total_cost": 0.0, "total_elapsed_s": 0.0,
    }
    buckets = defaultdict(list)

    for fp in files:
        try:
            d = json.loads(fp.read_text())
        except Exception:
            continue

        # Top-level list artifacts: each list-item is a row (or excluded as
        # failure-list if items have {error, request} shape).
        if isinstance(d, list):
            buckets[classify_bucket(d, set())].append(fp.name)
            if d and isinstance(d[0], dict) and "error" in d[0] and "request" in d[0]:
                continue  # failure-list envelope; excluded
            for row in d:
                _accumulate_row(row, counts)
            continue

        if not isinstance(d, dict):
            buckets[classify_bucket(d, set())].append(fp.name)
            continue

        top_keys = set(d.keys())
        buckets[classify_bucket(d, top_keys)].append(fp.name)

        # Top-level single-cell diagnostic artifacts: the whole file is one row.
        if "result" in top_keys and isinstance(d.get("result"), dict) \
                and "observations" in d["result"] and "rows" not in top_keys:
            _accumulate_top_level_single_cell(d, counts)
            continue

        for arr_key in COUNTED_ROW_ARRAYS:
            arr = d.get(arr_key)
            if not isinstance(arr, list):
                continue
            for row in arr:
                _accumulate_row(row, counts)

        # v0.2_extension_wave artifacts wrap rows inside `models[]` or
        # `families[]` containers. Descend one level and pick up nested
        # COUNTED_ROW_ARRAYS from each wrapper item.
        for wrapper_key in ("models", "families"):
            wrapper = d.get(wrapper_key)
            if not isinstance(wrapper, list):
                continue
            for item in wrapper:
                if not isinstance(item, dict):
                    continue
                for arr_key in COUNTED_ROW_ARRAYS:
                    arr = item.get(arr_key)
                    if not isinstance(arr, list):
                        continue
                    for row in arr:
                        _accumulate_row(row, counts)

    n_rows = counts["n_rows"]
    rows_with_obs = counts["rows_with_obs"]
    total_obs = counts["total_obs"]
    total_cost = counts["total_cost"]
    total_elapsed_s = counts["total_elapsed_s"]

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
