# Ministral 8B token-TOVA b=256 — substrate-fix verify cell

**Artifact**: `results/ministral-8b-tova-b256-substrate-fix-verify-2026-05-27.json`
**Date**: 2026-05-27
**Model**: `mistralai/Ministral-8B-Instruct-2410`
**Slice**: niah_multikey_1_4k validation[:50]
**Cell**: token-TOVA at b=256 (single comparison cell)
**Cost**: $0.1865
**Wall-clock**: 268.8s
**Harness**: vorn-mat main (post-PR #23 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)

## Result

**Hit rate: 0.000** (50/50). Wilson 95% CI: [0.000, 0.071].

## Purpose

Single-cell verification dispatched after PR #23 (PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True) merged to main, before launching the broader 8-cell Ministral re-run (Step 3 of multistep-fillout dispatch). The purpose was empirical: confirm the substrate-fix actually resolves the allocator-fragmentation OOM on the exact cell that crashed in the pre-fix wave.

## Reading

The substrate-fix MECHANISM is CONFIRMED. The cell that OOMed at fixture 0 in the pre-fix wave (with 27 GB reserved-but-unallocated fragmentation overhead) now completes n=50 cleanly under the canonical env-var. No fragmentation OOM in the cell's full run.

The hit-rate of 0.000 is the **actual measured value** for Ministral 8B token-TOVA at b=256, independent of the fix. The prior n=9 partial read from the column-fill artifact was also 0.00 (per the multistep-fillout dispatch table). This confirms the partial-data interpretation: Ministral 8B token-TOVA at b=256 truly floors at this budget.

Two distinct findings in one cell: (1) the fragmentation fix made the measurement possible, and (2) the measurement reveals the floor.

## Provenance / paper-claim impact

- Companion to the 8-cell Ministral re-run (Step 3 of multistep-fillout, vorn-mat#24): same family + b=256 cell, separately verified
- Provides the single-cell empirical anchor for PR #23's substrate-fix-confirmation narrative
- Paper §C.2 + §6.8 substrate-evolution arcs reference the allocator-class fix; this cell is the smoking-gun confirmation

## Reproducibility

- Harness: vorn-mat main (post-PR #23; commit 881b581 or later)
- A100-80GB on `layne-79000` profile (then) / `layne` profile (re-validation possible)
- Per-fixture observations preserved in artifact (50 entries; pattern is base-Ministral degeneracy under aggressive eviction, no special-token cascade)
- Wilson 95% CI: [0.000, 0.071] (n=50, p=0.00)
