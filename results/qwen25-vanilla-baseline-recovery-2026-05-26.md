# Qwen 2.5 7B Instruct vanilla baseline recovery on canonical bf16-fixed harness

**Artifact**: `results/qwen25-vanilla-baseline-recovery-2026-05-26.json`
**Date**: 2026-05-26
**Model**: `Qwen/Qwen2.5-7B-Instruct`
**Slice**: niah_multikey_1_4k validation[:50]
**Cost**: $0.1236
**Wall-clock**: 178.1s
**Modal app**: ap-dGOMLoHyemhxdVjzqoLdTW on layne-79000

## Result

**Hit rate: 0.640** (32/50). Wilson 95% CI: [0.502, 0.755].

## Interpretation

This is the Qwen 2.5 7B Instruct vanilla baseline re-run on the canonical bf16-fixed harness (Lane 1 from the 2026-05-26 morning Qwen 2.5 investigation). It recovers from the pre-bf16-fix reading of **0.040** (50/50 fixtures with `!`-cascade fingerprint per the 2026-05-14 artifact) to **0.640** under the canonical bf16-stability fix.

The bf16 fix MECHANISM is confirmed (16× improvement from 0.04). The `!`-cascade pattern is ELIMINATED in the new artifact (0/50 fixtures with 5+ `!` chars; old artifact had 50/50). Predictions are now coherent well-formed answers in Qwen 2.5's instruction-following format. The 18 wrong predictions are RETRIEVAL-wrong, not formatting-wrong — the model produces a well-formed 7-digit-bold answer, just not the correct one from the haystack.

The cell does NOT reach the 0.94-1.00 range that other vorn-favoring families hit on this slice (Mistral, Llama, Ministral, Gemma 2 = 1.00). 0.640 is consistent with Qwen 2.5's capability ceiling on multikey NIAH at 4k — a separate-from-fragmentation issue (the bf16 fix resolves the harness bug; the residual gap is a model property, not a harness artifact).

## Provenance / paper-claim impact

- Documented in vorn-mat#22 paper §C.2 family-restoration arc for Qwen 2.5 (bf16 fix lifts 0.04 → 0.640)
- Paper Table 1 Qwen 2.5 row: no-eviction = 0.64 ✓ matches this artifact
- Lane 2 STOP decision (per cascade rule): 0.640 < 0.94 ceiling threshold; Lane 2 full row refresh deferred until budgeted separately

## Reproducibility

- Harness: vorn-mat main (post-bf16-fix; circa 2026-05-23+)
- A100-80GB on `layne-79000` profile
- Per-fixture observations preserved in artifact (50 entries)
- Wilson 95% CI: [0.502, 0.755] (n=50, p=0.64)
