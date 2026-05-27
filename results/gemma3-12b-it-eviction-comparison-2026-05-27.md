# Gemma 3 12B-it eviction comparison — sentence-vorn b=1024

**Artifact**: `results/gemma3-12b-it-eviction-comparison-2026-05-27.json`
**Date**: 2026-05-27
**Model**: `google/gemma-3-12b-it` (instruction-tuned variant of the 12B family)
**Slice**: niah_multikey_1_4k validation[:50]
**Cell**: sentence-vorn at b=1024 (single comparison cell)
**Harness**: vorn-mat main post-PR#23 (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`)
**Hardware**: A100-80GB on `layne` profile

## Result

**Hit rate: 0.000** (50/50). Wilson 95% CI: [0.0000, 0.0714]. Cost: $0.3919. Wall-clock: 565s.

## Purpose

Single comparison cell to test the base-vs-instruct hypothesis for the 12B-pt = 0.000 eviction-panel finding. Specifically: if instruction-tuning rescues coherence under partial-context eviction, sentence-vorn at b=1024 on 12B-it should land near the instruction-tuned-family cluster (0.96-1.00); if it floors, base-vs-instruct is ruled out as the explanation.

## Reading

The instruction-tuned variant floors at 0.000 — IDENTICAL to the base 12B-pt variant on this cell. Base-vs-instruct is NOT the mechanism behind the 12B-pt 0.000 finding.

Prediction-text inspection (same fingerprint methodology as the 12B-pt analysis):

- **0/50 empty `''` predictions** (Form 1 pad-cascade RULED OUT)
- **0/50 `!`-cascade** (Form 2 cascade RULED OUT)
- **0/50 visible `<pad>` token** (no special-token cascade)
- **50/50 coherent token-level continuations with n-gram repetition + topic drift, no needle retrieval, no question-pattern reproduction**

Sample predictions:

```
'the last date the date the date the last date date date date date date date
 date date date date date date date date date date date date date date date'

', factory floor, factory floor.\n\nthe number, factory floor ability chase
 that and email etc etc etc etc etc " the floor etc'

'is like, the last. The first, the last. The over, the last the. the last
 the last thing, the last. the last. the'
```

Same fingerprint as 12B-pt (`"the the the to be the to grow"`, `"of a lot of a lot of"`, `"while before. while before."`): coherent tokens, n-gram repetition, topic drift, no needle.

## Cross-family comparison (b=1024 sentence-vorn, instruction-tuned)

| Family | b=1024 sentence-vorn |
|---|---|
| Mistral 7B v0.3 | 0.98+ |
| Llama 3.1 8B | 0.96 |
| Ministral 8B | 1.000 |
| Gemma 2 9B | 1.00 |
| **Gemma 3 12B-it** | **0.000** ← this cell |
| **Gemma 3 12B-pt** | **0.000** (Phase 3) |

Both Gemma 3 variants at 0.000 vs all other tested ~7-12B families at 0.96-1.00. The empirical anomaly is family-specific to Gemma 3.

## Mechanism — three candidates, NOT validated

Per Layne's 0-result discipline ("be careful with 0 results before claiming anything"), two-cells-both-zero is too thin to validate a specific causal mechanism. Three candidate explanations remain, no claim made:

1. **Gemma 3 hybrid sliding-window + global attention** interacts poorly with KV-cache eviction policies designed for full-attention models. Plausible architectural mechanism, but unconfirmed by independent ablation.
2. **Subtle harness artifact** specific to Gemma 3 that doesn't surface as the cascade-fingerprint we test for. Possible.
3. **Other unidentified cause**. Possible.

Distinguishing among these requires additional cells beyond what's in scope for this revision: e.g., Gemma 3 4B-it on same slice (size-axis), Gemma 3 12B-it at higher budgets (budget-driven recovery?), or independent harness validation (re-run on author-code if compatible).

## Paper-side disposition (Layne ratified 2026-05-27)

Gemma 3 is dropped from claim surface this revision:
- §C.2 entry: "observational anomaly, mechanism deferred"; documents empirical floor; names three candidate explanations; explicit that omitted from claim surface
- Figure 1 caption: "Gemma 3 is not plotted (observational anomaly; see Appendix C.2)"
- No mechanism claim made

This artifact ships as data + observational record, not paper-claim-supporting evidence.

## Reproducibility

- Harness: vorn-mat main post-PR#23 (env-var `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` baked into Modal image)
- Multi-EOS handling verified at `local_exec.py:342-343` (Gemma 3 12B-it has `eos_token_id=[1, 106]`)
- `skip_special_tokens=True` on decode at `local_exec.py:495`
- Chat template applied per standard Gemma family format
- A100-80GB; layne profile
- Single cell; full 50-fixture observations preserved in artifact JSON
