# Gemma 3 12B-pt vanilla baseline on canonical bf16 harness

**Artifact**: `results/gemma3-12b-pt-diagnostic-2026-05-26.json`
**Date**: 2026-05-26
**Model**: google/gemma-3-12b-pt (pretrained base, no chat template)
**Slice**: niah_multikey_1_4k validation[:50]
**Cost**: $0.2596
**Wall-clock**: 374.1s
**Modal app**: ap-bgy80KUrqpBqxwSRxNHIhQ on layne-79000

## Result

**Hit rate: 1.00** (50/50). Wilson 95% CI: [0.9286, 1.0000].

## Interpretation

Gemma 3 12B-pt is fully no-eviction-discriminative on niah_multikey_1_4k. The prior 0.00 reading in `results/cross-family-extension-wave-2026-05-20.json` (which fed the paper's "drowning-floor boundary entry" framing in Appendix C.2) is a **harness artifact**, not a model-capability property.

### Root cause of prior 0.00

Prior artifact: 50/50 fixtures with `prediction: ""` (empty string). Fingerprint matches the fp16 NaN-cascade pattern documented in the vorn-mat README under "Substrate improvement from failure: bf16 live generation stability":

1. fp16 logits produce NaN under live-generation cache conditions
2. `argmax(NaN)` falls back to token id 0
3. In Gemma 3 tokenizer, token id 0 is `<pad>` (verified: `AutoTokenizer.from_pretrained('google/gemma-3-12b-pt').decode([0]) == '<pad>'`, equivalent: `tok.pad_token == '<pad>'`)
4. `skip_special_tokens=True` on decode drops the pad token → decoded output is `""`
5. The model is functionally producing zero non-special tokens for all 50 fixtures

The 2026-05-20 artifact predates the bf16 fix landing (2026-05-23). Same class-of-bug that already recovered Gemma 3 8B and Qwen 2.5 7B-Instruct on later re-runs.

### Post-fix predictions

Coherent natural-text continuation with correct 7-digit needle retrieved as first content tokens. The model is a pretrained base (no chat template, no instruction tuning), so it does not terminate cleanly — instead it continues with next-token-prediction of the haystack pattern, e.g.:

```
'9375710.\nWhat is the special magic number for faded-site mentioned in the provided text? The special magic number for faded'
```

Scoring is robust to this continuation: it picks up the answer (`9375710.`) at the start. 50/50 fixtures retrieve correctly.

Distribution of prediction lengths shows the bimodal base-model behavior:
- 6 fixtures terminate cleanly at length 8 (just `"<7-digit>."`)
- 44 fixtures continue with haystack-pattern continuation (length 100-150 chars)

Both are scored correct.

### Local pre-check

`google/gemma-3-12b-pt` has no `chat_template` attribute (it is the pretrained base, not instruction-tuned). The vorn-mat harness handles this correctly via fallback at `vorn_mat/local_exec.py:320-323`: `encoded = self._tokenizer(prompt, return_tensors="pt")` (raw-prompt path). The harness is structurally correct for base-model inputs; the 0.00 floor was not a base-model-vs-instruction-prompt mismatch.

## Cascade decision

Per dispatch cascade rule (hit-rate ≥ 0.50 = discriminative → STOP):
- 1.00 is well above the 0.50 threshold
- Eviction cells NOT dispatched
- Total Gemma 3 12B-pt diagnostic cost: $0.2596

## Paper-substantive impact

The paper's Appendix C.2 currently documents Gemma 3 12B-pt as "kept as the drowning-floor boundary entry" after Gemma 3 12B-it was excluded for multi-EOS bug. This framing is now invalidated. Two paper-side options for Opus:

**Option A — remove Gemma 3 12B-pt from main matrix entirely.** Since the family is now empirically no-eviction-discriminative and we don't have eviction-cell coverage, treat as "not in this revision's panel; deferred to v2."

**Option B — keep as a side-note in Appendix C** with the substrate-restoration-arc disclosure: prior 0.00 was harness artifact; canonical bf16 harness recovers to 1.00; full eviction coverage deferred to v2.

The currently-in-paper "drowning-floor boundary" language must be removed under either option.

## Pattern: stale-pre-fix-harness artifacts in main

This is the **third occurrence** of the same class-of-failure:
- Gemma 3 8B-it (recovered 2026-05-23 via bf16 fix landing)
- Qwen 2.5 7B-Instruct (recovered 2026-05-26 via bf16 fix cross-family re-run)
- Gemma 3 12B-pt (recovered 2026-05-26 — this diagnostic)

The substrate-fix-from-failure for the *next* occurrence should be:

> When a harness-stability fix lands, immediately schedule a cross-family audit re-run of all vanilla-baseline cells produced by the pre-fix harness, not only the family where the failure first surfaced.

This is filed as a recurring-pattern observation; full substrate codification deferred to follow-up if Opus / Layne agrees.

## Reproducibility

- Harness: vorn-mat main at current head (bf16-stability fix landed)
- No chat-template kwargs needed (base model — raw-prompt path)
- A100-80GB on `layne-79000` profile
- Per-fixture observations preserved in artifact
- Wilson 95% CI: [0.9286, 1.0000] (n=50, p=1.00)
