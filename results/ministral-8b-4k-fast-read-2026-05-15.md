# Ministral 8B 4k Fast-Read — 2026-05-15

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Ministral-8B-Instruct-2410`

## Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Mean evicted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | full context | prefix + recent | 1.00 | [0.9286, 1.0000] | 198.71s | $0.1379 | 0.00% |
| TOVA | 1024 | prefix + recent | 0.44 | [0.3116, 0.5769] | 476.08s | $0.3304 | 73.76% |
| Sentence vorn | 1024 | prefix + recent | 1.00 | [0.9286, 1.0000] | 360.56s | $0.2502 | 74.02% |

## Pairwise Tests

- sentence_1024_guarded vs tova_1024_guarded, exact McNemar on `[[22, 28], [0, 0]]`: `p = 7.45058e-09`
- vanilla_floor vs tova_1024_guarded, exact McNemar on `[[22, 28], [0, 0]]`: `p = 7.45058e-09`
- sentence_1024_guarded vs vanilla_floor, exact McNemar on `[[50, 0], [0, 0]]`: `p = 1`

## Read

- Ministral 8B does **not** look Gemma-like on this slice. Its full-context ceiling is `1.00`, but `TOVA@1024` drops sharply to `0.44` while `sentence_vorn@1024` stays at `1.00`.
- That makes the fifth-family fast-read a strong architectural prediction test in the residual-direction direction, not in the attention-weight direction. The interleaved-attention family here aligns with the sentence-preserving channel rather than the Gemma-style attention-weight dominance.
- The immediate next question is whether the full cross-baseline matrix on Ministral keeps this separation once token_vorn and H2O are added, or whether the fast-read is only exposing the most extreme channel contrast.
