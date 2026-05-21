# Gemma 2 9B 4k Fast-Read — 2026-05-16

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `google/gemma-2-9b-it`

## Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Mean evicted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | full context | prefix + recent | 1.00 | [0.9286, 1.0000] | 262.90s | $0.1825 | 0.00% |
| TOVA | 1024 | prefix + recent | 0.60 | [0.4618, 0.7239] | 723.99s | $0.5024 | 73.99% |
| Sentence vorn | 1024 | prefix + recent | 0.86 | [0.7381, 0.9305] | 524.78s | $0.3642 | 74.21% |

## Pairwise Tests

- sentence_1024_guarded vs tova_1024_guarded, exact McNemar on `[[26, 17], [4, 3]]`: `p = 0.00719738`
- vanilla_floor vs tova_1024_guarded, exact McNemar on `[[30, 20], [0, 0]]`: `p = 1.90735e-06`
- sentence_1024_guarded vs vanilla_floor, exact McNemar on `[[43, 0], [7, 0]]`: `p = 0.015625`

## Read

- Gemma 2 does **not** reproduce the Gemma 4 pattern on this slice. Its full-context ceiling is `1.00`, but `sentence_vorn@1024` stays much closer to that ceiling (`0.86`) than `TOVA@1024` (`0.60`).
- That substantially weakens the “Google-lab-wide attention-weight preference” hypothesis. Same lab, different family outcome.
- The current best read is narrower: whatever drives Gemma 4’s attention-weight-favored behavior does not trivially generalize to Gemma 2, so the effect is more likely tied to Gemma-4-specific architectural or training details than to a Google-wide channel signature.
