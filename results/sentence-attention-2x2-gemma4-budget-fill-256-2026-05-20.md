# Gemma 4 E4B-it Sentence-Attention Budget Fill @ 256 — 2026-05-20

Run conditions:
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `google/gemma-4-E4B-it`
- Budget: `256`

| Method | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | KV savings |
| --- | --- | --- | --- | --- | --- |
| token_vorn | 0.00 | [0.0000, 0.0714] | 366.41s | $0.2543 | 93.51% |
| sentence_vorn | 0.00 | [0.0000, 0.0714] | 214.92s | $0.1492 | 93.95% |
| tova | 0.48 | [0.3480, 0.6149] | 490.58s | $0.3405 | 93.51% |
| sentence_tova | 1.00 | [0.9286, 1.0000] | 283.35s | $0.1966 | 93.74% |

## Pairwise Tests

- sentence_tova_256_guarded vs tova_256_guarded, exact McNemar on `[[24, 26], [0, 0]]`: `p = 2.98023e-08`
- sentence_tova_256_guarded vs sentence_256_guarded, exact McNemar on `[[0, 50], [0, 0]]`: `p = 1.77636e-15`
- sentence_256_guarded vs token_256_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`

## Read

- At `256`, both vorn rows are non-discriminative (`0.00`), token TOVA recovers partially (`0.48`), and sentence-TOVA reaches ceiling (`1.00`).
- This is the clean low-budget Gemma 4 rescue-spectrum edge for Figure 3.
