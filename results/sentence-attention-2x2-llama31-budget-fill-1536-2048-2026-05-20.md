# Llama 3.1 8B Sentence-Attention Budget Fill @ 1536 and 2048 — 2026-05-20

Run conditions:
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `meta-llama/Llama-3.1-8B-Instruct`

| Budget | Method | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | KV savings |
| --- | --- | --- | --- | --- | --- | --- |
| 1536 | token_vorn | 1.00 | [0.9286, 1.0000] | 403.86s | $0.2803 | 60.04% |
| 1536 | sentence_vorn | 1.00 | [0.9286, 1.0000] | 316.93s | $0.2200 | 60.30% |
| 1536 | tova | 0.96 | [0.8654, 0.9890] | 408.11s | $0.2832 | 60.04% |
| 1536 | sentence_tova | 1.00 | [0.9286, 1.0000] | 306.43s | $0.2127 | 60.30% |
| 2048 | token_vorn | 1.00 | [0.9286, 1.0000] | 515.95s | $0.3581 | 46.72% |
| 2048 | sentence_vorn | 1.00 | [0.9286, 1.0000] | 353.37s | $0.2452 | 47.00% |
| 2048 | tova | 1.00 | [0.9286, 1.0000] | 391.09s | $0.2714 | 46.72% |
| 2048 | sentence_tova | 1.00 | [0.9286, 1.0000] | 326.54s | $0.2266 | 46.95% |

## Pairwise Tests

- sentence_tova_1536_guarded vs tova_1536_guarded, exact McNemar on `[[48, 2], [0, 0]]`: `p = 0.5`
- sentence_tova_1536_guarded vs sentence_1536_guarded, exact McNemar on `[[50, 0], [0, 0]]`: `p = 1`
- sentence_1536_guarded vs token_1536_guarded, exact McNemar on `[[50, 0], [0, 0]]`: `p = 1`
- sentence_tova_2048_guarded vs tova_2048_guarded, exact McNemar on `[[50, 0], [0, 0]]`: `p = 1`
- sentence_tova_2048_guarded vs sentence_2048_guarded, exact McNemar on `[[50, 0], [0, 0]]`: `p = 1`
- sentence_2048_guarded vs token_2048_guarded, exact McNemar on `[[50, 0], [0, 0]]`: `p = 1`

## Read

- At `1536`, token_vorn, sentence_vorn, and sentence_tova are all at `1.00`, while token TOVA is `0.96`.
- At `2048`, all four methods are at ceiling (`1.00`).
