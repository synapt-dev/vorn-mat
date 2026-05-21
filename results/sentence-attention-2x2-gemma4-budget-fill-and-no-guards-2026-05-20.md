# Gemma 4 Sentence-Attention Budget Fill + No-Guards — 2026-05-20

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `google/gemma-4-E4B-it`

## Guarded budget fill

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost | Mean evicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sentence vorn @ 512 | 512 | prefix + recent | 0.04 | [0.0110, 0.1346] | historical_stale_reference | 269.72s | $0.1872 | 87.45% |
| TOVA-style @ 512 | 512 | prefix + recent | 0.34 | [0.2244, 0.4785] | historical_stale_reference | 383.05s | $0.2658 | 87.01% |
| Sentence TOVA-style @ 512 | 512 | prefix + recent | 0.72 | [0.5833, 0.8253] | current_n50_override | 276.98s | $0.1922 | 87.27% |
| Sentence H2O-style @ 512 | 512 | prefix + recent | 0.68 | [0.5419, 0.7924] | current_n50_override | 277.84s | $0.1928 | 87.32% |
| Sentence vorn @ 1536 | 1536 | prefix + recent | 0.68 | [0.5419, 0.7924] | historical_stale_reference | 394.20s | $0.2736 | 61.33% |
| TOVA-style @ 1536 | 1536 | prefix + recent | 0.98 | [0.8950, 0.9965] | historical_stale_reference | 623.58s | $0.4328 | 61.04% |
| Sentence TOVA-style @ 1536 | 1536 | prefix + recent | 0.98 | [0.8950, 0.9965] | current_n50_override | 394.51s | $0.2738 | 61.29% |
| Sentence H2O-style @ 1536 | 1536 | prefix + recent | 0.98 | [0.8950, 0.9965] | current_n50_override | 392.07s | $0.2721 | 61.30% |

## No-guards extension

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost | Mean evicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sentence vorn @ 1024 no-guards | 1024 | none | 0.22 | [0.1275, 0.3524] | historical_stale_reference | 290.25s | $0.2014 | 74.37% |
| TOVA-style @ 1024 no-guards | 1024 | none | 0.94 | [0.8378, 0.9794] | historical_stale_reference | 447.78s | $0.3108 | 74.03% |
| Sentence vorn @ 1024 no-guards fresh | 1024 | none | 0.22 | [0.1275, 0.3524] | current_n50_override | 300.51s | $0.2086 | 74.37% |
| Sentence TOVA-style @ 1024 no-guards | 1024 | none | 0.88 | [0.7619, 0.9438] | current_n50_override | 332.93s | $0.2311 | 74.30% |

## Pairwise Tests

- sentence_tova_512_guarded vs tova_512_guarded_historical, exact McNemar on `[[13, 23], [4, 10]]`: `p = 0.000310749`
- sentence_tova_512_guarded vs sentence_512_guarded_historical, exact McNemar on `[[0, 36], [2, 12]]`: `p = 5.39876e-09`
- sentence_h2o_512_guarded vs sentence_512_guarded_historical, exact McNemar on `[[0, 34], [2, 14]]`: `p = 1.94123e-08`
- sentence_tova_1536_guarded vs tova_1536_guarded_historical, exact McNemar on `[[48, 1], [1, 0]]`: `p = 1`
- sentence_tova_1536_guarded vs sentence_1536_guarded_historical, exact McNemar on `[[34, 15], [0, 1]]`: `p = 6.10352e-05`
- sentence_h2o_1536_guarded vs sentence_1536_guarded_historical, exact McNemar on `[[34, 15], [0, 1]]`: `p = 6.10352e-05`
- sentence_1024_noguards vs gemma_sentence_noguards, exact McNemar on `[[11, 0], [0, 39]]`: `p = 1`
- sentence_tova_1024_noguards vs gemma_tova_noguards, exact McNemar on `[[42, 2], [5, 1]]`: `p = 0.453125`
- sentence_tova_1024_noguards vs sentence_1024_noguards, exact McNemar on `[[10, 34], [1, 5]]`: `p = 2.09548e-09`

## Read

- This artifact fills the missing Gemma 4 sentence-attention budgets at 512 and 1536 and reruns the 1024 no-guards sentence surface for current-run confirmation.
- The guarded 512 and 1536 rows should be interpreted against historical token/sentence/TOVA controls, not as a fresh six-method rerun.
