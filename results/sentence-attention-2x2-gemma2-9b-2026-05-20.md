# Gemma 2 9B Sentence-Attention Extension — 2026-05-20

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `google/gemma-2-9b-it`

## Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost | Mean evicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | full context | prefix + recent | 1.00 | [0.9286, 1.0000] | historical_stale_reference | 262.90s | $0.1825 | 0.00% |
| Sentence vorn | 1024 | prefix + recent | 0.86 | [0.7381, 0.9305] | historical_stale_reference | 524.78s | $0.3642 | 74.21% |
| TOVA-style | 1024 | prefix + recent | 0.60 | [0.4618, 0.7239] | historical_stale_reference | 723.99s | $0.5024 | 73.99% |
| Sentence TOVA-style @ 256 | 256 | prefix + recent | 0.18 | [0.0977, 0.3080] | current_n50_override | 272.71s | $0.1893 | 93.81% |
| Sentence TOVA-style @ 512 | 512 | prefix + recent | 0.40 | [0.2761, 0.5382] | current_n50_override | 328.85s | $0.2282 | 87.30% |
| Sentence TOVA-style @ 1024 | 1024 | prefix + recent | 0.78 | [0.6476, 0.8725] | current_n50_override | 336.24s | $0.2334 | 74.32% |

## Pairwise Tests

- sentence_tova_1024_guarded vs tova_1024_guarded_historical, exact McNemar on `[[29, 10], [1, 10]]`: `p = 0.0117188`
- sentence_tova_1024_guarded vs sentence_1024_guarded_historical, exact McNemar on `[[35, 4], [8, 3]]`: `p = 0.387695`
- sentence_1024_guarded_historical vs tova_1024_guarded_historical, exact McNemar on `[[26, 17], [4, 3]]`: `p = 0.00719738`

## Read

- Gate verdict: `mistral_like_full_rescue`.
- Sentence-TOVA-style reaches the sentence-vorn regime at the gate budget.
- The 256 and 512 rows extend the same surface below the original 1024 fast-read gate without claiming same-runner six-method controls.
