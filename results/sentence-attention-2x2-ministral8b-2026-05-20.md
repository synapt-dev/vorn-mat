# Ministral 8B Sentence-Attention Extension — 2026-05-20

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Ministral-8B-Instruct-2410`

## Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost | Mean evicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | full context | prefix + recent | 1.00 | [0.9286, 1.0000] | historical_stale_reference | 198.71s | $0.1379 | 0.00% |
| Sentence vorn | 1024 | prefix + recent | 1.00 | [0.9286, 1.0000] | historical_stale_reference | 360.56s | $0.2502 | 74.02% |
| TOVA-style | 1024 | prefix + recent | 0.44 | [0.3116, 0.5769] | historical_stale_reference | 476.08s | $0.3304 | 73.76% |
| Sentence TOVA-style @ 256 | 256 | prefix + recent | 0.36 | [0.2414, 0.4986] | current_n50_override | 237.51s | $0.1648 | 93.75% |
| Sentence TOVA-style @ 512 | 512 | prefix + recent | 0.70 | [0.5625, 0.8090] | current_n50_override | 242.00s | $0.1680 | 87.12% |
| Sentence TOVA-style @ 1024 | 1024 | prefix + recent | 0.98 | [0.8950, 0.9965] | current_n50_override | 341.77s | $0.2372 | 74.00% |

## Pairwise Tests

- sentence_tova_1024_guarded vs tova_1024_guarded_historical, exact McNemar on `[[22, 27], [0, 1]]`: `p = 1.49012e-08`
- sentence_tova_1024_guarded vs sentence_1024_guarded_historical, exact McNemar on `[[49, 0], [1, 0]]`: `p = 1`
- sentence_1024_guarded_historical vs tova_1024_guarded_historical, exact McNemar on `[[22, 28], [0, 0]]`: `p = 7.45058e-09`

## Read

- Gate verdict: `mistral_like_full_rescue`.
- Sentence-TOVA-style reaches the sentence-vorn regime at the gate budget.
- The 256 and 512 rows extend the same surface below the original 1024 fast-read gate without claiming same-runner six-method controls.
