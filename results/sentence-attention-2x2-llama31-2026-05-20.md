# Sentence-Attention 2×2 on Llama 3.1 8B — 2026-05-20

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Pooling: `max`

## Ceiling Context

- Vanilla @ full context: `1.00` hit rate, `173.52s`, `$0.1204`.
- Ceiling status: `current_n50_override`.
- Historical vanilla reference: `1.00` with status `historical_stale_reference`.

## Rows

| Budget | Method | Hit rate | 95% Wilson CI | Guardrails | Ceiling status | Wall-clock | Inference cost | KV savings |
|--------|--------|----------|---------------|------------|----------------|------------|----------------|-----------|
| 256 | Token vorn | 0.56 | [0.4231, 0.6884] | prefix + recent | current_n50_override | 264.41s | $0.1835 | 93.34% |
| 256 | Sentence vorn | 1.00 | [0.9286, 1.0000] | prefix + recent | current_n50_override | 198.79s | $0.1380 | 93.72% |
| 256 | TOVA-style | 0.12 | [0.0562, 0.2381] | prefix + recent | current_n50_override | 277.00s | $0.1922 | 93.34% |
| 256 | Sentence TOVA-style | 0.74 | [0.6045, 0.8413] | prefix + recent | current_n50_override | 223.67s | $0.1552 | 93.64% |
| 256 | H2O-style | 0.08 | [0.0315, 0.1884] | prefix + recent | current_n50_override | 276.89s | $0.1922 | 93.34% |
| 256 | Sentence H2O-style | 0.74 | [0.6045, 0.8413] | prefix + recent | current_n50_override | 198.94s | $0.1381 | 93.64% |
| 512 | Token vorn | 0.96 | [0.8654, 0.9890] | prefix + recent | current_n50_override | 239.22s | $0.1660 | 86.68% |
| 512 | Sentence vorn | 1.00 | [0.9286, 1.0000] | prefix + recent | current_n50_override | 198.57s | $0.1378 | 86.98% |
| 512 | TOVA-style | 0.56 | [0.4231, 0.6884] | prefix + recent | current_n50_override | 204.36s | $0.1418 | 86.68% |
| 512 | Sentence TOVA-style | 0.94 | [0.8378, 0.9794] | prefix + recent | current_n50_override | 240.78s | $0.1671 | 86.97% |
| 512 | H2O-style | 0.56 | [0.4231, 0.6884] | prefix + recent | current_n50_override | 236.49s | $0.1641 | 86.68% |
| 512 | Sentence H2O-style | 0.94 | [0.8378, 0.9794] | prefix + recent | current_n50_override | 176.18s | $0.1223 | 86.98% |
| 1024 | Token vorn | 1.00 | [0.9286, 1.0000] | prefix + recent | current_n50_override | 359.17s | $0.2493 | 73.36% |
| 1024 | Sentence vorn | 1.00 | [0.9286, 1.0000] | prefix + recent | current_n50_override | 281.53s | $0.1954 | 73.62% |
| 1024 | TOVA-style | 0.90 | [0.7864, 0.9565] | prefix + recent | current_n50_override | 379.30s | $0.2632 | 73.36% |
| 1024 | Sentence TOVA-style | 1.00 | [0.9286, 1.0000] | prefix + recent | current_n50_override | 226.60s | $0.1573 | 73.64% |
| 1024 | H2O-style | 0.94 | [0.8378, 0.9794] | prefix + recent | current_n50_override | 310.66s | $0.2156 | 73.36% |
| 1024 | Sentence H2O-style | 1.00 | [0.9286, 1.0000] | prefix + recent | current_n50_override | 263.36s | $0.1828 | 73.64% |

## Paired Tests

- sentence_tova_256_guarded vs tova_256_guarded, exact McNemar on `[[6, 31], [0, 13]]`: `p = 9.31323e-10`
- sentence_h2o_256_guarded vs h2o_256_guarded, exact McNemar on `[[4, 33], [0, 13]]`: `p = 2.32831e-10`
- sentence_tova_256_guarded vs sentence_256_guarded, exact McNemar on `[[37, 0], [13, 0]]`: `p = 0.000244141`
- sentence_h2o_256_guarded vs sentence_256_guarded, exact McNemar on `[[37, 0], [13, 0]]`: `p = 0.000244141`
- tova_256_guarded vs sentence_256_guarded, exact McNemar on `[[6, 0], [44, 0]]`: `p = 1.13687e-13`
- h2o_256_guarded vs sentence_256_guarded, exact McNemar on `[[4, 0], [46, 0]]`: `p = 2.84217e-14`
- sentence_tova_512_guarded vs tova_512_guarded, exact McNemar on `[[28, 19], [0, 3]]`: `p = 3.8147e-06`
- sentence_h2o_512_guarded vs h2o_512_guarded, exact McNemar on `[[28, 19], [0, 3]]`: `p = 3.8147e-06`
- sentence_tova_512_guarded vs sentence_512_guarded, exact McNemar on `[[47, 0], [3, 0]]`: `p = 0.25`
- sentence_h2o_512_guarded vs sentence_512_guarded, exact McNemar on `[[47, 0], [3, 0]]`: `p = 0.25`
- tova_512_guarded vs sentence_512_guarded, exact McNemar on `[[28, 0], [22, 0]]`: `p = 4.76837e-07`
- h2o_512_guarded vs sentence_512_guarded, exact McNemar on `[[28, 0], [22, 0]]`: `p = 4.76837e-07`
- sentence_tova_1024_guarded vs tova_1024_guarded, exact McNemar on `[[45, 5], [0, 0]]`: `p = 0.0625`
- sentence_h2o_1024_guarded vs h2o_1024_guarded, exact McNemar on `[[47, 3], [0, 0]]`: `p = 0.25`
- sentence_tova_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[50, 0], [0, 0]]`: `p = 1`
- sentence_h2o_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[50, 0], [0, 0]]`: `p = 1`
- tova_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[45, 0], [5, 0]]`: `p = 0.0625`
- h2o_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[47, 0], [3, 0]]`: `p = 0.25`

## Read

- Budget `256`: token_vorn `0.56`, sentence_vorn `1.00`, TOVA-style `0.12`, sentence-TOVA-style `0.74`, H2O-style `0.08`, sentence-H2O-style `0.74`. Sentence-level attention-weight rows improve but still trail sentence_vorn by a material margin on this budget.
- Budget `512`: token_vorn `0.96`, sentence_vorn `1.00`, TOVA-style `0.56`, sentence-TOVA-style `0.94`, H2O-style `0.56`, sentence-H2O-style `0.94`. Sentence-level attention-weight rows get close to sentence_vorn but do not fully match it on this budget.
- Budget `1024`: token_vorn `1.00`, sentence_vorn `1.00`, TOVA-style `0.90`, sentence-TOVA-style `1.00`, H2O-style `0.94`, sentence-H2O-style `1.00`. Sentence-level attention-weight rows recover to the same regime as sentence_vorn on this budget.

- Overall outcome classification: `threshold_bounded_recovery`.

## Drift vs Historical Llama Surface

- Budget `256` historical deltas: token_vorn `+0.00`, sentence_vorn `+0.00`, TOVA-style `+0.00`, H2O-style `+0.00`
- Budget `512` historical deltas: token_vorn `+0.00`, sentence_vorn `+0.00`, TOVA-style `+0.00`, H2O-style `+0.00`
- Budget `1024` historical deltas: token_vorn `+0.00`, sentence_vorn `+0.00`, TOVA-style `+0.00`, H2O-style `+0.00`

## Coverage Notes

- Fresh same-runner controls rerun the full token/sentence/TOVA/H2O Llama threshold surface at `256`, `512`, and `1024`.
- Sentence-TOVA and sentence-H2O rows are added on the same paired `n=50` surface at all three budgets.
