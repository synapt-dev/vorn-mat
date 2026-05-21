# Sentence-Attention 2×2 on Gemma 4 E4B-it — 2026-05-19

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `google/gemma-4-E4B-it`
- Pooling: `max`

## Ceiling Context

- Vanilla @ full context: `1.00` hit rate, `202.10s`, `$0.1403`.
- Ceiling status: `current_n50_override`.
- Historical vanilla reference: `1.00` with status `historical_stale_reference`.

## Rows

| Budget | Method | Hit rate | 95% Wilson CI | Guardrails | Ceiling status | Wall-clock | Inference cost | KV savings |
|--------|--------|----------|---------------|------------|----------------|------------|----------------|-----------|
| 1024 | Token vorn | 0.02 | [0.0035, 0.1050] | prefix + recent | current_n50_override | 463.87s | $0.3219 | 74.03% |
| 1024 | Sentence vorn | 0.24 | [0.1430, 0.3741] | prefix + recent | current_n50_override | 314.18s | $0.2180 | 74.38% |
| 1024 | TOVA-style | 0.94 | [0.8378, 0.9794] | prefix + recent | current_n50_override | 453.21s | $0.3145 | 74.03% |
| 1024 | Sentence TOVA-style | 0.88 | [0.7619, 0.9438] | prefix + recent | current_n50_override | 293.39s | $0.2036 | 74.30% |
| 1024 | H2O-style | 0.94 | [0.8378, 0.9794] | prefix + recent | historical_stale_reference | 3141.36s | $2.1801 | 74.03% |
| 1024 | Sentence H2O-style | 0.86 | [0.7381, 0.9305] | prefix + recent | current_n50_override | 322.12s | $0.2236 | 74.29% |
| 2048 | Token vorn | 0.52 | [0.3851, 0.6520] | prefix + recent | current_n50_override | 845.31s | $0.5866 | 48.05% |
| 2048 | Sentence vorn | 0.96 | [0.8654, 0.9890] | prefix + recent | current_n50_override | 472.88s | $0.3282 | 48.31% |
| 2048 | TOVA-style | 1.00 | [0.9286, 1.0000] | prefix + recent | current_n50_override | 821.61s | $0.5702 | 48.05% |
| 2048 | Sentence TOVA-style | 1.00 | [0.9286, 1.0000] | prefix + recent | current_n50_override | 481.30s | $0.3340 | 48.28% |
| 2048 | Sentence H2O-style | 1.00 | [0.9286, 1.0000] | prefix + recent | current_n50_override | 506.69s | $0.3516 | 48.28% |

## Paired Tests

- sentence_tova_1024_guarded vs tova_1024_guarded, exact McNemar on `[[42, 2], [5, 1]]`: `p = 0.453125`
- sentence_h2o_1024_guarded vs h2o_1024_guarded, exact McNemar on `[[42, 1], [5, 2]]`: `p = 0.21875`
- sentence_tova_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[11, 33], [1, 5]]`: `p = 4.07454e-09`
- sentence_h2o_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[11, 32], [1, 6]]`: `p = 7.91624e-09`
- tova_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[12, 35], [0, 3]]`: `p = 5.82077e-11`
- h2o_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[11, 36], [1, 2]]`: `p = 5.52973e-10`
- sentence_tova_2048_guarded vs tova_2048_guarded, exact McNemar on `[[50, 0], [0, 0]]`: `p = 1`
- sentence_tova_2048_guarded vs sentence_2048_guarded, exact McNemar on `[[48, 2], [0, 0]]`: `p = 0.5`
- tova_2048_guarded vs sentence_2048_guarded, exact McNemar on `[[48, 2], [0, 0]]`: `p = 0.5`
- sentence_h2o_2048_guarded vs sentence_2048_guarded, exact McNemar on `[[48, 2], [0, 0]]`: `p = 0.5`

## Read

- Budget `1024`: token_vorn `0.02`, sentence_vorn `0.24`, TOVA-style `0.94`, sentence-TOVA-style `0.88`, H2O-style `0.94`, sentence-H2O-style `0.86`. Sentence-level attention-weight rows fail to preserve the ceiling cleanly or only improve modestly, which keeps Gemma in the attention-weight-favored family even after the granularity change.
- Budget `2048`: token_vorn `0.52`, sentence_vorn `0.96`, TOVA-style `1.00`, sentence-TOVA-style `1.00`, H2O-style `not rerun`, sentence-H2O-style `1.00`. Sentence-level attention-weight rows stay at ceiling and sentence_vorn also climbs to a competitive regime, which makes the Gemma outlier look token-granularity-specific rather than an absolute family-wide channel split.

- Overall outcome classification: `mixed_transition_surface`.

## Drift vs Historical Gemma Surface

- Budget `1024` historical deltas: token_vorn `+0.00`, sentence_vorn `+0.00`, TOVA-style `+0.00`, H2O-style `+0.00`
- Budget `2048` historical deltas: token_vorn `+0.00`, sentence_vorn `+0.00`, TOVA-style `+0.00`, H2O-style `n/a`.

## Coverage Notes

- Fresh same-runner controls confirm the historical Gemma token/sentence/TOVA rows at `1024` and `2048`.
- Token-level H2O was not rerun fresh because the H2O control path is the dominant cost/time tail on Gemma.
- Sentence-H2O `1024` is fresh. Sentence-H2O `2048` is included only if the optional row completed before artifact build.
