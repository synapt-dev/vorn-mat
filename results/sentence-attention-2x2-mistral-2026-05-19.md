# Sentence-Attention 2×2 on Mistral 4k — 2026-05-19

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Pooling: `max`

## Ceiling Context

- Vanilla @ full context: `1.00` hit rate, `191.87s`, `$0.1332`.
- Ceiling status: `current_n50_override`.
- The interpretation below is driven by the fresh paired `n=50` rows in this artifact. If the ceiling status is `historical_stale_reference`, treat the ceiling row as provenance context only, not as part of the mechanism read.

## Rows

| Budget | Method | Hit rate | 95% Wilson CI | Guardrails | Wall-clock | Inference cost | KV savings |
|--------|--------|----------|---------------|------------|------------|----------------|-----------|
| 512 | Token vorn | 0.96 | [0.8654, 0.9890] | prefix + recent | 343.98s | $0.2387 | 87.77% |
| 512 | Sentence vorn | 1.00 | [0.9286, 1.0000] | prefix + recent | 188.38s | $0.1307 | 88.03% |
| 512 | TOVA-style | 0.30 | [0.1910, 0.4375] | prefix + recent | 339.08s | $0.2353 | 87.77% |
| 512 | Sentence TOVA-style | 0.96 | [0.8654, 0.9890] | prefix + recent | 200.33s | $0.1390 | 88.13% |
| 512 | H2O-style | 0.32 | [0.2076, 0.4581] | prefix + recent | 306.19s | $0.2125 | 87.77% |
| 512 | Sentence H2O-style | 0.96 | [0.8654, 0.9890] | prefix + recent | 205.06s | $0.1423 | 88.13% |
| 1024 | Token vorn | 0.96 | [0.8654, 0.9890] | prefix + recent | 432.13s | $0.2999 | 75.53% |
| 1024 | Sentence vorn | 0.98 | [0.8950, 0.9965] | prefix + recent | 280.23s | $0.1945 | 75.87% |
| 1024 | TOVA-style | 0.86 | [0.7381, 0.9305] | prefix + recent | 420.45s | $0.2918 | 75.53% |
| 1024 | Sentence TOVA-style | 1.00 | [0.9286, 1.0000] | prefix + recent | 294.29s | $0.2042 | 75.83% |
| 1024 | H2O-style | 0.84 | [0.7149, 0.9166] | prefix + recent | 419.44s | $0.2911 | 75.53% |
| 1024 | Sentence H2O-style | 1.00 | [0.9286, 1.0000] | prefix + recent | 237.87s | $0.1651 | 75.88% |

## Paired Tests

- sentence_tova_512_guarded vs tova_512_guarded, exact McNemar on `[[14, 34], [1, 1]]`: `p = 2.09548e-09`
- sentence_h2o_512_guarded vs h2o_512_guarded, exact McNemar on `[[15, 33], [1, 1]]`: `p = 4.07454e-09`
- sentence_tova_512_guarded vs sentence_512_guarded, exact McNemar on `[[48, 0], [2, 0]]`: `p = 0.5`
- sentence_h2o_512_guarded vs sentence_512_guarded, exact McNemar on `[[48, 0], [2, 0]]`: `p = 0.5`
- tova_512_guarded vs token_512_guarded, exact McNemar on `[[15, 0], [33, 2]]`: `p = 2.32831e-10`
- h2o_512_guarded vs token_512_guarded, exact McNemar on `[[16, 0], [32, 2]]`: `p = 4.65661e-10`
- sentence_tova_1024_guarded vs tova_1024_guarded, exact McNemar on `[[43, 7], [0, 0]]`: `p = 0.015625`
- sentence_h2o_1024_guarded vs h2o_1024_guarded, exact McNemar on `[[42, 8], [0, 0]]`: `p = 0.0078125`
- sentence_tova_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[49, 1], [0, 0]]`: `p = 1`
- sentence_h2o_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[49, 1], [0, 0]]`: `p = 1`
- tova_1024_guarded vs token_1024_guarded, exact McNemar on `[[43, 0], [5, 2]]`: `p = 0.0625`
- h2o_1024_guarded vs token_1024_guarded, exact McNemar on `[[42, 0], [6, 2]]`: `p = 0.03125`

## Read

- Budget `512`: token_vorn `0.96`, sentence_vorn `1.00`, TOVA-style `0.30`, sentence-TOVA-style `0.96`, H2O-style `0.32`, sentence-H2O-style `0.96`. Sentence grouping alone recovers most of the sentence_vorn gain, with sentence-level attention-weight rows close to sentence_vorn.
- Budget `1024`: token_vorn `0.96`, sentence_vorn `0.98`, TOVA-style `0.86`, sentence-TOVA-style `1.00`, H2O-style `0.84`, sentence-H2O-style `1.00`. Sentence grouping alone recovers most of the sentence_vorn gain, with sentence-level attention-weight rows close to sentence_vorn.

- Overall outcome classification: `granularity_primary`.
