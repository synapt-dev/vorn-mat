# Sentence-Level 8k Budget Sweep — 2026-05-13

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_8k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Pooling: `max`

## Ceiling Context

- Vanilla @ full context: `0.42` hit rate, `242.69s`, `$0.1684`.

## Sentence Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
| Sentence vorn | 512 | prefix + recent | 0.18 | [0.0977, 0.3080] | 296.36s | $0.2057 | 0 | 93.87% |
| Sentence vorn | 512 | none | 0.00 | [0.0000, 0.0714] | 322.57s | $0.2239 | 0 | 93.87% |
| Sentence vorn | 1024 | prefix + recent | 0.62 | [0.4815, 0.7414] | 322.19s | $0.2236 | 0 | 87.64% |
| Sentence vorn | 1024 | none | 0.62 | [0.4815, 0.7414] | 379.56s | $0.2634 | 0 | 87.64% |
| Sentence vorn | 1536 | prefix + recent | 0.74 | [0.6045, 0.8413] | 377.69s | $0.2621 | 0 | 81.39% |
| Sentence vorn | 1536 | none | 0.74 | [0.6045, 0.8413] | 374.63s | $0.2600 | 0 | 81.39% |
| Sentence vorn | 2048 | prefix + recent | 0.62 | [0.4815, 0.7414] | 468.66s | $0.3253 | 0 | 75.11% |
| Sentence vorn | 2048 | none | 0.62 | [0.4815, 0.7414] | 414.96s | $0.2880 | 0 | 75.11% |

## Token Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
| Token vorn | 512 | prefix + recent | 0.38 | [0.2586, 0.5185] | 330.43s | $0.2293 | 0 | 93.74% |
| Token vorn | 1024 | prefix + recent | 0.38 | [0.2586, 0.5185] | 445.92s | $0.3095 | 0 | 87.47% |
| Token vorn | 1536 | prefix + recent | 0.52 | [0.3851, 0.6520] | 485.26s | $0.3368 | 0 | 81.21% |
| Token vorn | 2048 | prefix + recent | 0.68 | [0.5419, 0.7924] | 615.94s | $0.4275 | 0 | 74.94% |

## Paired Tests

- sentence_512_guarded vs token_512_guarded, exact McNemar on `[[7, 2], [12, 29]]`: `p = 0.0129395`
- sentence_512_noguards vs sentence_512_guarded, exact McNemar on `[[0, 0], [9, 41]]`: `p = 0.00390625`
- sentence_1024_guarded vs token_1024_guarded, exact McNemar on `[[16, 15], [3, 16]]`: `p = 0.00753784`
- sentence_1024_noguards vs sentence_1024_guarded, exact McNemar on `[[31, 0], [0, 19]]`: `p = 1`
- sentence_1536_guarded vs token_1536_guarded, exact McNemar on `[[23, 14], [3, 10]]`: `p = 0.0127258`
- sentence_1536_noguards vs sentence_1536_guarded, exact McNemar on `[[37, 0], [0, 13]]`: `p = 1`
- sentence_2048_guarded vs token_2048_guarded, exact McNemar on `[[25, 6], [9, 10]]`: `p = 0.607239`
- sentence_2048_noguards vs sentence_2048_guarded, exact McNemar on `[[31, 0], [0, 19]]`: `p = 1`

## Read

- The 8k crossover is regime-shaped, not monotonic. Sentence-level loses at `512` (`0.18` vs token `0.38`, paired `p = 0.0129395`), wins clearly at `1024` and `1536` (`0.62` vs `0.38`, `0.74` vs `0.52`), then stops separating at `2048` (`0.62` vs `0.68`, paired `p = 0.607239`).
- Guardrail-independence is budget-dependent. At `512`, removing guardrails collapses sentence-level to `0.00` (`p = 0.00390625`); at `1024`, `1536`, and `2048`, guarded and no-guardrails sentence rows are identical (`p = 1`).
- The strongest sentence-level zone on this slice is `1536`, where it reaches `0.74` at `81.39%` KV savings and still beats token-level on the same questions.
