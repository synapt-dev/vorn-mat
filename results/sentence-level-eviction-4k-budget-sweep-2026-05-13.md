# Sentence-Level 4k Budget Sweep — 2026-05-13

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Pooling: `max`

## Ceiling Context

- Vanilla @ full context: `0.28` hit rate, `160.23s`, `$0.1112`.

## Sentence Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
| Sentence vorn | 512 | prefix + recent | 0.26 | [0.1587, 0.3955] | 182.49s | $0.1266 | 0 | 88.03% |
| Sentence vorn | 512 | none | 0.02 | [0.0035, 0.1050] | 177.34s | $0.1231 | 0 | 87.99% |
| Sentence vorn | 1024 | prefix + recent | 0.68 | [0.5419, 0.7924] | 221.52s | $0.1537 | 0 | 75.87% |
| Sentence vorn | 1024 | none | 0.68 | [0.5419, 0.7924] | 274.61s | $0.1906 | 0 | 75.87% |
| Sentence vorn | 1536 | prefix + recent | 0.54 | [0.4040, 0.6703] | 324.53s | $0.2252 | 0 | 63.61% |
| Sentence vorn | 1536 | none | 0.54 | [0.4040, 0.6703] | 330.86s | $0.2296 | 0 | 63.61% |
| Sentence vorn | 2048 | prefix + recent | 0.36 | [0.2414, 0.4986] | 461.21s | $0.3201 | 0 | 51.40% |
| Sentence vorn | 2048 | none | 0.36 | [0.2414, 0.4986] | 452.52s | $0.3140 | 0 | 51.40% |

## Token Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
| Token vorn | 512 | prefix + recent | 0.20 | [0.1124, 0.3304] | 267.98s | $0.1860 | 0 | 87.77% |
| Token vorn | 1024 | prefix + recent | 0.26 | [0.1587, 0.3955] | 486.10s | $0.3374 | 0 | 75.53% |
| Token vorn | 1536 | prefix + recent | 0.16 | [0.0834, 0.2851] | 667.35s | $0.4631 | 0 | 63.30% |
| Token vorn | 2048 | prefix + recent | 0.16 | [0.0834, 0.2851] | 882.36s | $0.6124 | 0 | 51.06% |

## Paired Tests

- sentence_512_guarded vs token_512_guarded, exact McNemar on `[[4, 9], [6, 31]]`: `p = 0.607239`
- sentence_512_noguards vs sentence_512_guarded, exact McNemar on `[[1, 0], [12, 37]]`: `p = 0.000488281`
- sentence_1024_guarded vs token_1024_guarded, exact McNemar on `[[10, 24], [3, 13]]`: `p = 4.92334e-05`
- sentence_1024_noguards vs sentence_1024_guarded, exact McNemar on `[[34, 0], [0, 16]]`: `p = 1`
- sentence_1536_guarded vs token_1536_guarded, exact McNemar on `[[4, 23], [4, 19]]`: `p = 0.000310749`
- sentence_1536_noguards vs sentence_1536_guarded, exact McNemar on `[[27, 0], [0, 23]]`: `p = 1`
- sentence_2048_guarded vs token_2048_guarded, exact McNemar on `[[6, 12], [2, 30]]`: `p = 0.0129395`
- sentence_2048_noguards vs sentence_2048_guarded, exact McNemar on `[[18, 0], [0, 32]]`: `p = 1`

## Read

- The 4k curve is not the 8k U-shape shifted left. Sentence-level is roughly tied with token-level at `512` (`0.26` vs `0.20`, paired `p = 0.607239`), peaks sharply at `1024` (`0.68`), then decays through `1536` (`0.54`) and `2048` (`0.36`) while still beating token-level at each of those higher budgets.
- Guardrail-independence is again budget-dependent. At `512`, removing guardrails collapses sentence-level to `0.02` (`p = 0.000488281`); at `1024`, `1536`, and `2048`, guarded and no-guardrails sentence rows are identical (`p = 1`).
- The strongest 4k sentence-level zone is `1024`, not `1536`. That means the sweet-spot budget is context-dependent: the workable band shifts upward as context length grows from `4k` to `8k`.
