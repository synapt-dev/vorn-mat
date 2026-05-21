# Qwen 4k Budget Sweep — 2026-05-14

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Pooling: `max`

## Full-Context Floor

- Vanilla @ full context: `0.00` hit rate, `192.29s`, `$0.1335`.

## Sentence Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
| Sentence vorn | 256 | prefix + recent | 0.00 | [0.0000, 0.0714] | 160.95s | $0.1117 | 0 | 93.72% |
| Sentence vorn | 256 | none | 0.00 | [0.0000, 0.0714] | 189.30s | $0.1314 | 0 | 93.66% |
| Sentence vorn | 512 | prefix + recent | 0.00 | [0.0000, 0.0714] | 167.61s | $0.1163 | 0 | 87.05% |
| Sentence vorn | 512 | none | 0.04 | [0.0110, 0.1346] | 180.22s | $0.1251 | 0 | 87.08% |
| Sentence vorn | 1024 | prefix + recent | 0.00 | [0.0000, 0.0714] | 226.73s | $0.1573 | 0 | 73.85% |
| Sentence vorn | 1024 | none | 0.00 | [0.0000, 0.0714] | 326.50s | $0.2266 | 0 | 73.83% |
| Sentence vorn | 1536 | prefix + recent | 0.00 | [0.0000, 0.0714] | 236.78s | $0.1643 | 0 | 60.59% |
| Sentence vorn | 1536 | none | 0.00 | [0.0000, 0.0714] | 266.79s | $0.1851 | 0 | 60.59% |
| Sentence vorn | 2048 | prefix + recent | 0.00 | [0.0000, 0.0714] | 279.14s | $0.1937 | 0 | 47.31% |
| Sentence vorn | 2048 | none | 0.00 | [0.0000, 0.0714] | 275.92s | $0.1915 | 0 | 47.31% |

## Token Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
| Token vorn | 256 | prefix + recent | 0.00 | [0.0000, 0.0714] | 201.61s | $0.1399 | 0 | 93.38% |
| Token vorn | 512 | prefix + recent | 0.02 | [0.0035, 0.1050] | 273.85s | $0.1901 | 0 | 86.77% |
| Token vorn | 1024 | prefix + recent | 0.04 | [0.0110, 0.1346] | 294.38s | $0.2043 | 0 | 73.53% |
| Token vorn | 1536 | prefix + recent | 0.08 | [0.0315, 0.1884] | 379.25s | $0.2632 | 0 | 60.30% |
| Token vorn | 2048 | prefix + recent | 0.02 | [0.0035, 0.1050] | 532.43s | $0.3695 | 0 | 47.07% |

## Paired Tests

- sentence_256_guarded vs token_256_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`
- sentence_256_noguards vs sentence_256_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`
- sentence_512_guarded vs token_512_guarded, exact McNemar on `[[0, 0], [1, 49]]`: `p = 1`
- sentence_512_noguards vs sentence_512_guarded, exact McNemar on `[[0, 2], [0, 48]]`: `p = 0.5`
- sentence_1024_guarded vs token_1024_guarded, exact McNemar on `[[0, 0], [2, 48]]`: `p = 0.5`
- sentence_1024_noguards vs sentence_1024_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`
- sentence_1536_guarded vs token_1536_guarded, exact McNemar on `[[0, 0], [4, 46]]`: `p = 0.125`
- sentence_1536_noguards vs sentence_1536_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`
- sentence_2048_guarded vs token_2048_guarded, exact McNemar on `[[0, 0], [1, 49]]`: `p = 1`
- sentence_2048_noguards vs sentence_2048_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`

## Read

- Qwen's full-context floor on this slice is `0.00`, so every nonzero constrained-budget row is real recovery relative to the drowning baseline rather than degradation from a strong ceiling.
- Sentence-level does **not** replicate the Mistral 4k pattern here. The strongest guarded sentence row is only budget `256` at `0.00`, and the strongest no-guardrails sentence row is budget `512` at `0.04`.
- The only meaningful directional recovery band is token-level: token vorn peaks at budget `1536` with `0.08`. On this slice, the cross-model story is therefore not `sentence sweet-spot transfers`; it is `Qwen shows weak constrained-budget recovery and that recovery is token-shaped rather than sentence-shaped`.
- None of the sentence-vs-token paired tests clear a decisive threshold at `n=50` on this Qwen slice. Treat this as a negative or boundary-setting result for the Mistral sentence-level law, not as evidence that Qwen has the same sweet-spot structure.
