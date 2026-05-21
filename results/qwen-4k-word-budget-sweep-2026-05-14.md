# Qwen 4k Word-vorn Budget Sweep — 2026-05-14

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `Qwen/Qwen2.5-7B-Instruct`
- Pooling: `max`

## Full-Context Floor

- Vanilla @ full context: `0.00` hit rate, `192.29s`, `$0.1335`.

## Word Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
| Word vorn | 256 | prefix + recent | 0.00 | [0.0000, 0.0714] | 334.65s | $0.2322 | 0 | 93.40% |
| Word vorn | 256 | none | 0.00 | [0.0000, 0.0714] | 268.81s | $0.1866 | 0 | 93.40% |
| Word vorn | 512 | prefix + recent | 0.00 | [0.0000, 0.0714] | 303.62s | $0.2107 | 0 | 86.77% |
| Word vorn | 512 | none | 0.00 | [0.0000, 0.0714] | 289.11s | $0.2006 | 0 | 86.77% |
| Word vorn | 1024 | prefix + recent | 0.00 | [0.0000, 0.0714] | 279.92s | $0.1943 | 0 | 73.53% |
| Word vorn | 1024 | none | 0.00 | [0.0000, 0.0714] | 275.51s | $0.1912 | 0 | 73.53% |
| Word vorn | 1536 | prefix + recent | 0.00 | [0.0000, 0.0714] | 334.16s | $0.2319 | 0 | 60.30% |
| Word vorn | 1536 | none | 0.00 | [0.0000, 0.0714] | 365.64s | $0.2538 | 0 | 60.30% |
| Word vorn | 2048 | prefix + recent | 0.00 | [0.0000, 0.0714] | 419.92s | $0.2914 | 0 | 47.07% |
| Word vorn | 2048 | none | 0.00 | [0.0000, 0.0714] | 415.65s | $0.2885 | 0 | 47.07% |

## Reference Token Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
| Token vorn | 256 | prefix + recent | 0.00 | [0.0000, 0.0714] | 201.61s | $0.1399 | 0 | 93.38% |
| Token vorn | 512 | prefix + recent | 0.02 | [0.0035, 0.1050] | 273.85s | $0.1901 | 0 | 86.77% |
| Token vorn | 1024 | prefix + recent | 0.04 | [0.0110, 0.1346] | 294.38s | $0.2043 | 0 | 73.53% |
| Token vorn | 1536 | prefix + recent | 0.08 | [0.0315, 0.1884] | 379.25s | $0.2632 | 0 | 60.30% |
| Token vorn | 2048 | prefix + recent | 0.02 | [0.0035, 0.1050] | 532.43s | $0.3695 | 0 | 47.07% |

## Reference Sentence Rows

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

## Paired Tests

- word_256_guarded vs token_256_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`
- word_256_noguards vs word_256_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`
- word_256_guarded vs sentence_256_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`
- word_512_guarded vs token_512_guarded, exact McNemar on `[[0, 0], [1, 49]]`: `p = 1`
- word_512_noguards vs word_512_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`
- word_512_guarded vs sentence_512_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`
- word_1024_guarded vs token_1024_guarded, exact McNemar on `[[0, 0], [2, 48]]`: `p = 0.5`
- word_1024_noguards vs word_1024_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`
- word_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`
- word_1536_guarded vs token_1536_guarded, exact McNemar on `[[0, 0], [4, 46]]`: `p = 0.125`
- word_1536_noguards vs word_1536_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`
- word_1536_guarded vs sentence_1536_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`
- word_2048_guarded vs token_2048_guarded, exact McNemar on `[[0, 0], [1, 49]]`: `p = 1`
- word_2048_noguards vs word_2048_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`
- word_2048_guarded vs sentence_2048_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`

## Per-Fixture Deltas

- Token-only recoveries at budget `1536`: `['niah_multikey_1_4k-11', 'niah_multikey_1_4k-24', 'niah_multikey_1_4k-36', 'niah_multikey_1_4k-5']`
- Word-only recoveries versus token: `[]` at every tested budget.

## Read

- Word-vorn does **not** produce a Qwen sweet-spot on this slice. All ten word rows are `0.00`.
- Token-vorn remains the only directional recovery band, peaking at budget `1536` with `0.08`.
- Sentence-vorn remains weaker than token on this Qwen slice, with its best row at `512` no-guards = `0.04`.
- This is stronger than the prior negative sentence result: moving upward from token to word destroys the small Qwen recovery rather than improving it.
- The cross-model architectural-law claim sharpens rather than broadens: the minimum coherent retention unit is per-model, and on this Qwen slice it is smaller than word while Mistral's strongest regime remained sentence-shaped.
- Incremental Modal spend for the word matrix was `$2.2812`, within the dispatched `$3` cap.
