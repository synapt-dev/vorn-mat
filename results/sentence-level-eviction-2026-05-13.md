# Sentence-Level Eviction — 2026-05-13

## Run Conditions

- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Pooling: `max`

## Ceiling Context

- Vanilla @ full context: `0.28` hit rate, `160.23s`, `$0.1112`. This is a ceiling/context row, not a constrained-budget competitor.

## Comparison Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
| Sentence vorn | 1024 | prefix + recent | 0.68 | [0.5419, 0.7924] | 221.52s | $0.1537 | 0 | 75.87% |
| Sentence vorn | 1024 | none | 0.68 | [0.5419, 0.7924] | 274.61s | $0.1906 | 0 | 75.87% |
| Sentence vorn | 256 | prefix + recent | 0.06 | [0.0206, 0.1622] | 195.99s | $0.1360 | 0 | 94.41% |

## Token References

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Reference |
|--------|--------|------------|----------|---------------|-----------|
| Token vorn | 1024 | prefix + recent | 0.26 | [0.1587, 0.3955] | same slice |
| Token vorn | 1024 | none | 0.00 | [0.0000, 0.0714] | same slice |
| Token vorn | 256 | prefix + recent | 0.04 | [0.0110, 0.1346] | same slice |

## Paired Tests

- sentence_1024_guarded vs token_1024_guarded, exact McNemar on `[[10, 24], [3, 13]]`: `p = 4.92334e-05`
- sentence_1024_noguards vs token_1024_noguards, exact McNemar on `[[0, 34], [0, 16]]`: `p = 1.16415e-10`
- sentence_256_guarded vs token_256_guarded, exact McNemar on `[[0, 3], [2, 45]]`: `p = 1`

## Read

- The qualitative finding is unchanged: sentence-level eviction materially outperforms token-level vorn on the same NIAH questions at `1024`.
- The inferential layer is now correct for the design: these are paired same-slice comparisons, so the significance rows use exact McNemar instead of Fisher exact.
- At `256`, sentence-level does not separate from token-level on this slice.
