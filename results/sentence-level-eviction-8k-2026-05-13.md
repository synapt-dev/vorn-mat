# Sentence-Level Eviction @ 8k — 2026-05-13

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_8k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Pooling: `max`

## Ceiling Context

- Vanilla @ full context: `0.42` hit rate, `242.69s`, `$0.1684`. This is a ceiling/context row, not a constrained-budget competitor.

## Comparison Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
| Sentence vorn | 1024 | prefix + recent | 0.62 | [0.4815, 0.7414] | 322.19s | $0.2236 | 0 | 87.64% |
| Sentence vorn | 1024 | none | 0.62 | [0.4815, 0.7414] | 379.56s | $0.2634 | 0 | 87.64% |
| Sentence vorn | 2048 | prefix + recent | 0.62 | [0.4815, 0.7414] | 468.66s | $0.3253 | 0 | 75.11% |
| Sentence vorn | 2048 | none | 0.62 | [0.4815, 0.7414] | 414.96s | $0.2880 | 0 | 75.11% |

## Token References

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Reference |
|--------|--------|------------|----------|---------------|-----------|
| Token vorn | 1024 | prefix + recent | 0.38 | [0.2586, 0.5185] | same slice |
| Token vorn | 2048 | prefix + recent | 0.68 | [0.5419, 0.7924] | same slice |

## Paired Tests

- sentence_1024_guarded vs token_1024_guarded, exact McNemar on `[[16, 15], [3, 16]]`: `p = 0.00753784`
- sentence_1024_noguards vs sentence_1024_guarded, exact McNemar on `[[31, 0], [0, 19]]`: `p = 1`
- sentence_2048_guarded vs token_2048_guarded, exact McNemar on `[[25, 6], [9, 10]]`: `p = 0.607239`
- sentence_2048_noguards vs sentence_2048_guarded, exact McNemar on `[[31, 0], [0, 19]]`: `p = 1`
- sentence_2048_guarded vs sentence_1024_guarded, exact McNemar on `[[25, 6], [6, 13]]`: `p = 1`
- sentence_2048_noguards vs sentence_1024_noguards, exact McNemar on `[[25, 6], [6, 13]]`: `p = 1`

## Read

- At `1024`, sentence-level vorn extends the 4k coherence result to the 8k slice: `0.62` vs token-level `0.38`, with paired exact McNemar `p = 0.00753784`.
- At both `1024` and `2048`, sentence-level rows are guardrail-independent on this slice: guarded and no-guardrails outcomes are identical (`p = 1` in both paired comparisons).
- At `2048`, token-level vorn catches up and slightly exceeds the sentence row on point estimate (`0.68` vs `0.62`), but the paired same-slice comparison is not decisive at `n=50` (`p = 0.607239`).
