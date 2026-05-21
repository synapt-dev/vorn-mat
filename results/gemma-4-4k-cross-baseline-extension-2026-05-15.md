# Gemma 4 E4B-it 4k Cross-Baseline Extension — 2026-05-15

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `google/gemma-4-E4B-it`
- Budget: `1024`

## Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
| Vanilla | full context | prefix + recent | 1.00 | [0.9286, 1.0000] | 217.27s | $0.1508 | 0 | 0.00% |
| Token vorn | 1024 | prefix + recent | 0.02 | [0.0035, 0.1050] | 546.19s | $0.3791 | 0 | 74.03% |
| Sentence vorn | 1024 | prefix + recent | 0.24 | [0.1430, 0.3741] | 291.44s | $0.2023 | 0 | 74.38% |
| Sentence vorn | 1024 | none | 0.22 | [0.1275, 0.3524] | 290.25s | $0.2014 | 0 | 74.37% |
| TOVA | 1024 | prefix + recent | 0.94 | [0.8378, 0.9794] | 451.24s | $0.3132 | 0 | 74.03% |
| H2O | 1024 | prefix + recent | 0.94 | [0.8378, 0.9794] | 3141.36s | $2.1801 | 0 | 74.03% |

## Paired Tests

- tova_1024_guarded vs token_1024_guarded, exact McNemar on `[[1, 46], [0, 3]]`: `p = 2.84217e-14`
- h2o_1024_guarded vs token_1024_guarded, exact McNemar on `[[1, 46], [0, 3]]`: `p = 2.84217e-14`
- tova_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[12, 35], [0, 3]]`: `p = 5.82077e-11`
- h2o_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[11, 36], [1, 2]]`: `p = 5.52973e-10`
- tova_1024_guarded vs h2o_1024_guarded, exact McNemar on `[[46, 1], [1, 2]]`: `p = 1`
- tova_1024_guarded vs vanilla_floor, exact McNemar on `[[47, 0], [3, 0]]`: `p = 0.25`
- h2o_1024_guarded vs vanilla_floor, exact McNemar on `[[47, 0], [3, 0]]`: `p = 0.25`

## Read

- Gemma 4 does **not** reproduce the Mistral single-model pattern where residual-direction scoring is broadly comparable to attention-weight baselines. At the same `1024` budget and nearly identical retention ratios, `TOVA = 0.94` and `H2O = 0.94`, while `sentence_vorn = 0.24` and `token_vorn = 0.02`.
- The separation from the vorn rows is not marginal. On the shared 50-case slice, `TOVA` and `H2O` both separate from `token_vorn` with exact McNemar `p = 2.84e-14`; both also separate strongly from `sentence_vorn` (`p = 5.82e-11` and `5.53e-10` respectively).
- `TOVA` and `H2O` are effectively equivalent on quality here (`0.94` vs `0.94`; exact McNemar `p = 1.0`), but not on cost. `H2O` took `3141.36s` and `$2.1801` versus `TOVA` at `451.24s` and `$0.3132`.
- Vanilla remains at `1.00`, so Gemma still belongs to the strong-ceiling family on this slice. The new cross-baseline finding is narrower and stronger: on Gemma’s degradation regime, attention-weight baselines preserve nearly all of the ceiling while current residual-direction baselines do not.
