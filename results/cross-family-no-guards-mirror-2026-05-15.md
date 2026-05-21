# Cross-Family No-Guards Mirror @ 1024

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Families: `google/gemma-4-E4B-it`, `meta-llama/Llama-3.1-8B-Instruct`

## Rows

| Family | Method | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 | token_vorn | prefix + recent | 0.02 | [0.0035, 0.1050] | 546.19s | $0.3791 |
| Gemma 4 | token_vorn | none | 0.00 | [0.0000, 0.0714] | 230.63s | $0.1601 |
| Gemma 4 | sentence_vorn | prefix + recent | 0.24 | [0.1430, 0.3741] | 291.44s | $0.2023 |
| Gemma 4 | sentence_vorn | none | 0.22 | [0.1275, 0.3524] | 290.25s | $0.2014 |
| Gemma 4 | tova | prefix + recent | 0.94 | [0.8378, 0.9794] | 451.24s | $0.3132 |
| Gemma 4 | tova | none | 0.94 | [0.8378, 0.9794] | 447.78s | $0.3108 |
| Gemma 4 | h2o | prefix + recent | 0.94 | [0.8378, 0.9794] | 3141.36s | $2.1801 |
| Gemma 4 | h2o | none | 0.94 | [0.8378, 0.9794] | 456.36s | $0.3167 |
| Llama 3.1 | token_vorn | prefix + recent | 1.00 | [0.9286, 1.0000] | 276.25s | $0.1917 |
| Llama 3.1 | token_vorn | none | 1.00 | [0.9286, 1.0000] | 486.85s | $0.3379 |
| Llama 3.1 | sentence_vorn | prefix + recent | 1.00 | [0.9286, 1.0000] | 245.17s | $0.1702 |
| Llama 3.1 | sentence_vorn | none | 1.00 | [0.9286, 1.0000] | 217.60s | $0.1510 |
| Llama 3.1 | tova | prefix + recent | 0.90 | [0.7864, 0.9565] | 395.00s | $0.2741 |
| Llama 3.1 | tova | none | 0.90 | [0.7864, 0.9565] | 317.49s | $0.2203 |
| Llama 3.1 | h2o | prefix + recent | 0.94 | [0.8378, 0.9794] | 308.58s | $0.2142 |
| Llama 3.1 | h2o | none | 0.94 | [0.8378, 0.9794] | 317.76s | $0.2205 |

## Guarded vs No-Guards Paired Tests

| LHS | RHS | Table | Exact McNemar p |
| --- | --- | --- | --- |
| gemma_token_noguards | gemma_token_guarded | [[0, 0], [1, 49]] | 1 |
| gemma_sentence_noguards | gemma_sentence_guarded | [[11, 0], [1, 38]] | 1 |
| gemma_tova_noguards | gemma_tova_guarded | [[47, 0], [0, 3]] | 1 |
| gemma_h2o_noguards | gemma_h2o_guarded | [[47, 0], [0, 3]] | 1 |
| llama_token_noguards | llama_token_guarded | [[50, 0], [0, 0]] | 1 |
| llama_sentence_noguards | llama_sentence_guarded | [[50, 0], [0, 0]] | 1 |
| llama_tova_noguards | llama_tova_guarded | [[45, 0], [0, 5]] | 1 |
| llama_h2o_noguards | llama_h2o_guarded | [[47, 0], [0, 3]] | 1 |

## Read

- **Gemma 4**: the attention-weight preservation story is not a guardrail artifact on this slice. `TOVA` and `H2O` remain at `0.94` with or without prefix/recent-window guardrails, while `token_vorn` stays collapsed (`0.02` guarded, `0.00` no-guards) and `sentence_vorn` remains low (`0.24` guarded, `0.22` no-guards).
- **Llama 3.1**: the residual-direction preservation story is not a guardrail artifact either. Both `token_vorn` and `sentence_vorn` remain at `1.00` with guardrails removed, while `TOVA` and `H2O` stay at `0.90` / `0.94`.
- The mirror-image claim therefore survives guardrail removal cleanly on both families. Gemma's strong rows are genuinely attention-weight-dominant, and Llama's strong rows are genuinely residual-direction-preserving at this budget.
