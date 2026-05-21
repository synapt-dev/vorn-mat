# Llama 3.1 8B Threshold Cut @ 512

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `meta-llama/Llama-3.1-8B-Instruct`

## Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost |
| --- | --- | --- | --- | --- | --- | --- |
| token_vorn | 1024 | prefix + recent | 1.00 | [0.9286, 1.0000] | 276.25s | $0.1917 |
| sentence_vorn | 1024 | prefix + recent | 1.00 | [0.9286, 1.0000] | 245.17s | $0.1702 |
| tova | 1024 | prefix + recent | 0.90 | [0.7864, 0.9565] | 395.00s | $0.2741 |
| h2o | 1024 | prefix + recent | 0.94 | [0.8378, 0.9794] | 308.58s | $0.2142 |
| token_vorn | 512 | prefix + recent | 0.96 | [0.8654, 0.9890] | 249.42s | $0.1731 |
| sentence_vorn | 512 | prefix + recent | 1.00 | [0.9286, 1.0000] | 214.98s | $0.1492 |
| tova | 512 | prefix + recent | 0.56 | [0.4231, 0.6884] | 231.18s | $0.1604 |
| h2o | 512 | prefix + recent | 0.56 | [0.4231, 0.6884] | 235.93s | $0.1637 |

## 512 vs 1024 Paired Tests

| LHS | RHS | Table | Exact McNemar p |
| --- | --- | --- | --- |
| llama_token_512_guarded | llama_token_1024_guarded | [[48, 0], [2, 0]] | 0.5 |
| llama_sentence_512_guarded | llama_sentence_1024_guarded | [[50, 0], [0, 0]] | 1 |
| llama_tova_512_guarded | llama_tova_1024_guarded | [[28, 0], [17, 5]] | 1.52588e-05 |
| llama_h2o_512_guarded | llama_h2o_1024_guarded | [[28, 0], [19, 3]] | 3.8147e-06 |

## Read

- Llama's `1024` threshold regime does not extend cleanly down to `512`.
- `sentence_vorn` remains at ceiling (`1.00`), and `token_vorn` only drops slightly (`1.00 -> 0.96`).
- The attention-weight baselines crack much earlier: `TOVA` drops from `0.90` to `0.56`, and `H2O` drops from `0.94` to `0.56`.
- The immediate paper consequence is that the Llama family has a budget-sensitive break, but the break is **method-asymmetric**: at `512`, the attention-weight baselines lose far more quality than the residual-direction baselines.
