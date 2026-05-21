# Llama 3.1 8B Threshold Curve Through Budget 256 — 2026-05-16

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `meta-llama/Llama-3.1-8B-Instruct`

## Rows

| Method | Budget | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Mean evicted |
| --- | --- | --- | --- | --- | --- | --- |
| token_vorn | 1024 | 1.00 | [0.9286, 1.0000] | 276.25s | $0.1917 | 73.36% |
| token_vorn | 512 | 0.96 | [0.8654, 0.9890] | 249.42s | $0.1731 | 86.68% |
| token_vorn | 256 | 0.56 | [0.4231, 0.6884] | 219.33s | $0.1522 | 93.34% |
| sentence_vorn | 1024 | 1.00 | [0.9286, 1.0000] | 245.17s | $0.1702 | 73.62% |
| sentence_vorn | 512 | 1.00 | [0.9286, 1.0000] | 214.98s | $0.1492 | 86.98% |
| sentence_vorn | 256 | 1.00 | [0.9286, 1.0000] | 268.63s | $0.1864 | 93.72% |
| tova | 1024 | 0.90 | [0.7864, 0.9565] | 395.00s | $0.2741 | 73.36% |
| tova | 512 | 0.56 | [0.4231, 0.6884] | 231.18s | $0.1604 | 86.68% |
| tova | 256 | 0.12 | [0.0562, 0.2381] | 291.68s | $0.2024 | 93.34% |
| h2o | 1024 | 0.94 | [0.8378, 0.9794] | 308.58s | $0.2142 | 73.36% |
| h2o | 512 | 0.56 | [0.4231, 0.6884] | 235.93s | $0.1637 | 86.68% |
| h2o | 256 | 0.08 | [0.0315, 0.1884] | 273.93s | $0.1901 | 93.34% |

## Paired Tests

| LHS | RHS | Table | Exact McNemar p |
| --- | --- | --- | --- |
| token_512 | token_1024 | [[48, 0], [2, 0]] | 0.5 |
| token_256 | token_512 | [[28, 0], [20, 2]] | 1.90735e-06 |
| token_256 | token_1024 | [[28, 0], [22, 0]] | 4.76837e-07 |
| sentence_512 | sentence_1024 | [[50, 0], [0, 0]] | 1 |
| sentence_256 | sentence_512 | [[50, 0], [0, 0]] | 1 |
| sentence_256 | sentence_1024 | [[50, 0], [0, 0]] | 1 |
| tova_512 | tova_1024 | [[28, 0], [17, 5]] | 1.52588e-05 |
| tova_256 | tova_512 | [[6, 0], [22, 22]] | 4.76837e-07 |
| tova_256 | tova_1024 | [[6, 0], [39, 5]] | 3.63798e-12 |
| h2o_512 | h2o_1024 | [[28, 0], [19, 3]] | 3.8147e-06 |
| h2o_256 | h2o_512 | [[4, 0], [24, 22]] | 1.19209e-07 |
| h2o_256 | h2o_1024 | [[4, 0], [43, 3]] | 2.27374e-13 |

## Read

- The residual-direction advantage on Llama does not merely survive to `256`; it sharpens into a hard lower-bound contrast.
- `sentence_vorn` remains at `1.00` at all three budgets (`1024`, `512`, `256`), with zero discordance against the higher-budget rows.
- The token baseline breaks substantially at `256`, falling from `1.00` at `1024` to `0.56`.
- The attention-weight baselines break much harder still: `TOVA` falls from `0.90` to `0.56` to `0.12`, and `H2O` falls from `0.94` to `0.56` to `0.08`.
- The practical consequence is that the Llama family is not merely threshold-shaped. It exhibits a strongly method-asymmetric floor: the sentence-level residual-direction policy remains ceiling-stable even at roughly six percent retention, while token and attention-weight policies degrade sharply.
