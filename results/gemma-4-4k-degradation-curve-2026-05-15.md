# Gemma 4 E4B-it Degradation Curve @ NIAH Multikey 4k

## Setup
- Model: `google/gemma-4-E4B-it`
- Fixture: `rbiswasfc/ruler` `niah_multikey_1_4k` `validation[:50]`
- Profile: `author`
- Seed: `17`
- Budgets: `512`, `1024`, `1536`, `2048`
- Methods: vanilla, token-vorn, sentence-vorn, TOVA, sparse H2O anchor at `1024`

## Read
- Gemma's degradation curve has a sharp bend rather than a smooth decline.
- At `512`, the residual-direction methods are near floor: `sentence_vorn=0.04`, `token_vorn=0.00`, while `TOVA=0.34`.
- At `1024`, the earlier cross-baseline split remains the center anchor: `sentence_vorn=0.24`, `token_vorn=0.02`, `TOVA=0.94`, `H2O=0.94`.
- By `1536`, sentence-level retention becomes workable (`0.68`) but still trails the attention-weight baseline (`0.98`).
- By `2048`, sentence-level vorn nearly recovers the ceiling (`0.96`), while token-level is still materially degraded (`0.52`) and TOVA saturates (`1.00`).
- The honest local claim is now sharper: on Gemma 4, method ranking is budget-dependent inside the residual-direction family, but the attention-weight advantage persists across the whole observed bend.

## Rows
| Method | Budget | Guardrails | Hit rate | Wilson 95% CI | Elapsed | Cost | Positions evicted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vanilla | full context | prefix + recent | 1.00 | [0.9286, 1.0000] | 217.27s | $0.1508 | 0.00% |
| token_vorn | 512 | prefix + recent | 0.00 | [0.0000, 0.0714] | 403.67s | $0.2801 | 87.01% |
| sentence_vorn | 512 | prefix + recent | 0.04 | [0.0110, 0.1346] | 269.72s | $0.1872 | 87.45% |
| tova | 512 | prefix + recent | 0.34 | [0.2244, 0.4785] | 383.05s | $0.2658 | 87.01% |
| token_vorn | 1024 | prefix + recent | 0.02 | [0.0035, 0.1050] | 546.19s | $0.3791 | 74.03% |
| sentence_vorn | 1024 | prefix + recent | 0.24 | [0.1430, 0.3741] | 291.44s | $0.2023 | 74.38% |
| tova | 1024 | prefix + recent | 0.94 | [0.8378, 0.9794] | 451.24s | $0.3132 | 74.03% |
| h2o | 1024 | prefix + recent | 0.94 | [0.8378, 0.9794] | 3141.36s | $2.1801 | 74.03% |
| token_vorn | 1536 | prefix + recent | 0.22 | [0.1275, 0.3524] | 636.75s | $0.4419 | 61.04% |
| sentence_vorn | 1536 | prefix + recent | 0.68 | [0.5419, 0.7924] | 394.20s | $0.2736 | 61.33% |
| tova | 1536 | prefix + recent | 0.98 | [0.8950, 0.9965] | 623.58s | $0.4328 | 61.04% |
| token_vorn | 2048 | prefix + recent | 0.52 | [0.3851, 0.6520] | 848.17s | $0.5886 | 48.05% |
| sentence_vorn | 2048 | prefix + recent | 0.96 | [0.8654, 0.9890] | 478.76s | $0.3323 | 48.31% |
| tova | 2048 | prefix + recent | 1.00 | [0.9286, 1.0000] | 807.76s | $0.5606 | 48.05% |

## Pairwise Tests
| LHS | RHS | Table | Exact McNemar p |
| --- | --- | --- | --- |
| tova_512_guarded | sentence_512_guarded | [[1, 16], [1, 32]] | 0.000274658 |
| sentence_512_guarded | token_512_guarded | [[0, 2], [0, 48]] | 0.5 |
| tova_1024_guarded | sentence_1024_guarded | [[12, 35], [0, 3]] | 5.82077e-11 |
| tova_1536_guarded | sentence_1536_guarded | [[34, 15], [0, 1]] | 6.10352e-05 |
| sentence_1536_guarded | token_1536_guarded | [[10, 24], [1, 15]] | 1.54972e-06 |
| tova_2048_guarded | sentence_2048_guarded | [[48, 2], [0, 0]] | 0.5 |
| sentence_2048_guarded | token_2048_guarded | [[24, 24], [2, 0]] | 1.04904e-05 |

## Provenance
- Raw predictions are embedded directly in the JSON artifact under `rows[].observations[]`.
- Source raw-report filenames and run IDs are recorded per row.
- Raw Modal report home for replay/audit: `.benchmarks/cross-model/`.
