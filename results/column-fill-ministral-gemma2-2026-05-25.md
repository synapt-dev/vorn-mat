# Column fill: Ministral/Gemma2 low-budget gaps - 2026-05-25

Only 8 cells were truly missing after canonical dedupe; Qwen2.5 and B=1024 requested cells already existed.

| Family | Budget | Method | Status | Hits | Hit rate | Peak alloc MB |
|---|---:|---|---|---:|---:|---:|
| Ministral 8B | 256 | sentence_vorn | completed | 14/50 | 0.28 | 21275.7 |
| Ministral 8B | 256 | tova | failed_oom | 0/9 before OOM | n/a | 53774.4 |
| Ministral 8B | 512 | sentence_vorn | completed | 43/50 | 0.86 | 21275.7 |
| Ministral 8B | 512 | tova | failed_oom | 1/9 before OOM | n/a | 53774.4 |
| Gemma 2 9B | 256 | sentence_vorn | completed | 6/50 | 0.12 | 22642.6 |
| Gemma 2 9B | 256 | tova | completed | 4/50 | 0.08 | 42550.1 |
| Gemma 2 9B | 512 | sentence_vorn | completed | 23/50 | 0.46 | 22642.6 |
| Gemma 2 9B | 512 | tova | completed | 10/50 | 0.20 | 42550.1 |

Raw artifacts: `eval-results/vorn-mat/column-fill-2026-05-25` in `synapt/vorn-mat-cross-family-results`.
