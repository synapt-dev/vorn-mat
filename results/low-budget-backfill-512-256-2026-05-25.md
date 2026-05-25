# Low-Budget Backfill 512/256 (2026-05-25)

Continuation of the low-budget backfill wave on the canonical guarded `niah_multikey_1_4k` surface.

## Contract

- Harness: `vorn-mat` `a7911e78d45f0c77b401c76b432a12740f3a4977`
- Modal profile: `layne-79000`
- Dataset: `rbiswasfc/ruler`, `niah_multikey_1_4k`, `validation[:50]`
- Budgets: `512`, `256`
- Models: Ministral 8B, Gemma 2 9B, Qwen 2.5 7B Instruct
- Policies: `vorn`, `h2o`, `sentence_h2o`
- Memory telemetry: default-on per-case peak allocated/reserved
- Raw artifacts: HF `synapt/vorn-mat-cross-family-results`, path `eval-results/vorn-mat/low-budget-backfill-2026-05-25`

## Results

| Budget | Model | Policy | Hit Rate | Peak Alloc MB | Peak Reserved MB |
| --- | --- | --- | ---: | ---: | ---: |
| 512 | Ministral 8B | vorn | 28/50 = 0.56 | 21275.667 | 39950.0 |
| 512 | Gemma 2 9B | vorn | 0/50 = 0.00 | 22642.563 | 34244.0 |
| 512 | Gemma 2 9B | h2o | 10/50 = 0.20 | 42550.131 | 59800.0 |
| 512 | Gemma 2 9B | sentence_h2o | 21/50 = 0.42 | 42549.866 | 78906.0 |
| 512 | Qwen 2.5 7B | h2o | 23/50 = 0.46 | 41135.410 | 78908.0 |
| 512 | Qwen 2.5 7B | sentence_h2o | 33/50 = 0.66 | 41135.410 | 67186.0 |
| 256 | Ministral 8B | vorn | 5/50 = 0.10 | 21275.667 | 39948.0 |
| 256 | Gemma 2 9B | vorn | 0/50 = 0.00 | 22642.563 | 34246.0 |
| 256 | Gemma 2 9B | h2o | 5/50 = 0.10 | 42550.131 | 59802.0 |
| 256 | Gemma 2 9B | sentence_h2o | 8/50 = 0.16 | 42549.866 | 78906.0 |
| 256 | Qwen 2.5 7B | h2o | 6/50 = 0.12 | 41135.410 | 78906.0 |
| 256 | Qwen 2.5 7B | sentence_h2o | 37/50 = 0.74 | 41135.410 | 67184.0 |

## Excluded Cells

Both Ministral attention rows OOMed at both budgets with the same eager-attention softmax signature seen at B=1024: 77.55 GiB process memory in use, 1.82 GiB allocation attempted.

Per-case ledgers preserved partial observations:

| Budget | Policy | Partial Cases | Partial Hits | Peak Alloc MB | Peak Reserved MB |
| --- | --- | ---: | ---: | ---: | ---: |
| 512 | h2o | 9 | 1 | 53774.406 | 73360.0 |
| 512 | sentence_h2o | 9 | 8 | 53774.406 | 73362.0 |
| 256 | h2o | 9 | 0 | 53774.406 | 73358.0 |
| 256 | sentence_h2o | 9 | 6 | 53774.406 | 73358.0 |

## Notes

- The two-tier wave stayed under the $16 authorized ceiling. Successful-cell cost was approximately `$1.38` at B=512 and `$1.34` at B=256, excluding failed-cell partial burn.
- Qwen 2.5 `sentence_h2o` remains strong at low budgets (`0.66` at 512, `0.74` at 256) while token `h2o` collapses.
- Gemma 2 `sentence_h2o` degrades sharply from the B=1024 row (`0.82`) to B=512 (`0.42`) and B=256 (`0.16`).
- Ministral `vorn` degrades from the B=1024 row (`0.98`) to B=512 (`0.56`) and B=256 (`0.10`).
