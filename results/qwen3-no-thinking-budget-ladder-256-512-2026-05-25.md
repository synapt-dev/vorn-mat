# Qwen3 no-thinking budget ladder (256/512) - 2026-05-25

Surface: `Qwen/Qwen3-8B`, thinking disabled, `niah_multikey_1_4k`, `validation[:50]`, memory telemetry default-on.

| Budget | Method | Status | Hits | Hit rate | Peak alloc MB | Notes |
|---:|---|---|---:|---:|---:|---|
| 256 | h2o | completed_recovered | 0/50 | 0.00 | 53285.0 | Initial full-cell attention run OOMed around case 34; completed via case_offset_start recovery chunks. |
| 256 | sentence_h2o | completed_recovered | 20/50 | 0.40 | 53286.2 | Initial full-cell attention run OOMed around case 34; completed via case_offset_start recovery chunks. |
| 256 | sentence_tova | completed_recovered | 20/50 | 0.40 | 53286.2 | Initial full-cell attention run OOMed around case 34; completed via case_offset_start recovery chunks. |
| 256 | sentence_vorn | completed | 0/50 | 0.00 | 21435.0 |  |
| 256 | tova | completed_recovered | 0/50 | 0.00 | 53286.2 | Initial full-cell attention run OOMed around case 34; completed via case_offset_start recovery chunks. |
| 256 | vorn | completed | 0/50 | 0.00 | 21435.0 |  |
| 512 | h2o | completed_recovered | 0/50 | 0.00 | 53286.2 | Initial full-cell attention run OOMed around case 34; completed via case_offset_start recovery chunks. |
| 512 | sentence_h2o | completed_recovered | 36/50 | 0.72 | 53286.2 | Initial full-cell attention run OOMed around case 34; completed via case_offset_start recovery chunks.; Offset-2 recovery chunk hit an infra/GPU XID after cases 2-4; cases 5-33 recovered via microchunks and cases 34-49 via back-16 chunk. |
| 512 | sentence_tova | completed_recovered | 35/50 | 0.70 | 53286.2 | Initial full-cell attention run OOMed around case 34; completed via case_offset_start recovery chunks. |
| 512 | sentence_vorn | completed | 7/50 | 0.14 | 21435.0 |  |
| 512 | tova | completed_recovered | 0/50 | 0.00 | 53286.2 | Initial full-cell attention run OOMed around case 34; completed via case_offset_start recovery chunks. |
| 512 | vorn | completed | 0/50 | 0.00 | 21435.0 |  |
| 1024 | vanilla | completed | 50/50 | 1.00 | 20938.2 |  |

Raw artifacts: `eval-results/vorn-mat/qwen3-no-thinking-4k-budget-ladder-2026-05-25` in `synapt/vorn-mat-cross-family-results`.
