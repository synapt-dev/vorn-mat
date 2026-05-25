# 1024 Low-Budget Backfill

This wave closes the required 1024 matrix gaps on `niah_multikey_1_4k`, `validation[:50]`, budget `1024`, `max_new_tokens=32`, `bf16`, `A100-80GB`, profile `layne-79000`.

Raw per-cell reports and observation ledgers are synced under:

`eval-results/vorn-mat/low-budget-backfill-2026-05-25/`

## Results

| Family | Method | Status | Result | Peak allocated MB | Peak reserved MB |
| --- | --- | --- | --- | ---: | ---: |
| Ministral 8B | vorn | completed | 49/50 = 0.98 | 21275.667 | 39948.0 |
| Ministral 8B | h2o | failed_oom | 6/9 before OOM | 53774.406 | 73358.0 |
| Ministral 8B | sentence_h2o | failed_oom | 9/9 before OOM | 53774.406 | 73356.0 |
| Gemma 2 9B | vorn | completed | 2/50 = 0.04 | 22642.563 | 34244.0 |
| Gemma 2 9B | h2o | completed | 30/50 = 0.60 | 43291.875 | 59800.0 |
| Gemma 2 9B | sentence_h2o | completed | 41/50 = 0.82 | 43285.733 | 59800.0 |
| Qwen 2.5 7B Instruct | h2o | completed | 30/50 = 0.60 | 41166.796 | 62982.0 |
| Qwen 2.5 7B Instruct | sentence_h2o | completed | 35/50 = 0.70 | 41166.796 | 62982.0 |

## Read

Six cells completed at `n=50`. The two remaining Ministral attention-channel rows are no longer silent dashes: both hit the known eager-attention OOM ceiling after preserving 9 per-case observations. The OOM signature was the same for both: `modeling_ministral.py` eager attention softmax attempted a 1.82 GiB allocation with 77.55 GiB process memory in use.
