# Mistral 7B v0.3 Memory Rerun — 2026-05-25

Run conditions:
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- GPU: `A100-80GB`
- Precision/substrate: patched `vorn-mat` bf16 model load with default per-case memory telemetry
- Guardrails: `prefix_plus_recent`
- Raw artifacts: <https://huggingface.co/datasets/synapt/vorn-mat-cross-family-results/tree/main/eval-results/vorn-mat/mistral-memory-rerun-2026-05-25>

## Rows

| Budget | Method | Status | Hit rate | Cases | Peak allocated | Peak reserved | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1536 | `vorn` | completed | 0.96 | 50/50 | 20.44 GB | 35.61 GB | $0.4333 |
| 1536 | `sentence_vorn` | completed | 1.00 | 50/50 | 20.44 GB | 35.61 GB | $0.2125 |
| 1536 | `tova` | OOM | 3/3 before OOM | 3/50 | 54.96 GB before OOM | 72.44 GB before OOM | partial |
| 1536 | `sentence_tova` | OOM | 3/3 before OOM | 3/50 | 54.93 GB before OOM | 72.44 GB before OOM | partial |
| 2048 | `vorn` | completed | 0.98 | 50/50 | 20.44 GB | 35.61 GB | $0.6029 |
| 2048 | `sentence_vorn` | completed | 1.00 | 50/50 | 20.44 GB | 35.61 GB | $0.3021 |
| 2048 | `tova` | completed | 0.98 | 50/50 | 59.35 GB | 79.30 GB | $0.5245 |
| 2048 | `sentence_tova` | OOM | 3/3 before OOM | 3/50 | 59.10 GB before OOM | 78.86 GB before OOM | partial |

## Read

The current-substrate rerun preserves the vorn control story: both token-vorn and sentence-vorn complete with peak allocated memory around `20.44 GB` and reserved memory around `35.61 GB`.

The attention rows expose the memory boundary. `tova @2048` now completes, but only while running close to the A100-80GB ceiling (`79.30 GB` reserved). `tova @1536` and both sentence-TOVA rows reproduce the historical OOM failure after three completed fixtures, with the Mistral eager-attention softmax attempting an additional `2.09 GiB` allocation while the process already holds about `77.50 GiB`.

This supports a paper-safe framing: late-budget Mistral attention baselines are not uniformly missing because of lack of effort; they are operating at or beyond the current runner's memory ceiling, while matched vorn rows complete far below that ceiling on the same model, task, and substrate.
