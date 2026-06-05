# LongBench PassageRetrieval-en Tier 1 Probe — 2026-06-03

Public mirror of the preregistered config#316 LongBench PassageRetrieval-en probe. The run preserves the scientific contract: first 50 LongBench validation rows, B=1024, sentence_vorn vs sentence_tova, four locked model families, official LongBench retrieval_score mean primary, binary paragraph hit secondary, bf16, greedy decoding, max_new_tokens=32.

## Result

- sentence_vorn completed all 4 preregistered cells on a uniform H200 single-GPU substrate.
- sentence_tova is capacity-missing for all 4 preregistered cells: batched H200 failed all cells, and segmented one-case H200 recovery completed 2/200 chunks overall.
- The two segmented TOVA successes were both Gemma 4 chunks and both scored 0.00.
- Treat TOVA missingness as method-capacity missingness, not as zero accuracy for paired efficacy scoring.

## Cell Summary

| Family | Model | sentence_vorn score | sentence_vorn binary hit | sentence_vorn peak alloc GiB | sentence_tova status | sentence_tova segmented result |
|---|---|---:|---:|---:|---|---|
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` | 0.25 | 0.24 | 78.60 | capacity-missing | 0/50 successes; failures: cuda_oom=2, modal_deadline_exceeded=47, modal_resource_exhausted=1 |
| Llama 3.1 | `meta-llama/Llama-3.1-8B-Instruct` | 0.34 | 0.34 | 66.83 | capacity-missing | 0/50 successes; failures: cuda_oom=6, modal_deadline_exceeded=41, modal_resource_exhausted=3 |
| Gemma 4 | `google/gemma-4-E4B-it` | 0.04 | 0.04 | 31.02 | capacity-missing | success offsets [0, 10]; failures: cuda_oom=2, modal_deadline_exceeded=42, modal_resource_exhausted=4 |
| Qwen 3-NT | `Qwen/Qwen3-8B` | 0.34 | 0.34 | 69.66 | capacity-missing | 0/50 successes; failures: modal_deadline_exceeded=39, modal_resource_exhausted=11 |

## Modal Run IDs

- a100_diagnostic: `ap-N0hbLAbFtHdgaMD016Hkpz`
- h100_smoke: `ap-rY0Qnxxf6ARN2sNtn3cAhy`
- h200_full_wave: `ap-5tJP6Qg18nzuBzbjDm1elG`
- h200_tova_segmented: `ap-dxPlBLHBj15zXCzw85JGlT`

## Claim Bounds

- Do not report failed TOVA cells as zero-scored accuracy cells; treat them as capacity-missing / substrate-bound missingness.
- Do report the deployability asymmetry: vorn completed uniformly on single-H200; TOVA did not fit reliably even under one-case segmentation.
- Do not claim a completed LongBench family-conditional efficacy replication of the RULER matrix; the preregistered paired comparison was blocked by method-specific capacity.
- The four vorn cells are descriptive cross-task evidence, not a completed paired comparison against TOVA.

## Raw Sources

- h200_full_wave_reports: `.benchmarks/longbench-passage-wave-h200/reports.json`
- h200_full_wave_failures: `.benchmarks/longbench-passage-wave-h200/failures.json`
- h200_tova_segmented_reports: `.benchmarks/longbench-passage-wave-h200/tova-segmented-n1/reports.json`
- h200_tova_segmented_failures: `.benchmarks/longbench-passage-wave-h200/tova-segmented-n1/failures.json`
- h200_tova_segmented_summary: `.benchmarks/longbench-passage-wave-h200/tova-segmented-n1/segmented-tova-assembly-summary.json`

## HF Mirror

Canonical dataset target: `synapt/vorn-mat-cross-family-results` at `eval-results/vorn-mat/longbench-passage-retrieval-2026-06-03/`. The HF bundle contains one JSON artifact per preregistered cell plus this aggregate envelope.

