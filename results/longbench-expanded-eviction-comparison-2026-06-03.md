# LongBench Expanded Eviction Comparison Probe - 2026-06-03

Public mirror of the config#321 LongBench PassageRetrieval-en expanded comparison probe. The run preserves the original config#316 task surface: first 50 LongBench PassageRetrieval-en rows, B=1024 for eviction methods, four locked model families, official LongBench retrieval_score mean primary, binary paragraph hit secondary, bf16, greedy decoding, max_new_tokens=32. Config#321 adds deployable eviction comparisons (`sentence_snapkv`, `sentence_l2_norm`, `sentence_streaming_llm`) plus vanilla no-eviction baselines.

## Result

- All 16 config#321 cells completed on a uniform single-H200 substrate under Modal profile `layne1penney`.
- Vanilla baselines scored 0.96-1.00 across all four families, so low eviction scores are method-side behavior on this slice rather than baseline model inability.
- `sentence_snapkv` is the strongest added deployable eviction baseline in all four families.
- `sentence_l2_norm` is near-floor across all four families; the Gemma 4 L2 cell required an implementation fix for Gemma 4 nested decoder-layer resolution, then completed cleanly.
- Original config#316 `sentence_tova` remains capacity-missing and should not be scored as zero in method comparisons.

## Cell Summary

| Family | Method | retrieval_score mean | LongBench percent | binary hit | peak alloc GiB | elapsed sec | est. cost |
|---|---|---:|---:|---:|---:|---:|---:|
| Mistral | `sentence_snapkv` | 0.2 | 20.00 | 0.2 | 81.20 | 497.07 | $0.6268 |
| Mistral | `sentence_l2_norm` | 0.03667 | 3.67 | 0.02 | 80.28 | 376.07 | $0.4742 |
| Mistral | `sentence_streaming_llm` | 0.1 | 10.00 | 0.1 | 78.60 | 315.63 | $0.3980 |
| Mistral | `vanilla` | 0.96 | 96.00 | 0.96 | 77.02 | 265.93 | $0.3353 |
| Llama 3.1 | `sentence_snapkv` | 0.26 | 26.00 | 0.26 | 71.40 | 444.86 | $0.5610 |
| Llama 3.1 | `sentence_l2_norm` | 0.0275 | 2.75 | 0.02 | 68.33 | 337.25 | $0.4253 |
| Llama 3.1 | `sentence_streaming_llm` | 0.12 | 12.00 | 0.12 | 66.83 | 305.04 | $0.3847 |
| Llama 3.1 | `vanilla` | 1 | 100.00 | 1 | 65.43 | 253.97 | $0.3203 |
| Gemma 4 | `sentence_snapkv` | 0.1 | 10.00 | 0.1 | 37.94 | 278.06 | $0.3506 |
| Gemma 4 | `sentence_l2_norm` | 0.05 | 5.00 | 0.04 | 31.62 | 197.82 | $0.2495 |
| Gemma 4 | `sentence_streaming_llm` | 0.1 | 10.00 | 0.1 | 31.02 | 287.33 | $0.3623 |
| Gemma 4 | `vanilla` | 1 | 100.00 | 1 | 29.20 | 147.61 | $0.1861 |
| Qwen 3-NT | `sentence_snapkv` | 0.32 | 32.00 | 0.32 | 75.10 | 506.08 | $0.6382 |
| Qwen 3-NT | `sentence_l2_norm` | 0.0475 | 4.75 | 0.04 | 71.39 | 397.92 | $0.5018 |
| Qwen 3-NT | `sentence_streaming_llm` | 0.12 | 12.00 | 0.12 | 69.66 | 311.59 | $0.3929 |
| Qwen 3-NT | `vanilla` | 1 | 100.00 | 1 | 68.03 | 288.25 | $0.3635 |

## Modal Run IDs

- expanded_h200_full_wave: `ap-5cs1xz3u1oPaN08eZn2l7a`
- gemma4_l2_recovery_1: `ap-HrKfjmyZrcRVVkYRuT1FR1`
- gemma4_l2_recovery_2: `ap-IeovWSuZZXgn2IjE6ux0F5`
- gemma4_l2_recovery_3: `ap-yShCVueNWnRUwelWslVcO6`

## Claim Bounds

- Compare measured methods only where both cells completed; all config#321 expanded cells completed.
- Do not fold original sentence_tova capacity-missing cells into zero-score method comparisons.
- Report vanilla baselines separately as no-eviction ceilings, not as eviction methods.
- Treat n=50 per family as a small-sample probe; family-conditional patterns are descriptive unless followed by higher-N confirmation.

## Raw Sources

Canonical dataset target: `synapt/vorn-mat-cross-family-results` at `eval-results/vorn-mat/longbench-expanded-eviction-comparison-2026-06-03/`. The HF bundle contains one JSON artifact per config#321 cell plus this aggregate envelope.
HF upload commit: `8e28b174c9c3446d4b4d41093015bebcbbb14277`.

- expanded_h200_full_wave_reports: `.benchmarks/longbench-expanded-comparison-h200/reports.json`
- expanded_h200_full_wave_failures: `.benchmarks/longbench-expanded-comparison-h200/failures.json`
- gemma4_l2_rerun3_reports: `.benchmarks/longbench-expanded-comparison-h200/rerun3-gemma4-l2/reports.json`
