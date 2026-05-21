# Mistral 7B v0.3 Sentence-Attention Budget Fill @ 256, 1536, and 2048 — 2026-05-20

Run conditions:
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Hardware: original A100 wave plus authorized H100 retry for the late-budget cells

| Budget | Method | Hit rate | 95% Wilson CI | GPU | Wall-clock | Inference cost | KV savings |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256 | token_vorn | 0.90 | [0.7864, 0.9565] | A100-80GB | 199.17s | $0.1382 | 93.88% |
| 256 | sentence_vorn | 0.50 | [0.3664, 0.6336] | A100-80GB | 154.48s | $0.1072 | 94.41% |
| 256 | tova | 0.04 | [0.0110, 0.1346] | A100-80GB | 237.47s | $0.1648 | 93.88% |
| 256 | sentence_tova | 0.74 | [0.6045, 0.8413] | A100-80GB | 150.96s | $0.1048 | 94.12% |
| 1536 | token_vorn | 0.94 | [0.8378, 0.9794] | A100-80GB | 631.18s | $0.4380 | 63.30% |
| 1536 | sentence_vorn | 1.00 | [0.9286, 1.0000] | A100-80GB | 313.15s | $0.2173 | 63.61% |
| 2048 | token_vorn | 0.98 | [0.8950, 0.9965] | H100-80GB | 441.43s | $0.4842 | 51.06% |
| 2048 | sentence_vorn | 1.00 | [0.9286, 1.0000] | H100-80GB | 242.80s | $0.2664 | 51.39% |

## Runtime Unsupported

| Budget | Method | Status | Reason |
| --- | --- | --- | --- |
| 1536 | tova | runtime_unsupported_on_h100_80gb_current_runner | CUDA OOM on both the original A100 wave and the authorized H100-80GB retry. The H100 signature remained the same late-budget Mistral failure mode: attempted 2.10 GiB attention-softmax allocation with ~79.18 GiB total and ~1.2-1.3 GiB free. |
| 1536 | sentence_tova | runtime_unsupported_on_h100_80gb_current_runner | CUDA OOM on both the original A100 wave and the authorized H100-80GB retry. The H100 signature remained the same late-budget Mistral failure mode: attempted 2.10 GiB attention-softmax allocation with ~79.18 GiB total and ~1.2-1.3 GiB free. |
| 2048 | tova | runtime_unsupported_on_h100_80gb_current_runner | CUDA OOM on both the original A100 wave and the authorized H100-80GB retry. The H100 signature remained the same late-budget Mistral failure mode: attempted 2.10 GiB attention-softmax allocation with ~79.18 GiB total and ~1.2-1.3 GiB free. |
| 2048 | sentence_tova | runtime_unsupported_on_h100_80gb_current_runner | CUDA OOM on both the original A100 wave and the authorized H100-80GB retry. The H100 signature remained the same late-budget Mistral failure mode: attempted 2.10 GiB attention-softmax allocation with ~79.18 GiB total and ~1.2-1.3 GiB free. |

## Pairwise Tests

- sentence_tova_256_guarded vs tova_256_guarded, exact McNemar on `[[2, 35], [0, 13]]`: `p = 5.82077e-11`
- sentence_tova_256_guarded vs sentence_256_guarded, exact McNemar on `[[17, 20], [8, 5]]`: `p = 0.0356981`
- sentence_256_guarded vs token_256_guarded, exact McNemar on `[[23, 2], [22, 3]]`: `p = 3.5882e-05`
- sentence_1536_guarded vs token_1536_guarded, exact McNemar on `[[47, 3], [0, 0]]`: `p = 0.25`
- sentence_2048_guarded vs token_2048_guarded, exact McNemar on `[[49, 1], [0, 0]]`: `p = 1`

## Read

- At `256`, token_vorn remains strongest (`0.90`), sentence_vorn is `0.50`, token TOVA is `0.04`, and sentence-TOVA recovers to `0.74`.
- At `1536`, the available vorn rows are `token_vorn 0.94` and `sentence_vorn 1.00`; both attention-weight rows remained runtime-unsupported on the authorized H100 retry.
- At `2048`, H100 unblocked `token_vorn 0.98` and `sentence_vorn 1.00`, but both attention-weight rows remained runtime-unsupported.
