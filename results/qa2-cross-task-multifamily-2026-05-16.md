# Cross-Task Generalization (`qa_2_4k`, `n=200`) — 2026-05-16

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `qa_2_4k`
- Slice: `validation[:200]`
- Pooling: `max`

## Rows

| Family | Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | KV savings |
|--------|--------|--------|------------|----------|---------------|------------|----------------|-----------|
| Gemma 4 E4B-it | vanilla | full context | prefix + recent | 0.450 | [0.3826, 0.5192] | 255.19s | $0.1771 | 0.00% |
| Gemma 4 E4B-it | sentence_vorn | 1024 | prefix + recent | 0.260 | [0.2041, 0.3249] | 665.77s | $0.4620 | 69.97% |
| Gemma 4 E4B-it | tova | 1024 | prefix + recent | 0.310 | [0.2500, 0.3772] | 1296.15s | $0.8995 | 69.54% |
| Gemma 4 E4B-it | h2o | 1024 | prefix + recent | 0.305 | [0.2454, 0.3720] | 1668.18s | $1.1577 | 69.54% |
| Llama 3.1 8B | vanilla | full context | prefix + recent | 0.385 | [0.3203, 0.4540] | 264.73s | $0.1837 | 0.00% |
| Llama 3.1 8B | sentence_vorn | 1024 | prefix + recent | 0.460 | [0.3923, 0.5292] | 428.76s | $0.2976 | 68.99% |
| Llama 3.1 8B | tova | 1024 | prefix + recent | 0.330 | [0.2686, 0.3978] | 569.68s | $0.3954 | 68.49% |
| Llama 3.1 8B | h2o | 1024 | prefix + recent | 0.330 | [0.2686, 0.3978] | 538.68s | $0.3738 | 68.49% |
| Ministral 8B | vanilla | full context | prefix + recent | 0.575 | [0.5057, 0.6415] | 282.25s | $0.1959 | 0.00% |
| Ministral 8B | sentence_vorn | 1024 | prefix + recent | 0.510 | [0.4412, 0.5784] | 493.36s | $0.3424 | 70.25% |

## Paired Tests

- gemma4_sentence_1024 vs gemma4_vanilla, exact McNemar on `[[38, 14], [52, 96]]`: `p = 2.82202e-06`
- gemma4_sentence_1024 vs gemma4_tova_1024, exact McNemar on `[[30, 22], [32, 116]]`: `p = 0.220328`
- gemma4_tova_1024 vs gemma4_vanilla, exact McNemar on `[[52, 10], [38, 100]]`: `p = 6.16964e-05`
- gemma4_sentence_1024 vs gemma4_h2o_1024, exact McNemar on `[[29, 23], [32, 116]]`: `p = 0.28061`
- gemma4_h2o_1024 vs gemma4_vanilla, exact McNemar on `[[51, 10], [39, 100]]`: `p = 3.84591e-05`
- llama31_sentence_1024 vs llama31_vanilla, exact McNemar on `[[69, 23], [8, 100]]`: `p = 0.0106738`
- llama31_sentence_1024 vs llama31_tova_1024, exact McNemar on `[[56, 36], [10, 98]]`: `p = 0.000156417`
- llama31_tova_1024 vs llama31_vanilla, exact McNemar on `[[52, 14], [25, 109]]`: `p = 0.108129`
- llama31_sentence_1024 vs llama31_h2o_1024, exact McNemar on `[[56, 36], [10, 98]]`: `p = 0.000156417`
- llama31_h2o_1024 vs llama31_vanilla, exact McNemar on `[[52, 14], [25, 109]]`: `p = 0.108129`
- ministral_sentence_1024 vs ministral_vanilla, exact McNemar on `[[92, 10], [23, 75]]`: `p = 0.035082`

## Execution Boundaries

- ministral_tova_1024: Repeated CUDA OOM in Ministral attention-weight path on qa_2_4k validation[:200], including one retry with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
- ministral_h2o_1024: Repeated CUDA OOM in Ministral attention-weight path on qa_2_4k validation[:200], including one retry with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.

## Read

- Gemma 4 E4B-it: vanilla `0.450`; sentence `0.260`; TOVA `0.310`; H2O `0.305`. Best completed compressed method is `tova` at `0.310`. Sentence vs TOVA McNemar `p = 0.220328`. Sentence vs H2O McNemar `p = 0.28061`.
- Llama 3.1 8B: vanilla `0.385`; sentence `0.460`; TOVA `0.330`; H2O `0.330`. Best completed compressed method is `sentence_vorn` at `0.460`. Sentence vs TOVA McNemar `p = 0.000156417`. Sentence vs H2O McNemar `p = 0.000156417`.
- Ministral 8B: vanilla `0.575`; sentence `0.510`; TOVA `OOM at n=200 on current stack`; H2O `OOM at n=200 on current stack`. Best completed compressed method is `sentence_vorn` at `0.510`.
- Execution boundary: Ministral attention-weight rows at `qa_2_4k`, `n=200`, `budget=1024` repeatedly OOMed on the current Modal/A100 stack, including one retry with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Treat the missing TOVA/H2O cells as runtime-unsupported on this surface, not as scored negative rows.
