# Sentence-Attention Cross-Task Generalization (`qa_2_4k`, `n=200`) — 2026-05-20

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `qa_2_4k`
- Slice: `validation[:200]`
- Budget: `1024`
- Guardrails: `prefix + recent`

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
| Gemma 4 E4B-it | sentence_tova | 1024 | prefix + recent | 0.370 | [0.3061, 0.4388] | 774.45s | $0.5375 | 70.06% |
| Gemma 4 E4B-it | sentence_h2o | 1024 | prefix + recent | 0.370 | [0.3061, 0.4388] | 864.16s | $0.5997 | 70.08% |
| Llama 3.1 8B | sentence_tova | 1024 | prefix + recent | 0.425 | [0.3585, 0.4943] | 401.85s | $0.2789 | 69.03% |
| Llama 3.1 8B | sentence_h2o | 1024 | prefix + recent | 0.425 | [0.3585, 0.4943] | 393.65s | $0.2732 | 69.06% |

## New Sentence-Attention Contrasts

- gemma4_sentence_1024 vs gemma4_sentence_tova_1024, exact McNemar on `[[39, 13], [35, 113]]`: `p = 0.00208811`
- gemma4_sentence_1024 vs gemma4_sentence_h2o_1024, exact McNemar on `[[39, 13], [35, 113]]`: `p = 0.00208811`
- gemma4_sentence_tova_1024 vs gemma4_tova_1024, exact McNemar on `[[48, 26], [14, 112]]`: `p = 0.0806905`
- gemma4_sentence_h2o_1024 vs gemma4_h2o_1024, exact McNemar on `[[48, 26], [13, 113]]`: `p = 0.0532519`
- llama31_sentence_1024 vs llama31_sentence_tova_1024, exact McNemar on `[[75, 17], [10, 98]]`: `p = 0.247789`
- llama31_sentence_1024 vs llama31_sentence_h2o_1024, exact McNemar on `[[75, 17], [10, 98]]`: `p = 0.247789`
- llama31_sentence_tova_1024 vs llama31_tova_1024, exact McNemar on `[[53, 32], [13, 102]]`: `p = 0.00660882`
- llama31_sentence_h2o_1024 vs llama31_h2o_1024, exact McNemar on `[[53, 32], [13, 102]]`: `p = 0.00660882`

## Execution Boundaries

- ministral_tova_1024: Repeated CUDA OOM in Ministral attention-weight path on qa_2_4k validation[:200], including one retry with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
- ministral_h2o_1024: Repeated CUDA OOM in Ministral attention-weight path on qa_2_4k validation[:200], including one retry with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
- ministral_sentence_tova_1024: Repeated CUDA OOM in Ministral sentence-attention path on qa_2_4k validation[:200] on the current Modal/A100 stack.
- ministral_sentence_h2o_1024: Repeated CUDA OOM in Ministral sentence-attention path on qa_2_4k validation[:200] on the current Modal/A100 stack.

## Read

- Gemma 4 E4B-it: sentence_vorn `0.260`; token-TOVA `0.310`; token-H2O `0.305`; sentence-TOVA `0.370`; sentence-H2O `0.370`.
- Llama 3.1 8B: sentence_vorn `0.460`; token-TOVA `0.330`; token-H2O `0.330`; sentence-TOVA `0.425`; sentence-H2O `0.425`.
- Ministral 8B: both sentence-attention rows OOMed on the current A100 stack, so the cross-task sentence-attention comparison remains unresolved for this family.
