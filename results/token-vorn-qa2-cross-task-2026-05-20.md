# Phase 2E — Token-vorn `qa_2_4k` Cross-Task Addendum (2026-05-20)

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `qa_2_4k`
- Slice: `validation[:200]`
- Budget: `1024`
- Pooling: `max`
- Sentence top-k: `3`

## Rows

| Family | Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | KV savings |
|--------|--------|--------|------------|----------|---------------|------------|----------------|-----------|
| Gemma 4 E4B-it | vanilla | full context | prefix + recent | 0.450 | [0.3826, 0.5192] | 255.19s | $0.1771 | 0.00% |
| Gemma 4 E4B-it | vorn | 1024 | prefix + recent | 0.050 | [0.0274, 0.0896] | 1901.96s | $1.3200 | 69.54% |
| Gemma 4 E4B-it | sentence_vorn | 1024 | prefix + recent | 0.260 | [0.2041, 0.3249] | 665.77s | $0.4620 | 69.97% |
| Llama 3.1 8B | vanilla | full context | prefix + recent | 0.385 | [0.3203, 0.4540] | 264.73s | $0.1837 | 0.00% |
| Llama 3.1 8B | vorn | 1024 | prefix + recent | 0.260 | [0.2041, 0.3249] | 976.62s | $0.6778 | 68.49% |
| Llama 3.1 8B | sentence_vorn | 1024 | prefix + recent | 0.460 | [0.3923, 0.5292] | 428.76s | $0.2976 | 68.99% |

## Paired Tests

- gemma4_sentence_1024 vs gemma4_token_1024, exact McNemar on `[[7, 45], [3, 145]]`: `p = 1.31259e-10`
- gemma4_sentence_1024 vs gemma4_vanilla, exact McNemar on `[[38, 14], [52, 96]]`: `p = 2.82202e-06`
- gemma4_token_1024 vs gemma4_vanilla, exact McNemar on `[[10, 0], [80, 110]]`: `p = 1.65436e-24`
- llama31_sentence_1024 vs llama31_token_1024, exact McNemar on `[[46, 46], [6, 102]]`: `p = 1.03258e-08`
- llama31_sentence_1024 vs llama31_vanilla, exact McNemar on `[[69, 23], [8, 100]]`: `p = 0.0106738`
- llama31_token_1024 vs llama31_vanilla, exact McNemar on `[[42, 10], [35, 113]]`: `p = 0.000247088`

## Read

- Gemma 4 E4B-it: sentence-vorn `0.260` (52/200) at `113` correct/$1 versus token-vorn `0.050` (10/200) at `8` correct/$1. Sentence-vorn vs token-vorn McNemar `p = 1.31259e-10`. Sentence-vorn vs vanilla `p = 2.82202e-06`. Token-vorn vs vanilla `p = 1.65436e-24`.
- Llama 3.1 8B: sentence-vorn `0.460` (92/200) at `309` correct/$1 versus token-vorn `0.260` (52/200) at `77` correct/$1. Sentence-vorn vs token-vorn McNemar `p = 1.03258e-08`. Sentence-vorn vs vanilla `p = 0.0106738`. Token-vorn vs vanilla `p = 0.000247088`.
