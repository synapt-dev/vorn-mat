# Tail gap fill - 2026-05-25

Surface: `niah_multikey_1_4k`, `validation[:50]`, `max_new_tokens=32`, prefix+recent guardrails, bf16, A100-80GB, memory telemetry default-on.

| Family | Budget | Method | Status | Hits | Hit rate | Peak alloc MB |
|---|---:|---|---|---:|---:|---:|
| Mistral 7B v0.3 | 256 | h2o | oom_partial | 0/3 before OOM | 0.00 | 53493.8 |
| Mistral 7B v0.3 | 256 | sentence_h2o | oom_partial | 3/3 before OOM | 1.00 | 53493.8 |
| Gemma 4 E4B-it | 256 | h2o | completed | 28/50 | 0.56 | 29944.7 |
| Gemma 4 E4B-it | 256 | sentence_h2o | completed | 49/50 | 0.98 | 29944.7 |
| Gemma 4 E4B-it | 512 | vorn | completed | 0/50 | 0.00 | 19948.3 |
| Gemma 4 E4B-it | 512 | h2o | completed | 24/50 | 0.48 | 29944.7 |

Mistral failures match the known attention-memory path: CUDA OOM in `transformers.models.mistral.modeling_mistral` eager-attention softmax, attempted 2.09 GiB allocation with 79.03 GiB process memory in use.

Raw artifacts: `eval-results/vorn-mat/tail-gap-fill-2026-05-25` in `synapt/vorn-mat-cross-family-results`.
