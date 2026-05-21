# Cross-Family Extension Wave — 2026-05-20

Fresh wave cost recorded so far: `$3.0212`.

## Gemma 3

- Model choice: `google/gemma-3-12b-pt` (12B primary)
- Read: `drowning_floor_non_discriminative`
- Interpretation: Vanilla is non-discriminative on this slice, so the family remains an observational boundary rather than a valid five-method comparison surface.

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost | Mean evicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | full context | prefix + recent | 0.00 | [0.0000, 0.0714] | current_n50_override | 342.23s | $0.2375 | 0.00% |

## Qwen 2.5 7B

- Model choice: `Qwen/Qwen2.5-7B-Instruct` (7B primary)
- Read: `gemma_like_channel_persistence`
- Interpretation: Sentence-TOVA-style stays close to token TOVA-style at the 1024 gate.

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost | Mean evicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | full context | prefix + recent | 0.04 | [0.0110, 0.1346] | current_n50_override | 181.80s | $0.1262 | 0.00% |
| Token vorn | 1024 | prefix + recent | 0.56 | [0.4231, 0.6884] | current_n50_override | 307.05s | $0.2131 | 73.53% |
| Sentence vorn | 1024 | prefix + recent | 0.96 | [0.8654, 0.9890] | current_n50_override | 256.54s | $0.1780 | 73.85% |
| TOVA-style | 1024 | prefix + recent | 0.36 | [0.2414, 0.4986] | current_n50_override | 577.11s | $0.4005 | 73.53% |
| Sentence TOVA-style | 1024 | prefix + recent | 0.08 | [0.0315, 0.1884] | current_n50_override | 368.12s | $0.2555 | 73.72% |
| Token vorn | 512 | prefix + recent | 0.06 | [0.0206, 0.1622] | current_n50_override | 272.94s | $0.1894 | 86.77% |
| Sentence vorn | 512 | prefix + recent | 0.96 | [0.8654, 0.9890] | current_n50_override | 199.37s | $0.1384 | 87.05% |
| TOVA-style | 512 | prefix + recent | 0.10 | [0.0435, 0.2136] | current_n50_override | 329.44s | $0.2286 | 86.77% |
| Sentence TOVA-style | 512 | prefix + recent | 0.26 | [0.1587, 0.3955] | current_n50_override | 222.62s | $0.1545 | 86.97% |
| Token vorn | 256 | prefix + recent | 0.00 | [0.0000, 0.0714] | current_n50_override | 201.74s | $0.1400 | 93.38% |
| Sentence vorn | 256 | prefix + recent | 0.86 | [0.7381, 0.9305] | current_n50_override | 200.56s | $0.1392 | 93.72% |
| TOVA-style | 256 | prefix + recent | 0.10 | [0.0435, 0.2136] | current_n50_override | 256.62s | $0.1781 | 93.38% |
| Sentence TOVA-style | 256 | prefix + recent | 0.32 | [0.2076, 0.4581] | current_n50_override | 220.47s | $0.1530 | 93.58% |

Pairwise tests:
- sentence_tova_1024_guarded vs tova_1024_guarded, exact McNemar on `[[2, 2], [16, 30]]`: `p = 0.00131226`
- sentence_tova_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[4, 0], [44, 2]]`: `p = 1.13687e-13`
- sentence_1024_guarded vs tova_1024_guarded, exact McNemar on `[[17, 31], [1, 1]]`: `p = 1.53668e-08`

## Qwen 3 30B-A3B

- Model choice: `Qwen/Qwen3-30B-A3B` (30B-A3B primary)
- Read: `drowning_floor_non_discriminative`
- Interpretation: Vanilla is non-discriminative on this slice, so the family remains an observational boundary rather than a valid five-method comparison surface.

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost | Mean evicted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla | full context | prefix + recent | 0.00 | [0.0000, 0.0714] | current_n50_override | 416.73s | $0.2892 | 0.00% |
