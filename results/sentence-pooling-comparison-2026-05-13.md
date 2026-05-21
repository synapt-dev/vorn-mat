# Sentence Pooling Comparison — 2026-05-13

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Budget: `1024`
- Retention policy: `sentence_vorn`
- Comparison axis: sentence aggregation only (`max` vs `mean` vs `top-k`)

| Pooling | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Retention |
|---------|------------|----------|---------------|------------|----------------|-----------|
| Max | prefix + recent | 0.68 | [0.5419, 0.7924] | 221.52s | $0.1537 | 24.13% |
| Max | none | 0.68 | [0.5419, 0.7924] | 274.61s | $0.1906 | 24.13% |
| Mean | prefix + recent | 0.42 | [0.2937, 0.5577] | 276.89s | $0.1922 | 24.17% |
| Mean | none | 0.42 | [0.2937, 0.5577] | 279.81s | $0.1942 | 24.17% |
| Top-k mean (`k=3`) | prefix + recent | 0.68 | [0.5419, 0.7924] | 249.78s | $0.1734 | 24.17% |
| Top-k mean (`k=3`) | none | 0.68 | [0.5419, 0.7924] | 252.72s | $0.1754 | 24.17% |

Paired same-slice inference:

- max_guarded vs mean_guarded, exact McNemar on `[[19, 15], [2, 14]]`: `p = 0.00234985`
- topk_guarded vs mean_guarded, exact McNemar on `[[20, 14], [1, 15]]`: `p = 0.000976562`
- max_noguards vs mean_noguards, exact McNemar on `[[19, 15], [2, 14]]`: `p = 0.00234985`
- topk_noguards vs mean_noguards, exact McNemar on `[[20, 14], [1, 15]]`: `p = 0.000976562`

Read:

- Pooling still matters materially. `mean` blurs the signal, while `max` and `top-k` preserve the sentence-level win.
- This artifact now preserves per-case outcomes, so the aggregation comparisons can be re-tested downstream without re-deriving fixture-level data.
