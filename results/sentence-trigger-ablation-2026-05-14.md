# Sentence Trigger Ablation — 2026-05-14

Run conditions:
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_8k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Retention policy: `sentence_vorn`
- Budget: `1536`
- Pooling: `max`

| Variant | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | KV savings |
|--------|----------|---------------|------------|----------------|-----------|
| Threshold trigger | 0.74 | [0.6045, 0.8413] | 373.62s | $0.2593 | 81.39% |
| Sentence-boundary trigger | 0.72 | [0.5833, 0.8253] | 378.75s | $0.2629 | 81.35% |

## Paired test

- sentence_boundary_trigger vs threshold_trigger, exact McNemar on `[[36, 0], [1, 13]]`: `p = 1`

## Read

- This run does not support a decisive quality difference at n=50.
- The higher point estimate in this pair is `sentence_1536_threshold_trigger` at `0.74`.
- `observations[]` is preserved in the JSON artifact, so the paired claim is reconstructible downstream.
