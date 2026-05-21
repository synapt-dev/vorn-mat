# Vorn Seed Reproducibility — 2026-05-13

This is a determinism check, not a stochastic-ablation check. In the current
`vorn` implementation, `random_seed` does not affect token retention or model
generation. Varying the seed should therefore reproduce the same hit-rate curve
unless there is hidden runtime nondeterminism.

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Slice: `niah_multikey_1_4k`, `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Retention policy: `vorn`

## Results

| Budget | Seed 17 hit rate | Seed 42 hit rate | Seed 100 hit rate | Stable? |
|--------|------------------|------------------|-------------------|---------|
| 256 | 0.04 | 0.04 | 0.04 | yes |
| 512 | 0.18 | 0.18 | 0.18 | yes |
| 1024 | 0.26 | 0.26 | 0.26 | yes |
| 2048 | 0.20 | 0.20 | 0.20 | yes |

Latency and cost varied across Modal containers, but the retrieval metric did
not:

| Budget | Seed | Wall-clock | Inference cost |
|--------|------|------------|----------------|
| 256 | 17 | 202.27s | $0.1404 |
| 256 | 42 | 201.44s | $0.1398 |
| 256 | 100 | 217.53s | $0.1510 |
| 512 | 17 | 286.13s | $0.1986 |
| 512 | 42 | 306.29s | $0.2126 |
| 512 | 100 | 234.50s | $0.1627 |
| 1024 | 17 | 379.60s | $0.2634 |
| 1024 | 42 | 406.35s | $0.2820 |
| 1024 | 100 | 337.60s | $0.2343 |
| 2048 | 17 | 597.56s | $0.4147 |
| 2048 | 42 | 570.08s | $0.3956 |
| 2048 | 100 | 631.99s | $0.4386 |

## Read

- The observed `vorn` hit-rate curve is stable across seeds on this slice.
- That means the current headline curve (`0.04 / 0.18 / 0.26 / 0.20`) is not a
  `seed=17` artifact.
- The meaningful remaining uncertainty is not seed variance; it is benchmark
  transfer and larger-sample stability across other tasks and slices.
