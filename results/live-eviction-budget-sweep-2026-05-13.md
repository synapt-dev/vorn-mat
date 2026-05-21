# Live Eviction Budget Sweep — 2026-05-13

Context ceiling, not a competing row:

| Method | Budget regime | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|---------------|----------|---------------|------------|----------------|---------------|------------|
| Vanilla | Full context, unconstrained | 0.28 | [0.175, 0.417] | 160.23s | $0.1112 | 0.00s / $0.0000 | 0% |

Constrained-budget comparison rows on the same slice:

| Method | Budget | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|----------|---------------|------------|----------------|---------------|------------|
| Random eviction | 256 | 0.00 | [0.000, 0.071] | 237.06s | $0.1645 | 0.00s / $0.0000 | 93.88% |
| Sliding window | 256 | 0.00 | [0.000, 0.071] | 224.45s | $0.1558 | 0.00s / $0.0000 | 93.88% |
| Sliding window | 512 | 0.00 | [0.000, 0.071] | 271.09s | $0.1881 | 0.00s / $0.0000 | 87.77% |
| Sliding window | 1024 | 0.00 | [0.000, 0.071] | 359.69s | $0.2496 | 0.00s / $0.0000 | 75.53% |
| Sliding window | 2048 | 0.00 | [0.000, 0.071] | 451.53s | $0.3134 | 0.00s / $0.0000 | 51.06% |
| Prefix + suffix | 256 | 0.04 | [0.011, 0.135] | 227.79s | $0.1581 | 0.00s / $0.0000 | 93.88% |
| Prefix + suffix | 512 | 0.00 | [0.000, 0.071] | 285.41s | $0.1981 | 0.00s / $0.0000 | 87.77% |
| Prefix + suffix | 1024 | 0.06 | [0.021, 0.162] | 439.56s | $0.3051 | 0.00s / $0.0000 | 75.53% |
| Prefix + suffix | 2048 | 0.10 | [0.043, 0.214] | 527.17s | $0.3659 | 0.00s / $0.0000 | 51.06% |
| Summarize compact | 256 | 0.12 | [0.056, 0.238] | 425.12s | $0.2950 | 267.54s / $0.1857 | 94.54% |
| Summarize compact | 512 | 0.12 | [0.056, 0.238] | 596.90s | $0.4142 | 356.10s / $0.2471 | 94.45% |
| Summarize compact | 1024 | 0.12 | [0.056, 0.238] | 330.79s | $0.2296 | 177.45s / $0.1232 | 94.45% |
| Summarize compact | 2048 | 0.12 | [0.056, 0.238] | 605.38s | $0.4201 | 360.36s / $0.2501 | 94.45% |
| Vorn eviction | 256 | 0.04 | [0.011, 0.135] | 202.27s | $0.1404 | 0.00s / $0.0000 | 93.88% |
| Vorn eviction | 512 | 0.18 | [0.098, 0.308] | 286.13s | $0.1986 | 0.00s / $0.0000 | 87.77% |
| Vorn eviction | 1024 | 0.26 | [0.159, 0.396] | 379.60s | $0.2634 | 0.00s / $0.0000 | 75.53% |
| Vorn eviction | 2048 | 0.20 | [0.112, 0.330] | 597.56s | $0.4147 | 0.00s / $0.0000 | 51.06% |

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Slice: `niah_multikey_1_4k`, `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Live-eviction summary contract: `canonical_hidden_state_float32_per_token_from_layer_L_star`
- Summarize compact contract: `whole_context_summary_without_question_then_answer_with_question`
- Sample size per row: `50` cases

Read:
- The first constrained-budget signal is that vorn beats random at the same pathological `256` budget.
- The first production-realistic control is that vorn also beats a naive sliding-window truncation at the same `256` budget.
- Prefix+suffix at `256` matches `vorn@256` on this 50-case slice. That weakens a “vorn alone works” claim and points instead toward positional-bias analysis for this benchmark slice and heuristic family.
- Summarize compact is the best observed constrained method at the pathological `256` point (`0.12`), but it pays a large preprocessing tax before the answer pass (`177s` to `360s`, `$0.1232` to `$0.2501` on this slice).
- Summarize compact stays flat at `0.12` across `256/512/1024/2048` in this configuration. On this slice, adding more summary budget does not translate into better retrieval performance.
- The higher-budget control curves separate the mechanisms. Sliding-window stays flat at `0.00` across `256/512/1024/2048`. Prefix+suffix climbs from `0.00` at `512` to `0.06` at `1024` and `0.10` at `2048`, but it remains well below the vorn curve (`0.18/0.26/0.20`).
- That means the first clean differentiation claim is not uniqueness at `256`; it is curve advantage above `256`. Vorn scales with additional budget more effectively than the production-standard truncation policy, the positional prefix+suffix heuristic, and the two-pass summarize compact baseline on this slice.
- From `512` upward, vorn overtakes summarize while avoiding preprocessing overhead entirely: `0.18` vs `0.12` at `512`, `0.26` vs `0.12` at `1024`, and `0.20` vs `0.12` at `2048`.
- The best observed point on this slice is `vorn@1024`, which reaches `0.26` while retaining `24.47%` of prompt positions.
- `vorn@2048` regresses on this 50-case slice, so the budget-response curve is not yet stable enough to treat as monotonic.
- Future comparison rows should stay inside the constrained budget regime: sliding-window, prefix+suffix, and summarization at the same budget. Vanilla remains context only.

95% Wilson confidence intervals and pairwise significance:
- `vorn@1024 vs summarize@1024`: `13/50 (0.26, [0.159, 0.396])` vs `6/50 (0.12, [0.056, 0.238])` -> `p=0.1247` (not significant at `alpha=0.05`).
- `vorn@256 vs prefix+suffix@256`: `2/50 (0.04, [0.011, 0.135])` vs `2/50 (0.04, [0.011, 0.135])` -> `p=1.0000` (not significant at `alpha=0.05`).
- `vorn@2048 vs vorn@1024`: `10/50 (0.20, [0.112, 0.330])` vs `13/50 (0.26, [0.159, 0.396])` -> `p=0.6353` (not significant at `alpha=0.05`).

Claims that survive significance testing on this 50-case slice:
- `vorn@512 vs sliding-window@512` -> `p=0.0026` (significant at `alpha=0.05`).
- `vorn@1024 vs sliding-window@1024` -> `p=0.0001` (significant at `alpha=0.05`).
- `vorn@1024 vs prefix+suffix@1024` -> `p=0.0122` (significant at `alpha=0.05`).

External-claim guidance:
- Safe to claim: vorn separates from the zero-hit sliding-window controls above `256`, and `vorn@1024` beats `prefix+suffix@1024` on this slice.
- Directional only, not publication-safe without caveat: `vorn@1024` over `summarize@1024`, `vorn@2048` dipping below `vorn@1024`, and any claim that `vorn@256` is uniquely better than the simpler `prefix+suffix@256` heuristic.
- Do not publish a monotonic budget-scaling claim from this matrix alone; the Wilson intervals overlap heavily and the `1024 -> 2048` dip is not significant at this sample size.
