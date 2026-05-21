# Vorn vs TOVA-style vs H2O-style at 1024 Budget — Large Sample

Context ceiling, not a competing row:

| Method | Budget regime | Hit rate | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|---------------|----------|------------|----------------|---------------|------------|
| Vanilla | Full context, unconstrained | 0.28 | 160.23s | $0.1112 | 0.00s / $0.0000 | 0% |

Large-sample constrained comparison on the same slice:

| Method | Cases | Hits | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|-------|------|----------|---------------|------------|----------------|---------------|------------|
| Vorn eviction @ 1024 | 200 | 50 | 0.25 | [0.1951, 0.3143] | 1029.17s | $0.7142 | 0.00s / $0.0000 | 75.54% |
| TOVA-style @ 1024 | 200 | 47 | 0.235 | [0.1816, 0.2984] | 1324.87s | $0.9195 | 0.00s / $0.0000 | 75.54% |
| H2O-style @ 1024 | 200 | 46 | 0.23 | [0.1771, 0.2931] | 1359.50s | $0.9435 | 0.00s / $0.0000 | 75.54% |

Pairwise significance against the existing vorn row:

- Vorn vs TOVA-style, Fisher exact on `[[50, 150], [47, 153]]`: `p = 0.8156`, odds ratio `1.0851`
- Vorn vs H2O-style, Fisher exact on `[[50, 150], [46, 154]]`: `p = 0.7255`, odds ratio `1.1159`

Run conditions:

- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Slice: `niah_multikey_1_4k`, `validation[:200]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Random seed: `17`
- Cache budget: `1024`
- Vorn contract: `canonical_hidden_state_float32_per_token_from_layer_L_star`
- TOVA-style contract: `last_token_attention_mean_over_final_layer_heads`
- H2O-style contract: `accumulated_last_token_attention_mean_over_final_layer_heads`

Approximation boundary:

- These are **TOVA-style** and **H2O-style** controls implemented inside the same vorn-MAT live runner, not the original author codebases.
- The TOVA-style arm ranks retained token positions by the final layer's most-recent-token attention weights averaged across heads.
- The H2O-style arm accumulates that same scalar attention signal over time, then evicts by the running heavy-hitter mass while preserving the same prefix and recent-window guardrails as the other controls.
- That makes these rows valid for `does residual-space cosine beat simple current-attention or accumulated-attention controls in this exact runner?`, but not for `does vorn beat the full reference implementations on every engineering detail?`

Read:

- The current data does **not** support the strong claim `residual-space cosine beats softmax-attention scoring` at this operating point. Vorn is numerically highest at `0.25`, but the gaps to the TOVA-style (`0.235`) and H2O-style (`0.23`) controls are small and not statistically significant at `n = 200`.
- That means the literature-grounded novelty claim must stay architectural rather than superiority-based for now. We can say the vorn mechanism is a distinct residual-space, rotary-safe, deterministic variant inside the current-direction-conditioned family. We cannot yet say it empirically outperforms the nearest conceptual cousin on this slice.
- The cost/latency profile still mildly favors vorn over the two attention-weight controls in this runner, but the dominant result is non-separation, not a win.
- The stronger empirical headline from the existing matrix remains `vorn@1024` vs summarize compact at the same budget. That result cleared significance at `n = 200`. The TOVA/H2O ablation weakens any claim that the scoring family itself is already decisively better than neighboring attention-based policies.

## Determinism appendix

A follow-up determinism question asked whether the TOVA-style and H2O-style rows themselves have run-to-run variance under identical settings. If they varied materially across identical replays, the single-run Fisher comparisons against vorn would be under-specified.

Determinism replay conditions:

- same Modal profile: `author`
- same dataset slice: `niah_multikey_1_4k`, `validation[:200]`
- same model: `mistralai/Mistral-7B-Instruct-v0.3`
- same seed: `17`
- same cache budget: `1024`
- same implementation revision: `feat/vorn-mat-live-eviction-modal` with eager-attention forcing enabled

Replay results:

| Method | Original hit rate | Replay hit rate | Deterministic verdict |
|--------|-------------------|-----------------|-----------------------|
| TOVA-style @ 1024 | 0.235 | 0.235 | exact match |
| H2O-style @ 1024 | 0.23 | 0.23 | exact match |

Read:

- On this runner, these two attention-based controls are deterministic at the hit-rate level under identical replay conditions.
- That means the earlier pairwise Fisher results stand as written: `p = 0.8156` for vorn vs TOVA-style and `p = 0.7255` for vorn vs H2O-style.
- No multi-run variance estimate is needed for the current paper draft at this slice/budget, because the replay did not expose run-to-run output drift.
