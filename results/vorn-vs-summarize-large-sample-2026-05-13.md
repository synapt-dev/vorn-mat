# Vorn vs Summarize at 1024 Budget — Large Sample

Context ceiling, not a competing row:

| Method | Budget regime | Hit rate | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|---------------|----------|------------|----------------|---------------|------------|
| Vanilla | Full context, unconstrained | 0.28 | 160.23s | $0.1112 | 0.00s / $0.0000 | 0% |

Large-sample constrained comparison on the same slice:

| Method | Cases | Hits | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|-------|------|----------|---------------|------------|----------------|---------------|------------|
| Vorn eviction @ 1024 | 200 | 50 | 0.25 | [0.1951, 0.3143] | 1029.17s | $0.7142 | 0.00s / $0.0000 | 75.54% |
| Summarize compact @ 1024 | 200 | 32 | 0.16 | [0.1157, 0.2171] | 1380.48s | $0.9581 | 1049.54s / $0.7284 | 94.52% |

Significance:

- Fisher exact test on `[[50, 150], [32, 168]]`: `p = 0.0348`
- Odds ratio: `1.75`
- Absolute hit-rate delta: `+0.09` in favor of vorn

Run conditions:

- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Slice: `niah_multikey_1_4k`, `validation[:200]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Random seed: `17`
- Vorn summary contract: `canonical_hidden_state_float32_per_token_from_layer_L_star`
- Summarize contract: `whole_context_summary_without_question_then_answer_with_question`

Read:

- The headline `vorn@1024` vs `summarize@1024` comparison clears `alpha = 0.05` at `n = 200`. The earlier `n = 50` ambiguity is resolved without needing to spend up to `n = 300`.
- The direction of the result is stable with the earlier sweep: vorn still beats summarize at the `1024` budget. The larger sample tightened the estimates and moved summarize upward from `0.12` to `0.16`, but not enough to close the gap.
- The product-shape difference remains large. Vorn reaches higher hit rate with zero preprocessing, while summarize pays a full extra summarize pass that dominates both latency and dollar cost on this slice.
- The methods are not matched on realized retention ratio. Summarize compresses more aggressively by design, while vorn preserves more live prompt positions. That means this artifact supports the paper claim `vorn beats summarize at the 1024 constrained-budget operating point on this slice`, not the stronger claim `vorn dominates every compression regime`.
