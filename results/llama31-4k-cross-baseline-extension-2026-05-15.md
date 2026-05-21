# Llama 3.1 8B 4k Cross-Baseline Extension — 2026-05-15

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Budget: `1024`

## Rows

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|---------------|-----------|
| Vanilla | full context | prefix + recent | 1.00 | [0.9286, 1.0000] | 236.73s | $0.1643 | 0 | 0.00% |
| Token vorn | 1024 | prefix + recent | 1.00 | [0.9286, 1.0000] | 276.25s | $0.1917 | 0 | 73.36% |
| Sentence vorn | 1024 | prefix + recent | 1.00 | [0.9286, 1.0000] | 245.17s | $0.1702 | 0 | 73.62% |
| Sentence vorn | 1024 | none | 1.00 | [0.9286, 1.0000] | 217.60s | $0.1510 | 0 | 73.62% |
| TOVA | 1024 | prefix + recent | 0.90 | [0.7864, 0.9565] | 395.00s | $0.2741 | 0 | 73.36% |
| H2O | 1024 | prefix + recent | 0.94 | [0.8378, 0.9794] | 308.58s | $0.2142 | 0 | 73.36% |

## Paired Tests

- tova_1024_guarded vs token_1024_guarded, exact McNemar on `[[45, 0], [5, 0]]`: `p = 0.0625`
- h2o_1024_guarded vs token_1024_guarded, exact McNemar on `[[47, 0], [3, 0]]`: `p = 0.25`
- tova_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[45, 0], [5, 0]]`: `p = 0.0625`
- h2o_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[47, 0], [3, 0]]`: `p = 0.25`
- tova_1024_guarded vs h2o_1024_guarded, exact McNemar on `[[45, 0], [2, 3]]`: `p = 0.5`
- tova_1024_guarded vs vanilla_floor, exact McNemar on `[[45, 0], [5, 0]]`: `p = 0.0625`
- h2o_1024_guarded vs vanilla_floor, exact McNemar on `[[47, 0], [3, 0]]`: `p = 0.25`

## Read

- Llama 3.1 does **not** stay in the all-methods threshold class once the attention-weight baselines are included. At `1024`, the vorn rows remain at `1.00`, while `TOVA = 0.90` and `H2O = 0.94`.
- Unlike Qwen 3, Llama emits direct numeric or short declarative answers inside the existing `32`-token budget, so this extension remains a clean competence-comparison surface rather than a runner-surface mismatch case.
- The paper consequence is structural rather than headline-competitive: Llama is not just "Gemma with higher scores." Gemma preserves the ceiling under attention-weight baselines while residual-direction baselines collapse; Llama preserves the ceiling under residual-direction baselines while attention-weight baselines lose a small amount of quality.
- The paired rows are suggestive rather than decisive at `n = 50` (`TOVA` vs token/sentence `p = 0.0625`; `H2O` vs token/sentence `p = 0.25`). The honest cross-family claim is therefore about **effect-class shape**, not about a settled superiority result for any one baseline family on Llama.
