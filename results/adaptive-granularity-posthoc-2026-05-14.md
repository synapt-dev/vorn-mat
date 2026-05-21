# Adaptive Granularity Post-Hoc Analysis — 2026-05-14

This artifact is descriptive and post-hoc. It adds no new Modal runs and no new inferential claims. It reuses the already-published paired-result artifacts for Mistral and the raw cross-model reports for Qwen to ask a narrower question: if an adaptive selector could choose the best granularity/unit per regime after the fact, what would it have chosen?

Selection policy:
- Primary criterion: highest hit rate within the regime
- Tie-breakers: lower wall-clock, then lower inference cost, then guardrail-free variant when other metrics tie
- All-zero regimes remain unselected (`none`) rather than inventing a winner

Source surfaces:
- [sentence-level-eviction-4k-budget-sweep-2026-05-13.md](sentence-level-eviction-4k-budget-sweep-2026-05-13.md)
- [sentence-level-eviction-8k-budget-sweep-2026-05-13.md](sentence-level-eviction-8k-budget-sweep-2026-05-13.md)
- raw Qwen cross-model reports under the benchmark cache directory

## Common-Budget Selector Map

| Model | Context | Budget | Ceiling | Token G | Sentence G | Sentence NG | Word G | Word NG | Selected | Selected hit | Status |
|------|---------|--------|---------|---------|------------|-------------|--------|---------|----------|--------------|--------|
| mistralai/Mistral-7B-Instruct-v0.3 | 4k | 512 | 0.28 | 0.20 | 0.26 | 0.02 | n/a | n/a | sentence:G | 0.26 | unique_max |
| mistralai/Mistral-7B-Instruct-v0.3 | 4k | 1024 | 0.28 | 0.26 | 0.68 | 0.68 | n/a | n/a | sentence:G | 0.68 | tie_broken_by_efficiency |
| mistralai/Mistral-7B-Instruct-v0.3 | 4k | 1536 | 0.28 | 0.16 | 0.54 | 0.54 | n/a | n/a | sentence:G | 0.54 | tie_broken_by_efficiency |
| mistralai/Mistral-7B-Instruct-v0.3 | 4k | 2048 | 0.28 | 0.16 | 0.36 | 0.36 | n/a | n/a | sentence:NG | 0.36 | tie_broken_by_efficiency |
| mistralai/Mistral-7B-Instruct-v0.3 | 8k | 512 | 0.42 | 0.38 | 0.18 | 0.00 | n/a | n/a | token:G | 0.38 | unique_max |
| mistralai/Mistral-7B-Instruct-v0.3 | 8k | 1024 | 0.42 | 0.38 | 0.62 | 0.62 | n/a | n/a | sentence:G | 0.62 | tie_broken_by_efficiency |
| mistralai/Mistral-7B-Instruct-v0.3 | 8k | 1536 | 0.42 | 0.52 | 0.74 | 0.74 | n/a | n/a | sentence:NG | 0.74 | tie_broken_by_efficiency |
| mistralai/Mistral-7B-Instruct-v0.3 | 8k | 2048 | 0.42 | 0.68 | 0.62 | 0.62 | n/a | n/a | token:G | 0.68 | unique_max |
| Qwen/Qwen2.5-7B-Instruct | 4k | 512 | 0.00 | 0.02 | 0.00 | 0.04 | 0.00 | 0.00 | sentence:NG | 0.04 | unique_max |
| Qwen/Qwen2.5-7B-Instruct | 4k | 1024 | 0.00 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | token:G | 0.04 | unique_max |
| Qwen/Qwen2.5-7B-Instruct | 4k | 1536 | 0.00 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | token:G | 0.08 | unique_max |
| Qwen/Qwen2.5-7B-Instruct | 4k | 2048 | 0.00 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | token:G | 0.02 | unique_max |

## Model-Specific Appendix

| Model | Context | Budget | Ceiling | Token G | Sentence G | Sentence NG | Word G | Word NG | Selected | Selected hit | Status |
|------|---------|--------|---------|---------|------------|-------------|--------|---------|----------|--------------|--------|
| Qwen/Qwen2.5-7B-Instruct | 4k | 256 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | none | 0.00 | no_positive_recovery |

## Read

- Across the 12 common-budget regimes, the post-hoc selector chooses sentence-level in 7 regimes, token-level in 5 regimes, word-level in 0 regimes, and leaves 0 regime unselected.
- Mistral 4k is sentence-dominant across the shared budget band. The peak stays at `1024`, which matches the earlier 4k sweet-spot result.
- Mistral 8k is regime-shaped rather than monotonic: token wins at the edge budgets (`512`, `2048`), while sentence wins in the mid-budget band (`1024`, `1536`).
- Qwen 4k does not reproduce the Mistral sentence law. Its only positive recovery band is token-level, peaking at `1536`, while word-level never wins and the `256` appendix is an all-zero regime.
- Word-level never wins in any observed regime. That keeps the cross-model claim narrow: the architectural need is adaptive unit selection, not a universal move upward from token to larger semantic units.
- The selector result should be read as evidence for where a forward adaptive policy is worth implementing, not as proof that an ex post oracle is deployable without additional online selection logic.
