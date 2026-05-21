# No-Guardrails Ablation — 2026-05-13

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|-----------|
| Vorn | 1024 | prefix + recent | 0.26 | [0.1587, 0.3955] | 486.10s | $0.3374 | 75.53% |
| Vorn | 1024 | none | 0.00 | [0.0000, 0.0714] | 307.48s | $0.2134 | 75.53% |

Paired same-slice inference:

- Guarded vs no-guardrails vorn, exact McNemar on `[[0, 13], [0, 37]]`: `p = 0.000244141`

Read:

- The qualitative read is unchanged and now formally supported with a paired test: guardrails are load-bearing for token-level vorn on this slice.
- This artifact now carries per-fixture observations for both rows, so the paired discordance table is reconstructible downstream.

References:

- [live-eviction-budget-sweep-2026-05-13.md](live-eviction-budget-sweep-2026-05-13.md)
- [vanilla-observation-neighborhood-2026-05-13.md](vanilla-observation-neighborhood-2026-05-13.md)
