# Token No-Guardrails Comparison — 2026-05-13

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Wall-clock | Inference cost | KV savings |
|--------|--------|------------|----------|---------------|------------|----------------|-----------|
| Vorn | 1024 | none | 0.00 | [0.0000, 0.0714] | 307.48s | $0.2134 | 75.53% |
| TOVA | 1024 | none | 0.16 | [0.0834, 0.2851] | 402.54s | $0.2794 | 75.53% |

Paired same-slice inference:

- TOVA no-guardrails vs vorn no-guardrails, exact McNemar on `[[0, 8], [0, 42]]`: `p = 0.0078125`

Read:

- The honest boundary stays the same: token-level vorn collapses without guardrails on this slice, while token-level TOVA retains some middle-only signal.
- The inferential row is now a paired exact test over the same 50 fixtures, not a Fisher exact test over aggregates.
