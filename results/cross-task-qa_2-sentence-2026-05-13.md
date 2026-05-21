# Cross-Task Sentence Generalization (`qa_2_4k`) — 2026-05-13

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `qa_2_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Metric: exact match against any acceptable output
- Pooling: `max`

| Method | Budget regime | Accuracy | 95% Wilson CI | Wall-clock | Inference cost | Preprocessing | Retention |
|--------|---------------|----------|---------------|------------|----------------|---------------|-----------|
| Vanilla | full context | 0.48 | [0.3480, 0.6149] | 181.74s | $0.1261 | 0 | 100.00% |
| Token vorn | 1024 guarded | 0.32 | [0.2076, 0.4581] | 314.56s | $0.2183 | 0 | 26.73% |
| Sentence vorn | 1024 guarded | 0.40 | [0.2761, 0.5382] | 241.15s | $0.1674 | 0 | 26.24% |
| Sentence vorn | 1024 no guardrails | 0.44 | [0.3116, 0.5769] | 206.38s | $0.1432 | 0 | 26.17% |

Paired same-slice inference:

- sentence_guarded_1024 vs token_vorn_1024, exact McNemar on `[[14, 6], [2, 28]]`: `p = 0.289062`
- sentence_noguards_1024 vs token_vorn_1024, exact McNemar on `[[15, 7], [1, 27]]`: `p = 0.0703125`
- sentence_guarded_1024 vs vanilla, exact McNemar on `[[19, 1], [5, 25]]`: `p = 0.21875`
- sentence_noguards_1024 vs vanilla, exact McNemar on `[[21, 1], [3, 25]]`: `p = 0.625`

Read:

- The directional generalization beyond NIAH remains: sentence-level vorn improves the point estimate over token-level vorn on `qa_2_4k` and approaches vanilla on this slice.
- The inferential layer is now paired-correct. At `n=50`, these rows are still best read as directional rather than decisive.
