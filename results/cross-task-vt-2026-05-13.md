# Cross-Task — vt_4k — 2026-05-13

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `vt_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Comparator: vanilla full-context vs `vorn@1024`
- Scoring note: exact-match against any acceptable output in `outputs[]`

| Method | Budget regime | Accuracy | Wall-clock | Inference cost | Preprocessing | Retention |
|--------|---------------|----------|------------|----------------|---------------|-----------|
| Vanilla | Full context | 0.00 | 144.61s | $0.1004 | 0.00s / $0.0000 | 100% |
| Vorn eviction | 1024 token positions | 0.00 | 367.00s | $0.2547 | 0.00s / $0.0000 | 25.42% |

Read:
- This task does not currently discriminate the mechanism because the base model is already at `0.00` under full context.
- `vorn@1024` does not recover anything on `vt_4k`, but that is not evidence against the mechanism specifically; it is evidence that this task/model pairing is dead before compression.
