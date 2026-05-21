# Cross-Task — qa_2_4k — 2026-05-13

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `qa_2_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Comparator: vanilla full-context vs `vorn@1024`
- Scoring note: exact-match against any acceptable output in `outputs[]`

| Method | Budget regime | Accuracy | Wall-clock | Inference cost | Preprocessing | Retention |
|--------|---------------|----------|------------|----------------|---------------|-----------|
| Vanilla | Full context | 0.48 | 136.57s | $0.0948 | 0.00s / $0.0000 | 100% |
| Vorn eviction | 1024 token positions | 0.32 | 335.06s | $0.2325 | 0.00s / $0.0000 | 26.73% |

Read:
- This is the strongest cross-task transfer result from the first Tier 1 batch.
- `vorn@1024` preserves two-thirds of the vanilla task accuracy (`0.32` vs `0.48`) while retaining only `26.73%` of prompt positions.
- That is not as strong as the NIAH sweet-spot result, but it is real transfer outside the needle-in-haystack shape.
