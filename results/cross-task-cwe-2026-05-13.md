# Cross-Task — cwe_4k — 2026-05-13

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `cwe_4k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Comparator: vanilla full-context vs `vorn@1024`
- Scoring note: exact-match against any acceptable output in `outputs[]`

| Method | Budget regime | Accuracy | Wall-clock | Inference cost | Preprocessing | Retention |
|--------|---------------|----------|------------|----------------|---------------|-----------|
| Vanilla | Full context | 0.00 | 181.86s | $0.1262 | 0.00s / $0.0000 | 100% |
| Vorn eviction | 1024 token positions | 0.00 | 456.02s | $0.3165 | 0.00s / $0.0000 | 18.49% |

Read:
- This task also fails to discriminate the mechanism because the base model is already at `0.00` under full context.
- Like `vt_4k`, `cwe_4k` currently tells us more about the model/task pairing than about the quality of the compression policy.
