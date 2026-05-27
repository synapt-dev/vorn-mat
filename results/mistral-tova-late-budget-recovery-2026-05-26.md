# Mistral late-budget TOVA recovery (2026-05-26)

Recovered/composed late-budget Mistral TOVA cells for Figure 3 integrity.

| Method | Budget | Status | Hits | Hit rate | Notes |
|---|---:|---|---:|---:|---|
| `tova` | 1536 | `completed_recovered` | 49/50 | 0.98 | 1 OOM microchunks; single-case fills completed |
| `sentence_tova` | 1536 | `completed_recovered` | 49/50 | 0.98 | 3 OOM microchunks; single-case fills completed |
| `tova` | 2048 | `completed` | 49/50 | 0.98 | single-shot from 2026-05-25 |
| `sentence_tova` | 2048 | `completed_recovered` | 49/50 | 0.98 | 3 OOM microchunks; single-case fills completed |

Figure 3 values from this artifact:

- `mistral_token_tova = [0.04, 0.30, 0.86, 0.98, 0.98]`
- `mistral_sentence_tova = [0.74, 0.96, 1.00, 0.98, 0.98]`

Premium boundary: pure OSS benchmark data artifact.
