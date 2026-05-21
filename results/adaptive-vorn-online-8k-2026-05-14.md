# Adaptive Vorn Online 8k Comparison — 2026-05-14

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_8k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Adaptive selector: `choose_token_or_sentence_by_peak_zscore_over_current_alignment_scores`

| Budget | Adaptive hit | Token ref | Sentence ref | Adaptive token steps | Adaptive sentence steps | Wall-clock | Cost |
|--------|--------------|-----------|--------------|----------------------|-------------------------|------------|------|
| 512 | 0.38 | 0.38 | 0.18 | 380 | 186 | 363.84s | $0.2525 |
| 1024 | 0.40 | 0.38 | 0.62 | 408 | 155 | 556.93s | $0.3865 |
| 1536 | 0.50 | 0.52 | 0.74 | 365 | 115 | 509.53s | $0.3536 |
| 2048 | 0.66 | 0.68 | 0.62 | 408 | 62 | 568.95s | $0.3949 |

## Paired Tests

- adaptive_512_guarded vs token_512_guarded, exact McNemar on `[[19, 0], [0, 31]]`: `p = 1`
- adaptive_512_guarded vs sentence_512_guarded, exact McNemar on `[[7, 12], [2, 29]]`: `p = 0.0129395`
- adaptive_1024_guarded vs token_1024_guarded, exact McNemar on `[[19, 1], [0, 30]]`: `p = 1`
- adaptive_1024_guarded vs sentence_1024_guarded, exact McNemar on `[[17, 3], [14, 16]]`: `p = 0.0127258`
- adaptive_1536_guarded vs token_1536_guarded, exact McNemar on `[[25, 0], [1, 24]]`: `p = 1`
- adaptive_1536_guarded vs sentence_1536_guarded, exact McNemar on `[[22, 3], [15, 10]]`: `p = 0.00753784`
- adaptive_2048_guarded vs token_2048_guarded, exact McNemar on `[[33, 0], [1, 16]]`: `p = 1`
- adaptive_2048_guarded vs sentence_2048_guarded, exact McNemar on `[[24, 9], [7, 10]]`: `p = 0.803619`
