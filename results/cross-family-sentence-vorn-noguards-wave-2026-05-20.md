# Cross-Family Sentence-Vorn No-Guards Wave — 2026-05-20

Fresh wave cost recorded so far: `$1.7042`.

## Mistral 7B v0.3

- Read: `guardrail_robust`
- Interpretation: Sentence-vorn stays in the same regime with or without prefix/recent guardrails.
- Delta vs guarded sentence-vorn: `+0.00`

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla reference | 512 | prefix + recent | 1.00 | [0.9286, 1.0000] | current_n50_override | 191.87s | $0.1332 |
| Sentence vorn guarded | 512 | prefix + recent | 1.00 | [0.9286, 1.0000] | current_n50_override | 188.38s | $0.1307 |
| Sentence vorn no-guards | 512 | none | 1.00 | [0.9286, 1.0000] | current_n50_override | 200.26s | $0.1390 |

Pairwise tests:
- sentence_512_noguards vs sentence_512_guarded, exact McNemar on `[[50, 0], [0, 0]]`: `p = 1`

## Ministral 8B

- Read: `guardrail_robust`
- Interpretation: Sentence-vorn stays in the same regime with or without prefix/recent guardrails.
- Delta vs guarded sentence-vorn: `+0.00`

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla reference | 1024 | prefix + recent | 1.00 | [0.9286, 1.0000] | historical_stale_reference | 198.71s | $0.1379 |
| Sentence vorn guarded | 1024 | prefix + recent | 1.00 | [0.9286, 1.0000] | historical_stale_reference | 360.56s | $0.2502 |
| Sentence vorn no-guards | 1024 | none | 1.00 | [0.9286, 1.0000] | current_n50_override | 351.14s | $0.2437 |

Pairwise tests:
- sentence_1024_noguards vs sentence_1024_guarded, exact McNemar on `[[50, 0], [0, 0]]`: `p = 1`

## Gemma 2 9B

- Read: `guardrail_robust`
- Interpretation: Sentence-vorn stays in the same regime with or without prefix/recent guardrails.
- Delta vs guarded sentence-vorn: `+0.00`

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla reference | 1024 | prefix + recent | 1.00 | [0.9286, 1.0000] | historical_stale_reference | 262.90s | $0.1825 |
| Sentence vorn guarded | 1024 | prefix + recent | 0.86 | [0.7381, 0.9305] | historical_stale_reference | 524.78s | $0.3642 |
| Sentence vorn no-guards | 1024 | none | 0.86 | [0.7381, 0.9305] | current_n50_override | 457.14s | $0.3173 |

Pairwise tests:
- sentence_1024_noguards vs sentence_1024_guarded, exact McNemar on `[[43, 0], [0, 7]]`: `p = 1`

## Qwen 3 8B

- Read: `observational_floor`
- Interpretation: Vanilla is non-discriminative on this slice, so the no-guards row is observational rather than a valid guarded-vs-no-guards contrast.
- Delta vs guarded sentence-vorn: `+0.00`

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla reference | 1024 | prefix + recent | 0.00 | [0.0000, 0.0714] | historical_stale_reference | 245.54s | $0.1704 |
| Sentence vorn guarded | 1024 | prefix + recent | 0.00 | [0.0000, 0.0714] | historical_stale_reference | 2782.82s | $1.9313 |
| Sentence vorn no-guards | 1024 | none | 0.00 | [0.0000, 0.0714] | current_n50_override | 413.30s | $0.2868 |

Pairwise tests:
- sentence_1024_noguards vs sentence_1024_guarded, exact McNemar on `[[0, 0], [0, 50]]`: `p = 1`

## Qwen 2.5 7B

- Read: `guardrail_robust`
- Interpretation: Sentence-vorn stays in the same regime with or without prefix/recent guardrails.
- Delta vs guarded sentence-vorn: `+0.00`

| Method | Budget | Guardrails | Hit rate | 95% Wilson CI | Ceiling status | Wall-clock | Inference cost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla reference | 1024 | prefix + recent | 0.04 | [0.0110, 0.1346] | current_n50_override | 181.80s | $0.1262 |
| Sentence vorn guarded | 1024 | prefix + recent | 0.96 | [0.8654, 0.9890] | current_n50_override | 256.54s | $0.1780 |
| Sentence vorn no-guards | 1024 | none | 0.96 | [0.8654, 0.9890] | current_n50_override | 215.21s | $0.1494 |

Pairwise tests:
- sentence_1024_noguards vs sentence_1024_guarded, exact McNemar on `[[48, 0], [0, 2]]`: `p = 1`
