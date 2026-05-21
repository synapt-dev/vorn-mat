# Score Distribution Observation — 8k — 2026-05-14

Run conditions:
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_8k`
- Slice: `validation[:50]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Vorn measurement: canonical layer `L*=16`, recent window `W=16`
- Word/sentence aggregation: `max`
- Observation mode: score-shape metrics logged before each live eviction decision; metrics did not drive selection

## Initial findings

- Peak-zscore does not track the oracle winner. Token or word has the highest mean peak-zscore at every budget, including the sentence-oracle mid-band.
- Sentence-oracle budgets `1024` and `1536` are the regimes where sentence-level mass concentration and tail spread separate most strongly from token-level. The token-edge budgets `512` and `2048` show materially smaller sentence-vs-token gaps on those same metrics.
- Word-level score shape shadows token-level score shape throughout this sweep and does not surface a distinct winning regime.

## Oracle Runs

| Budget | Oracle granularity | Retention policy | Hit rate | Wall-clock | Cost |
|--------|--------------------|------------------|----------|------------|------|
| 512 | token | vorn | 0.38 | 620.58s | $0.4307 |
| 1024 | sentence | sentence_vorn | 0.62 | 465.57s | $0.3231 |
| 1536 | sentence | sentence_vorn | 0.74 | 471.55s | $0.3273 |
| 2048 | token | vorn | 0.68 | 836.69s | $0.5807 |

## Budget 512 (token oracle)

| Granularity | Step count | Mean positions | Mean peak-z | Mean top10 mass | Mean entropy(norm) | Mean KL-from-uniform | Mean q90-q50 | Mean above-threshold frac | Mean spatial coherence |
|-------------|------------|----------------|-------------|-----------------|--------------------|----------------------|--------------|---------------------------|------------------------|
| token | 619 | 1131.69 | 5.209 | 0.029 | 0.999 | 0.008 | 0.148 | 0.136 | 0.677 |
| word | 619 | 1015.97 | 5.135 | 0.033 | 0.999 | 0.008 | 0.147 | 0.124 | 0.699 |
| sentence | 619 | 164.61 | 4.673 | 0.096 | 0.998 | 0.010 | 0.176 | 0.124 | 0.635 |

## Budget 1024 (sentence oracle)

| Granularity | Step count | Mean positions | Mean peak-z | Mean top10 mass | Mean entropy(norm) | Mean KL-from-uniform | Mean q90-q50 | Mean above-threshold frac | Mean spatial coherence |
|-------------|------------|----------------|-------------|-----------------|--------------------|----------------------|--------------|---------------------------|------------------------|
| token | 71 | 6058.38 | 10.771 | 0.007 | 1.000 | 0.003 | 0.083 | 0.115 | 0.645 |
| word | 71 | 5535.00 | 10.833 | 0.007 | 1.000 | 0.003 | 0.078 | 0.110 | 0.693 |
| sentence | 71 | 312.44 | 6.698 | 0.099 | 0.999 | 0.006 | 0.131 | 0.119 | 0.266 |

## Budget 1536 (sentence oracle)

| Granularity | Step count | Mean positions | Mean peak-z | Mean top10 mass | Mean entropy(norm) | Mean KL-from-uniform | Mean q90-q50 | Mean above-threshold frac | Mean spatial coherence |
|-------------|------------|----------------|-------------|-----------------|--------------------|----------------------|--------------|---------------------------|------------------------|
| token | 71 | 6209.82 | 11.043 | 0.005 | 1.000 | 0.003 | 0.069 | 0.104 | 0.642 |
| word | 71 | 5669.92 | 11.106 | 0.006 | 1.000 | 0.003 | 0.066 | 0.101 | 0.689 |
| sentence | 71 | 319.85 | 6.731 | 0.078 | 0.999 | 0.006 | 0.139 | 0.130 | 0.285 |

## Budget 2048 (token oracle)

| Granularity | Step count | Mean positions | Mean peak-z | Mean top10 mass | Mean entropy(norm) | Mean KL-from-uniform | Mean q90-q50 | Mean above-threshold frac | Mean spatial coherence |
|-------------|------------|----------------|-------------|-----------------|--------------------|----------------------|--------------|---------------------------|------------------------|
| token | 488 | 2676.40 | 8.198 | 0.008 | 1.000 | 0.003 | 0.070 | 0.093 | 0.657 |
| word | 488 | 2417.66 | 8.313 | 0.009 | 1.000 | 0.003 | 0.066 | 0.086 | 0.691 |
| sentence | 488 | 327.32 | 6.729 | 0.045 | 0.999 | 0.005 | 0.072 | 0.084 | 0.596 |
