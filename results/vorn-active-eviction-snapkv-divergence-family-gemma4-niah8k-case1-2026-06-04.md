# Vorn-Active Eviction Pressure Sweep

- Run id: `vorn-active-eviction-snapkv-divergence-family-gemma4-niah8k-case1-2026-06-04`
- Case id: `niah_multikey_1_8k-1`
- Full-context hit: `True`
- Record status counts: `{"PROMPT_QUALITY_SUCCESS": 4}`

| Arm | Pressure N | SEMU ids | Counterfactual hit | Delta quality |
|---|---:|---|---|---:|
| vorn_high | 1 | 423 | True | 0.000 |
| vorn_high | 3 | 423,422,421 | False | 1.000 |
| snapkv_high | 1 | 403 | True | 0.000 |
| snapkv_high | 3 | 403,415,411 | True | 0.000 |
