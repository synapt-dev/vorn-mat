# SnapKV Divergence Result Analysis

- Run id: `vorn-active-eviction-snapkv-divergence-family-llama31-niah8k-case1-2026-06-04`
- Lane: `family-llama31-case1`
- Case id: `niah_multikey_1_8k-1`
- Family: `Llama 3.1 8B`
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Modal run: https://modal.com/apps/layne1penney/main/ap-lzhc6OTIwYFR8wjzlw6hqp
- Wall-clock elapsed seconds: `198.399`
- Wall-clock cost USD: `0.1377`
- Full-context hit: `True`
- Full-context prediction: `4374754`
- Full-context per-row generation cost USD: `0.002838`
- Full-context peak allocated MB: `33962.8`
- Phase 0 source JSON: `results/phase0-source-snapkv-divergence-family-llama31-case1-2026-06-04.json`
- Phase 0 source MD: `results/phase0-source-snapkv-divergence-family-llama31-case1-2026-06-04.md`

## Arm Results

| Arm | N | SEMU ids | Hit | Delta quality | Per-row generation cost USD | Runtime seconds | Peak allocated MB |
|---|---:|---|---|---:|---:|---:|---:|
| snapkv_high | 1 | 0 | True | 0.000 | 0.002036 | 2.934 | 33943.6 |
| vorn_high | 1 | 426 | True | 0.000 | 0.002440 | 3.515 | 33876.7 |

## First-Failure Comparison

- `snapkv_high` first failure N: `none in measured subset`
- `vorn_high` first failure N: `none in measured subset`

## Scope Note

This divergence cell intentionally restricts selector arms to `vorn_high` and `snapkv_high` and restricts pressures to the high-arm non-overlap subset. It measures which high-arm selector identifies structural essentials before the broad overlap gate would make the cell an honest-negative overlap finding.

Detailed SEMU labels for both high arms are in the Phase 0 source MD linked above.
