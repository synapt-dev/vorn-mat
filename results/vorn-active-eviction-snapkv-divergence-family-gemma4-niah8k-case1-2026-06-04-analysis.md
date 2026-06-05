# SnapKV Divergence Result Analysis

- Run id: `vorn-active-eviction-snapkv-divergence-family-gemma4-niah8k-case1-2026-06-04`
- Lane: `family-gemma4-case1`
- Case id: `niah_multikey_1_8k-1`
- Family: `Gemma 4`
- Model: `google/gemma-4-E4B-it`
- Modal run: https://modal.com/apps/layne1penney/main/ap-XAacXUZN6piwOxHfdfpZJK
- Wall-clock elapsed seconds: `141.955`
- Wall-clock cost USD: `0.0985`
- Full-context hit: `True`
- Full-context prediction: `The special magic number for faint-smolt mentioned in the provided text is **4374754**.`
- Full-context per-row generation cost USD: `0.003082`
- Full-context peak allocated MB: `21058.8`
- Phase 0 source JSON: `results/phase0-source-snapkv-divergence-family-gemma4-case1-2026-06-04.json`
- Phase 0 source MD: `results/phase0-source-snapkv-divergence-family-gemma4-case1-2026-06-04.md`

## Arm Results

| Arm | N | SEMU ids | Hit | Delta quality | Per-row generation cost USD | Runtime seconds | Peak allocated MB |
|---|---:|---|---|---:|---:|---:|---:|
| snapkv_high | 1 | 403 | True | 0.000 | 0.002372 | 3.418 | 21016.0 |
| vorn_high | 1 | 423 | True | 0.000 | 0.002184 | 3.147 | 21038.1 |
| snapkv_high | 3 | 403,415,411 | True | 0.000 | 0.002404 | 3.464 | 20933.7 |
| vorn_high | 3 | 423,422,421 | False | 1.000 | 0.001681 | 2.422 | 20967.9 |

## First-Failure Comparison

- `snapkv_high` first failure N: `none in measured subset`
- `vorn_high` first failure N: `3`

## Scope Note

This divergence cell intentionally restricts selector arms to `vorn_high` and `snapkv_high` and restricts pressures to the high-arm non-overlap subset. It measures which high-arm selector identifies structural essentials before the broad overlap gate would make the cell an honest-negative overlap finding.

Detailed SEMU labels for both high arms are in the Phase 0 source MD linked above.
