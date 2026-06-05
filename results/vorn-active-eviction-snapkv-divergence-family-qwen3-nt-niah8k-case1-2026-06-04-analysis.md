# SnapKV Divergence Result Analysis

- Run id: `vorn-active-eviction-snapkv-divergence-family-qwen3-nt-niah8k-case1-2026-06-04`
- Lane: `family-qwen3-nt-case1`
- Case id: `niah_multikey_1_8k-1`
- Family: `Qwen 3-NT 8B`
- Model: `Qwen/Qwen3-8B`
- Modal run: https://modal.com/apps/layne1penney/main/ap-LV9bEduW4Ou73KJrqNuY5w
- Wall-clock elapsed seconds: `167.562`
- Wall-clock cost USD: `0.1163`
- Full-context hit: `True`
- Full-context prediction: `The special magic number for faint-smolt mentioned in the provided text is **4374754**.`
- Full-context per-row generation cost USD: `0.003728`
- Full-context peak allocated MB: `34540.3`
- Phase 0 source JSON: `results/phase0-source-snapkv-divergence-family-qwen3-nt-case1-2026-06-04.json`
- Phase 0 source MD: `results/phase0-source-snapkv-divergence-family-qwen3-nt-case1-2026-06-04.md`

## Arm Results

| Arm | N | SEMU ids | Hit | Delta quality | Per-row generation cost USD | Runtime seconds | Peak allocated MB |
|---|---:|---|---|---:|---:|---:|---:|
| snapkv_high | 1 | 0 | True | 0.000 | 0.003043 | 4.384 | 34530.7 |
| vorn_high | 1 | 423 | True | 0.000 | 0.002903 | 4.182 | 34468.0 |
| snapkv_high | 3 | 0,426,353 | True | 0.000 | 0.002792 | 4.023 | 34429.5 |
| vorn_high | 3 | 423,422,299 | False | 1.000 | 0.002452 | 3.533 | 34285.5 |

## First-Failure Comparison

- `snapkv_high` first failure N: `none in measured subset`
- `vorn_high` first failure N: `3`

## Scope Note

This divergence cell intentionally restricts selector arms to `vorn_high` and `snapkv_high` and restricts pressures to the high-arm non-overlap subset. It measures which high-arm selector identifies structural essentials before the broad overlap gate would make the cell an honest-negative overlap finding.

Detailed SEMU labels for both high arms are in the Phase 0 source MD linked above.
