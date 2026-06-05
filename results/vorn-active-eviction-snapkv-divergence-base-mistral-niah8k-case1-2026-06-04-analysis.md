# SnapKV Divergence Result Analysis

- Run id: `vorn-active-eviction-snapkv-divergence-base-mistral-niah8k-case1-2026-06-04`
- Lane: `base-mistral-case1`
- Case id: `niah_multikey_1_8k-1`
- Family: `Mistral 7B`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Modal run: https://modal.com/apps/layne1penney/main/ap-cu8VVYlDt6FJdDbfitcUl6
- Wall-clock elapsed seconds: `150.926`
- Wall-clock cost USD: `0.1047`
- Full-context hit: `True`
- Full-context prediction: `4374754`
- Full-context per-row generation cost USD: `0.002819`
- Full-context peak allocated MB: `35748.9`
- Phase 0 source JSON: `results/phase0-source-snapkv-divergence-base-mistral-case1-2026-06-04.json`
- Phase 0 source MD: `results/phase0-source-snapkv-divergence-base-mistral-case1-2026-06-04.md`

## Arm Results

| Arm | N | SEMU ids | Hit | Delta quality | Per-row generation cost USD | Runtime seconds | Peak allocated MB |
|---|---:|---|---|---:|---:|---:|---:|
| snapkv_high | 1 | 0 | True | 0.000 | 0.002448 | 3.528 | 35681.5 |
| vorn_high | 1 | 422 | True | 0.000 | 0.002412 | 3.475 | 35671.1 |

## First-Failure Comparison

- `snapkv_high` first failure N: `none in measured subset`
- `vorn_high` first failure N: `none in measured subset`

## Scope Note

This divergence cell intentionally restricts selector arms to `vorn_high` and `snapkv_high` and restricts pressures to the high-arm non-overlap subset. It measures which high-arm selector identifies structural essentials before the broad overlap gate would make the cell an honest-negative overlap finding.

Detailed SEMU labels for both high arms are in the Phase 0 source MD linked above.
