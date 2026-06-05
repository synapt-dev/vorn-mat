# SnapKV Divergence Result Analysis

- Run id: `vorn-active-eviction-snapkv-divergence-lateral-niah8k-case3-2026-06-04`
- Lane: `lateral-case3`
- Case id: `niah_multikey_1_8k-3`
- Family: `Mistral 7B`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Modal run: https://modal.com/apps/layne1penney/main/ap-oDfDspnijMKoF6UCJ1oIj1
- Wall-clock elapsed seconds: `153.509`
- Wall-clock cost USD: `0.1065`
- Full-context hit: `True`
- Full-context prediction: `5672073`
- Full-context per-row generation cost USD: `0.003110`
- Full-context peak allocated MB: `35790.5`
- Phase 0 source JSON: `results/phase0-source-snapkv-divergence-lateral-case3-2026-06-04.json`
- Phase 0 source MD: `results/phase0-source-snapkv-divergence-lateral-case3-2026-06-04.md`

## Arm Results

| Arm | N | SEMU ids | Hit | Delta quality | Per-row generation cost USD | Runtime seconds | Peak allocated MB |
|---|---:|---|---|---:|---:|---:|---:|
| snapkv_high | 1 | 0 | True | 0.000 | 0.002428 | 3.498 | 35723.0 |
| vorn_high | 1 | 422 | True | 0.000 | 0.001988 | 2.865 | 35697.1 |

## First-Failure Comparison

- `snapkv_high` first failure N: `none in measured subset`
- `vorn_high` first failure N: `none in measured subset`

## Scope Note

This divergence cell intentionally restricts selector arms to `vorn_high` and `snapkv_high` and restricts pressures to the high-arm non-overlap subset. It measures which high-arm selector identifies structural essentials before the broad overlap gate would make the cell an honest-negative overlap finding.

Detailed SEMU labels for both high arms are in the Phase 0 source MD linked above.
