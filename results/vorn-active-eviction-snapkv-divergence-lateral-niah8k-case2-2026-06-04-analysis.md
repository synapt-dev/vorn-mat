# SnapKV Divergence Result Analysis

- Run id: `vorn-active-eviction-snapkv-divergence-lateral-niah8k-case2-2026-06-04`
- Lane: `lateral-case2`
- Case id: `niah_multikey_1_8k-2`
- Family: `Mistral 7B`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Modal run: https://modal.com/apps/layne1penney/main/ap-e6ft90jFnNgcsqcNUzVigt
- Wall-clock elapsed seconds: `150.168`
- Wall-clock cost USD: `0.1042`
- Full-context hit: `True`
- Full-context prediction: `4694634.`
- Full-context per-row generation cost USD: `0.003316`
- Full-context peak allocated MB: `35723.0`
- Phase 0 source JSON: `results/phase0-source-snapkv-divergence-lateral-case2-2026-06-04.json`
- Phase 0 source MD: `results/phase0-source-snapkv-divergence-lateral-case2-2026-06-04.md`

## Arm Results

| Arm | N | SEMU ids | Hit | Delta quality | Per-row generation cost USD | Runtime seconds | Peak allocated MB |
|---|---:|---|---|---:|---:|---:|---:|
| snapkv_high | 1 | 0 | True | 0.000 | 0.002068 | 2.980 | 35655.6 |
| vorn_high | 1 | 422 | True | 0.000 | 0.001870 | 2.695 | 35645.2 |

## First-Failure Comparison

- `snapkv_high` first failure N: `none in measured subset`
- `vorn_high` first failure N: `none in measured subset`

## Scope Note

This divergence cell intentionally restricts selector arms to `vorn_high` and `snapkv_high` and restricts pressures to the high-arm non-overlap subset. It measures which high-arm selector identifies structural essentials before the broad overlap gate would make the cell an honest-negative overlap finding.

Detailed SEMU labels for both high arms are in the Phase 0 source MD linked above.
