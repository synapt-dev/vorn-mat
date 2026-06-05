# Phase 0 SnapKV Divergence Source

- Run id: `vorn-active-eviction-snapkv-divergence-family-llama31-niah8k-case1-2026-06-04`
- Lane: `snapkv-divergence-family-llama31-case1`
- Case id: `niah_multikey_1_8k-1`
- Dataset config: `niah_multikey_1_8k`
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Pressure Ns: `[1]`
- Selector arms: `vorn_high`, `snapkv_high`

## Selector Replay

| Arm | N | SEMU ids |
|---|---:|---|
| vorn_high | 1 | [426] |
| snapkv_high | 1 | [0] |

## vorn_high Trace

- N=1: 426: The special magic number for faint-smolt mentioned in the provided text is<|eot_id|><|start_header_id|>assistant<|end_he

## snapkv_high Trace

- N=1: 0: <|begin_of_text|><|start_header_id|>system<|end_header_id|>
