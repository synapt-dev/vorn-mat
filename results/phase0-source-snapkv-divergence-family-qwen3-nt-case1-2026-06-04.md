# Phase 0 SnapKV Divergence Source

- Run id: `vorn-active-eviction-snapkv-divergence-family-qwen3-nt-niah8k-case1-2026-06-04`
- Lane: `snapkv-divergence-family-qwen3-nt-case1`
- Case id: `niah_multikey_1_8k-1`
- Dataset config: `niah_multikey_1_8k`
- Model: `Qwen/Qwen3-8B`
- Pressure Ns: `[1, 3]`
- Selector arms: `vorn_high`, `snapkv_high`

## Selector Replay

| Arm | N | SEMU ids |
|---|---:|---|
| vorn_high | 1 | [423] |
| vorn_high | 3 | [423, 422, 299] |
| snapkv_high | 1 | [0] |
| snapkv_high | 3 | [0, 426, 353] |

## vorn_high Trace

- N=1: 423: The special magic number for faint-smolt mentioned in the provided text is<|im_end|>
- N=3: 423: The special magic number for faint-smolt mentioned in the provided text is<|im_end|>; 422: What is the special magic number for faint-smolt mentioned in the provided text?; 299 answer_overlap: One of the special magic numbers for faint-smolt is: 4374754.

## snapkv_high Trace

- N=1: 0: <|im_start|>user
- N=3: 0: <|im_start|>user; 426: </think>; 353: Another unusual thing about this software was that it was written primarily in a programming language called Lisp.
