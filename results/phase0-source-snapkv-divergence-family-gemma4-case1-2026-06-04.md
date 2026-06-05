# Phase 0 SnapKV Divergence Source

- Run id: `vorn-active-eviction-snapkv-divergence-family-gemma4-niah8k-case1-2026-06-04`
- Lane: `snapkv-divergence-family-gemma4-case1`
- Case id: `niah_multikey_1_8k-1`
- Dataset config: `niah_multikey_1_8k`
- Model: `google/gemma-4-E4B-it`
- Pressure Ns: `[1, 3]`
- Selector arms: `vorn_high`, `snapkv_high`

## Selector Replay

| Arm | N | SEMU ids |
|---|---:|---|
| vorn_high | 1 | [423] |
| vorn_high | 3 | [423, 422, 421] |
| snapkv_high | 1 | [403] |
| snapkv_high | 3 | [403, 415, 411] |

## vorn_high Trace

- N=1: 423: The special magic number for faint-smolt mentioned in the provided text is<turn|>
- N=3: 423: The special magic number for faint-smolt mentioned in the provided text is<turn|>; 422: What is the special magic number for faint-smolt mentioned in the provided text?; 421: Our hypothesis was that if we wrote our software in Lisp, we'd be able to get features done faster than our competitors,

## snapkv_high Trace

- N=1: 403: But with Web-based software, especially when you have the source code of both the language and the operating system, you
- N=3: 403: But with Web-based software, especially when you have the source code of both the language and the operating system, you; 415: We didn't know anything about marketing, or hiring people, or raising money, or getting customers.; 411: We knew Lisp was a really good language for writing software quickly, and server-based applications magnify the effect o
