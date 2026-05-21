# Cross-Model Gate (`Qwen 2.5 7B`, `Llama 3.1 8B`) — 2026-05-13

Run conditions:
- Profile: `author`
- Dataset: `rbiswasfc/ruler`
- Config: `niah_multikey_1_4k`
- Slice: `validation[:50]`
- Budget target: `1024` sweet-spot gate on the 4k slice

| Model | Method | Budget regime | Hit rate | Wall-clock | Inference cost |
|------|--------|---------------|----------|------------|----------------|
| `Qwen/Qwen2.5-7B-Instruct` | Vanilla | full context | 0.00 | 192.29s | $0.1335 |
| `Qwen/Qwen2.5-7B-Instruct` | Token vorn | 1024 guarded | 0.04 | 294.38s | $0.2043 |
| `Qwen/Qwen2.5-7B-Instruct` | Sentence vorn | 1024 guarded | 0.00 | 226.73s | $0.1573 |
| `Qwen/Qwen2.5-7B-Instruct` | Sentence vorn | 1024 no guardrails | 0.00 | 326.50s | $0.2266 |

Qwen read:

- This is not a valid cross-model architecture comparison on this slice because the full-context ceiling is already `0.00`.
- The first incorrect predictions are visibly garbled rather than merely wrong answers, which points to a model-specific prompt/runtime compatibility problem in the current runner surface:
  - `The!pecial!mag!c!!!n!um!b!er!!!!!f!or!!!!!f!`
  - `The! special!!!!!!!!!!!!!!!!!!! magic!!! number!! for!!`
  - `The!!!! special!! magic!!!!!!!!!!!!!!!!!!! number! for!`

Llama gate:

- `meta-llama/Llama-3.1-8B-Instruct` could not be run because the Modal Hugging Face secret does not have access to the gated repository (`403`).

Read:

- The lane still paid off technically: the Modal runner now accepts `model_id` explicitly, tags remote outputs by model, and can support future cross-model sweeps without ad hoc code edits.
- The empirical result is narrower: cross-model validity remains open. Qwen 2.5 7B is non-discriminative on the target NIAH slice, and Llama 3.1 8B is operationally blocked.
- The honest next step is not to overclaim generalization. It is to either fix Qwen compatibility, obtain Llama access, or select a different open model with demonstrated vanilla headroom first.
