# Qwen 2.5 7B Instruct multi-budget refresh on canonical bf16-fixed harness

**Artifact**: `results/qwen25-multibudget-refresh-2026-05-26.json`
**Date**: 2026-05-26
**Model**: Qwen/Qwen2.5-7B-Instruct
**Slice**: niah_multikey_1_4k validation[:50]
**Cost**: $1.9263
**Total cells**: 12 (6 methods × 2 budgets b={256, 512})
**Modal app**: ap-jaZGtQPU5THBQW5HnRoqTb on layne-79000

## Context

Replaces the 2026-05-14 pre-fix artifact (`qwen-4k-budget-sweep-2026-05-14.json`) which contained the `!`-cascade harness artifact (fp16 NaN logits → token-0 emission). Companion to the Sub-lane B b=1024 row refresh (also 2026-05-26).

The pre-fix `!`-cascade differentially suppressed attention-weight methods more than vorn methods on Qwen 2.5; the post-fix data reveals Qwen 2.5 is **channel-tolerant** rather than vorn-favoring as the pre-fix data suggested. The original 'mirror-inverse to Gemma 4' framing was a harness artifact, not a model property.

## Hit-rates

| Method | b=256 | b=512 |
|--------|-------|-------|
| token-vorn | 0.0 | 0.1 |
| sentence-vorn | 0.76 | 0.74 |
| token-TOVA | 0.14 | 0.44 |
| sentence-TOVA | 0.72 | 0.66 |
| token-H2O | 0.12 | 0.46 |
| sentence-H2O | 0.74 | 0.66 |

## Rescue spectrum pattern

- **Token-level methods at b=256**: all near-floor (vorn 0.00, TOVA 0.14, H2O 0.12)
- **Sentence-level methods at b=256**: all rescue to 0.72-0.76 range
- **At b=512**: token methods recover partially (0.10-0.46); sentence stays 0.66-0.76
- Pattern matches the **rescue-resilient** profile (sentence-grouping rescues   attention-weight + vorn at constrained budgets)

## Methodology

- Harness: vorn-mat main at commit `b0223af` (bf16-stability fix applied 2026-05-23)
- Dispatched via canonical parallel orchestrator (`run_modal_cells_parallel.py`,   Layer 5 fire-and-forget per vorn-mat#18)
- A100-80GB profile on `layne-79000`
- Per-fixture observations + memory telemetry preserved in artifact

## Build reproducibility

```bash
modal run --detach examples/run_modal_cells_parallel.py \
  --cell-spec-path .benchmarks/qwen25-b256-b512-specs.json \
  --output-dir .benchmarks/qwen25-multibudget-refresh
```
