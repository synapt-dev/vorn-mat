# Qwen 3 30B-A3B eviction panel (Phase 1 + Phase 3 fill-out)

**Artifact**: `results/gemma3-qwen3-30b-eviction-phase1-phase3-2026-05-26.json` (cells filtered to Qwen 3 30B-A3B)
**Date**: 2026-05-26
**Model**: `Qwen/Qwen3-30B-A3B` (MoE; ~3B active params; `enable_thinking=False` auto-applied via `_chat_template_kwargs()` substring match)
**Slice**: niah_multikey_1_4k validation[:50]
**Harness**: vorn-mat main at `de7d633` (post-PR#21)
**Hardware**: A100-80GB on `layne-79000` profile (attention-channel cells deferred — see "Hardware-bounded coverage limit" below)

## Hit-rate panel — vorn-channel methods (covered this session)

| Method        | b=256 | b=512 | b=1024 | b=1536 | b=2048 |
|---------------|-------|-------|--------|--------|--------|
| token-vorn    | 0.000 | 0.000 | 0.000  | n/a    | n/a    |
| sentence-vorn | 0.020 | 0.020 | 0.420  | 0.980  | 1.000  |

Wilson 95% CIs (n=50):
- token-vorn at any covered budget: hr=0.000, CI [0.000, 0.071]
- sentence-vorn at b=256: hr=0.020, CI [0.004, 0.105]
- sentence-vorn at b=512: hr=0.020, CI [0.004, 0.105]
- sentence-vorn at b=1024: hr=0.420, CI [0.294, 0.558] (Phase 1 cell)
- sentence-vorn at b=1536: hr=0.980, CI [0.895, 0.996]
- sentence-vorn at b=2048: hr=1.000, CI [0.929, 1.000]

## Hit-rate panel — attention-channel methods (hardware-bounded coverage limit)

| Method        | b=256 | b=512 | b=1024 | b=1536 | b=2048 |
|---------------|-------|-------|--------|--------|--------|
| token-TOVA    | n/a   | n/a   | **OOM**    | n/a    | n/a    |
| sentence-TOVA | n/a   | n/a   | **OOM**    | n/a    | n/a    |
| token-H2O     | n/a   | n/a   | **OOM**    | n/a    | n/a    |
| sentence-H2O  | n/a   | n/a   | **OOM**    | n/a    | n/a    |

All 4 cells at b=1024 hit identical CUDA OOM on both A100-80GB (Phase 1) and H100 80GB (Phase 3):
```
OutOfMemoryError: Tried to allocate 1.77 GiB.
GPU 0 has total capacity of ~80 GiB of which <1 GiB free.
```

Root cause: Qwen 3 30B-A3B bf16 weights occupy ~60GB; combined with KV cache + activations + attention-state-tracking buffer for TOVA/H2O scoring, total exceeds the 80GB envelope of either A100-80GB or H100. The 1.77 GiB allocation that OOMs is the softmax computation buffer at the attention layer.

**Hardware-bounded coverage limit** (Layne ratified 2026-05-26 evening, Option (a)): the 4 attention-channel cells at b=1024 are skipped this revision and documented as VRAM-bounded coverage limits. Future work covers via H200 (141GB VRAM), B100/B200 (192GB+), or tensor-parallel sharding of the attention-state buffer.

This is the same disclosure shape as the §6.8 cross-runner caveat (Mistral late-budget cells filled via 3-case micro-recovery composition; A100 vs H100 boundary). Hardware-bounded coverage limits are honestly disclosable, not claim-undercutting.

## Vanilla reference

`results/qwen3-30b-a3b-no-thinking-vanilla-2026-05-26.json`: vanilla no-eviction baseline at hit-rate **1.000** (50/50). Wilson 95% CI [0.929, 1.000].

## Substantive read

**Sentence-vorn budget curve shows sharp threshold-bounded recovery:**

```
   b=256   b=512   b=1024   b=1536   b=2048
   0.020   0.020   0.420    0.980    1.000
   floor   floor   partial  near-ceil ceiling
```

The recovery threshold lies between b=1024 and b=1536. Two-cell jump from 0.42 → 0.98 across that budget step is steeper than the gradual budget-curve shapes observed on Mistral, Llama 3.1, or Gemma 4 in the main paper panels.

**Token-vorn floors completely across b={256, 512, 1024}.** No partial-discrimination cells; uniform 0.000. Combined with the sentence-vorn threshold-bounded recovery, this is a clear **granularity-rescue** signal at MoE scale (sentence grouping unlocks signal that token-level scoring cannot retain).

**Channel-favoritism analysis blocked** by the VRAM-bounded attention-channel coverage limit. Cannot compare sentence-vorn vs sentence-TOVA/H2O at any budget; cannot characterize Qwen 3 30B-A3B's position on the channel-favoritism axis used in Figure 1.

**Proposed paper positioning** (Opus's lane to draft + Layne ratify):
- **Granularity-rescue surface**: include Qwen 3 30B-A3B as a 4th supporting-replication entry (alongside Mistral, Llama 3.1, Gemma 4) on the vorn-channel side. Threshold-bounded recovery shape similar to Llama 3.1's pattern but with sharper threshold (jump from 0.42 → 0.98 in one budget step).
- **Cross-family channel-favoritism axis (Figure 1)**: cannot place Qwen 3 30B-A3B on this axis due to attention-channel coverage gap. Document as deferred-coverage in figure caption.
- **MoE scaling note**: the MoE architecture is a NEW dimension in the rescue spectrum. Worth a brief §6 paragraph noting that the granularity-rescue mechanism applies at MoE scale (~3B active, 30B total) too, with the caveat that attention-channel coverage requires VRAM beyond what's available on the current eval substrate.

## Cost (Qwen 3 30B-A3B slice)

| Phase | Cells | Cost |
|---|---|---|
| Phase 1 (b=1024 × vorn + sentence-vorn) | 2 | $1.4831 |
| Phase 3 A100 (token-vorn b=256/512 + sentence-vorn b=256/512/1536/2048) | 6 | $4.1160 |
| Phase 3 H100 (attention-channel at b=1024, OOM) | 0 (4 wasted) | ~$1.50-2 wasted |
| **Total** | **8 succeeded** | **~$7.10** |

## Reproducibility

- Harness: vorn-mat main at `de7d633` (post-PR#21)
- `enable_thinking=False` auto-applied via substring match `"qwen3" in model_id.lower()` at `vorn_mat/local_exec.py:292-294`
- Specs: random_seed=17, always_keep_prefix_tokens=1, preserve_recent_window=True, sentence_pooling=max, sentence_top_k=3, eviction_trigger=budget_threshold, sentence_boundary_lookahead_tokens=25, force_eviction_overflow_ratio=1.2
- Per-fixture observations preserved in merged JSON artifact
- A100-80GB on Modal `layne-79000` profile (vorn-channel cells); attention-channel cells require future H200+
