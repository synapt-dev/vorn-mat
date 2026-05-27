# Multi-step fillout 2026-05-27 — Ministral gap-fill + b=1536/2048 panel + Mistral micro-recovery

**Artifact**: `results/multistep-fillout-2026-05-27.json`
**Date**: 2026-05-27
**Slice**: niah_multikey_1_4k validation[:50] (n=50)
**Harness**: vorn-mat main at post-PR#23 (commit e57f2a7 + 882d546); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` applied org-wide via `default_modal_app_spec()`
**Profiles**: layne-79000 (capped mid-session) + layne (resume)
**Total cells succeeded**: 69 across 7 families

## Session arc

| Step | Wave | Profile | Cells | Cost | Notes |
|---|---|---|---|---|---|
| 1 | Ministral gap-fill (initial) | layne-79000 | 0/8 | ~$1.28 wasted | PyTorch allocator fragmentation OOM; led to PR #23 substrate fix |
| 2 | 1-cell verify (PR #23 fix) | layne-79000 | 1/1 | $0.19 | Fragmentation fix CONFIRMED |
| 3 | 8-cell Ministral re-run | layne-79000 | 8/8 | $1.57 | TOVA/H2O at b=256/512/1024 |
| 4a | 56-cell b=1536/2048 fill-out (partial) | layne-79000 | 31/56 | ~$9-10 | Modal billing cap hit at $99.36/$100 mid-wave |
| 4b | 25 Step 4 missing + 5 Step 5 Mistral | layne | 30/30 | $15.29 | Profile-switched resume on layne |

**Total cells delivered**: 69 (Step 3: 8; Step 4: 56; Step 5: 5)
**Total cost**: ~$27-28 across the lane.

## Substantive panel by family

### Ministral 8B (20 cells) — rescue-spectrum-anchor-grade ✓

5-budget panel (b=256, 512, 1024, 1536, 2048) across all 6 methods:

| method | b=256 | b=512 | b=1024 | b=1536 | b=2048 |
|---|---|---|---|---|---|
| token-vorn | (prior 1.00) | (prior 1.00) | (prior 1.00) | 0.980 | 1.000 |
| sentence-vorn | (prior 1.00) | (prior 1.00) | (prior 1.00) | 1.000 | 1.000 |
| token-TOVA | 0.000 | 0.060 | (prior 0.44) | 0.740 | 0.900 |
| sentence-TOVA | (prior 0.98) | (prior 1.00) | (prior 0.98) | 1.000 | 1.000 |
| token-H2O | 0.000 | 0.060 | 0.460 | 0.720 | 0.920 |
| sentence-H2O | 0.400 | 0.700 | 0.980 | 1.000 | 1.000 |

Pattern: **sentence-level methods rescue all the way to ceiling by b=1024** (all 3 sentence methods at 0.98-1.00 from b=1024 onward). Token-attention methods (TOVA, H2O) show smooth budget-driven recovery from floor (b=256) to near-ceiling (b=2048). Token-vorn is at ceiling at b=1024+ (prior data) and slightly below at b=1536 (0.98).

Anchor-grade multi-budget surface; supports promotion from "supporting replication" to "rescue spectrum anchor" in paper §5.6.

### Gemma 2 9B (12 cells) — rescue-spectrum-anchor-grade ✓

| method | b=1536 | b=2048 |
|---|---|---|
| token-vorn | 0.600 | 0.860 |
| sentence-vorn | 0.960 | 0.980 |
| token-TOVA | 0.860 | 0.920 |
| sentence-TOVA | 0.920 | 0.980 |
| token-H2O | 0.860 | 0.920 |
| sentence-H2O | 0.920 | 0.980 |

Pattern: **clear granularity-rescue at b=1536** (token-vorn 0.60 → sentence-vorn 0.96 = 0.36 lift). Token-attention methods already near-ceiling. At b=2048 the budget-driven recovery is in evidence (token-vorn rises 0.60 → 0.86).

Anchor-grade multi-budget surface; same promotion shape as Ministral.

### Qwen 2.5 7B (12 cells) — capability-bounded plateau

| method | b=1536 | b=2048 |
|---|---|---|
| token-vorn | 0.620 | 0.680 |
| sentence-vorn | 0.700 | 0.660 |
| token-TOVA | 0.700 | 0.680 |
| sentence-TOVA | 0.680 | 0.700 |
| token-H2O | 0.700 | 0.680 |
| sentence-H2O | 0.680 | 0.700 |

Pattern: all 12 cells cluster in **0.62-0.70 range** with **no method × budget axis emerging as discriminative**. Consistent with paper's existing "supporting replication" framing and capability-bounded interpretation (Qwen 2.5 vanilla = 0.64). NOT anchor-grade.

### Qwen 3 8B (12 cells) — STRONG granularity-rescue with attention-channel favoring at sentence-level

| method | b=1536 | b=2048 |
|---|---|---|
| token-vorn | 0.000 | 0.100 |
| sentence-vorn | 0.820 | 0.920 |
| token-TOVA | 0.260 | 0.460 |
| sentence-TOVA | 1.000 | 1.000 |
| token-H2O | 0.260 | 0.460 |
| sentence-H2O | 1.000 | 1.000 |

Pattern: **sharp granularity-rescue** — token-vorn floors (0.00/0.10) while sentence-vorn recovers to 0.82/0.92. **Channel-favoritism at sentence-level**: sentence-TOVA + sentence-H2O hit 1.000 at both budgets, beating sentence-vorn (0.82/0.92). This is the sentence-attention-favoring extreme similar to Gemma 4 — Qwen 3 8B may be a NEW attention-favoring outlier candidate for the family-conditional axis.

### Qwen 3 30B-A3B token-vorn (2 cells) — budget-curve continuation

| method | b=1024 (Phase 3) | b=1536 | b=2048 |
|---|---|---|---|
| token-vorn | 0.000 | 0.100 | 0.380 |

Combined with sentence-vorn from Phase 3 (0.020 → 0.020 → 0.420 → 0.980 → 1.000 across b=256→2048): full granularity-rescue picture for Qwen 3 30B-A3B vorn-channel. Attention-channel still VRAM-bounded (>80GB needed), per existing §C.2 disclosure.

### Gemma 3 12B-pt sentence (6 cells) — UNIFORM FLOOR across sentence methods

| method | b=1536 | b=2048 |
|---|---|---|
| sentence-vorn | 0.000 | 0.000 |
| sentence-TOVA | 0.000 | 0.000 |
| sentence-H2O | 0.000 | 0.000 |

Pattern: confirms "vanilla-discriminative-but-comprehensively-eviction-fragile" profile at sentence-level too. Combined with prior 12 cells from Phase 3 (all 0.000), the family now has **18/18 eviction cells at 0.000** — a stable family-property across all method × budget combinations tested.

### Mistral 7B v0.3 micro-recovery promotion (5 cells)

| budget | method | n=3 micro-recovered (prior) | n=50 (this run) |
|---|---|---|---|
| 256 | token-H2O | 0.00 | 0.040 |
| 256 | sentence-H2O | 1.00 | 0.760 |
| 1536 | token-TOVA | 1.00 | 0.980 |
| 1536 | sentence-TOVA | 1.00 | 0.980 |
| 2048 | sentence-TOVA | 1.00 | 1.000 |

Pattern: **Form 2 (fragmentation) confirmed** — all 5 cells completed cleanly on A100-80GB under expandable_segments, unblocking the previous OOMs. 3-case micro-recovery values were directionally faithful for b≥1536 cells (0.98-1.00); the b=256 sentence-H2O small-n estimate over-shot (0.76 actual vs 1.00 partial).

Paper §6.8 "3-case agreement methodology" caveat can be removed for the b=1536 token-TOVA, b=1536 sentence-TOVA, and b=2048 sentence-TOVA cells (n=50 now available and matches prior reading). The b=256 cells reveal the partial-data over-estimate boundary — useful methodological note.

## OOM events for OOM Registry (config#283)

**Allocator-class fragmentation OOMs (Form 2)** — pre-PR-23 harness:
- 8 cells Ministral 8B (token+sentence × TOVA+H2O at b=256/512/1024 subset) — all recovered post-PR-23

**Budget-rejection events** — NEW class:
- 25 cells of Step 4 wave were Modal-rejected when layne-79000 hit $99.36/$100 cap mid-wave. Cells: 1 Gemma 2 (vorn b=2048), 4 Qwen 2.5 (×4 b=2048), 12 Qwen 3 8B, 2 Qwen 3 30B-A3B, 6 Gemma 3 12B-pt. ALL successfully re-run on `layne` profile in resume wave.

These are a substrate event class distinct from OOM (account-level cap vs hardware-level resource exhaustion). May warrant a new registry section.

## Paper-impact summary

Substantive upgrades supported by this data (Opus's lane to draft + Layne ratify):

- **§3.5 Table 1**: Ministral row extends from single-budget to 5-budget; Gemma 2 row extends from single-budget to 2-budget
- **§5.6**: Ministral + Gemma 2 promote from "supporting replication" to "rescue spectrum anchor" (3-family anchor → 5-family anchor)
- **§6.1**: hypothesis-narrowing language can shift from "narrow" to "falsify" for Ministral + Gemma 2 architectural-pattern tests
- **§6.8**: "3-case agreement methodology" caveat removable for 3 Mistral cells (b=1536 tokenTOVA + sentenceTOVA + b=2048 sentenceTOVA)
- **Abstract**: hedge sentence about "single-budget coverage on Ministral and Gemma 2" can be removed
- **§C.2 Gemma 3 12B-pt entry**: extend with "18/18 eviction cells at 0.000 across all methods × budgets tested" — confirms comprehensive-eviction-fragility profile
- **Figure 1 / §5.8**: consider Qwen 3 8B as a new attention-favoring outlier candidate (sentence-attention beats sentence-vorn 1.00 vs 0.82-0.92)

## Reproducibility

- Harness: vorn-mat main at e57f2a7 (post-PR#23)
- env_vars: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` baked into Modal image via `ModalAppSpec.env_vars`
- Specs preserved at:
  - `.benchmarks/fillout-all-b1536-b2048-specs.json` (Step 4 spec, 56 cells)
  - `.benchmarks/ministral-tova-h2o-gap-fill-specs.json` (Step 3 spec, 8 cells)
  - `.benchmarks/step5-mistral-microrecovery-promotion-specs.json` (Step 5 spec, 5 cells)
  - `.benchmarks/resume-wave-step4-step5-on-layne-specs.json` (Step 4+5 resume, 30 cells)
- A100-80GB throughout
- Modal Volume artifacts pulled via `modal volume get synapt-vorn-mat-vol /results/vorn-mat`
