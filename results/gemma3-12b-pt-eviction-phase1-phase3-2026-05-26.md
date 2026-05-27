# Gemma 3 12B-pt eviction panel (Phase 1 + Phase 3 fill-out)

**Artifact**: `results/gemma3-qwen3-30b-eviction-phase1-phase3-2026-05-26.json` (cells filtered to Gemma 3 12B-pt)
**Date**: 2026-05-26
**Model**: `google/gemma-3-12b-pt` (pretrained base, no chat template)
**Slice**: niah_multikey_1_4k validation[:50]
**Harness**: vorn-mat main at `de7d633` (post-PR#21 base-model sentence-eviction offset-fix)
**Hardware**: A100-80GB on `layne-79000` profile

## Hit-rate panel

| Method        | b=1024 | b=1536 | b=2048 |
|---------------|--------|--------|--------|
| token-vorn    | 0.000  | 0.000  | 0.000  |
| sentence-vorn | 0.000  | n/a    | n/a    |
| token-TOVA    | 0.000  | 0.000  | 0.000  |
| sentence-TOVA | 0.000  | n/a    | n/a    |
| token-H2O     | 0.000  | 0.000  | 0.000  |
| sentence-H2O  | 0.000  | n/a    | n/a    |

All 9 eviction cells across {b=1024, b=1536, b=2048} × {token, sentence} × {vorn, TOVA, H2O} have hit-rate **0.000** with Wilson 95% CI [0.000, 0.071] (n=50).

Cells in scope but not run this session: sentence-level at b=1536 and b=2048 (deferred; given the uniform-floor pattern at b=1024 across all methods + token-level floor at b=1536/2048, expected to floor as well; cheap to add in a follow-up if needed).

## Vanilla reference

`results/gemma3-12b-pt-diagnostic-2026-05-26.json` (canonical bf16-fixed harness, same slice):
- **hit-rate 1.000** (50/50). Wilson 95% CI [0.929, 1.000].

## Substantive read

Gemma 3 12B-pt is **vanilla-discriminative but comprehensively eviction-fragile** on this slice. The recovery story:

- **Pre-bf16-fix (2026-05-20)**: 0.000 vanilla — harness artifact (pad-cascade fingerprint)
- **Post-bf16-fix (2026-05-26 vanilla diagnostic)**: 1.000 vanilla — full capability recovered
- **Eviction panel (this artifact)**: 0.000 across ALL methods at ALL tested budgets (b=1024, b=1536, b=2048)

This is a third profile distinct from the prior §C.2 binary options:
- NOT "drowning-floor boundary entry" (Option A description) — the family is fully vanilla-discriminative
- NOT a partial-recovery arc (Option B description) — eviction-fragility is uniform, not gradual

**Proposed §C.2 framing**: "vanilla-discriminative-but-eviction-fragile" family-class. Distinct profile from the rescue-spectrum families (where SOME method retains signal under eviction). Worth documenting as a distinct outcome shape if the panel-expansion ratifies this language.

### Hypotheses for the uniform floor

Speculative — not investigated this session:
- **H1 (base-model x retrieval prompt)**: base models do next-token continuation; their first-token prediction is correct for vanilla (the answer is the most likely continuation) but eviction reduces attention to the needle-bearing region, and base models lack the instruction-following robustness that helps instruction-tuned models recover from partial-context. Tests: re-run with Gemma 3 12B-it post-multi-EOS-fix would isolate base-vs-instruction-tuned dimension.
- **H2 (Gemma 3 architecture-specific)**: Gemma 3's hybrid sliding-window-attention architecture may interact with eviction policies differently than non-sliding architectures. Tests: budget sweep at b=4096+ (full context, no eviction) would confirm graceful degradation as cache grows.
- **H3 (4k context boundary)**: the niah_multikey_1_4k slice is at 4k context; eviction at any budget < 4096 forces some content drop. Base models may not handle partial-haystack at all. Tests: drop-percentage analysis across budgets.

Reserved as future-work; not in scope for this revision.

## Cost (Gemma 3 12B-pt slice)

| Phase | Cells | Cost |
|---|---|---|
| Phase 1 (b=1024 × token-methods × 3) | 3 | $1.8038 |
| Phase 3 A100 (b=1024 sentence × 3 + b=1536/2048 token × 6) | 9 | $7.4689 |
| **Total** | **12** | **$9.27** |

## Reproducibility

- Harness: vorn-mat main at `de7d633` (post-PR#21)
- Specs: random_seed=17, always_keep_prefix_tokens=1, preserve_recent_window=True, sentence_pooling=max, sentence_top_k=3, eviction_trigger=budget_threshold, sentence_boundary_lookahead_tokens=25, force_eviction_overflow_ratio=1.2
- Per-fixture observations preserved in merged JSON artifact
- A100-80GB on Modal `layne-79000` profile
