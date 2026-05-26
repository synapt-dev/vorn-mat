# Qwen 3 30B-A3B vanilla baseline with enable_thinking=False

**Artifact**: `results/qwen3-30b-a3b-no-thinking-vanilla-2026-05-26.json`
**Date**: 2026-05-26
**Model**: Qwen/Qwen3-30B-A3B (MoE; ~3B active)
**Slice**: niah_multikey_1_4k validation[:50]
**Cost**: $0.3099
**Wall-clock**: 446.6s
**Modal app**: ap-vw7ZQZ4BZiqN2Lj7hPQVes on layne-79000

## Result

**Hit rate: 1.00** (50/50). Wilson 95% CI: [0.9286, 1.0000].

## Interpretation

Qwen 3 30B-A3B is no-eviction-discriminative when `enable_thinking=False` is applied — same recovery pattern as Qwen 3 8B → Qwen 3-NT 8B (also 0.00 → 1.00 with the same fix). The default-thinking drowning-floor in earlier panel data was **configuration-bounded**, not capability/architecture-bounded.

## Scope

Diagnostic-only — vanilla baseline only. Full eviction-cell coverage (token + sentence × vorn / TOVA / H2O at b=1024) is deferred to v2 per current revision-scope. Paper central claims unchanged this revision; this artifact is documented in Appendix C as recovery-pattern extension to the MoE family.

## Cascade decision

Per dispatch cascade rule (hit-rate ≥ 0.50 = discriminative → STOP):
- 1.00 is well above the 0.50 threshold
- Eviction cells NOT dispatched
- Total Qwen 3 30B-A3B investigation cost: $0.3099

## Reproducibility

- Harness: vorn-mat main at `b0223af` (bf16-stability fix)
- Chat-template kwarg `enable_thinking=False` auto-applied via `_chat_template_kwargs()` substring match `"qwen3" in model_id.lower()` at `vorn_mat/local_exec.py:292-294`
- A100-80GB on `layne-79000` profile
- Per-fixture observations preserved in artifact
