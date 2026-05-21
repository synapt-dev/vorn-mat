# Vanilla Observation — 2026-05-13

## Run

- Dataset config: `niah_multikey_1_4k`
- Split: `validation[:50]`
- Cases: `50`
- Wall-clock: `679.37s`
- Estimated cost: `$0.4715`
- Top-K tracked: `10`
- Observation manifest: `vanilla-observation-2026-05-13.json`
- Case shards: `vanilla-observation-2026-05-13-shards/`

## Initial findings

- Success cases with any answer token entering top-10: `0/14` (0.000); failure cases: `0/36` (0.000).
- Mean first answer-in-top-10 step: success `n/a`; failure `n/a`.
- Final-step alignment gap (answer vs strongest non-answer position): success `-0.3409`, failure `-0.5657`.
- Final-step residual-norm gap (answer mean minus non-answer mean): success `-0.8020`, failure `-0.8509`.

## Figures

- [answer-topk-hit-rate-by-step.png](figures/answer-topk-hit-rate-by-step.png)
- [alignment-gap-by-step.png](figures/alignment-gap-by-step.png)
- [ranking-stability-by-step.png](figures/ranking-stability-by-step.png)
- [residual-gap-by-step.png](figures/residual-gap-by-step.png)

## Interpretation boundary

- This run is pure observation under vanilla inference: no eviction, no replay, no cache intervention.
- The patterns here describe what the unmodified model naturally does on this slice. They do not by themselves validate an eviction policy.
