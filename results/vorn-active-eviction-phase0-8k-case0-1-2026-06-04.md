# Vorn-Active Eviction Phase 0 SEMU Trajectory Findings

Date: 2026-06-03

## Scope

This is an offline Phase 0 analysis over existing vorn-mat observation artifacts. It does not run fresh generations and does not estimate counterfactual SEMU contribution.

Current trajectory-bearing coverage is narrower than the 21,600 method-level evaluation observations in the public corpus. The committed run analyzes the `vanilla-observation-2026-05-13` sharded observation report, which contains Mistral `niah_multikey_1_4k` generation traces with token-level vorn alignment scores.

## Corpus

- Observation report: `.benchmarks/vanilla-observation-8k-case0-1-2026-06-04.json`
- Dataset config: `niah_multikey_1_8k`
- Split: `validation[:2]`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Cases: 2 (2 success / 0 failure)
- Sentence SEMUs per case, mean: 423.0
- Steps per case, mean: 9.5
- Matrix artifact: `.benchmarks/vorn-active-eviction-phase0-8k-case0-1-2026-06-04.json`

## Findings

### F1. Existing trajectory data is targeted, not corpus-wide

The 21,600 public observations remain method-level fixture outcomes. They do not carry per-token or per-SEMU trajectory traces. The available SEMU trajectory mining path is through targeted observation artifacts with positional score arrays, currently the sharded vanilla observation report. The 8k score-distribution observation runs carry per-step distribution summaries but not the positional score arrays needed for SEMU ranking extraction.

### F1b. Additional SEMU-bearing substrate is captured as bounded evidence

This artifact now indexes the SEMU-bearing substrate available without fresh compute:

- Positional-score sources: 10 JSON/JSON.GZ artifacts with `alignment_scores` arrays, currently the vanilla observation shards.
- Neighborhood probe: `results/vanilla-observation-neighborhood-2026-05-13.json` (6 probe families) — answer-neighborhood proxy aggregates.
- Score-distribution probe: `results/score-distribution-observation-8k-2026-05-14.json` (4 budget runs) — token/word/sentence distribution summaries.
- Method-level semantic-granularity inventory: 52 repository-stable JSON/JSON.GZ artifacts with sentence/word method rows. Transient `.benchmarks` files are excluded by default because their counts drift with local run state and require an exact manifest/checksum opt-in.

Interpretation: these sources are useful for stratification and substrate inventory, but only the positional-score observation report supports sentence-SEMU ranking trajectories.

### F1c. Existing telemetry coverage is heterogeneous

The SEMU-bearing substrate also differs by telemetry shape:

- Positional-score sources: 10 artifacts with per-step positional score arrays, answer spans, top-alignment positions, and ranking-stability fields.
- Method-level semantic inventory: 52 artifacts; memory telemetry in 9, cost telemetry in 49, runtime telemetry in 49, retention telemetry in 41, outcome metrics in 47.
- Counterfactual SEMU quality labels in method-level rows: 0.
- Positional score arrays in method-level rows: 0.

Interpretation: existing artifacts are enough to audit SEMU-granularity coverage and some runner telemetry, but a Phase 1+ probe must instrument score trajectories, deletion labels, decision-event markers, and telemetry consistently in the same records.

### F2. Sentence-level vorn scores vary, but rankings are mostly stable

- Mean per-case mean SEMU score range: 0.018517
- Median per-case mean SEMU score range: 0.018517
- P90 per-case mean SEMU score range: 0.019612
- Mean top-1 SEMU switches per case: 0.0
- Cases with at least one top-1 SEMU switch: 0/2
- Mean top-5 Jaccard across adjacent steps: 0.958333

Interpretation: score values are not static snapshots, but rank order is mostly stable in this targeted Mistral NIAH observation corpus. That is a narrower finding than the strongest temporal/Jenga hypothesis. It supports recording trajectories, but suggests tool-result-before/after-decision fixtures are the better test surface for true temporal decay.

### F3. Answer-bearing SEMUs are recoverable as a ranking probe

- Cases with answer-overlapping sentence SEMU: 2/2
- Mean answer SEMU first-step rank: 3.0
- Mean answer SEMU final-step rank: 3.5
- Mean answer SEMU best rank: 3.0
- Median answer SEMU best rank: 3.0
- Answer SEMU reached top-10 at least once: 2/2
- Answer SEMU reached top-20 at least once: 2/2

Interpretation: answer-span overlap gives a useful proxy probe for Phase 0 ranking analysis, but it is still a proxy. Counterfactual deletion is required to label a SEMU as load-bearing.

### F4. Cross-family SEMU trajectory comparison is not available yet

The current trajectory-bearing observation artifacts are not seven-family. Cross-family active-eviction claims therefore require fresh instrumentation or new observation runs on the same family panel. Existing seven-family rows can select strata and families, but cannot answer whether families have different SEMU trajectory shapes.

## Implications for the design doc

- Keep Phase 0 framed as proxy signal mining.
- Keep Pilot A as signal-detection, not threshold calibration.
- Preserve sentence-level SEMU as the first intervention granularity.
- Require fresh counterfactual runs for causal contribution labels.
- Require fresh cross-family observation instrumentation before claiming family-conditional SEMU trajectories.
- Treat method-level sentence/word rows as SEMU-granularity outcome evidence, not per-SEMU contribution labels.

## Honest negatives

- This analysis only sees generation-time vorn movement on targeted Mistral observation traces.
- It does not include tool-result-before/after-decision workflows.
- It does not establish that low-vorn SEMUs are safe to drop.
- It does not compare seven families at trajectory level.
- Distribution-only and method-level SEMU artifacts do not expose positional score arrays.
- Sentence segmentation is a practical first granularity, not proven optimal.
