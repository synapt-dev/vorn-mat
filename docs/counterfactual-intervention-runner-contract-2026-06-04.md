# Counterfactual Intervention Runner Contract

Date: 2026-06-04

This note defines the Pilot A runner substrate needed before any
vorn-active-eviction Modal cells are authorized. The implementation surface in
`src/vorn_mat/counterfactual_intervention_runner.py` is intentionally a
contract and skeleton: it selects SEMUs, renders counterfactual prompts, and
serializes auditable run records. It does not call a model, launch Modal, or
claim fresh quality results.

## Scope

- Input substrate: Phase 0 SEMU extraction artifacts and the run-ready manifest
  locked in config.
- Output substrate: one JSONL-style `CounterfactualRunRecord` per attempted
  targeted-drop generation.
- Granularity: sentence SEMUs only for Pilot A.
- Checkpoints: `T0_PRE_GENERATION` and `T1_CONTINUATION`.
- Selector arms: `vorn_high`, `vorn_low`, `attention_high`, `recency_high`,
  `length_high`, and `random_length_matched`.
- Deletion modes: `delete` for primary Pilot A probes and `mask` for optional
  boundary-preserving diagnostics.

The runner contract is separate from the world-and-portal substrate. Future
product-layer verbs such as `evict_semu` or `protect_semu` can consume these
records as measurement evidence, but this package does not implement
authority, Sigtam grants, or MeasurementReceipt semantics.

## Sentence-Drop Policy

The runner builds sentence SEMUs from rendered prompt text and tokenizer
offsets. Each SEMU carries:

- stable `semu_id` assigned after sentence-span extraction;
- original character span and token span;
- original text;
- protected-class labels;
- answer-overlap flag where answer token spans are known.

Prompt interventions are fail-closed:

- Protected SEMUs cannot be selected or rendered for deletion.
- Missing vorn scores fail closed because every eligible SEMU must be ranked.
- `attention_high` fails closed as capacity-missing when attention scores are
  unavailable.
- `random_length_matched` requires a reference SEMU and selects an eligible
  non-reference SEMU with nearest token length. It cannot silently delete the
  same SEMU as the vorn target.

The primary deletion policy removes the exact original character span and
collapses one duplicated gap space if the deletion creates it. The mask policy
replaces the original span with `[MASKED_SEMU]` repeated to the original SEMU
token length. The mask path preserves positional pressure for diagnostics but
is not the Pilot A primary intervention unless the preregistration manifest is
updated before cells run.

## Determinism Guarantees

The contract pins deterministic fields before model execution:

- `random_seed = sha256(run_id|case_id|selector_arm|model_id)[:8]`
- original prompt hash and counterfactual prompt hash are SHA-256 over UTF-8
  text;
- selector tie-breaks are deterministic;
- JSON payloads serialize with sorted keys.

These guarantees are sigtam-style at the research-substrate layer: the record
does not prove authority, but it binds the intervention view to the exact
prompt, SEMU, selector arm, and model identifiers that produced it.

## Artifact Schema

`CounterfactualRunRecord` contains:

- `schema_version`: currently `vorn-active-eviction-counterfactual/v1`;
- `record_status`: one of `PROMPT_QUALITY_SUCCESS`, `RUNTIME_FAILURE`, or
  `CAPACITY_MISSING`;
- `intervention`: family, model revisions, case id, checkpoint, selector arm,
  SEMU id, original spans, score/rank, deletion mode, deterministic seed;
- `prompt_record`: original/counterfactual hashes, deletion/mask policy,
  optional token counts, and sentence-alignment audit;
- `quality_record`: full-context prediction, counterfactual prediction,
  primary metric, binary hit, delta quality, runtime, cost, and optional memory
  telemetry;
- `failure`: capacity-missing or runner-failure record with hardware, Modal
  profile, retry count, and partial artifact path.

The dataclass enforces terminal-path invariants. Successful runs carry
`record_status=PROMPT_QUALITY_SUCCESS`, `prompt_record`, and `quality_record`
with no `failure`. Runtime failures carry `record_status=RUNTIME_FAILURE`,
`failure`, no `quality_record`, and may preserve `prompt_record` when prompt
rendering succeeded before model execution failed. Capacity-missing records
carry `record_status=CAPACITY_MISSING`, `failure.capacity_missing_class`, and no
`quality_record`. An all-`None` terminal record is invalid by construction.

## Implementation Still Required

Before Pilot A cells can run, the next implementation layer must add:

- loader from Phase 0 SEMU/ranking artifacts into this contract;
- full-context and counterfactual model execution;
- scorer integration matching the preregistered primary metric;
- Modal entrypoint using the locked profile and hardware tier;
- JSONL writer to `results/` plus HF upload integration;
- consumer-validation smoke fixture proving one end-to-end targeted-drop record
  before the T0 primary matrix starts.

Until those pieces exist and the run-ready manifest is locked, this contract
does not authorize Modal execution.
