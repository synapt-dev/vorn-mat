# Pilot C 8k Selector Derivation

This note records the reproducibility trace for the Pilot C 8k pressure-sweep
selector inputs. It closes the gap between the committed pressure-sweep result
rows and the SEMU source used to derive selector arms.

## Source Artifact

- Phase 0 source: `results/vorn-active-eviction-phase0-8k-case0-1-2026-06-04.json`
- Summary: `results/vorn-active-eviction-phase0-8k-case0-1-2026-06-04.md`
- Dataset config: `niah_multikey_1_8k`
- Case: `niah_multikey_1_8k-1`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- SEMU granularity: sentence

`niah_multikey_1_16k` was not available in the canonical RULER HF mirror.
Pilot C therefore used `niah_multikey_1_8k-1` as the nearest canonical same-
structure context-length fallback.

## Protected SEMU Derivation

The 8k fixture protected set was re-derived from the 8k SEMU layout:

- `0`, `1`, `2`: instruction/setup SEMUs at the beginning of the rendered prompt
- `298`: answer-bearing SEMU for `4374754`
- `421`, `422`: final question and answer-prefix SEMUs

Protected SEMUs used by the run: `{0,1,2,298,421,422}`.

## N=1 Selector Smoke IDs

These are the first-row selector outputs for the pressure sweep. They are
included here because they are the easiest way to detect accidental reuse of
the prior 4k fixture positions.

| Arm | N=1 SEMU id |
|---|---:|
| `vorn_high` | 75 |
| `vorn_low` | 200 |
| `length_high` | 342 |
| `random_length_matched` | 109 |

## Replay Command

Run from the vorn-mat repo root:

```bash
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from vorn_mat.active_eviction_consumer_validation import (
    build_semus_and_scores_from_phase0,
    load_phase0_case,
)
from vorn_mat.active_eviction_pressure_sweep import select_semus_for_arm

phase0_artifact = Path("results/vorn-active-eviction-phase0-8k-case0-1-2026-06-04.json")
case_id = "niah_multikey_1_8k-1"
protected = (0, 1, 2, 298, 421, 422)
run_id = "vorn-active-eviction-pilot-c-8k-pressure-sweep-2026-06-04"
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
seed = 534588164691762844

phase0_case = load_phase0_case(phase0_artifact, case_id)
semus, scores = build_semus_and_scores_from_phase0(
    phase0_case,
    protected_semu_ids=protected,
)

for pressure_n in (1, 3, 5, 10, 20, 50):
    vorn_high = select_semus_for_arm(
        selector_arm="vorn_high",
        pressure_n=pressure_n,
        semus=semus,
        scores=scores,
        run_id=run_id,
        case_id=case_id,
        model_id=model_id,
        base_seed=seed,
    )
    arms = [("vorn_high", vorn_high)]
    for arm in ("vorn_low", "length_high", "random_length_matched"):
        arms.append(
            (
                arm,
                select_semus_for_arm(
                    selector_arm=arm,
                    pressure_n=pressure_n,
                    semus=semus,
                    scores=scores,
                    run_id=run_id,
                    case_id=case_id,
                    model_id=model_id,
                    base_seed=seed,
                    reference_semus=vorn_high
                    if arm == "random_length_matched"
                    else None,
                ),
            )
        )
    print("N=", pressure_n)
    for arm, selected in arms:
        ids = tuple(semu.semu_id for semu in selected)
        print(" ", arm, ids)
PY
```

The replayed IDs should match the `selected_semu_ids` fields in
`results/vorn-active-eviction-pilot-c-8k-pressure-sweep-2026-06-04.jsonl`.
