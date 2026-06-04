# Pilot D 8k No-Guards Selector Derivation

Pilot D reruns the Pilot C 8k pressure sweep with an empty protected set. This
tests whether vorn's sentence-SEMU scoring ranks structural essentials highly
without external protection scaffolding.

## Source Artifact

- Phase 0 source: `results/vorn-active-eviction-phase0-8k-case0-1-2026-06-04.json`
- Result JSONL: `results/vorn-active-eviction-pilot-d-8k-no-guards-pressure-sweep-2026-06-04.jsonl`
- Dataset config: `niah_multikey_1_8k`
- Case: `niah_multikey_1_8k-1`
- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- SEMU granularity: sentence
- Protected SEMUs: `{}`

## Structural SEMUs In The 8k Fixture

- Setup/instruction SEMUs: `0`, `1`, `2`
- Answer-bearing SEMU: `298`
- Final question SEMU: `421`
- Answer-prefix SEMU: `422`

## No-Guards Vorn-High Signal

With no protected set, `vorn_high` immediately selects structurally essential
SEMU IDs:

| Pressure N | vorn_high SEMU ids | Structural overlap |
|---:|---|---|
| 1 | `422` | answer-prefix |
| 3 | `422,421,75` | question + answer-prefix |
| 5 | `422,421,75,298,152` | question + answer-prefix + answer-bearing |
| 10 | `422,421,75,298,152,161,88,23,3,25` | question + answer-prefix + answer-bearing |
| 20 | `422,421,75,298,152,161,88,23,3,25,2,0,338,69,36,1,333,334,184,14` | setup + question + answer-prefix + answer-bearing |
| 50 | `422,421,75,298,152,161,88,23,3,25,2,0,338,69,36,1,333,334,184,14,24,57,13,345,342,123,46,346,160,76,344,299,300,81,51,162,84,71,181,272,354,66,172,7,370,6,110,169,156,351` | setup + question + answer-prefix + answer-bearing |

This is the load-bearing selector signal: the first answer loss appears at
`vorn_high` N=3, after the selector removes both the final question and answer
prefix. The answer-bearing SEMU enters the dropped set at N=5.

## High-N Arm Selections

At N=50, the arms evicted these SEMU IDs:

- `vorn_high`: `422,421,75,298,152,161,88,23,3,25,2,0,338,69,36,1,333,334,184,14,24,57,13,345,342,123,46,346,160,76,344,299,300,81,51,162,84,71,181,272,354,66,172,7,370,6,110,169,156,351`
- `vorn_low`: `200,149,210,199,143,43,201,234,134,232,212,142,205,244,216,388,140,198,135,316,213,269,211,284,133,385,256,245,148,41,417,27,282,415,255,414,400,139,145,126,118,376,372,68,209,238,379,125,243,127`
- `length_high`: `342,357,123,221,226,45,399,179,23,49,203,354,392,48,178,228,88,359,3,19,64,208,278,410,173,326,257,271,350,351,366,69,86,89,222,275,320,321,420,79,101,297,128,353,57,84,106,110,119,180`
- `random_length_matched`: `91,211,290,26,72,263,359,392,64,27,142,317,108,321,60,417,106,292,215,269,262,180,138,158,357,221,159,287,205,30,194,390,303,207,61,289,119,38,382,379,203,267,87,4,276,398,353,80,240,366`

Only `vorn_high` removes the explicit setup/question/answer-bearing SEMUs at
high N. The other arms preserve the answer through N=50 in this fixture.

## Arm-Overlap Audit

Overlap appears at larger pressure levels. This is a selector-correlation signal,
not a protected-set failure, because Pilot D intentionally has no protected set.

- N=10: `vorn_high` ∩ `length_high` = `23`; `length_high` ∩ `random_length_matched` = `49`
- N=20: `vorn_high` ∩ `length_high` = `3,23,88`; `vorn_low` ∩ `random_length_matched` = `143,212`; `length_high` ∩ `random_length_matched` = `19,359,392`
- N=50: `vorn_high` ∩ `length_high` = `3,23,57,69,84,88,110,123,342,351,354`; `vorn_low` ∩ `random_length_matched` = `27,142,205,211,269,379,417`; `length_high` ∩ `random_length_matched` = `64,106,119,180,203,221,321,353,357,359,366,392`

## Replay Command

Run from the vorn-mat repo root:

```bash
PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path
from vorn_mat.active_eviction_consumer_validation import (
    build_semus_and_scores_from_phase0,
    load_phase0_case,
)
from vorn_mat.active_eviction_pressure_sweep import select_semus_for_arm

phase0_artifact = Path("results/vorn-active-eviction-phase0-8k-case0-1-2026-06-04.json")
result_jsonl = Path("results/vorn-active-eviction-pilot-d-8k-no-guards-pressure-sweep-2026-06-04.jsonl")
case_id = "niah_multikey_1_8k-1"
run_id = "vorn-active-eviction-pilot-d-8k-no-guards-pressure-sweep-2026-06-04"
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
seed = 534588164691762844

phase0_case = load_phase0_case(phase0_artifact, case_id)
semus, scores = build_semus_and_scores_from_phase0(
    phase0_case,
    protected_semu_ids=(),
)

expected = {}
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
    expected[("vorn_high", pressure_n)] = tuple(semu.semu_id for semu in vorn_high)
    for arm in ("vorn_low", "length_high", "random_length_matched"):
        selected = select_semus_for_arm(
            selector_arm=arm,
            pressure_n=pressure_n,
            semus=semus,
            scores=scores,
            run_id=run_id,
            case_id=case_id,
            model_id=model_id,
            base_seed=seed,
            reference_semus=vorn_high if arm == "random_length_matched" else None,
        )
        expected[(arm, pressure_n)] = tuple(semu.semu_id for semu in selected)

observed = {}
for line in result_jsonl.read_text().splitlines():
    row = json.loads(line)
    if row.get("record_type") == "FULL_CONTEXT_CONTROL":
        continue
    intervention = row["intervention"]
    observed[(intervention["selector_arm"], intervention["pressure_n"])] = tuple(
        intervention["selected_semu_ids"]
    )

assert expected == observed
print("selector_replay_ok rows=", len(expected))
PY
```
