# Pilot D Vorn-High Smoking-Gun Trace

Pilot D asks whether vorn scoring identifies structural essentials without an
external protected set. This trace cross-references the `vorn_high` evictions
against the committed 8k Phase 0 SEMU source:

- Phase 0 source: `results/vorn-active-eviction-phase0-8k-case0-1-2026-06-04.json`
- Result JSONL: `results/vorn-active-eviction-pilot-d-8k-no-guards-pressure-sweep-2026-06-04.jsonl`
- Case: `niah_multikey_1_8k-1`
- Answer: `4374754`

## Structural SEMUs

| SEMU id | Label | Text preview |
|---:|---|---|
| 0 | setup | `<s>[INST] A special magic number is hidden within the following text.` |
| 1 | setup | `Make sure to memorize it.` |
| 2 | setup | `I will quiz you about the number afterwards.` |
| 298 | needle / answer-bearing | `One of the special magic numbers for faint-smolt is: 4374754.` |
| 421 | final question | `What is the special magic number for faint-smolt mentioned in the provided text?` |
| 422 | answer-prefix | `The special magic number for faint-smolt mentioned in the provided text is[/INST]` |

## Vorn-High Eviction Trace

| N | Hit | Delta | vorn_high SEMUs | Structural SEMUs Evicted |
|---:|---|---:|---|---|
| 1 | true | 0.0 | `422` | `422` answer-prefix |
| 3 | false | 1.0 | `422,421,75` | `422` answer-prefix; `421` final question |
| 5 | false | 1.0 | `422,421,75,298,152` | `422` answer-prefix; `421` final question; `298` needle / answer-bearing |
| 10 | false | 1.0 | `422,421,75,298,152,161,88,23,3,25` | `422` answer-prefix; `421` final question; `298` needle / answer-bearing |
| 20 | false | 1.0 | `422,421,75,298,152,161,88,23,3,25,2,0,338,69,36,1,333,334,184,14` | `422` answer-prefix; `421` final question; `298` needle / answer-bearing; `2,0,1` setup |
| 50 | false | 1.0 | `422,421,75,298,152,161,88,23,3,25,2,0,338,69,36,1,333,334,184,14,24,57,13,345,342,123,46,346,160,76,344,299,300,81,51,162,84,71,181,272,354,66,172,7,370,6,110,169,156,351` | `422` answer-prefix; `421` final question; `298` needle / answer-bearing; `2,0,1` setup |

## Interpretation

The discriminative transition is between N=1 and N=3:

- N=1 drops only SEMU `422`, the answer-prefix. The model still answers
  correctly (`4374754`).
- N=3 drops SEMU `422`, SEMU `421` (the final question), and SEMU `75` (an
  unrelated context needle for `massive-creation`). The model fails and emits
  surrounding context prose instead of the requested number.
- N=5 adds SEMU `298`, the actual needle / answer-bearing SEMU for `faint-smolt`,
  and continues failing.

This does not prove whether SEMU `421` alone is sufficient to cause the N=3
failure, because N=3 also removes SEMU `75`. It does show that vorn ranks the
final question and answer-prefix as the highest structural units without
external guards, and that the first failure appears exactly when the final
question enters the dropped set.

The N=5 row strengthens the substrate interpretation: vorn's top five include
the answer-prefix, final question, and answer-bearing needle. That is the
operationally complete selector shape this probe was designed to test.
