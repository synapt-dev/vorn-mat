# Vanilla Observation Neighborhood Probe — 2026-05-13

## Run

- Dataset config: `niah_multikey_1_4k`
- Split: `validation[:50]`
- Cases: `50`
- Top-K tracked: `10`

## Probe comparison

### `exact_answer`

- Success top-K hit: `0/14` (0.000); failure top-K hit: `0/36` (0.000).
- Mean first hit step: success `n/a`, failure `n/a`.
- Mean best rank: success `36.2857`, failure `33.0833`.
- Mean final alignment gap: success `-0.3409`, failure `-0.3504`.

### `answer_window_5`

- Success top-K hit: `0/14` (0.000); failure top-K hit: `0/36` (0.000).
- Mean first hit step: success `n/a`, failure `n/a`.
- Mean best rank: success `33.0714`, failure `31.2778`.
- Mean final alignment gap: success `-0.3396`, failure `-0.3456`.
- Note: answer span expanded by +/-5 tokens.

### `answer_window_10`

- Success top-K hit: `0/14` (0.000); failure top-K hit: `0/36` (0.000).
- Mean first hit step: success `n/a`, failure `n/a`.
- Mean best rank: success `31.6429`, failure `30.9722`.
- Mean final alignment gap: success `-0.3346`, failure `-0.3435`.
- Note: answer span expanded by +/-10 tokens.

### `sentence`

- Success top-K hit: `0/14` (0.000); failure top-K hit: `0/36` (0.000).
- Mean first hit step: success `n/a`, failure `n/a`.
- Mean best rank: success `32.7857`, failure `30.9722`.
- Mean final alignment gap: success `-0.3396`, failure `-0.3420`.

### `line`

- Success top-K hit: `0/14` (0.000); failure top-K hit: `0/36` (0.000).
- Mean first hit step: success `n/a`, failure `n/a`.
- Mean best rank: success `23.9286`, failure `23.4167`.
- Mean final alignment gap: success `-0.3101`, failure `-0.3075`.

### `paragraph`

- Success top-K hit: `14/14` (1.000); failure top-K hit: `36/36` (1.000).
- Mean first hit step: success `0.0000`, failure `0.0000`.
- Mean best rank: success `1.0000`, failure `1.0000`.
- Mean final alignment gap: success `0.0167`, failure `-0.0251`.
- Degenerate cases: `50` (probe collapsed to nearly the whole prompt; not discriminative).
- Note: single-paragraph prompt serialization makes the paragraph probe degenerate.

## Interpretation boundary

- This is offline re-processing of the existing vanilla observation traces. No new Modal inference was run.
- The probe asks whether vorn-aligned positions cluster around wider neighborhoods of the needle even when they do not land on the exact answer tokens.

## Interpretation

- Widening the probe from the exact answer to `answer_window_10` improves the success-case mean best rank from `36.2857` to `31.6429`, but still leaves the needle neighborhood well outside the tracked top-10.
- The full sentence probe behaves similarly (`32.7857` success-case mean best rank) and also never enters top-10 on any successful case.
- The coarse-granularity rescue story is therefore weak on this slice: the neighborhood probes are only modestly less bad than exact answer tokens, not qualitatively different.
- Success/failure separation remains weak. On these probes, successful cases do not rank the needle neighborhood materially better than failures, so the current mechanism story likely depends on a different structure than direct needle-span alignment.