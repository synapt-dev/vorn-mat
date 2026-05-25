# B=256 vorn verification - 2026-05-25

Current-substrate verification run for Layne's two low-budget vorn questions:
Mistral 7B v0.3 sentence-vorn at B=256 and Llama 3.1 8B token-vorn at B=256.

Surface: `niah_multikey_1_4k`, `validation[:50]`, `max_new_tokens=32`,
prefix+recent guardrails, bf16, A100-80GB, memory telemetry default-on.

Raw artifacts: `eval-results/vorn-mat/b256-vorn-verification-2026-05-25` in
`synapt/vorn-mat-cross-family-results`.

| Family | Budget | Method | Status | Hits | Hit rate | Peak alloc MB | Cost |
|---|---:|---|---|---:|---:|---:|---:|
| Mistral 7B v0.3 | 256 | vorn | completed | 45/50 | 0.90 | 20437.1 | $0.1778 |
| Mistral 7B v0.3 | 256 | sentence_vorn | completed | 28/50 | 0.56 | 20437.1 | $0.1442 |
| Llama 3.1 8B | 256 | vorn | completed | 27/50 | 0.54 | 20984.1 | $0.1638 |
| Llama 3.1 8B | 256 | sentence_vorn | completed | 50/50 | 1.00 | 20984.1 | $0.1406 |

All four cells completed at `n=50`; no OOMs or recovery chunks were needed.
Total estimated Modal cost was `$0.6263`.

Read:

- Mistral sentence-vorn remains below token-vorn at B=256 under the current
  patched substrate. The counterintuitive low-budget sentence regression is
  therefore not explained by stale harness drift.
- Llama token-vorn remains near the prior low-budget value while sentence-vorn
  remains ceiling-preserving at B=256.
