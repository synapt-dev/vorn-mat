# B=1536/B=2048 Mistral/Llama/Gemma 4 Fill — 2026-05-26

Public mirror of the 2026-05-26 late-budget fill artifact originally landed in the companion site data surface.

## Scope

- Mistral 7B v0.3: token-H2O and sentence-H2O at B=1536 and B=2048.
- Llama 3.1 8B: token-H2O and sentence-H2O at B=1536 and B=2048.
- Gemma 4 E4B-it: token-vorn and token-H2O at B=1536 and B=2048.

All rows use the RULER `niah_multikey_1_4k` validation slice with n=50.

## Recovery Note

The Mistral B=1536 H2O rows are marked `completed_recovered`: the original single-shot run OOMed early, so the cell was composed from 3-case micro-recovery chunks plus one-case fills where needed. This mirrors the recovery-status taxonomy used by the site playground.

## Provenance

- Source artifact: `site/scripts/artifacts/2026-05-26/b1536-b2048-mistral-llama-gemma4-fill.json`
- Public result artifact: `results/b1536-b2048-mistral-llama-gemma4-fill-2026-05-26.json`
- Premium boundary: pure OSS data refresh.
