# Qwen 3 no-thinking b=1024 attention composition (2026-05-26)

Composed aggregate for Qwen 3 8B no-thinking attention rows at `B=1024` on `rbiswasfc/ruler`, `niah_multikey_1_4k`, `validation[:50]`.

| Method | Status | Hits | Hit rate | Source chunks |
|---|---:|---:|---:|---|
| `tova` | `completed_recovered` | 5/50 | 0.10 | 0+34 (2 hits), 34+16 (3 hits) |
| `h2o` | `completed_recovered` | 5/50 | 0.10 | 0+34 (2 hits), 34+16 (3 hits) |
| `sentence_tova` | `completed_recovered` | 50/50 | 1.00 | 0+34 (34 hits), 34+16 (16 hits) |
| `sentence_h2o` | `completed_recovered` | 50/50 | 1.00 | 0+34 (34 hits), 34+16 (16 hits) |

The first-34 ledgers were recovered from Modal volume `synapt-vorn-mat-vol:/results/vorn-mat/` on 2026-05-26. The offset-34 back-16 reports were already represented in the HF dataset under `eval-results/vorn-mat/qwen3-no-thinking-4k-b1024-back16-2026-05-25`; this artifact embeds all 50 observations per row so the composition is durable in-repo.

Premium boundary: pure OSS benchmark data artifact.
