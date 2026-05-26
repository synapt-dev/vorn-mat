# Qwen 2.5 b=1024 bf16 verification - 2026-05-26

Independent Atlas rerun of Apollo's Qwen 2.5 7B Instruct post-bf16-fix values on `niah_multikey_1_4k`, `validation[:50]`, B=1024, max_new_tokens=32, bf16, A100-80GB, prefix+recent guardrails.

Harness commit: `b0223afa2c6cda598321ca1f6dd545c88211ce66`. Modal apps: `ap-weGRh7NXBlqR8L2OmiF1Fj` (vanilla), `ap-uaYBI0wmCZb2tWdxyJLP9b` (eviction wave). Total estimated cost: `$1.5714`.

| Cell | Atlas | Apollo | Delta | Verdict |
|---|---:|---:|---:|---|
| Qwen 2.5 x no-eviction x b=1024 | 32/50 (0.64) | 32/50 (0.64) | 0/50 | AGREE |
| Qwen 2.5 x token-vorn x b=1024 | 22/50 (0.44) | 22/50 (0.44) | 0/50 | AGREE |
| Qwen 2.5 x sentence-vorn x b=1024 | 38/50 (0.76) | 38/50 (0.76) | 0/50 | AGREE |
| Qwen 2.5 x token-TOVA x b=1024 | 28/50 (0.56) | 28/50 (0.56) | 0/50 | AGREE |
| Qwen 2.5 x sentence-TOVA x b=1024 | 35/50 (0.70) | 35/50 (0.70) | 0/50 | AGREE |
| Qwen 2.5 x token-H2O x b=1024 | 30/50 (0.60) | 30/50 (0.60) | 0/50 | AGREE |
| Qwen 2.5 x sentence-H2O x b=1024 | 35/50 (0.70) | 35/50 (0.70) | 0/50 | AGREE |

All cells are exact matches to Apollo (0 fixture delta). This clears the cross-agent verification gate for using the refreshed Qwen 2.5 values in the paper revision.

Raw report paths before HF sync: `.benchmarks/qwen25-b1024-bf16-verification-2026-05-26/vanilla-report.json` and `.benchmarks/qwen25-b1024-bf16-verification-2026-05-26/live-eviction/reports.json`.
