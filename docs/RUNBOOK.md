# vorn-mat Live Eval Runbook

This runbook is the fastest way to verify that the public release repository can run a real Modal-backed evaluation and emit the same artifact shape used in the paper.

All commands below assume you are in the repository root:

```bash
cd /tmp/vorn-mat-release
```

## 1. Confirm authentication and environment

The Modal app runs remotely, so there are two separate auth checks:

1. Your local CLI must be authenticated with Modal.
2. The remote Modal app must have a Hugging Face token available as a Modal secret named `huggingface-secret`.

### Check local auth

```bash
test -n "${HF_TOKEN:-}" && echo "HF_TOKEN is set" || echo "HF_TOKEN is missing"
modal token info
modal secret list | rg 'huggingface-secret'
```

Expected results:

- `HF_TOKEN is set`
- `modal token info` prints the active authenticated account
- `modal secret list` includes `huggingface-secret`

If `huggingface-secret` is missing, create or replace it from your current shell environment:

```bash
modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"
```

If `modal token info` fails, authenticate the CLI first:

```bash
modal token new
```

On current Modal CLI versions, `modal token info` is the supported status check.

## 2. Confirm model access

The release repo uses public Hugging Face model IDs directly. Some of them are gated and require access approval on the Hugging Face account behind `HF_TOKEN`.

### Gated models in the current experiment surface

- Llama 3.1 8B Instruct: <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>
- Gemma 2 9B IT: <https://huggingface.co/google/gemma-2-9b-it>
- Gemma 3 12B PT: <https://huggingface.co/google/gemma-3-12b-pt>
- Gemma 3n E4B IT: <https://huggingface.co/google/gemma-3n-E4B-it>

If your token does not already have access, open the model card and use the access-request flow there before retrying the run.

### Public models in the current experiment surface

As of 2026-05-21, these model cards are public and do not require a gated-access request:

- Ministral 8B Instruct 2410: <https://huggingface.co/mistralai/Ministral-8B-Instruct-2410>
- Mistral 7B Instruct v0.3: <https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3>
- Qwen 2.5 7B Instruct: <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct>
- Qwen 3-NT 8B (Qwen 3 8B with `enable_thinking=False` rendering): <https://huggingface.co/Qwen/Qwen3-8B>
- Qwen 3 30B-A3B: <https://huggingface.co/Qwen/Qwen3-30B-A3B>

If you substitute a different model or revision, re-check the current model card before assuming access behavior is unchanged.

## 3. One-shot smoke test

Use this first. It is cheap, fast, and produces a real JSON report on the current runner.

This command runs a 5-case live-eval smoke on the Llama 3.1 sentence-level cell at budget `1024`:

```bash
modal run examples/run_modal_live_eviction.py \
  --dataset-config niah_multikey_1_4k \
  --split validation \
  --case-limit 5 \
  --cache-budget-tokens 1024 \
  --retention-policy sentence_vorn \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --output .benchmarks/llama31-niah-smoke-sentence-1024-n5.json
```

Expected budget:

- cost: about `$0.02`
- wall clock: about `3-5 minutes`

What success looks like:

- the command exits cleanly
- it writes `.benchmarks/llama31-niah-smoke-sentence-1024-n5.json`
- stdout includes lines like:
  - `run_id=...`
  - `metrics={...}`
  - `elapsed_seconds=...`
  - `estimated_cost_usd=...`

Quick sanity check:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path(".benchmarks/llama31-niah-smoke-sentence-1024-n5.json")
data = json.loads(path.read_text())
print("run_id:", data["result"]["run_id"])
print("metrics:", data["result"]["metrics"])
print("estimated_cost_usd:", data["estimated_cost_usd"])
PY
```

## 4. Reproduce a paper headline cell

The simplest headline reproduction surface in this repo is the Llama 3.1 `qa_2_4k` within-vorn comparison at `b=1024`. That is the cross-task anchor used for the `4.04x` cost-per-correct comparison.

To reproduce it from scratch, run three rows:

### 4.1 Llama 3.1 vanilla row

```bash
modal run examples/run_modal_vanilla.py \
  --dataset-config qa_2_4k \
  --split validation \
  --case-limit 200 \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --output .benchmarks/cross-model/llama31-qa2-vanilla-n200.json
```

### 4.2 Llama 3.1 token-vorn row

```bash
modal run examples/run_modal_live_eviction.py \
  --dataset-config qa_2_4k \
  --split validation \
  --case-limit 200 \
  --cache-budget-tokens 1024 \
  --retention-policy vorn \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --output .benchmarks/cross-model/llama31-qa2-token-1024-guarded-n200.json
```

### 4.3 Llama 3.1 sentence-vorn row

```bash
modal run examples/run_modal_live_eviction.py \
  --dataset-config qa_2_4k \
  --split validation \
  --case-limit 200 \
  --cache-budget-tokens 1024 \
  --retention-policy sentence_vorn \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --output .benchmarks/cross-model/llama31-qa2-sentence-1024-guarded-n200.json
```

Then rebuild the paired artifact:

```bash
python examples/build_token_vorn_qa2_cross_task_artifact.py
```

The rebuilt artifact will be written to:

- `results/token-vorn-qa2-cross-task-2026-05-20.json`
- `results/token-vorn-qa2-cross-task-2026-05-20.md`

### Where to look in the artifact

The JSON contains:

- `rows[]` with per-row hit rate, cost, and per-fixture observations
- `pairwise_tests[]` with exact McNemar results

The paired comparison you want is:

- `lhs = "llama31_sentence_1024"`
- `rhs = "llama31_token_1024"`

Example extraction:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path("results/token-vorn-qa2-cross-task-2026-05-20.json")
data = json.loads(path.read_text())
pair = next(
    item for item in data["pairwise_tests"]
    if item["lhs"] == "llama31_sentence_1024" and item["rhs"] == "llama31_token_1024"
)
rows = {row["label"]: row for row in data["rows"]}
sentence = rows["llama31_sentence_1024"]
token = rows["llama31_token_1024"]
print("sentence hit_rate:", sentence["hit_rate"])
print("sentence cost:", sentence["estimated_cost_usd"])
print("token hit_rate:", token["hit_rate"])
print("token cost:", token["estimated_cost_usd"])
print("paired p-value:", pair["p_value"])
PY
```

On the release artifact, the expected surface is:

- sentence-vorn: `92/200` correct at `$0.2976`
- token-vorn: `52/200` correct at `$0.6778`
- paired McNemar `p = 1.0325821975243343e-08`

## 5. Common failure modes

### `403` or gated-access error when loading a model

Symptoms:

- the Modal run fails while fetching model weights
- logs mention access denied, gated repo, or license acceptance

Checks:

- confirm `HF_TOKEN` is set locally
- confirm `huggingface-secret` exists in Modal
- open the model card and confirm the account behind `HF_TOKEN` has access

Recovery:

- request access on the model card if needed
- recreate the Modal secret after changing tokens:

```bash
modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"
```

### `modal token info` fails or Modal reports you are not authenticated

Symptoms:

- `modal token info` errors
- `modal run ...` fails before job submission

Recovery:

```bash
modal token new
```

Then rerun `modal token info`.

### Modal quota or spending limit exceeded

Symptoms:

- submission fails before execution
- Modal reports quota exhaustion or spending-limit issues

Recovery:

- use the 5-case smoke test first
- keep verification runs to single cells
- avoid multi-budget sweeps and `n=200` baselines unless you are reproducing a specific headline artifact

### CUDA OOM on higher-budget cells

Symptoms:

- the job starts and then fails with CUDA out-of-memory

Typical cause:

- large-budget attention-heavy runs, especially at later budgets or on larger models

Recovery:

- lower `--cache-budget-tokens`
- reduce `--case-limit`
- switch to a smaller verification cell
- if you are only checking the pipeline, prefer the `niah_multikey_1_4k` smoke cell above

Do not use expensive late-budget sweeps as a first verification step.

### Local package mismatch, especially `transformers`

Symptoms:

- builder or local-run code fails because of dependency drift

Recovery:

- use `modal run` for live evaluation instead of trying to validate the live cell locally
- keep local post-processing to the repo’s builder scripts
- if you must run local code beyond the builders, match the repo’s Modal image expectations, including `transformers>=4.44.0`

## Verification budget guidance

For a fast verification pass:

- start with the 5-case smoke test
- then run one headline reproduction surface
- avoid any single cell expected to cost more than about `$1`

That is enough to verify:

- auth works
- the Modal app can fetch the model
- the runner emits the expected JSON structure
- the paired artifact rebuild path works end to end
