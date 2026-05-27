# Vorn-Mat

Reference implementation and result artifacts for the paper *Vorn-Mat: Family-Conditional KV-Cache Eviction and the Granularity Rescue Spectrum*.

This repository contains the prototype source, all released result artifacts, supplementary analysis scripts, and a runbook for reproducing the headline cells. The companion HuggingFace dataset at [`synapt/vorn-mat-cross-family-results`](https://huggingface.co/datasets/synapt/vorn-mat-cross-family-results) mirrors the result artifacts for citeable, discoverable access independent of this code repository.

## What's in this repository

- **`src/vorn_mat/`**: prototype source.
  - `vorn.py`: residual-direction scoring at the canonical mid-depth layer (`L* = L // 2`) under a prefill-time cache selection contract with full-prompt visibility.
  - `baselines/live_eviction.py`: token-level and sentence-level retention policies, plus TOVA-style and H2O-style attention-weight baselines under the same one-shot prefill contract.
  - `plan.py`, `runner.py`, `remote_exec.py`: experiment plan dataclasses, result-envelope JSON schemas, and Modal job dispatch.
  - `paired_stats.py`: exact paired McNemar tests over per-fixture observations preserved in each result row.
- **`results/`**: 67 released JSON artifacts (each paired with a Markdown summary) covering the seven-family active claim panel (Mistral 7B v0.3, Llama 3.1 8B, Ministral 8B, Gemma 2 9B, Gemma 4 E4B-it, Qwen 2.5 7B, Qwen 3-NT 8B), the granularity rescue spectrum, the cross-task validation surface, two observational-boundary entries (Gemma 3 12B-pt, Qwen 3 30B-A3B), and the supporting probes documented in the paper. The cross-family finding is family-conditional: five families are channel-tolerant (Mistral, Llama 3.1, Ministral, Gemma 2, Qwen 2.5) and two families are attention-favoring at the shared b=1024 gate (Gemma 4 and Qwen 3-NT 8B).
- **`scripts/`**: the Appendix A artifact-accounting recompute script (`appendix_a_recompute.py`) and the cross-family statistics script (`vorn_mat_cross_family_stats.py`) referenced in the paper.
- **`examples/`**: Modal job harness for live-eviction experiments.
- **`tests/`**: pytest suite covering plan, results, paired_stats.
- **`docs/RUNBOOK.md`**: environment setup, smoke test, and headline-cell reproduction instructions.

## Quickstart

The lightweight quickstart installs only the lightweight dev dependencies (`pytest`, `numpy`) and runs the 133 torch-free tests — the plan/result/paired-stats/orchestration layer that doesn't need a GPU:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ --ignore=tests/test_live_eviction_runner.py --ignore=tests/test_local_exec.py
```

For the full 186-test suite (which exercises the live-eviction runner and the local-execution path through `transformers`), install the `[local]` extras as well. `[local]` pulls torch + transformers + accelerate + datasets + faiss-cpu + sentencepiece + huggingface_hub at the canonical pin set (~5 GB total; CPU-only is fine for running the tests, GPU is needed for the headline cells):

```bash
pip install -e ".[dev,local]"
pytest tests/
```

For local validation against the bundled 5-case NIAH smoke fixture (requires sufficient RAM):

```bash
pip install -e ".[local]"
python examples/run_local_vanilla.py --limit 5
```

For Modal-backed reproduction of the headline cells in the paper, see `docs/RUNBOOK.md`.

## Reproducibility substrate

The canonical reproduction path is a hash-locked Docker image built from the
repo-root `Dockerfile`. The base image is `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
and the Python dependency closure is pinned by `requirements.lock` (generated
via `uv pip compile --generate-hashes`).

```bash
docker build --platform linux/amd64 -t vorn-mat:canonical .
docker run --gpus all -v $(pwd):/app vorn-mat:canonical pytest tests/
```

The `--platform linux/amd64` flag is required on macOS arm64 and other non-x86 hosts: the base image `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` and the pinned torch wheel target Linux x86_64. Without the flag, Docker silently pulls an emulated image (or fails to find one) and reproduction drifts from the canonical Modal-run platform.

The Modal job entry-points (`examples/run_modal_*.py`) build the same image
through `Image.from_dockerfile(...)`, so local Docker reproduction and Modal
reproduction share an identical software stack.

### Substrate improvement from failure: multi-token EOS support

A 2026-05-23 Gemma 3 instruct rerun surfaced a harness defect in the
hand-rolled live-eviction generation loops. Some chat-tuned models expose
multiple terminal token ids through `generation_config.eos_token_id`
(for example `google/gemma-3-12b-it` uses both `<eos>` and
`<end_of_turn>`). The earlier harness stopped only on the tokenizer's
singular `eos_token_id`, which could let the loop consume terminal special
tokens and decode them away to an empty string. The live-eviction and
streaming loops now honor the full terminal-token set from
`generation_config`.

The same patch also strengthens metadata isolation in the Modal wrappers by
overwriting `metadata.model` with the request model id. This prevents stale
canonical-plan metadata from leaking into rerun artifacts when the request
model differs from the default family.

### Substrate improvement from failure: bf16 live generation stability

The same Gemma 3 rerun later exposed a deeper numerical-correctness defect:
manual live generation on CUDA float16 could emit NaN next-token logits.
Because the hand-rolled greedy loop did not suppress non-terminal pad tokens
or treat NaN logits explicitly, the pad-mask path could surface NaN top
candidates and collapse into immediate blank / terminal behavior. Those
blank constrained-row outputs were harness artifacts, not evidence about
Gemma 3 retrieval behavior.

The live harness now loads CUDA models in `bfloat16`, casts hidden-state and
attention tensors back to `float32` before NumPy scoring boundaries, suppresses
non-terminal pad tokens during greedy selection, and raises explicitly if a
row contains only NaN logits. Live-eviction generation also emits structured
`generation_step_*` and `token_step` diagnostic lines so token-level failures
are visible in Modal logs.

Verify-by-fruit anchor: a one-case Gemma 3 control rerun
(`google/gemma-3-12b-it`, `sentence_vorn`, `B=8192`, `n=1`,
`max_new_tokens=8`) produced prediction `9375710` with `correct=true`,
`hit_rate=1.0`, and estimated cost `$0.1119` after the bf16 + cast fix.

Two-layer note: the Dockerfile installs the `vorn_mat` source via
`pip install --no-deps -e /app` during image build to prime the editable
install layer. At Modal run time the volume mount overlays the live source,
so the in-container `vorn_mat` import points at whatever the Modal task
mounts (not a stale build-time snapshot). Local `docker run -v $(pwd):/app`
gets the same behavior.

To regenerate `requirements.lock` after an intentional pin change in
`pyproject.toml`:

```bash
uv pip compile \
    --python-version 3.11 \
    --python-platform linux \
    --generate-hashes \
    pyproject.toml --extra local \
    -o requirements.lock
```

The `--python-platform linux` flag is load-bearing: without it, the resolver
runs against the host platform (macOS / Windows) and silently omits the
Linux+CUDA transitive closure (`cuda-toolkit`, `nvidia-cublas`, etc.). The
Docker image then fails at `pip install --require-hashes` because those
transitives are pulled at install-time but have no hash entries in the lock.
Always regenerate with the Linux target since that is what the Dockerfile
installs into.

### Reproducibility disclosure

The pin set in `pyproject.toml` and `requirements.lock` is a **best-guess
reconstruction** based on raw-report timestamps and PyPI release chronology.
The original public canonical runs did not preserve:

- a `pip freeze` / lockfile inside the prototype
- a Modal image hash or image id in the result artifacts
- a persisted `environment_versions` block in the result envelopes

So the pinned substrate above is the closest defensible reconstruction of the
canonical family-wave software stack, not a recovered exact lockfile. Two
documented caveats follow from this:

- the earliest May 13 / 14 seed reports likely ran on
  `huggingface_hub==1.14.0` (since `1.15.0` had not released yet at those
  timestamps)
- some late May 20 budget-fill rows may have crossed into
  `transformers==5.9.0` if Modal rebuilt the image after the `5.9.0` release
  at `2026-05-20T14:50:45Z`

Going forward, every cell run on the pinned substrate captures
`env_versions` (transformers / torch / accelerate / datasets / sentencepiece /
huggingface_hub / faiss-cpu) and CUDA peak-memory telemetry
(`peak_memory_allocated_gb`, `peak_memory_reserved_gb`, `oom_near_miss`) into
the result envelope (`vorn_mat.results.RunResult`). This closes the
provenance hole for all post-2026-05-23 artifacts.

## Resilience: per-case incremental persistence

Every baseline runner (`run_vanilla` / `run_vorn` / `run_live_eviction`)
accepts an `on_case` callback that fires once per completed case with the
case's `CaseObservation`. The Modal and local wrappers wire this callback to
`vorn_mat.results.append_observation`, which appends a single JSONL line +
`fsync()` to a ledger file at `output_path.with_suffix(".observations.jsonl")`
**before the next case runs**.

Why this matters: cell runs are minutes-to-an-hour long. Mid-run failures
(server-side OOM, Modal container kill, network blip, manual kill, account
GPU cap hit, laptop sleep) used to lose all completed cases because the
summary `RunResult` envelope only landed on disk after all cases finished.
The per-case ledger persists every completed case incrementally, so a mid-run
kill at case N of 50 preserves cases 1..N on disk and only cases N+1..50 are
lost.

To recover from a mid-run kill:
- The summary file at `output_path` is missing or partial. Ignore it.
- The ledger at `output_path.with_suffix(".observations.jsonl")` has all
  completed cases. Reload with `vorn_mat.results.load_observations(path)`.
- Re-run the cell with `case_offset_start` (or your wrapper's equivalent) to
  resume from case N+1.

### Modal-native parallel cell execution (Layer 3: cloud-side orchestrator)

`examples/run_modal_cells_parallel.py` is the canonical entrypoint for firing
many cells in parallel. The local entrypoint makes ONE
`orchestrate_wave.remote(specs)` call. The cloud-side `orchestrate_wave`
function is decorated with `@app.function(timeout=86400, ...)` and is what
issues `binding.remote_fn.spawn()` per cell + per-handle `.get()`
collection under `max_containers=10` on the cell function binding.

Why this shape: Modal's launch warning under `modal run --detach` is
"running a local entrypoint in detached mode only keeps the last triggered
Modal function alive after the parent process has been killed or
disconnected." Moving the `.spawn()` loop one altitude up makes the local
entrypoint's single `.remote()` call the only thing `--detach` needs to
protect; the spawned cells become children of that protected call and
inherit its protection. Local can disconnect freely after kickoff.

Per-cell exceptions surface as entries in the JSON-safe wave-report dict
that `orchestrate_wave` returns (`{"reports": [...], "failures": [...]}`),
so partial-wave failures do not kill the whole batch. Each cell still
benefits from per-case persistence on the Modal Volume mount, so even
within a failed cell, completed cases are preserved.

This pattern replaces, in successive layers:
- (Layer 1) `pip_install` of loose-pin tuples at image-build time, which
  could not reproduce the dependency closure that produced canonical results.
- (Layer 2) user-side parallelism (ThreadPoolExecutor wrapping per-cell
  `modal run`), which created N independent local-client lifecycles = N
  independent disconnect-class failure points.
- (Layer 3) local-entrypoint server-side fanout, which had ambiguous
  protection semantics under `--detach`.

```bash
# Build a cell spec JSON, then fire the wave with one detached invocation:
modal run --detach examples/run_modal_cells_parallel.py \
  --cell-spec-path .benchmarks/cell-specs.json

# Aggregate per-cell results into per-family canonical artifacts:
python examples/build_matrix_backfill_artifact.py --family mistral
```

### Observability (Layer 4: Modal-visible progress logging)

Long-running cells (30-60min A100 inference) used to go silent after the
HuggingFace model-load phase. Modal captures stdout to the dashboard and
`modal app logs` CLI, but cells emitted nothing during the silent inference
phase, so from outside the local terminal there was no mid-cell progress
visibility.

Every baseline runner (`run_vanilla` / `run_vorn` / `run_live_eviction`)
accepts a `progress_logger: Callable[[str], None] | None` keyword. When set,
the runner emits a line-based progress trace to Modal's stdout capture:

- Once at start: `vorn-mat: dataset_loaded n_cases=N`
- Per case: `vorn-mat: case I/N correct=true running_accuracy=0.XXX`
- Once at end: `vorn-mat: complete n_cases=N hit_rate=0.XXX`

The Modal entry-points (`run_modal_*_niah`) default `progress_logger` to
`vorn_mat.progress.default_progress_logger`, which prints with `flush=True`
so Modal sees output immediately rather than buffered until container exit.
Modal auto-timestamps each line in the dashboard, so no local timestamping
is added. To suppress emissions in non-Modal contexts (tests, library use),
pass `progress_logger=None`.

## Reproducing the paper's headline numbers

The Appendix A v1.1 totals (67 artifacts / 446 counted rows / 342 with observations / 21,600 per-fixture observations / $119.59 / 49.46h) are reproducible from this repository's `results/` directory by running:

```bash
python scripts/appendix_a_recompute.py
```

The script defines the explicit counting contract (which row-array fields are counted versus excluded, and why) and recomputes the totals against the released artifacts directly. The script handles the canonical result-envelope schemas plus the Phase 3 composed-artifact schemas (`phase1_cells`, `phase3_a100_cells`), top-level single-cell diagnostic artifacts, top-level list envelopes, `models[]`/`families[]` wrapper-descent for v0.2 extension-wave artifacts, and nested `row.result.observations[]` descent for rows that wrap a `result` sub-dict. `cells_by_family` (merged-view summary) and failure-list envelopes are excluded by design.

The paired McNemar p-values cited in the paper are recoverable from the per-fixture `observations[]` arrays in the claim-bearing result rows. 342 of the 446 counted rows carry observations. See Appendix A for the counting contract. For example, the Llama 3.1 vorn cross-task headline cell (sentence-vorn 92/200 versus token-vorn 52/200 at b=1024 on qa_2_4k, paired exact McNemar `p = 1.03e-08`) traces to `results/token-vorn-qa2-cross-task-2026-05-20.json`.

## Citation

If you use this repository or the released artifacts in your work, please cite the paper:

```bibtex
@article{penney2026vornmat,
  title={Vorn-Mat: Family-Conditional KV-Cache Eviction and the Granularity Rescue Spectrum},
  author={Penney, L.},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT. See `LICENSE`.
