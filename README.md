# Vorn-Mat

Reference implementation and result artifacts for the paper *Vorn-Mat: Family-Conditional KV-Cache Eviction and the Granularity Rescue Spectrum*.

This repository contains the prototype source, all released result artifacts, supplementary analysis scripts, and a runbook for reproducing the headline cells. The companion HuggingFace dataset at [`synapt/vorn-mat-cross-family-results`](https://huggingface.co/datasets/synapt/vorn-mat-cross-family-results) mirrors the result artifacts for citeable, discoverable access independent of this code repository.

## What's in this repository

- **`src/vorn_mat/`**: prototype source.
  - `vorn.py`: residual-direction scoring at the canonical mid-depth layer (`L* = L // 2`) under a prefill-time cache selection contract with full-prompt visibility.
  - `baselines/live_eviction.py`: token-level and sentence-level retention policies, plus TOVA-style and H2O-style attention-weight baselines under the same one-shot prefill contract.
  - `plan.py`, `runner.py`, `remote_exec.py`: experiment plan dataclasses, result-envelope JSON schemas, and Modal job dispatch.
  - `paired_stats.py`: exact paired McNemar tests over per-fixture observations preserved in each result row.
- **`results/`**: 49 released JSON artifacts (each paired with a Markdown summary) covering the nine-model panel, the granularity rescue spectrum, the cross-task validation surface, and the supporting probes documented in the paper.
- **`scripts/`**: the Appendix A artifact-accounting recompute script (`appendix_a_recompute.py`) and the cross-family statistics script (`vorn_mat_cross_family_stats.py`) referenced in the paper.
- **`examples/`**: Modal job harness for live-eviction experiments.
- **`tests/`**: pytest suite covering plan, results, paired_stats.
- **`docs/RUNBOOK.md`**: environment setup, smoke test, and headline-cell reproduction instructions.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
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
docker build -t vorn-mat:canonical .
docker run --gpus all -v $(pwd):/app vorn-mat:canonical pytest tests/
```

The Modal job entry-points (`examples/run_modal_*.py`) build the same image
through `Image.from_dockerfile(...)`, so local Docker reproduction and Modal
reproduction share an identical software stack.

To regenerate `requirements.lock` after an intentional pin change in
`pyproject.toml`:

```bash
uv pip compile --generate-hashes pyproject.toml --extra local -o requirements.lock
```

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

## Reproducing the paper's headline numbers

The Appendix A totals in the paper (49 artifacts / 299 counted rows / 270 with observations / 18,000 per-fixture observations / $77.78 / 32.72h) are reproducible from this repository's `results/` directory by running:

```bash
python scripts/appendix_a_recompute.py
```

The script defines the explicit counting contract (which row-array fields are counted versus excluded, and why) and recomputes the totals against the released artifacts directly.

The paired McNemar p-values cited in the paper are recoverable from the per-fixture `observations[]` arrays in the claim-bearing result rows. 270 of the 299 counted rows carry observations. See Appendix A for the counting contract. For example, the Llama 3.1 vorn cross-task headline cell (sentence-vorn 92/200 versus token-vorn 52/200 at b=1024 on qa_2_4k, paired exact McNemar `p = 1.03e-08`) traces to `results/token-vorn-qa2-cross-task-2026-05-20.json`.

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
