"""Modal app metadata for Week 1 baseline reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ModalAppSpec:
    app_name: str
    python_version: str
    pip_dependencies: tuple[str, ...]
    volume_name: str
    volume_path: str
    model_cache: str
    results_root: str
    benchmark_cache: str
    hf_secret_name: str
    gpu: str
    timeout_seconds: int


@dataclass(frozen=True)
class ModalArtifacts:
    spec: ModalAppSpec
    app: object
    image: object
    volume: object


@dataclass(frozen=True)
class ModalBinding:
    spec: ModalAppSpec
    app: object
    entrypoint_name: str
    remote_fn: object


def default_modal_app_spec(gpu: str = "A100-80GB") -> ModalAppSpec:
    """Canonical Modal app spec.

    The pip_dependencies tuple is the canonical pin set (Atlas archaeology
    2026-05-23) and is the source-of-truth surface that tests assert against.
    The Modal image itself is built from the repo-root Dockerfile via
    Image.from_dockerfile in build_modal_artifacts; pip_dependencies is kept
    here for inspection and for the test surface contract.
    """
    volume_path = "/vol"
    return ModalAppSpec(
        app_name="vorn-mat-week1-baselines",
        python_version="3.11",
        pip_dependencies=(
            "torch==2.12.0",
            "transformers==5.8.1",
            "accelerate==1.13.0",
            "datasets==4.8.5",
            "faiss-cpu==1.13.2",
            "huggingface_hub==1.15.0",
            "sentencepiece==0.2.1",
        ),
        volume_name="synapt-vorn-mat-vol",
        volume_path=volume_path,
        model_cache=f"{volume_path}/models",
        results_root=f"{volume_path}/results/vorn-mat",
        benchmark_cache=f"{volume_path}/benchmarks",
        hf_secret_name="huggingface-secret",
        gpu=gpu,
        timeout_seconds=60 * 60,
    )


def _default_dockerfile_path() -> Path:
    # src/vorn_mat/modal_app.py -> ../../../Dockerfile
    return Path(__file__).resolve().parent.parent.parent / "Dockerfile"


def build_modal_artifacts(
    modal_module: object | None = None,
    app_spec: ModalAppSpec | None = None,
    dockerfile_path: Path | str | None = None,
) -> ModalArtifacts:
    """Build Modal app primitives lazily so tests do not require Modal.

    The image is built from the repo-root Dockerfile via Image.from_dockerfile.
    This is the hash-locked reproducibility substrate (see Dockerfile + the
    requirements.lock file). pip_install of loose-pin tuples is no longer used
    because the canonical published runs depended on dependency snapshots that
    pip-install at image-build time cannot reproduce.
    """
    if modal_module is None:
        import modal as modal_module  # type: ignore[no-redef]

    spec = app_spec or default_modal_app_spec()
    app = modal_module.App(spec.app_name)
    resolved_dockerfile = Path(dockerfile_path) if dockerfile_path else _default_dockerfile_path()
    image = modal_module.Image.from_dockerfile(
        str(resolved_dockerfile),
        context_dir=str(resolved_dockerfile.parent),
    )
    volume = modal_module.Volume.from_name(spec.volume_name, create_if_missing=True)
    return ModalArtifacts(spec=spec, app=app, image=image, volume=volume)


def build_vanilla_entrypoint(
    run_callable: Callable[..., object],
    modal_module: object | None = None,
    app_spec: ModalAppSpec | None = None,
) -> ModalBinding:
    """Bind the vanilla runner to a real Modal app.function surface."""
    return _build_entrypoint_binding(
        entrypoint_name="run_vanilla_entrypoint",
        run_callable=run_callable,
        modal_module=modal_module,
        app_spec=app_spec,
    )


def build_vorn_entrypoint(
    run_callable: Callable[..., object],
    modal_module: object | None = None,
    app_spec: ModalAppSpec | None = None,
) -> ModalBinding:
    """Bind the Step 1 vorn runner to a real Modal app.function surface."""
    return _build_entrypoint_binding(
        entrypoint_name="run_vorn_entrypoint",
        run_callable=run_callable,
        modal_module=modal_module,
        app_spec=app_spec,
    )


def _build_entrypoint_binding(
    *,
    entrypoint_name: str,
    run_callable: Callable[..., object],
    modal_module: object | None = None,
    app_spec: ModalAppSpec | None = None,
) -> ModalBinding:
    artifacts = build_modal_artifacts(modal_module, app_spec=app_spec)
    if modal_module is None:
        import modal as modal_module  # type: ignore[no-redef]

    decorator = artifacts.app.function(
        image=artifacts.image,
        gpu=artifacts.spec.gpu,
        timeout=artifacts.spec.timeout_seconds,
        volumes={artifacts.spec.volume_path: artifacts.volume},
        secrets=[modal_module.Secret.from_name(artifacts.spec.hf_secret_name)],
    )
    remote_fn = decorator(run_callable)
    return ModalBinding(
        spec=artifacts.spec,
        app=artifacts.app,
        entrypoint_name=entrypoint_name,
        remote_fn=remote_fn,
    )
