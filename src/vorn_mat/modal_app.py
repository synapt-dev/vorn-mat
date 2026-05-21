"""Modal app metadata for Week 1 baseline reproduction."""

from __future__ import annotations

from dataclasses import dataclass
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
    volume_path = "/vol"
    return ModalAppSpec(
        app_name="vorn-mat-week1-baselines",
        python_version="3.11",
        pip_dependencies=(
            "torch",
            "transformers>=4.44.0",
            "accelerate",
            "datasets",
            "faiss-cpu",
            "huggingface_hub",
            "sentencepiece",
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


def build_modal_artifacts(
    modal_module: object | None = None,
    app_spec: ModalAppSpec | None = None,
) -> ModalArtifacts:
    """Build Modal app primitives lazily so tests do not require Modal."""
    if modal_module is None:
        import modal as modal_module  # type: ignore[no-redef]

    spec = app_spec or default_modal_app_spec()
    app = modal_module.App(spec.app_name)
    image = modal_module.Image.debian_slim(
        python_version=spec.python_version
    ).pip_install(*spec.pip_dependencies)
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
