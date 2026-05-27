from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat import (
    BASELINES,
    BENCHMARKS,
    ExperimentDefaults,
    RunResult,
    append_result,
    build_execution_plans,
    build_vanilla_entrypoint,
    build_vorn_entrypoint,
    build_week1_run_matrix,
    default_modal_app_spec,
    load_results,
)


def test_modal_app_spec_locks_expected_week1_defaults():
    spec = default_modal_app_spec()

    assert spec.app_name == "vorn-mat-week1-baselines"
    assert spec.python_version == "3.11"
    assert spec.volume_name == "synapt-vorn-mat-vol"
    assert spec.model_cache == "/vol/models"
    assert spec.results_root == "/vol/results/vorn-mat"
    assert spec.benchmark_cache == "/vol/benchmarks"
    assert spec.hf_secret_name == "huggingface-secret"
    assert spec.gpu == "A100-80GB"
    assert spec.timeout_seconds == 3600
    # Canonical Atlas-archaeology pin set (2026-05-23). The Modal image is
    # built from the repo-root Dockerfile via Image.from_dockerfile; this
    # tuple stays as the test-surface contract on the canonical pin shape.
    assert "torch==2.12.0" in spec.pip_dependencies
    assert "transformers==5.8.1" in spec.pip_dependencies
    assert "accelerate==1.13.0" in spec.pip_dependencies
    assert "datasets==4.8.5" in spec.pip_dependencies
    assert "sentencepiece==0.2.1" in spec.pip_dependencies
    assert "faiss-cpu==1.13.2" in spec.pip_dependencies
    assert "huggingface_hub==1.15.0" in spec.pip_dependencies


def test_modal_app_spec_carries_pytorch_cuda_alloc_conf_expandable_segments():
    """PyTorch's documented fix for caching-allocator fragmentation under
    varying-size allocations across many fixtures must be baked into the
    canonical app spec. Filed 2026-05-27 after Ministral 8B TOVA/H2O n=50
    cells OOMed on A100-80GB with ~27 GB reserved-but-unallocated overhead.
    """
    spec = default_modal_app_spec()

    env_dict = dict(spec.env_vars)
    assert env_dict.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True", (
        f"PYTORCH_CUDA_ALLOC_CONF must default to expandable_segments:True; "
        f"got env_vars={spec.env_vars!r}"
    )


def test_build_modal_artifacts_applies_env_vars_to_image():
    """When ModalAppSpec.env_vars is non-empty, build_modal_artifacts must
    apply them to the image via image.env(...) so PyTorch reads
    PYTORCH_CUDA_ALLOC_CONF at module import time in the per-cell container.
    """
    from vorn_mat.modal_app import build_modal_artifacts

    env_calls: list[dict] = []

    class FakeImage:
        def env(self, env_dict):
            env_calls.append(dict(env_dict))
            return self

    class FakeApp:
        def __init__(self, name):
            self.name = name

    class FakeVolume:
        @classmethod
        def from_name(cls, name, create_if_missing=False):
            return cls()

    class FakeImageFactory:
        @staticmethod
        def from_dockerfile(path, context_dir):
            return FakeImage()

    class FakeModal:
        App = FakeApp
        Image = FakeImageFactory
        Volume = FakeVolume

    artifacts = build_modal_artifacts(modal_module=FakeModal, dockerfile_path="/tmp/fake-dockerfile")

    assert len(env_calls) == 1, f"image.env() should be invoked exactly once; got {len(env_calls)} calls"
    assert env_calls[0].get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True", (
        f"image.env() must receive PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; "
        f"got {env_calls[0]!r}"
    )


def test_benchmark_registry_contains_week1_targets():
    assert set(BENCHMARKS) == {"ruler", "niah"}
    assert BENCHMARKS["ruler"].metric_name == "task_accuracy"
    assert BENCHMARKS["niah"].metric_name == "needle_hit_rate"


def test_baseline_registry_contains_week1_baselines():
    assert set(BASELINES) == {
        "vanilla",
        "vorn",
        "vorn_live",
        "sentence_vorn_live",
        "word_vorn_live",
        "adaptive_vorn_live",
        "tova_live",
        "sentence_tova_live",
        "h2o_live",
        "sentence_h2o_live",
        "random_live",
        "sliding_window_live",
        "prefix_suffix_live",
        "summarize_compact",
        "h2o",
        "streamingllm",
    }
    assert BASELINES["vorn"].cache_strategy == "canonical_layer_prompt_retention_proxy"
    assert BASELINES["vorn_live"].cache_strategy == "live_token_position_eviction"
    assert BASELINES["sentence_vorn_live"].cache_strategy == "live_sentence_eviction"
    assert BASELINES["word_vorn_live"].cache_strategy == "live_word_eviction"
    assert BASELINES["adaptive_vorn_live"].cache_strategy == (
        "online_token_or_sentence_eviction"
    )
    assert BASELINES["tova_live"].cache_strategy == (
        "last_token_attention_weight_eviction"
    )
    assert BASELINES["h2o_live"].cache_strategy == (
        "accumulated_attention_weight_eviction"
    )
    assert BASELINES["random_live"].cache_strategy == (
        "uniform_random_token_position_eviction"
    )
    assert BASELINES["sliding_window_live"].cache_strategy == (
        "recent_only_token_position_eviction"
    )
    assert BASELINES["prefix_suffix_live"].cache_strategy == (
        "prefix_and_recent_suffix_token_position_eviction"
    )
    assert BASELINES["summarize_compact"].cache_strategy == (
        "whole_context_summary_then_answer"
    )
    assert BASELINES["h2o"].cache_strategy == "heavy_hitter_oracle"
    assert BASELINES["streamingllm"].cache_strategy == "attention_sinks_plus_recent_window"


def test_execution_plan_expands_run_matrix_into_runner_contract():
    defaults = ExperimentDefaults()
    plans = build_execution_plans(build_week1_run_matrix(defaults))

    assert len(plans) == 6
    assert plans[0].run.run_id == "week1-ruler-vanilla"
    assert plans[0].benchmark.name == "ruler"
    assert plans[0].baseline.name == "vanilla"
    assert plans[0].results_path.endswith("/week1-ruler-vanilla.jsonl")
    assert all(plan.run.model == defaults.model for plan in plans)
    assert all(plan.run.canonical_layer == defaults.canonical_layer for plan in plans)
    assert all(plan.run.recent_token_window == defaults.recent_token_window for plan in plans)


def test_result_sink_round_trips_jsonl(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    result = RunResult(
        run_id="week1-ruler-vanilla",
        benchmark="ruler",
        baseline="vanilla",
        metrics={"task_accuracy": 0.75},
        metadata={"gpu": "A100-80GB"},
        preprocessing_elapsed_seconds=1.25,
        preprocessing_cost_usd=0.031,
    )

    append_result(path, result)
    loaded = load_results(path)

    assert loaded == [result]


class _FakeApp:
    def __init__(self, name: str):
        self.name = name
        self.calls: list[dict[str, object]] = []

    def function(self, **kwargs):
        self.calls.append(kwargs)

        def decorator(fn):
            fn._modal_kwargs = kwargs
            return fn

        return decorator


class _FakeImage:
    def __init__(self, python_version: str):
        self.python_version = python_version
        self.packages: tuple[str, ...] = ()

    def pip_install(self, *packages: str):
        self.packages = packages
        return self


class _FakeDockerfileImage:
    def __init__(self, dockerfile_path: str, context_dir: str):
        self.dockerfile_path = dockerfile_path
        self.context_dir = context_dir
        self.env_vars: dict[str, str] = {}

    def env(self, env_dict):
        self.env_vars = dict(env_dict)
        return self


class _FakeImageFactory:
    @staticmethod
    def debian_slim(*, python_version: str):
        return _FakeImage(python_version)

    @staticmethod
    def from_dockerfile(path: str, *, context_dir: str):
        return _FakeDockerfileImage(path, context_dir)


class _FakeVolumeFactory:
    @staticmethod
    def from_name(name: str, create_if_missing: bool):
        return {"name": name, "create_if_missing": create_if_missing}


class _FakeSecretFactory:
    @staticmethod
    def from_name(name: str):
        return {"secret_name": name}


class _FakeModal:
    App = _FakeApp
    Image = _FakeImageFactory
    Volume = _FakeVolumeFactory
    Secret = _FakeSecretFactory


def test_build_vanilla_entrypoint_binds_modal_function_contract():
    def run_callable(run_id: str) -> str:
        return run_id

    binding = build_vanilla_entrypoint(run_callable, modal_module=_FakeModal)

    assert binding.entrypoint_name == "run_vanilla_entrypoint"
    assert binding.spec.app_name == "vorn-mat-week1-baselines"
    assert binding.app.name == "vorn-mat-week1-baselines"
    assert binding.remote_fn is run_callable
    assert run_callable._modal_kwargs["gpu"] == "A100-80GB"
    assert run_callable._modal_kwargs["timeout"] == 3600
    assert run_callable._modal_kwargs["volumes"] == {
        "/vol": {"name": "synapt-vorn-mat-vol", "create_if_missing": True}
    }
    assert run_callable._modal_kwargs["secrets"] == [
        {"secret_name": "huggingface-secret"}
    ]


def test_build_vorn_entrypoint_binds_modal_function_contract():
    def run_callable(run_id: str) -> str:
        return run_id

    binding = build_vorn_entrypoint(run_callable, modal_module=_FakeModal)

    assert binding.entrypoint_name == "run_vorn_entrypoint"
    assert binding.spec.app_name == "vorn-mat-week1-baselines"
    assert binding.app.name == "vorn-mat-week1-baselines"
    assert binding.remote_fn is run_callable
    assert run_callable._modal_kwargs["gpu"] == "A100-80GB"
    assert run_callable._modal_kwargs["timeout"] == 3600
    assert run_callable._modal_kwargs["volumes"] == {
        "/vol": {"name": "synapt-vorn-mat-vol", "create_if_missing": True}
    }
    assert run_callable._modal_kwargs["secrets"] == [
        {"secret_name": "huggingface-secret"}
    ]
