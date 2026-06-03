from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import vorn_mat.remote_exec as remote_exec
from vorn_mat.benchmarks.common import BenchmarkCase
from vorn_mat.counterfactual_intervention_runner import sha256_text
from vorn_mat.observation import ObservationCase
from vorn_mat.results import CaseObservation, RunResult
from vorn_mat.score_distribution_observation import (
    ScoreDistributionObservationCase,
    ScoreDistributionObservationReport,
    ScoreDistributionObservationStep,
    ScoreDistributionStats,
)


_CONSUMER_RENDERED_PROMPT = (
    "[INST] Setup. "
    "High magic clue. "
    "Long neutral sentence with many extra words. "
    "Length match. "
    "Low filler. "
    "Question? [/INST]"
)


def test_run_modal_vanilla_niah_enriches_result_metadata(monkeypatch):
    monkeypatch.setattr(
        remote_exec,
        "load_ruler_hf_niah_slice",
        lambda dataset_config, split, case_limit, **kwargs: (
            BenchmarkCase("c1", "prompt", "answer", {}),
        ),
    )
    monkeypatch.setattr(
        remote_exec,
        "select_week1_plan",
        lambda benchmark, baseline="vanilla": "plan",
    )

    class FakeGenerator:
        def __init__(self, config):
            self.config = config
            assert config.model_id == "meta-llama/Llama-3.1-8B-Instruct"

    monkeypatch.setattr(remote_exec, "TransformersTextGenerator", FakeGenerator)

    def fake_run_vanilla(plan, cases, generator, **kwargs):
        assert plan == "plan"
        assert len(cases) == 1
        return (
            RunResult(
                run_id="week1-niah-vanilla",
                benchmark="niah",
                baseline="vanilla",
                metrics={"needle_hit_rate": 1.0},
                metadata={"gpu": "A100-80GB"},
                observations=(
                    CaseObservation(
                        fixture_id="c1",
                        correct=True,
                        prediction="answer",
                    ),
                ),
            ),
            (),
        )

    monkeypatch.setattr(remote_exec, "run_vanilla", fake_run_vanilla)
    monkeypatch.setattr(remote_exec.time, "perf_counter", lambda: next(counter))

    counter = iter([10.0, 22.5])
    report = remote_exec.run_modal_vanilla_niah(
        remote_exec.ModalVanillaRunRequest(
            case_limit=1,
            model_id="meta-llama/Llama-3.1-8B-Instruct",
        )
    )

    assert report.dataset_config == "niah_multikey_1_4k"
    assert report.case_count == 1
    assert report.elapsed_seconds == 12.5
    assert report.estimated_cost_usd == 12.5 * remote_exec.A100_80GB_PER_SECOND
    assert report.result.metadata["case_count"] == "1"
    assert report.result.metadata["dataset_id"] == "rbiswasfc/ruler"
    assert report.result.metadata["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert report.result.metadata["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert report.result.observations[0].fixture_id == "c1"


def test_run_modal_live_eviction_niah_enriches_result_metadata(monkeypatch):
    loader_calls = []

    def fake_load_slice(dataset_config, split, case_limit, **kwargs):
        loader_calls.append((dataset_config, split, case_limit, kwargs))
        return (BenchmarkCase("c1", "prompt", "answer", {}),)

    monkeypatch.setattr(
        remote_exec,
        "load_ruler_hf_niah_slice",
        fake_load_slice,
    )
    class Plan:
        class Run:
            cache_budget_tokens = 256
            retention_policy = "random"
            random_seed = 23
            always_keep_prefix_tokens = 0
            preserve_recent_window = False
            eviction_trigger = "sentence_boundary"
            sentence_boundary_lookahead_tokens = 25
            force_eviction_overflow_ratio = 1.2

        run = Run()

    def fake_select_live_eviction_plan(**kwargs):
        assert kwargs == {
            "cache_budget_tokens": 256,
            "retention_policy": "random",
            "random_seed": 23,
            "always_keep_prefix_tokens": 0,
            "preserve_recent_window": False,
            "sentence_pooling": "max",
            "sentence_top_k": 3,
            "eviction_trigger": "sentence_boundary",
            "sentence_boundary_lookahead_tokens": 25,
            "force_eviction_overflow_ratio": 1.2,
        }
        return Plan()

    monkeypatch.setattr(
        remote_exec,
        "select_live_eviction_plan",
        fake_select_live_eviction_plan,
    )

    class FakeGenerator:
        def __init__(self, config):
            self.config = config
            assert config.model_id == "meta-llama/Llama-3.1-8B-Instruct"

    monkeypatch.setattr(remote_exec, "TransformersLiveEvictionGenerator", FakeGenerator)

    def fake_run_live_eviction(plan, cases, generator, **kwargs):
        assert plan.run.cache_budget_tokens == 256
        assert len(cases) == 1
        return (
            RunResult(
                run_id="step2-niah-vorn-live",
                benchmark="niah",
                baseline="vorn_live",
                metrics={"needle_hit_rate": 0.5},
                metadata={"gpu": "A100-80GB", "cache_budget_tokens": "256"},
                preprocessing_elapsed_seconds=3.0,
                observations=(
                    CaseObservation(
                        fixture_id="c1",
                        correct=True,
                        prediction="answer",
                    ),
                ),
            ),
            (),
        )

    monkeypatch.setattr(remote_exec, "run_live_eviction", fake_run_live_eviction)
    monkeypatch.setattr(remote_exec.time, "perf_counter", lambda: next(counter))

    counter = iter([20.0, 36.0])
    report = remote_exec.run_modal_live_eviction_niah(
        remote_exec.ModalLiveEvictionRunRequest(
            case_limit=16,
            case_offset_start=34,
            cache_budget_tokens=256,
            retention_policy="random",
            random_seed=23,
            always_keep_prefix_tokens=0,
            preserve_recent_window=False,
            eviction_trigger="sentence_boundary",
            model_id="meta-llama/Llama-3.1-8B-Instruct",
        )
    )

    assert report.dataset_config == "niah_multikey_1_4k"
    assert loader_calls == [
        (
            "niah_multikey_1_4k",
            "validation",
            16,
            {"case_offset_start": 34},
        )
    ]
    assert report.case_count == 1
    assert report.case_offset_start == 34
    assert report.cache_budget_tokens == 256
    assert report.retention_policy == "random"
    assert report.elapsed_seconds == 16.0
    assert report.estimated_cost_usd == 16.0 * remote_exec.A100_80GB_PER_SECOND
    assert report.result.metadata["case_count"] == "1"
    assert report.result.metadata["case_offset_start"] == "34"
    assert report.result.metadata["dataset_id"] == "rbiswasfc/ruler"
    assert report.result.metadata["model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert report.result.metadata["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert report.result.metadata["cache_budget_tokens"] == "256"
    assert report.result.metadata["retention_policy"] == "random"
    assert report.result.metadata["random_seed"] == "23"
    assert report.result.metadata["always_keep_prefix_tokens"] == "0"
    assert report.result.metadata["preserve_recent_window"] == "false"
    assert report.result.metadata["sentence_pooling"] == "max"
    assert report.result.metadata["sentence_top_k"] == "3"
    assert report.result.metadata["eviction_trigger"] == "sentence_boundary"
    assert report.result.metadata["sentence_boundary_lookahead_tokens"] == "25"
    assert report.result.metadata["force_eviction_overflow_ratio"] == "1.20"
    assert report.eviction_trigger == "sentence_boundary"
    assert report.sentence_boundary_lookahead_tokens == 25
    assert report.force_eviction_overflow_ratio == 1.2
    assert report.result.preprocessing_elapsed_seconds == 3.0
    assert report.result.preprocessing_cost_usd == (
        3.0 * remote_exec.A100_80GB_PER_SECOND
    )
    assert report.result.observations[0].fixture_id == "c1"


def test_run_modal_live_eviction_niah_allows_budget_sweep_variants(monkeypatch):
    monkeypatch.setattr(
        remote_exec,
        "load_ruler_hf_niah_slice",
        lambda dataset_config, split, case_limit, **kwargs: (
            BenchmarkCase("c1", "prompt", "answer", {}),
        ),
    )

    class Plan:
        class Run:
            cache_budget_tokens = 256
            retention_policy = "sliding_window"
            random_seed = 17
            always_keep_prefix_tokens = 1
            preserve_recent_window = True
            eviction_trigger = "budget_threshold"
            sentence_boundary_lookahead_tokens = 25
            force_eviction_overflow_ratio = 1.2

        run = Run()

    def fake_select_live_eviction_plan(**kwargs):
        assert kwargs == {
            "cache_budget_tokens": 256,
            "retention_policy": "sliding_window",
            "random_seed": 17,
            "always_keep_prefix_tokens": 1,
            "preserve_recent_window": True,
            "sentence_pooling": "max",
            "sentence_top_k": 3,
            "eviction_trigger": "budget_threshold",
            "sentence_boundary_lookahead_tokens": 25,
            "force_eviction_overflow_ratio": 1.2,
        }
        return Plan()

    monkeypatch.setattr(remote_exec, "select_live_eviction_plan", fake_select_live_eviction_plan)

    class FakeGenerator:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(remote_exec, "TransformersLiveEvictionGenerator", FakeGenerator)
    monkeypatch.setattr(
        remote_exec,
        "run_live_eviction",
        lambda plan, cases, generator, **kwargs: (
            RunResult(
                run_id="step2-niah-sliding-window-live-b256",
                benchmark="niah",
                baseline="sliding_window_live",
                metrics={"needle_hit_rate": 0.75},
                metadata={"gpu": "A100-80GB", "cache_budget_tokens": "256"},
            ),
            (),
        ),
    )
    monkeypatch.setattr(remote_exec.time, "perf_counter", lambda: next(counter))

    counter = iter([5.0, 8.0])
    report = remote_exec.run_modal_live_eviction_niah(
        remote_exec.ModalLiveEvictionRunRequest(
            case_limit=1,
            cache_budget_tokens=256,
            retention_policy="sliding_window",
        )
    )

    assert report.cache_budget_tokens == 256
    assert report.retention_policy == "sliding_window"
    assert report.result.run_id == "step2-niah-sliding-window-live-b256"


def test_run_modal_live_eviction_niah_supports_sentence_level_variants(monkeypatch):
    monkeypatch.setattr(
        remote_exec,
        "load_ruler_hf_niah_slice",
        lambda dataset_config, split, case_limit, **kwargs: (
            BenchmarkCase("c1", "prompt", "answer", {}),
        ),
    )

    class Plan:
        class Run:
            cache_budget_tokens = 1024
            retention_policy = "sentence_vorn"
            random_seed = 17
            always_keep_prefix_tokens = 0
            preserve_recent_window = False
            sentence_pooling = "max"
            sentence_top_k = 3
            eviction_trigger = "budget_threshold"
            sentence_boundary_lookahead_tokens = 25
            force_eviction_overflow_ratio = 1.2

        run = Run()

    def fake_select_live_eviction_plan(**kwargs):
        assert kwargs == {
            "cache_budget_tokens": 1024,
            "retention_policy": "sentence_vorn",
            "random_seed": 17,
            "always_keep_prefix_tokens": 0,
            "preserve_recent_window": False,
            "sentence_pooling": "max",
            "sentence_top_k": 3,
            "eviction_trigger": "budget_threshold",
            "sentence_boundary_lookahead_tokens": 25,
            "force_eviction_overflow_ratio": 1.2,
        }
        return Plan()

    monkeypatch.setattr(
        remote_exec,
        "select_live_eviction_plan",
        fake_select_live_eviction_plan,
    )

    class FakeGenerator:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(remote_exec, "TransformersLiveEvictionGenerator", FakeGenerator)
    monkeypatch.setattr(
        remote_exec,
        "run_live_eviction",
        lambda plan, cases, generator, **kwargs: (
            RunResult(
                run_id="step2-niah-sentence-vorn-live-b1024-noguards",
                benchmark="niah",
                baseline="sentence_vorn_live",
                metrics={"needle_hit_rate": 0.2},
                metadata={"gpu": "A100-80GB", "cache_budget_tokens": "1024"},
            ),
            (),
        ),
    )
    monkeypatch.setattr(remote_exec.time, "perf_counter", lambda: next(counter))

    counter = iter([3.0, 9.0])
    report = remote_exec.run_modal_live_eviction_niah(
        remote_exec.ModalLiveEvictionRunRequest(
            case_limit=1,
            cache_budget_tokens=1024,
            retention_policy="sentence_vorn",
            always_keep_prefix_tokens=0,
            preserve_recent_window=False,
            sentence_pooling="max",
            sentence_top_k=3,
            model_id="meta-llama/Llama-3.1-8B-Instruct",
        )
    )

    assert report.retention_policy == "sentence_vorn"
    assert report.always_keep_prefix_tokens == 0
    assert report.preserve_recent_window is False
    assert report.sentence_pooling == "max"
    assert report.sentence_top_k == 3
    assert report.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert report.result.run_id == "step2-niah-sentence-vorn-live-b1024-noguards"


def test_run_modal_live_eviction_niah_supports_sentence_level_tova_variant(monkeypatch):
    monkeypatch.setattr(
        remote_exec,
        "load_ruler_hf_niah_slice",
        lambda dataset_config, split, case_limit, **kwargs: (
            BenchmarkCase("c1", "prompt", "answer", {}),
        ),
    )

    class Plan:
        class Run:
            cache_budget_tokens = 1024
            retention_policy = "sentence_tova"
            random_seed = 17
            always_keep_prefix_tokens = 1
            preserve_recent_window = True
            sentence_pooling = "max"
            sentence_top_k = 3
            eviction_trigger = "budget_threshold"
            sentence_boundary_lookahead_tokens = 25
            force_eviction_overflow_ratio = 1.2

        run = Run()

    def fake_select_live_eviction_plan(**kwargs):
        assert kwargs == {
            "cache_budget_tokens": 1024,
            "retention_policy": "sentence_tova",
            "random_seed": 17,
            "always_keep_prefix_tokens": 1,
            "preserve_recent_window": True,
            "sentence_pooling": "max",
            "sentence_top_k": 3,
            "eviction_trigger": "budget_threshold",
            "sentence_boundary_lookahead_tokens": 25,
            "force_eviction_overflow_ratio": 1.2,
        }
        return Plan()

    monkeypatch.setattr(
        remote_exec,
        "select_live_eviction_plan",
        fake_select_live_eviction_plan,
    )

    class FakeGenerator:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(remote_exec, "TransformersLiveEvictionGenerator", FakeGenerator)
    monkeypatch.setattr(
        remote_exec,
        "run_live_eviction",
        lambda plan, cases, generator, **kwargs: (
            RunResult(
                run_id="step2-niah-sentence-tova-live-b1024",
                benchmark="niah",
                baseline="sentence_tova_live",
                metrics={"needle_hit_rate": 0.22},
                metadata={"gpu": "A100-80GB", "cache_budget_tokens": "1024"},
            ),
            (),
        ),
    )
    monkeypatch.setattr(remote_exec.time, "perf_counter", lambda: next(counter))

    counter = iter([4.0, 11.0])
    report = remote_exec.run_modal_live_eviction_niah(
        remote_exec.ModalLiveEvictionRunRequest(
            case_limit=1,
            cache_budget_tokens=1024,
            retention_policy="sentence_tova",
            sentence_pooling="max",
            sentence_top_k=3,
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
        )
    )

    assert report.retention_policy == "sentence_tova"
    assert report.sentence_pooling == "max"
    assert report.sentence_top_k == 3
    assert report.result.run_id == "step2-niah-sentence-tova-live-b1024"


def test_run_modal_live_eviction_longbench_uses_preregistered_contract(monkeypatch):
    calls = {}

    def fake_load_longbench(**kwargs):
        calls["load_longbench"] = kwargs
        return (
            BenchmarkCase(
                "case-1",
                "prompt",
                "Paragraph 15",
                {"scoring_contract": "longbench_retrieval_score_v1"},
            ),
        )

    monkeypatch.setattr(
        remote_exec,
        "load_longbench_passage_retrieval_en_slice",
        fake_load_longbench,
    )

    class FakeGenerator:
        def __init__(self, config):
            self.config = config
            assert config.model_id == "meta-llama/Llama-3.1-8B-Instruct"
            assert config.max_new_tokens == 32

        def ensure_model_loaded(self):
            return 4.25

        def unload_model(self):
            return 0.75

    monkeypatch.setattr(remote_exec, "TransformersLiveEvictionGenerator", FakeGenerator)

    def fake_build_execution_plans(runs):
        run = runs[0]
        calls["run"] = run
        assert run.benchmark == "longbench_passage_retrieval_en"
        assert run.retention_policy == "sentence_vorn"
        assert run.cache_budget_tokens == 1024
        assert run.sentence_pooling == "max"
        assert run.sentence_top_k == 3
        return ("longbench-plan",)

    monkeypatch.setattr(remote_exec, "build_execution_plans", fake_build_execution_plans)

    def fake_run_live_eviction(plan, cases, generator, **kwargs):
        assert plan == "longbench-plan"
        assert len(cases) == 1
        return (
            RunResult(
                run_id="step2-longbench-passage-sentence-vorn-live-b1024",
                benchmark="longbench_passage_retrieval_en",
                baseline="sentence_vorn_live",
                metrics={
                    "mean_official_score": 1.0,
                    "longbench_percent": 100.0,
                    "binary_paragraph_hit_rate": 1.0,
                },
                metadata={"gpu": "A100-80GB"},
                observations=(
                    CaseObservation(
                        fixture_id="case-1",
                        correct=True,
                        prediction="Paragraph 15",
                        official_score=1.0,
                        binary_paragraph_hit=True,
                        numbers_extracted=("15",),
                    ),
                ),
            ),
            (),
        )

    monkeypatch.setattr(remote_exec, "run_live_eviction", fake_run_live_eviction)
    monkeypatch.setattr(remote_exec.time, "perf_counter", lambda: next(counter))

    counter = iter([20.0, 50.0])
    report = remote_exec.run_modal_live_eviction_longbench_passage_retrieval(
        remote_exec.ModalLongBenchLiveEvictionRunRequest(
            case_limit=1,
            model_id="meta-llama/Llama-3.1-8B-Instruct",
            retention_policy="sentence_vorn",
            gpu="H200",
        )
    )

    assert calls["load_longbench"]["case_limit"] == 1
    assert calls["load_longbench"]["case_offset_start"] == 0
    assert report.dataset_config == "passage_retrieval_en"
    assert report.split == "test[:1]"
    assert report.cache_budget_tokens == 1024
    assert report.retention_policy == "sentence_vorn"
    assert report.result.metadata["dataset_id"] == "THUDM/LongBench"
    assert report.result.metadata["prompt_template_id"] == (
        "longbench_passage_retrieval_en_v1"
    )
    assert report.result.metadata["primary_metric"] == "mean_official_score"
    assert report.result.metadata["secondary_metric"] == "binary_paragraph_hit_rate"
    assert report.result.metadata["preregistration"] == "config#316"
    assert report.result.metadata["gpu_hours"] == "0.008333"
    assert report.result.metadata["model_load_elapsed_seconds"] == "4.250000"
    assert report.result.metadata["model_unload_elapsed_seconds"] == "0.750000"
    assert report.result.metadata["vanilla_delta_available"] == "false"
    assert report.estimated_cost_usd == 30.0 * remote_exec.per_second_rate_for_gpu("H200")
    assert report.elapsed_seconds == 30.0


def test_run_modal_live_eviction_longbench_rejects_out_of_spec_policy():
    with pytest.raises(ValueError, match="sentence_vorn"):
        remote_exec.run_modal_live_eviction_longbench_passage_retrieval(
            remote_exec.ModalLongBenchLiveEvictionRunRequest(
                retention_policy="h2o",
            )
        )


def test_run_modal_live_eviction_longbench_rejects_changed_max_new_tokens():
    with pytest.raises(ValueError, match="max_new_tokens"):
        remote_exec.run_modal_live_eviction_longbench_passage_retrieval(
            remote_exec.ModalLongBenchLiveEvictionRunRequest(
                max_new_tokens=64,
            )
        )


def test_run_modal_live_eviction_niah_supports_sentence_level_h2o_variant(monkeypatch):
    monkeypatch.setattr(
        remote_exec,
        "load_ruler_hf_niah_slice",
        lambda dataset_config, split, case_limit, **kwargs: (
            BenchmarkCase("c1", "prompt", "answer", {}),
        ),
    )

    class Plan:
        class Run:
            cache_budget_tokens = 512
            retention_policy = "sentence_h2o"
            random_seed = 17
            always_keep_prefix_tokens = 1
            preserve_recent_window = True
            sentence_pooling = "max"
            sentence_top_k = 3
            eviction_trigger = "budget_threshold"
            sentence_boundary_lookahead_tokens = 25
            force_eviction_overflow_ratio = 1.2

        run = Run()

    def fake_select_live_eviction_plan(**kwargs):
        assert kwargs == {
            "cache_budget_tokens": 512,
            "retention_policy": "sentence_h2o",
            "random_seed": 17,
            "always_keep_prefix_tokens": 1,
            "preserve_recent_window": True,
            "sentence_pooling": "max",
            "sentence_top_k": 3,
            "eviction_trigger": "budget_threshold",
            "sentence_boundary_lookahead_tokens": 25,
            "force_eviction_overflow_ratio": 1.2,
        }
        return Plan()

    monkeypatch.setattr(
        remote_exec,
        "select_live_eviction_plan",
        fake_select_live_eviction_plan,
    )

    class FakeGenerator:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(remote_exec, "TransformersLiveEvictionGenerator", FakeGenerator)
    monkeypatch.setattr(
        remote_exec,
        "run_live_eviction",
        lambda plan, cases, generator, **kwargs: (
            RunResult(
                run_id="step2-niah-sentence-h2o-live-b512",
                benchmark="niah",
                baseline="sentence_h2o_live",
                metrics={"needle_hit_rate": 0.24},
                metadata={"gpu": "A100-80GB", "cache_budget_tokens": "512"},
            ),
            (),
        ),
    )
    monkeypatch.setattr(remote_exec.time, "perf_counter", lambda: next(counter))

    counter = iter([6.0, 14.0])
    report = remote_exec.run_modal_live_eviction_niah(
        remote_exec.ModalLiveEvictionRunRequest(
            case_limit=1,
            cache_budget_tokens=512,
            retention_policy="sentence_h2o",
            sentence_pooling="max",
            sentence_top_k=3,
        )
    )

    assert report.retention_policy == "sentence_h2o"
    assert report.sentence_pooling == "max"
    assert report.sentence_top_k == 3
    assert report.result.run_id == "step2-niah-sentence-h2o-live-b512"


def test_run_modal_live_eviction_niah_supports_word_level_variants(monkeypatch):
    monkeypatch.setattr(
        remote_exec,
        "load_ruler_hf_niah_slice",
        lambda dataset_config, split, case_limit, **kwargs: (
            BenchmarkCase("c1", "prompt", "answer", {}),
        ),
    )

    class Plan:
        class Run:
            cache_budget_tokens = 1536
            retention_policy = "word_vorn"
            random_seed = 17
            always_keep_prefix_tokens = 0
            preserve_recent_window = False
            sentence_pooling = "max"
            sentence_top_k = 3

        run = Run()

    def fake_select_live_eviction_plan(**kwargs):
        assert kwargs == {
            "cache_budget_tokens": 1536,
            "retention_policy": "word_vorn",
            "random_seed": 17,
            "always_keep_prefix_tokens": 0,
            "preserve_recent_window": False,
            "sentence_pooling": "max",
            "sentence_top_k": 3,
            "eviction_trigger": "budget_threshold",
            "sentence_boundary_lookahead_tokens": 25,
            "force_eviction_overflow_ratio": 1.2,
        }
        return Plan()

    monkeypatch.setattr(
        remote_exec,
        "select_live_eviction_plan",
        fake_select_live_eviction_plan,
    )

    class FakeGenerator:
        def __init__(self, config):
            self.config = config
            assert config.model_id == "Qwen/Qwen2.5-7B-Instruct"

    monkeypatch.setattr(remote_exec, "TransformersLiveEvictionGenerator", FakeGenerator)
    monkeypatch.setattr(
        remote_exec,
        "run_live_eviction",
        lambda plan, cases, generator, **kwargs: (
            RunResult(
                run_id="step2-niah-word-vorn-live-b1536-noguards",
                benchmark="niah",
                baseline="word_vorn_live",
                metrics={"needle_hit_rate": 0.12},
                metadata={"gpu": "A100-80GB", "cache_budget_tokens": "1536"},
            ),
            (),
        ),
    )
    monkeypatch.setattr(remote_exec.time, "perf_counter", lambda: next(counter))

    counter = iter([2.0, 10.0])
    report = remote_exec.run_modal_live_eviction_niah(
        remote_exec.ModalLiveEvictionRunRequest(
            case_limit=1,
            cache_budget_tokens=1536,
            retention_policy="word_vorn",
            always_keep_prefix_tokens=0,
            preserve_recent_window=False,
            sentence_pooling="max",
            sentence_top_k=3,
            model_id="Qwen/Qwen2.5-7B-Instruct",
        )
    )

    assert report.retention_policy == "word_vorn"
    assert report.always_keep_prefix_tokens == 0
    assert report.preserve_recent_window is False
    assert report.sentence_pooling == "max"
    assert report.sentence_top_k == 3
    assert report.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert report.result.run_id == "step2-niah-word-vorn-live-b1536-noguards"


def test_run_modal_live_eviction_niah_supports_adaptive_vorn_variants(monkeypatch):
    monkeypatch.setattr(
        remote_exec,
        "load_ruler_hf_niah_slice",
        lambda dataset_config, split, case_limit, **kwargs: (
            BenchmarkCase("c1", "prompt", "answer", {}),
        ),
    )

    class Plan:
        class Run:
            cache_budget_tokens = 1536
            retention_policy = "adaptive_vorn"
            random_seed = 17
            always_keep_prefix_tokens = 1
            preserve_recent_window = True
            sentence_pooling = "max"
            sentence_top_k = 3

        run = Run()

    def fake_select_live_eviction_plan(**kwargs):
        assert kwargs == {
            "cache_budget_tokens": 1536,
            "retention_policy": "adaptive_vorn",
            "random_seed": 17,
            "always_keep_prefix_tokens": 1,
            "preserve_recent_window": True,
            "sentence_pooling": "max",
            "sentence_top_k": 3,
            "eviction_trigger": "budget_threshold",
            "sentence_boundary_lookahead_tokens": 25,
            "force_eviction_overflow_ratio": 1.2,
        }
        return Plan()

    monkeypatch.setattr(
        remote_exec,
        "select_live_eviction_plan",
        fake_select_live_eviction_plan,
    )

    class FakeGenerator:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(remote_exec, "TransformersLiveEvictionGenerator", FakeGenerator)
    monkeypatch.setattr(
        remote_exec,
        "run_live_eviction",
        lambda plan, cases, generator, **kwargs: (
            RunResult(
                run_id="step2-niah-adaptive-vorn-live-b1536",
                benchmark="niah",
                baseline="adaptive_vorn_live",
                metrics={"needle_hit_rate": 0.6},
                metadata={
                    "gpu": "A100-80GB",
                    "cache_budget_tokens": "1536",
                    "adaptive_token_steps": "5",
                    "adaptive_sentence_steps": "7",
                    "adaptive_selector_contract": (
                        "choose_token_or_sentence_by_peak_zscore_over_current_alignment_scores"
                    ),
                },
            ),
            (),
        ),
    )
    monkeypatch.setattr(remote_exec.time, "perf_counter", lambda: next(counter))

    counter = iter([11.0, 21.0])
    report = remote_exec.run_modal_live_eviction_niah(
        remote_exec.ModalLiveEvictionRunRequest(
            case_limit=1,
            cache_budget_tokens=1536,
            retention_policy="adaptive_vorn",
            sentence_pooling="max",
            sentence_top_k=3,
        )
    )

    assert report.retention_policy == "adaptive_vorn"
    assert report.cache_budget_tokens == 1536
    assert report.result.run_id == "step2-niah-adaptive-vorn-live-b1536"
    assert report.result.metadata["adaptive_token_steps"] == "5"
    assert report.result.metadata["adaptive_sentence_steps"] == "7"


def test_run_modal_vanilla_observation_niah_returns_structured_report(monkeypatch):
    monkeypatch.setattr(
        remote_exec,
        "load_ruler_hf_niah_slice",
        lambda dataset_config, split, case_limit, **kwargs: (
            BenchmarkCase("c1", "prompt", "answer", {}),
        ),
    )

    class FakeGenerator:
        def __init__(self, config):
            self.config = config

        def observe_vanilla_case(
            self,
            case,
            canonical_layer,
            recent_token_window,
            top_k,
            attention_last_n_layers,
        ):
            assert canonical_layer == 16
            assert recent_token_window == 16
            assert top_k == 10
            assert attention_last_n_layers == 4
            return ObservationCase(
                case_id=case.case_id,
                expected_answer=case.expected_answer,
                prediction="answer",
                success=True,
                prompt_token_count=4,
                answer_token_spans=((1, 2),),
                steps=(),
            )

    monkeypatch.setattr(remote_exec, "TransformersObservationGenerator", FakeGenerator)
    monkeypatch.setattr(remote_exec.time, "perf_counter", lambda: next(counter))

    counter = iter([10.0, 15.5])
    report = remote_exec.run_modal_vanilla_observation_niah(
        remote_exec.ModalVanillaObservationRequest(case_limit=1)
    )

    assert report.dataset_config == "niah_multikey_1_4k"
    assert report.case_count == 1
    assert report.elapsed_seconds == 5.5
    assert report.estimated_cost_usd == 5.5 * remote_exec.A100_80GB_PER_SECOND
    assert len(report.cases) == 1
    assert report.cases[0].success is True


def test_run_modal_score_distribution_observation_niah_returns_structured_report(
    monkeypatch,
):
    monkeypatch.setattr(
        remote_exec,
        "load_ruler_hf_niah_slice",
        lambda dataset_config, split, case_limit, **kwargs: (
            BenchmarkCase("c1", "prompt", "answer", {}),
        ),
    )

    class FakeGenerator:
        def __init__(self, config):
            self.config = config
            assert config.model_id == "mistralai/Mistral-7B-Instruct-v0.3"

        def observe_live_case(self, case, *, config):
            assert case.case_id == "c1"
            assert config.cache_budget_tokens == 1024
            assert config.retention_policy == "sentence_vorn"
            return ScoreDistributionObservationCase(
                case_id="c1",
                expected_answer="answer",
                prediction="answer",
                success=True,
                observations=(
                    CaseObservation(
                        fixture_id="c1",
                        correct=True,
                        prediction="answer",
                    ),
                ),
                steps=(
                    ScoreDistributionObservationStep(
                        step_index=0,
                        active_token_count=1025,
                        granularity_stats={
                            "token": ScoreDistributionStats(
                                position_count=4,
                                score_min=0.1,
                                score_max=0.9,
                                score_mean=0.4,
                                score_median=0.35,
                                score_std=0.2,
                                score_q10=0.1,
                                score_q25=0.2,
                                score_q75=0.5,
                                score_q90=0.8,
                                peak_zscore=2.5,
                                top10_mass_fraction=1.0,
                                top25_mass_fraction=1.0,
                                top50_mass_fraction=1.0,
                                entropy=0.9,
                                normalized_entropy=0.7,
                                kl_divergence_from_uniform=0.1,
                                q90_minus_q50=0.45,
                                q75_minus_q25=0.3,
                                above_median_plus_std_count=1,
                                above_median_plus_std_fraction=0.25,
                                spatial_coherence=0.5,
                            )
                        },
                    ),
                ),
            )

    monkeypatch.setattr(
        remote_exec,
        "TransformersScoreDistributionObservationGenerator",
        FakeGenerator,
    )
    monkeypatch.setattr(remote_exec.time, "perf_counter", lambda: next(counter))

    counter = iter([100.0, 112.0])
    report = remote_exec.run_modal_score_distribution_observation_niah(
        remote_exec.ModalScoreDistributionObservationRequest(
            dataset_config="niah_multikey_1_8k",
            case_limit=1,
            cache_budget_tokens=1024,
            retention_policy="sentence_vorn",
        )
    )

    assert isinstance(report, ScoreDistributionObservationReport)
    assert report.dataset_config == "niah_multikey_1_8k"
    assert report.case_count == 1
    assert report.cache_budget_tokens == 1024
    assert report.retention_policy == "sentence_vorn"
    assert report.elapsed_seconds == 12.0
    assert report.estimated_cost_usd == 12.0 * remote_exec.A100_80GB_PER_SECOND
    assert report.cases[0].observations[0].fixture_id == "c1"


def test_run_modal_consumer_validation_niah_writes_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        remote_exec,
        "load_ruler_hf_niah_slice",
        lambda dataset_config, split, case_limit, **kwargs: (
            BenchmarkCase("case-1", "raw prompt", "blue", {}),
        ),
    )

    class FakeConsumerGenerator:
        def __init__(self, config):
            self.config = config

        def generate(self, prompt):
            return "blue"

        def generate_rendered_prompt(self, rendered_prompt):
            if "High magic clue." in rendered_prompt:
                return "blue"
            return "wrong"

        def render_prompt_text_with_offsets(self, prompt):
            return _CONSUMER_RENDERED_PROMPT, ()

        def count_rendered_prompt_tokens(self, rendered_prompt):
            return len(rendered_prompt.split())

    monkeypatch.setattr(
        remote_exec,
        "TransformersTextGenerator",
        FakeConsumerGenerator,
    )
    reset_calls = []
    monkeypatch.setattr(
        remote_exec,
        "reset_runtime_telemetry",
        lambda: reset_calls.append("reset"),
    )
    monkeypatch.setattr(
        remote_exec,
        "capture_runtime_telemetry",
        lambda: {
            "peak_memory_allocated_gb": 0.25,
            "peak_memory_reserved_gb": 0.5,
        },
    )
    monkeypatch.setattr(remote_exec.time, "perf_counter", lambda: next(counter))
    counter = iter([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
    jsonl_path = tmp_path / "consumer-validation.jsonl"
    summary_path = tmp_path / "consumer-validation.md"

    report = remote_exec.run_modal_consumer_validation_niah(
        remote_exec.ModalConsumerValidationRunRequest(
            output_jsonl_path=str(jsonl_path),
            output_summary_path=str(summary_path),
            phase0_case=_consumer_phase0_case(),
            protected_semu_ids=(0,),
            selector_arms=("vorn_high", "vorn_low"),
            expected_selected_semu_ids=(
                ("vorn_high", 1),
                ("vorn_low", 4),
            ),
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            cost_per_second=0.01,
        )
    )

    assert report.case_id == "case-1"
    assert report.case_count == 1
    assert report.elapsed_seconds == 7.0
    assert report.estimated_cost_usd == 0.07
    assert report.output_jsonl_path == str(jsonl_path)
    assert jsonl_path.exists()
    assert summary_path.exists()
    assert len(reset_calls) == 3
    assert report.report.records[0].record_status == "PROMPT_QUALITY_SUCCESS"
    assert (
        report.report.records[0].quality_record.peak_memory_allocated_mb
        == 256.0
    )


def test_modal_consumer_validation_request_defaults_to_phase3_a100():
    request = remote_exec.ModalConsumerValidationRunRequest()

    assert request.gpu == "A100-80GB"
    assert request.cost_per_second == remote_exec.A100_80GB_PER_SECOND


def test_run_modal_pressure_sweep_niah_writes_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(
        remote_exec,
        "load_ruler_hf_niah_slice",
        lambda dataset_config, split, case_limit, **kwargs: (
            BenchmarkCase("case-1", "raw prompt", "blue", {}),
        ),
    )

    class FakePressureGenerator:
        def __init__(self, config):
            self.config = config

        def generate(self, prompt):
            return "blue"

        def generate_rendered_prompt(self, rendered_prompt):
            if "High magic clue." in rendered_prompt:
                return "blue"
            return "wrong"

        def render_prompt_text_with_offsets(self, prompt):
            return _CONSUMER_RENDERED_PROMPT, ()

        def count_rendered_prompt_tokens(self, rendered_prompt):
            return len(rendered_prompt.split())

    monkeypatch.setattr(
        remote_exec,
        "TransformersTextGenerator",
        FakePressureGenerator,
    )
    reset_calls = []
    monkeypatch.setattr(
        remote_exec,
        "reset_runtime_telemetry",
        lambda: reset_calls.append("reset"),
    )
    monkeypatch.setattr(
        remote_exec,
        "capture_runtime_telemetry",
        lambda: {
            "peak_memory_allocated_gb": 0.25,
            "peak_memory_reserved_gb": 0.5,
        },
    )
    monkeypatch.setattr(remote_exec.time, "perf_counter", lambda: next(counter))
    counter = iter(
        [
            100.0,
            101.0,
            102.0,
            103.0,
            104.0,
            105.0,
            106.0,
            107.0,
            108.0,
            109.0,
            110.0,
            111.0,
        ]
    )
    jsonl_path = tmp_path / "pressure-sweep.jsonl"
    summary_path = tmp_path / "pressure-sweep.md"

    report = remote_exec.run_modal_pressure_sweep_niah(
        remote_exec.ModalPressureSweepRunRequest(
            output_jsonl_path=str(jsonl_path),
            output_summary_path=str(summary_path),
            phase0_case=_consumer_phase0_case(),
            protected_semu_ids=(0,),
            selector_arms=("vorn_high", "vorn_low"),
            pressure_ns=(1, 2),
            model_id="mistralai/Mistral-7B-Instruct-v0.3",
            cost_per_second=0.01,
        )
    )

    assert report.case_id == "case-1"
    assert report.case_count == 1
    assert report.elapsed_seconds == 11.0
    assert report.estimated_cost_usd == 0.11
    assert report.output_jsonl_path == str(jsonl_path)
    assert jsonl_path.exists()
    assert summary_path.exists()
    assert len(reset_calls) == 5
    assert len(report.report.records) == 4
    assert report.report.records[0].intervention.pressure_n == 1
    assert report.report.records[0].intervention.selected_semu_ids == (1,)


def _consumer_phase0_case() -> dict[str, object]:
    return {
        "case_id": "case-1",
        "prompt_hash": sha256_text(_CONSUMER_RENDERED_PROMPT),
        "prompt_token_count": len(_CONSUMER_RENDERED_PROMPT.split()),
        "semu_matrix": (
            _consumer_semu_payload(0, "[INST] Setup.", 0.99, 0, 0, 2),
            _consumer_semu_payload(1, "High magic clue.", 0.90, 1, 2, 6),
            _consumer_semu_payload(
                2,
                "Long neutral sentence with many extra words.",
                0.50,
                2,
                6,
                14,
            ),
            _consumer_semu_payload(3, "Length match.", 0.40, 3, 14, 18),
            _consumer_semu_payload(4, "Low filler.", 0.10, 4, 18, 20),
        ),
    }


def _consumer_semu_payload(
    semu_id: int,
    text: str,
    final_score: float,
    final_rank: int,
    token_start: int,
    token_end: int,
) -> dict[str, object]:
    char_start = _CONSUMER_RENDERED_PROMPT.index(text)
    return {
        "semu_id": semu_id,
        "char_start": char_start,
        "char_end": char_start + len(text),
        "token_start": token_start,
        "token_end": token_end,
        "text_preview": text,
        "final_score": final_score,
        "final_rank": final_rank,
        "answer_overlap": False,
    }
