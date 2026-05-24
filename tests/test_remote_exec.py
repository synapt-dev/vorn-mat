from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import vorn_mat.remote_exec as remote_exec
from vorn_mat.benchmarks.common import BenchmarkCase
from vorn_mat.observation import ObservationCase
from vorn_mat.results import CaseObservation, RunResult
from vorn_mat.score_distribution_observation import (
    ScoreDistributionObservationCase,
    ScoreDistributionObservationReport,
    ScoreDistributionObservationStep,
    ScoreDistributionStats,
)


def test_run_modal_vanilla_niah_enriches_result_metadata(monkeypatch):
    monkeypatch.setattr(
        remote_exec,
        "load_ruler_hf_niah_slice",
        lambda dataset_config, split, case_limit: (
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
    monkeypatch.setattr(
        remote_exec,
        "load_ruler_hf_niah_slice",
        lambda dataset_config, split, case_limit: (
            BenchmarkCase("c1", "prompt", "answer", {}),
        ),
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
            case_limit=1,
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
    assert report.case_count == 1
    assert report.cache_budget_tokens == 256
    assert report.retention_policy == "random"
    assert report.elapsed_seconds == 16.0
    assert report.estimated_cost_usd == 16.0 * remote_exec.A100_80GB_PER_SECOND
    assert report.result.metadata["case_count"] == "1"
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
        lambda dataset_config, split, case_limit: (
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
        lambda dataset_config, split, case_limit: (
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
        lambda dataset_config, split, case_limit: (
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


def test_run_modal_live_eviction_niah_supports_sentence_level_h2o_variant(monkeypatch):
    monkeypatch.setattr(
        remote_exec,
        "load_ruler_hf_niah_slice",
        lambda dataset_config, split, case_limit: (
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
        lambda dataset_config, split, case_limit: (
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
        lambda dataset_config, split, case_limit: (
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
        lambda dataset_config, split, case_limit: (
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
        lambda dataset_config, split, case_limit: (
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
