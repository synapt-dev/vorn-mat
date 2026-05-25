from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat import (
    LocalModelConfig,
    build_summary_answer_prompt,
    build_summary_prompt,
    load_results,
    run_local_live_eviction_smoke,
    run_local_step1_pair,
    run_local_vanilla_smoke,
    select_live_eviction_plan,
    select_step1_plans,
    split_context_and_question_for_summary,
    select_week1_plan,
)
from vorn_mat.baselines.live_eviction import LiveEvictionStats, LiveEvictionTextGenerator
from vorn_mat.baselines.vanilla import TextGenerator
from vorn_mat.baselines.vorn import RetentionStats, VornTextGenerator


class FakeGenerator(TextGenerator):
    def __init__(self, answers: dict[str, str]):
        self.answers = answers

    def generate(self, prompt: str) -> str:
        return self.answers[prompt]


class FakeVornGenerator(VornTextGenerator):
    def __init__(self, answers: dict[str, str], stats: RetentionStats):
        self.answers = answers
        self.stats = stats

    def generate_with_retention(self, prompt: str, config):
        return self.answers[prompt], self.stats


class FakeLiveGenerator(LiveEvictionTextGenerator):
    def __init__(self, answers: dict[str, str], stats: LiveEvictionStats):
        self.answers = answers
        self.stats = stats

    def generate_with_live_eviction(self, prompt: str, config, **kwargs):
        return self.answers[prompt], self.stats


def test_local_model_config_defaults_to_step0_shape():
    config = LocalModelConfig()

    assert config.model_id == "mistralai/Mistral-7B-Instruct-v0.3"
    assert config.max_new_tokens == 32
    assert config.trust_remote_code is True
    assert config.attention_implementation == "eager"


def test_select_week1_plan_targets_niah_vanilla():
    plan = select_week1_plan("niah")

    assert plan.run.run_id == "week1-niah-vanilla"
    assert plan.benchmark.name == "niah"
    assert plan.baseline.name == "vanilla"


def test_select_step1_plans_returns_vanilla_and_vorn_pair():
    vanilla, vorn = select_step1_plans()

    assert vanilla.run.run_id == "step1-niah-vanilla"
    assert vorn.run.run_id == "step1-niah-vorn"
    assert vanilla.baseline.name == "vanilla"
    assert vorn.baseline.name == "vorn"


def test_select_live_eviction_plan_targets_step2_vorn_live():
    plan = select_live_eviction_plan()

    assert plan.run.run_id == "step2-niah-vorn-live"
    assert plan.benchmark.name == "niah"
    assert plan.baseline.name == "vorn_live"


def test_select_live_eviction_plan_supports_random_control_and_budget_override():
    plan = select_live_eviction_plan(
        cache_budget_tokens=512,
        retention_policy="random",
        random_seed=23,
    )

    assert plan.run.run_id == "step2-niah-random-live-b512"
    assert plan.run.cache_budget_tokens == 512
    assert plan.run.retention_policy == "random"
    assert plan.run.random_seed == 23
    assert plan.baseline.name == "random_live"


def test_select_live_eviction_plan_supports_sliding_window_control():
    plan = select_live_eviction_plan(
        cache_budget_tokens=256,
        retention_policy="sliding_window",
        random_seed=17,
    )

    assert plan.run.run_id == "step2-niah-sliding-window-live-b256"
    assert plan.run.cache_budget_tokens == 256
    assert plan.run.retention_policy == "sliding_window"
    assert plan.baseline.name == "sliding_window_live"


def test_select_live_eviction_plan_supports_prefix_suffix_control():
    plan = select_live_eviction_plan(
        cache_budget_tokens=256,
        retention_policy="prefix_suffix",
        random_seed=17,
    )

    assert plan.run.run_id == "step2-niah-prefix-suffix-live-b256"
    assert plan.run.cache_budget_tokens == 256
    assert plan.run.retention_policy == "prefix_suffix"
    assert plan.baseline.name == "prefix_suffix_live"


def test_select_live_eviction_plan_supports_summarize_control():
    plan = select_live_eviction_plan(
        cache_budget_tokens=256,
        retention_policy="summarize",
        random_seed=17,
    )

    assert plan.run.run_id == "step2-niah-summarize-compact-b256"
    assert plan.run.cache_budget_tokens == 256
    assert plan.run.retention_policy == "summarize"
    assert plan.baseline.name == "summarize_compact"


def test_select_live_eviction_plan_supports_tova_control():
    plan = select_live_eviction_plan(
        cache_budget_tokens=1024,
        retention_policy="tova",
        random_seed=17,
    )

    assert plan.run.run_id == "step2-niah-tova-live-b1024"
    assert plan.run.cache_budget_tokens == 1024
    assert plan.run.retention_policy == "tova"
    assert plan.baseline.name == "tova_live"


def test_select_live_eviction_plan_supports_h2o_control():
    plan = select_live_eviction_plan(
        cache_budget_tokens=1024,
        retention_policy="h2o",
        random_seed=17,
    )

    assert plan.run.run_id == "step2-niah-h2o-live-b1024"
    assert plan.run.cache_budget_tokens == 1024
    assert plan.run.retention_policy == "h2o"
    assert plan.baseline.name == "h2o_live"


def test_select_live_eviction_plan_supports_sentence_level_tova_control():
    plan = select_live_eviction_plan(
        cache_budget_tokens=1024,
        retention_policy="sentence_tova",
        random_seed=17,
        sentence_pooling="max",
        sentence_top_k=3,
    )

    assert plan.run.run_id == "step2-niah-sentence-tova-live-b1024"
    assert plan.run.cache_budget_tokens == 1024
    assert plan.run.retention_policy == "sentence_tova"
    assert plan.run.eviction_unit == "sentence"
    assert plan.run.sentence_pooling == "max"
    assert plan.run.sentence_top_k == 3
    assert plan.baseline.name == "sentence_tova_live"


def test_select_live_eviction_plan_supports_sentence_level_h2o_control():
    plan = select_live_eviction_plan(
        cache_budget_tokens=1024,
        retention_policy="sentence_h2o",
        random_seed=17,
        sentence_pooling="max",
        sentence_top_k=3,
    )

    assert plan.run.run_id == "step2-niah-sentence-h2o-live-b1024"
    assert plan.run.cache_budget_tokens == 1024
    assert plan.run.retention_policy == "sentence_h2o"
    assert plan.run.eviction_unit == "sentence"
    assert plan.run.sentence_pooling == "max"
    assert plan.run.sentence_top_k == 3
    assert plan.baseline.name == "sentence_h2o_live"


def test_select_live_eviction_plan_supports_no_guardrails_vorn():
    plan = select_live_eviction_plan(
        cache_budget_tokens=1024,
        retention_policy="vorn",
        random_seed=17,
        always_keep_prefix_tokens=0,
        preserve_recent_window=False,
    )

    assert plan.run.run_id == "step2-niah-vorn-live-b1024-noguards"
    assert plan.run.always_keep_prefix_tokens == 0
    assert plan.run.preserve_recent_window is False
    assert plan.baseline.name == "vorn_live"


def test_select_live_eviction_plan_supports_sentence_level_vorn():
    plan = select_live_eviction_plan(
        cache_budget_tokens=1024,
        retention_policy="sentence_vorn",
        random_seed=17,
        sentence_pooling="max",
        sentence_top_k=3,
    )

    assert plan.run.run_id == "step2-niah-sentence-vorn-live-b1024"
    assert plan.run.retention_policy == "sentence_vorn"
    assert plan.run.eviction_unit == "sentence"
    assert plan.run.sentence_pooling == "max"
    assert plan.run.sentence_top_k == 3
    assert plan.baseline.name == "sentence_vorn_live"


def test_select_live_eviction_plan_supports_sentence_level_no_guardrails():
    plan = select_live_eviction_plan(
        cache_budget_tokens=1024,
        retention_policy="sentence_vorn",
        random_seed=17,
        always_keep_prefix_tokens=0,
        preserve_recent_window=False,
        sentence_pooling="max",
        sentence_top_k=3,
    )

    assert plan.run.run_id == "step2-niah-sentence-vorn-live-b1024-noguards"
    assert plan.run.always_keep_prefix_tokens == 0
    assert plan.run.preserve_recent_window is False
    assert plan.run.eviction_unit == "sentence"

def test_select_live_eviction_plan_supports_sentence_boundary_trigger():
    plan = select_live_eviction_plan(
        cache_budget_tokens=1536,
        retention_policy="sentence_vorn",
        random_seed=17,
        sentence_pooling="max",
        sentence_top_k=3,
        eviction_trigger="sentence_boundary",
        sentence_boundary_lookahead_tokens=25,
        force_eviction_overflow_ratio=1.2,
    )

    assert plan.run.run_id == "step2-niah-sentence-vorn-live-b1536-sentbound"
    assert plan.run.eviction_trigger == "sentence_boundary"
    assert plan.run.sentence_boundary_lookahead_tokens == 25
    assert plan.run.force_eviction_overflow_ratio == 1.2


def test_select_live_eviction_plan_supports_word_level_vorn():
    plan = select_live_eviction_plan(
        cache_budget_tokens=1024,
        retention_policy="word_vorn",
        random_seed=17,
        sentence_pooling="max",
        sentence_top_k=3,
    )

    assert plan.run.run_id == "step2-niah-word-vorn-live-b1024"
    assert plan.run.retention_policy == "word_vorn"
    assert plan.run.eviction_unit == "word"
    assert plan.run.sentence_pooling == "max"
    assert plan.run.sentence_top_k == 3
    assert plan.baseline.name == "word_vorn_live"


def test_select_live_eviction_plan_supports_word_level_no_guardrails():
    plan = select_live_eviction_plan(
        cache_budget_tokens=1024,
        retention_policy="word_vorn",
        random_seed=17,
        always_keep_prefix_tokens=0,
        preserve_recent_window=False,
        sentence_pooling="max",
        sentence_top_k=3,
    )

    assert plan.run.run_id == "step2-niah-word-vorn-live-b1024-noguards"
    assert plan.run.always_keep_prefix_tokens == 0
    assert plan.run.preserve_recent_window is False
    assert plan.run.eviction_unit == "word"


def test_select_live_eviction_plan_supports_adaptive_vorn():
    plan = select_live_eviction_plan(
        cache_budget_tokens=1536,
        retention_policy="adaptive_vorn",
        random_seed=17,
        sentence_pooling="max",
        sentence_top_k=3,
    )

    assert plan.run.run_id == "step2-niah-adaptive-vorn-live-b1536"
    assert plan.run.retention_policy == "adaptive_vorn"
    assert plan.run.eviction_unit == "adaptive_token_or_sentence"
    assert plan.run.sentence_pooling == "max"
    assert plan.baseline.name == "adaptive_vorn_live"


def test_split_context_and_question_for_summary_removes_final_query():
    prompt = (
        "Context body line one.\nContext body line two.\n"
        "What is the special magic number for fair-sprout mentioned in the provided text?"
    )

    context, question = split_context_and_question_for_summary(prompt)

    assert context == "Context body line one.\nContext body line two."
    assert question == (
        "What is the special magic number for fair-sprout mentioned in the provided text?"
    )


def test_build_summary_prompt_and_answer_prompt_are_distinct():
    summary_prompt = build_summary_prompt("context goes here")
    answer_prompt = build_summary_answer_prompt(
        summary="summary goes here",
        question="What is the code?",
    )

    assert "Do not answer any question." in summary_prompt
    assert "summary goes here" in answer_prompt
    assert "What is the code?" in answer_prompt


def test_run_local_vanilla_smoke_writes_result_for_fixture_cases(tmp_path: Path):
    dataset_path = Path(__file__).parent.parent / "examples" / "niah_smoke.jsonl"
    output_path = tmp_path / "local-smoke.jsonl"

    prompts = [line["prompt"] for line in _load_fixture_prompts(dataset_path)]
    answers = {
        prompts[0]: "amber",
        prompts[1]: "lagoon",
        prompts[2]: "cedar",
    }

    result, traces = run_local_vanilla_smoke(
        benchmark="niah",
        dataset_path=dataset_path,
        output_path=output_path,
        case_limit=3,
        generator=FakeGenerator(answers),
    )

    assert result.run_id == "week1-niah-vanilla"
    assert result.metrics == {"needle_hit_rate": 1.0}
    assert len(traces) == 3
    assert load_results(output_path) == [result]


def test_run_local_step1_pair_writes_both_result_arms(tmp_path: Path):
    dataset_path = Path(__file__).parent.parent / "examples" / "niah_smoke.jsonl"
    prompts = [line["prompt"] for line in _load_fixture_prompts(dataset_path)]
    vanilla_answers = {
        prompts[0]: "amber",
        prompts[1]: "lagoon",
        prompts[2]: "cedar",
    }
    vorn_answers = {
        prompts[0]: "amber",
        prompts[1]: "lagoon",
        prompts[2]: "wrong",
    }
    retention = RetentionStats(
        original_token_count=24,
        kept_token_count=10,
        kept_positions=(0, 2, 4, 6, 8, 12, 16, 20, 22, 23),
        dropped_positions=(1, 3, 5, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21),
    )

    (vanilla_result, vanilla_traces), (vorn_result, vorn_traces) = run_local_step1_pair(
        dataset_path=dataset_path,
        output_dir=tmp_path,
        case_limit=3,
        vanilla_generator=FakeGenerator(vanilla_answers),
        vorn_generator=FakeVornGenerator(vorn_answers, retention),
    )

    assert vanilla_result.run_id == "step1-niah-vanilla"
    assert vanilla_result.metrics == {"needle_hit_rate": 1.0}
    assert len(vanilla_traces) == 3

    assert vorn_result.run_id == "step1-niah-vorn"
    assert vorn_result.metrics == {"needle_hit_rate": 2 / 3}
    assert len(vorn_traces) == 3

    assert load_results(tmp_path / "step1-niah-vanilla.jsonl") == [vanilla_result]
    assert load_results(tmp_path / "step1-niah-vorn.jsonl") == [vorn_result]


def test_run_local_live_eviction_smoke_writes_result_for_fixture_cases(tmp_path: Path):
    dataset_path = Path(__file__).parent.parent / "examples" / "niah_smoke.jsonl"
    output_path = tmp_path / "live-eviction.jsonl"

    prompts = [line["prompt"] for line in _load_fixture_prompts(dataset_path)]
    answers = {
        prompts[0]: "amber",
        prompts[1]: "lagoon",
        prompts[2]: "wrong",
    }

    result, traces = run_local_live_eviction_smoke(
        dataset_path=dataset_path,
        output_path=output_path,
        case_limit=3,
        generator=FakeLiveGenerator(
            answers,
            LiveEvictionStats(
                prompt_token_count=36,
                generated_token_count=2,
                mean_kept_token_count=18.0,
                final_kept_token_count=19,
                eviction_steps=1,
                mean_retention_ratio=0.5,
                retention_policy="vorn",
                summary_contract="canonical_hidden_state_float32_per_token_from_layer_L_star",
                summary_fingerprint="abc123",
            ),
        ),
    )

    assert result.run_id == "step2-niah-vorn-live"
    assert result.metrics == {"needle_hit_rate": 2 / 3}
    assert len(traces) == 3
    assert load_results(output_path) == [result]


def test_run_local_live_eviction_smoke_supports_random_control_arm(tmp_path: Path):
    dataset_path = Path(__file__).parent.parent / "examples" / "niah_smoke.jsonl"
    output_path = tmp_path / "live-eviction-random.jsonl"

    prompts = [line["prompt"] for line in _load_fixture_prompts(dataset_path)]
    answers = {
        prompts[0]: "amber",
        prompts[1]: "wrong",
        prompts[2]: "cedar",
    }

    result, traces = run_local_live_eviction_smoke(
        dataset_path=dataset_path,
        output_path=output_path,
        case_limit=3,
        cache_budget_tokens=512,
        retention_policy="random",
        random_seed=23,
        generator=FakeLiveGenerator(
            answers,
            LiveEvictionStats(
                prompt_token_count=36,
                generated_token_count=2,
                mean_kept_token_count=24.0,
                final_kept_token_count=25,
                eviction_steps=1,
                mean_retention_ratio=2 / 3,
                retention_policy="random",
                summary_contract="canonical_hidden_state_float32_per_token_from_layer_L_star",
                summary_fingerprint="xyz789",
            ),
        ),
    )

    assert result.run_id == "step2-niah-random-live-b512"
    assert result.baseline == "random_live"
    assert result.metrics == {"needle_hit_rate": 2 / 3}
    assert len(traces) == 3
    assert load_results(output_path) == [result]


def test_render_prompt_accepts_mapping_output_from_chat_template():
    import torch

    from vorn_mat.local_exec import _TransformersGeneratorBase

    class FakeTokenizer:
        chat_template = "{{ messages }}"

        def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
            assert messages == [{"role": "user", "content": "needle prompt"}]
            assert add_generation_prompt is True
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([[11, 22, 33]])}

    generator = _TransformersGeneratorBase()
    object.__setattr__(generator, "_tokenizer", FakeTokenizer())
    object.__setattr__(generator, "_model", object())
    object.__setattr__(generator, "_device", "cpu")

    input_ids, attention_mask = generator._render_prompt("needle prompt")

    assert input_ids.tolist() == [[11, 22, 33]]
    assert attention_mask.tolist() == [[1, 1, 1]]


def test_forward_with_hidden_states_allows_attention_outputs_to_be_disabled():
    import torch

    from vorn_mat.local_exec import _TransformersGeneratorBase

    class FakeModel:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return object()

    generator = _TransformersGeneratorBase()
    fake_model = FakeModel()
    object.__setattr__(generator, "_tokenizer", object())
    object.__setattr__(generator, "_model", fake_model)
    object.__setattr__(generator, "_device", "cpu")

    result = generator._forward_with_hidden_states(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        position_ids=torch.tensor([[0, 1, 2]]),
        output_attentions=False,
    )

    assert result is not None
    assert fake_model.calls[0]["output_hidden_states"] is True
    assert fake_model.calls[0]["output_attentions"] is False


def test_forward_with_hidden_states_uses_compact_cache_position_for_gemma3():
    import torch

    from vorn_mat.local_exec import _TransformersGeneratorBase

    class FakeModel:
        class config:
            model_type = "gemma3"

        def __init__(self):
            self.prepare_calls = []
            self.calls = []

        def prepare_inputs_for_generation(self, **kwargs):
            self.prepare_calls.append(kwargs)
            return {
                "input_ids": kwargs["input_ids"],
                "attention_mask": kwargs["attention_mask"],
                "position_ids": kwargs["position_ids"],
                "cache_position": kwargs["cache_position"],
            }

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return object()

    generator = _TransformersGeneratorBase()
    fake_model = FakeModel()
    object.__setattr__(generator, "_tokenizer", object())
    object.__setattr__(generator, "_model", fake_model)
    object.__setattr__(generator, "_device", "cpu")

    position_ids = torch.tensor([[0, 7, 42]])
    result = generator._forward_with_hidden_states(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        position_ids=position_ids,
        output_attentions=False,
    )

    assert result is not None
    assert fake_model.prepare_calls[0]["cache_position"].tolist() == [0, 1, 2]
    assert fake_model.prepare_calls[0]["position_ids"].tolist() == [[0, 7, 42]]
    assert fake_model.calls[0]["cache_position"].tolist() == [0, 1, 2]
    assert fake_model.calls[0]["position_ids"].tolist() == [[0, 7, 42]]
    assert fake_model.calls[0]["output_hidden_states"] is True
    assert fake_model.calls[0]["output_attentions"] is False


def test_answer_retention_payload_reports_retained_and_dropped_answer_tokens():
    from vorn_mat.local_exec import _answer_retention_payload

    payload = _answer_retention_payload(
        case_id="fixture-1",
        expected_answer="1234",
        answer_token_spans=((10, 14),),
        active_absolute_positions=(0, 10, 11, 12, 13, 20),
        keep_positions=(0, 1, 3, 5),
    )

    assert payload["case_id"] == "fixture-1"
    assert payload["answer_positions"] == [10, 11, 12, 13]
    assert payload["retained_answer_positions"] == [10, 12]
    assert payload["dropped_answer_positions"] == [11, 13]
    assert payload["active_answer_token_count_before"] == 4
    assert payload["retained_answer_token_count_after"] == 2
    assert payload["dropped_answer_token_count_this_step"] == 2
    assert payload["answer_fully_active_before"] is True
    assert payload["answer_fully_retained_after"] is False
    assert payload["answer_fully_dropped_after"] is False


def test_terminal_token_ids_prefer_generation_config_over_tokenizer_eos():
    from types import SimpleNamespace

    from vorn_mat.local_exec import _TransformersGeneratorBase

    class FakeTokenizer:
        eos_token_id = 1

    fake_model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[1, 106])
    )

    generator = _TransformersGeneratorBase()
    object.__setattr__(generator, "_tokenizer", FakeTokenizer())
    object.__setattr__(generator, "_model", fake_model)
    object.__setattr__(generator, "_device", "cpu")

    assert generator._terminal_token_ids() == (1, 106)
    assert generator._is_terminal_token_id(1) is True
    assert generator._is_terminal_token_id(106) is True
    assert generator._is_terminal_token_id(999) is False


def test_select_next_token_supports_standard_3d_logits():
    import torch

    from vorn_mat.local_exec import _TransformersGeneratorBase

    generator = _TransformersGeneratorBase()
    logits = torch.tensor(
        [[[0.1, 0.2, 0.3], [0.4, 0.9, 0.1]]],
        dtype=torch.float32,
    )

    next_token = generator._select_next_token(logits)

    assert next_token.tolist() == [[1]]


def test_select_next_token_supports_extra_singleton_axis():
    import torch

    from vorn_mat.local_exec import _TransformersGeneratorBase

    generator = _TransformersGeneratorBase()
    logits = torch.tensor(
        [[[[0.1, 0.2, 0.3], [0.4, 0.9, 0.1]]]],
        dtype=torch.float32,
    )

    next_token = generator._select_next_token(logits)

    assert next_token.tolist() == [[1]]


def test_select_next_token_supports_sequence_by_vocab_logits():
    import torch

    from vorn_mat.local_exec import _TransformersGeneratorBase

    generator = _TransformersGeneratorBase()
    logits = torch.tensor(
        [[0.1, 0.9, 0.0], [0.5, 0.2, 0.8]],
        dtype=torch.float32,
    )

    next_token = generator._select_next_token(logits)

    assert next_token.tolist() == [[2]]


def test_select_next_token_suppresses_non_terminal_pad_token():
    import torch
    from types import SimpleNamespace

    from vorn_mat.local_exec import _TransformersGeneratorBase

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

    generator = _TransformersGeneratorBase()
    object.__setattr__(generator, "_tokenizer", FakeTokenizer())
    object.__setattr__(
        generator,
        "_model",
        SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[1, 106])),
    )
    object.__setattr__(generator, "_device", "cpu")
    logits = torch.tensor(
        [[[10.0, 0.1, 9.0]]],
        dtype=torch.float32,
    )

    next_token = generator._select_next_token(logits)

    assert next_token.tolist() == [[2]]


def test_select_next_token_ignores_nan_when_finite_candidates_remain():
    import torch

    from vorn_mat.local_exec import _TransformersGeneratorBase

    generator = _TransformersGeneratorBase()
    logits = torch.tensor(
        [[[float("nan"), 0.5, 0.9]]],
        dtype=torch.float32,
    )

    next_token = generator._select_next_token(logits)

    assert next_token.tolist() == [[2]]


def test_select_next_token_rejects_all_nan_logits():
    import pytest
    import torch

    from vorn_mat.local_exec import _TransformersGeneratorBase

    generator = _TransformersGeneratorBase()
    logits = torch.tensor(
        [[[float("nan"), float("nan"), float("nan")]]],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="only NaN values"):
        generator._select_next_token(logits)


def test_runtime_event_logging_emits_json_line(capsys):
    import json

    from vorn_mat.local_exec import _TransformersGeneratorBase

    generator = _TransformersGeneratorBase()

    generator._emit_runtime_event(
        "token_step",
        {
            "step_index": 0,
            "next_token_id": 106,
            "is_terminal": True,
        },
    )

    captured = capsys.readouterr().out.strip()
    prefix, payload = captured.split(" ", 1)
    assert prefix == "token_step"
    assert json.loads(payload) == {
        "step_index": 0,
        "next_token_id": 106,
        "is_terminal": True,
    }


def _load_fixture_prompts(path: Path) -> list[dict[str, str]]:
    import json

    rows: list[dict[str, str]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
