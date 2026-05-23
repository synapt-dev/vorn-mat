"""Local Step 0 execution path for Week 1 validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from types import MethodType
from typing import Any

import numpy as np

from .baselines.vanilla import PredictionTrace, TextGenerator, run_vanilla
from .baselines.live_eviction import (
    ADAPTIVE_SELECTOR_CONTRACT,
    H2O_ATTENTION_CONTRACT,
    LiveEvictionConfig,
    LiveEvictionStats,
    LiveEvictionTextGenerator,
    SEMANTIC_SUMMARY_CONTRACT,
    SUMMARIZE_COMPACT_CONTRACT,
    TOVA_ATTENTION_CONTRACT,
    accumulate_attention_scores,
    aggregate_unit_alignment_scores,
    assign_word_ids_from_offsets,
    select_adaptive_retained_positions,
    assign_sentence_ids_from_offsets,
    build_word_unit_ids_for_active_positions,
    build_sentence_unit_ids_for_active_positions,
    build_unit_ids_for_active_positions,
    extract_canonical_token_summaries,
    extract_last_token_attention_scores,
    select_prefix_suffix_retained_positions,
    select_attention_score_retained_positions,
    run_live_eviction,
    select_random_retained_positions,
    select_sentence_retained_positions,
    select_sliding_window_retained_positions,
    should_trigger_sentence_boundary_eviction,
    summary_fingerprint,
)
from .baselines.vorn import (
    RetentionStats,
    VornPredictionTrace,
    VornRetentionConfig,
    VornTextGenerator,
    compute_vorn_direction,
    cosine_similarity,
    run_vorn,
    select_retained_positions,
)
from .benchmarks import load_cases
from .benchmarks.common import BenchmarkCase, _acceptable_answers, normalize_answer
from .observation import (
    ObservationCase,
    ObservationStep,
    find_subsequence_spans,
    jaccard_similarity,
    residual_l2_norms,
    round_float_list,
    select_top_alignment_positions,
)
from .plan import (
    DEFAULT_LIVE_EVICTION_CACHE_BUDGET,
    DEFAULT_MODEL,
    LiveEvictionDefaults,
    build_live_eviction_run,
    build_step1_run_matrix,
)
from .results import CaseObservation, RunResult, append_result
from .runner import ExecutionPlan, build_execution_plans
from .score_distribution_observation import (
    ScoreDistributionObservationCase,
    ScoreDistributionObservationStep,
    score_distribution_stats,
)


@dataclass(frozen=True)
class LocalModelConfig:
    model_id: str = DEFAULT_MODEL
    max_new_tokens: int = 32
    trust_remote_code: bool = True
    attention_implementation: str = "eager"


_TELEMETRY_NEAR_MISS_RATIO = 0.75


def reset_runtime_telemetry() -> None:
    """Reset CUDA peak-memory counters before a measured cell begins."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _capture_env_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    # NOTE: pkg_name is the PyPI distribution name (matches pyproject.toml +
    # requirements.lock); import_name is the Python module name. They differ
    # for faiss-cpu (imports as `faiss`). If the canonical pin switches to
    # faiss-gpu, update both the pkg_name string and the import target.
    for pkg_name, import_name in (
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("accelerate", "accelerate"),
        ("datasets", "datasets"),
        ("sentencepiece", "sentencepiece"),
        ("huggingface_hub", "huggingface_hub"),
        ("faiss-cpu", "faiss"),
    ):
        try:
            module = __import__(import_name)
        except ImportError:
            versions[pkg_name] = "not_installed"
            continue
        versions[pkg_name] = getattr(module, "__version__", "unknown")
    return versions


def capture_runtime_telemetry() -> dict[str, object]:
    """Snapshot CUDA peak memory + installed package versions for the envelope."""
    snapshot: dict[str, object] = {
        "peak_memory_allocated_gb": None,
        "peak_memory_reserved_gb": None,
        "oom_near_miss": False,
        "env_versions": _capture_env_versions(),
    }
    try:
        import torch
    except ImportError:
        return snapshot
    if not torch.cuda.is_available():
        return snapshot
    allocated_bytes = int(torch.cuda.max_memory_allocated())
    reserved_bytes = int(torch.cuda.max_memory_reserved())
    snapshot["peak_memory_allocated_gb"] = allocated_bytes / (1024 ** 3)
    snapshot["peak_memory_reserved_gb"] = reserved_bytes / (1024 ** 3)
    try:
        total_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    except (RuntimeError, AssertionError):
        total_bytes = 0
    if total_bytes > 0:
        snapshot["oom_near_miss"] = bool(allocated_bytes > _TELEMETRY_NEAR_MISS_RATIO * total_bytes)
    return snapshot


def attach_runtime_telemetry(result: RunResult) -> RunResult:
    """Return a new RunResult with capture_runtime_telemetry() fields filled."""
    from dataclasses import replace

    return replace(result, **capture_runtime_telemetry())


SUMMARY_PROMPT_PREFIX = (
    "Summarize the following context concisely while preserving factual details, "
    "numbers, named entities, and identifiers. Do not answer any question.\n\n"
    "Context:\n"
)
SUMMARY_PROMPT_SUFFIX = "\n\nSummary:"
ANSWER_FROM_SUMMARY_PREFIX = (
    "Use the summary below to answer the question. Output only the final answer "
    "with no explanation.\n\nSummary:\n"
)
ANSWER_FROM_SUMMARY_MIDDLE = "\n\nQuestion:\n"
ANSWER_FROM_SUMMARY_SUFFIX = "\n\nAnswer:"


class _TransformersGeneratorBase:
    """Shared lazy model loader for local and remote smoke paths."""

    def __init__(self, config: LocalModelConfig = LocalModelConfig()):
        self.config = config
        self._tokenizer = None
        self._model = None
        self._device = None

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._device is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        elif torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            trust_remote_code=self.config.trust_remote_code,
            low_cpu_mem_usage=True,
            attn_implementation=self.config.attention_implementation,
        )
        model.to(device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def _render_prompt(self, prompt: str) -> tuple[Any, Any]:
        self._ensure_model()

        import torch

        assert self._tokenizer is not None
        assert self._device is not None

        messages = [{"role": "user", "content": prompt}]
        if getattr(self._tokenizer, "chat_template", None):
            inputs = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if hasattr(inputs, "keys"):
                input_ids = inputs["input_ids"]
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
            else:
                input_ids = inputs
                attention_mask = torch.ones_like(input_ids)
        else:
            encoded = self._tokenizer(prompt, return_tensors="pt")
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)
        return input_ids, attention_mask

    def _render_prompt_text_with_offsets(
        self,
        prompt: str,
    ) -> tuple[str, tuple[tuple[int, int], ...]]:
        self._ensure_model()

        assert self._tokenizer is not None

        messages = [{"role": "user", "content": prompt}]
        if getattr(self._tokenizer, "chat_template", None):
            rendered_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered_prompt = prompt

        encoded = self._tokenizer(
            rendered_prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = tuple(
            (int(start), int(end))
            for start, end in encoded["offset_mapping"]
        )
        return rendered_prompt, offsets

    def _generate_from_tensors(
        self,
        input_ids: Any,
        attention_mask: Any,
        *,
        max_new_tokens: int | None = None,
    ) -> str:
        self._ensure_model()

        import torch

        assert self._tokenizer is not None
        assert self._model is not None

        with torch.no_grad():
            output_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = output_ids[0][input_ids.shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _count_prompt_tokens(self, prompt: str) -> int:
        input_ids, _attention_mask = self._render_prompt(prompt)
        return int(input_ids.shape[-1])

    def _trim_summary_to_budget(
        self,
        *,
        summary: str,
        question: str,
        cache_budget_tokens: int,
    ) -> str:
        self._ensure_model()

        assert self._tokenizer is not None

        static_prompt = build_summary_answer_prompt(summary="", question=question)
        static_tokens = self._count_prompt_tokens(static_prompt)
        available_summary_tokens = max(1, cache_budget_tokens - static_tokens)
        summary_ids = self._tokenizer(
            summary,
            add_special_tokens=False,
        )["input_ids"]
        if len(summary_ids) <= available_summary_tokens:
            return summary

        trimmed_ids = summary_ids[:available_summary_tokens]
        trimmed_summary = self._tokenizer.decode(
            trimmed_ids,
            skip_special_tokens=True,
        ).strip()
        while trimmed_summary:
            answer_prompt = build_summary_answer_prompt(
                summary=trimmed_summary,
                question=question,
            )
            if self._count_prompt_tokens(answer_prompt) <= cache_budget_tokens:
                return trimmed_summary
            trimmed_ids = trimmed_ids[:-1]
            trimmed_summary = self._tokenizer.decode(
                trimmed_ids,
                skip_special_tokens=True,
            ).strip()
        return ""

    def _forward_with_hidden_states(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        position_ids: Any,
        output_attentions: bool = False,
    ) -> Any:
        self._ensure_model()

        import torch

        assert self._model is not None

        with torch.no_grad():
            return self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                output_attentions=output_attentions,
                use_cache=False,
                return_dict=True,
            )

    def _canonical_token_summaries(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        canonical_layer: int,
    ) -> np.ndarray:
        self._ensure_model()

        import torch

        assert self._model is not None

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states
        layer_index = canonical_layer + 1  # hidden_states[0] is the embedding output
        if hidden_states is None or layer_index >= len(hidden_states):
            raise ValueError(
                f"canonical_layer {canonical_layer} is unavailable for this model"
            )
        return (
            hidden_states[layer_index][0]
            .detach()
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )


class TransformersTextGenerator(_TransformersGeneratorBase, TextGenerator):
    """Lazy local text generator for Step 0 smoke validation."""

    def generate(self, prompt: str) -> str:
        input_ids, attention_mask = self._render_prompt(prompt)
        return self._generate_from_tensors(input_ids, attention_mask)


class TransformersObservationGenerator(_TransformersGeneratorBase):
    """Vanilla observer: no eviction, only per-step measurements."""

    def __init__(self, config: LocalModelConfig | None = None):
        super().__init__(
            config
            or LocalModelConfig(
                attention_implementation="sdpa",
            )
        )

    def observe_vanilla_case(
        self,
        case: BenchmarkCase,
        *,
        canonical_layer: int = 16,
        recent_token_window: int = 16,
        top_k: int = 10,
        attention_last_n_layers: int = 4,
    ) -> ObservationCase:
        self._ensure_model()

        import torch

        assert self._tokenizer is not None

        prompt_input_ids, prompt_attention_mask = self._render_prompt(case.prompt)
        prompt_token_count = int(prompt_input_ids.shape[-1])
        prompt_token_ids = prompt_input_ids[0].detach().cpu().tolist()
        answer_ids = self._tokenizer(
            case.expected_answer,
            add_special_tokens=False,
        )["input_ids"]
        answer_token_spans = find_subsequence_spans(prompt_token_ids, answer_ids)

        active_input_ids = prompt_input_ids
        active_attention_mask = prompt_attention_mask
        active_position_ids = torch.arange(
            prompt_token_count,
            device=prompt_input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)
        next_absolute_position = prompt_token_count
        generated_token_ids: list[int] = []
        steps: list[ObservationStep] = []
        previous_top_positions: tuple[int, ...] | None = None
        assert self._model is not None
        selected_attention_layers = tuple(
            range(
                max(0, len(self._model.model.layers) - attention_last_n_layers),
                len(self._model.model.layers),
            )
        )

        for step_index in range(self.config.max_new_tokens):
            outputs, attention_by_layer = self._forward_with_hidden_states_and_selected_attentions(
                input_ids=active_input_ids,
                attention_mask=active_attention_mask,
                position_ids=active_position_ids,
                capture_attention_layers=selected_attention_layers,
            )
            token_summaries = extract_canonical_token_summaries(
                outputs.hidden_states,
                canonical_layer=canonical_layer,
            )
            vorn_direction = compute_vorn_direction(
                token_summaries,
                recent_token_window=recent_token_window,
            )
            alignment_scores = np.asarray(
                [
                    cosine_similarity(token_summaries[position], vorn_direction)
                    for position in range(token_summaries.shape[0])
                ],
                dtype=np.float32,
            )
            norms = residual_l2_norms(token_summaries)
            top_positions = select_top_alignment_positions(
                alignment_scores=alignment_scores,
                residual_norms=norms,
                answer_token_spans=answer_token_spans,
                top_k=top_k,
            )

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_token_id = int(next_token.item())
            next_token_text = self._tokenizer.decode(
                [next_token_id],
                skip_special_tokens=False,
            )
            top_position_ids = tuple(item.position for item in top_positions)
            steps.append(
                ObservationStep(
                    step_index=step_index,
                    generated_token_id=next_token_id,
                    generated_token_text=next_token_text,
                    vorn_vector=round_float_list(vorn_direction),
                    alignment_scores=round_float_list(alignment_scores),
                    residual_norms=round_float_list(norms),
                    attention_by_layer=attention_by_layer,
                    top_alignment_positions=top_position_ids,
                    top_alignment_scores=tuple(
                        round(item.alignment_score, 6) for item in top_positions
                    ),
                    ranking_stability_with_prev=(
                        None
                        if previous_top_positions is None
                        else round(
                            jaccard_similarity(previous_top_positions, top_position_ids),
                            6,
                        )
                    ),
                )
            )
            previous_top_positions = top_position_ids

            if next_token_id == int(self._tokenizer.eos_token_id):
                break

            generated_token_ids.append(next_token_id)
            active_input_ids = torch.cat([active_input_ids, next_token], dim=1)
            active_attention_mask = torch.cat(
                [
                    active_attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=active_attention_mask.dtype,
                        device=active_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            active_position_ids = torch.cat(
                [
                    active_position_ids,
                    torch.tensor(
                        [[next_absolute_position]],
                        dtype=active_position_ids.dtype,
                        device=active_position_ids.device,
                    ),
                ],
                dim=1,
            )
            next_absolute_position += 1

        prediction = self._tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        ).strip()
        success = normalize_answer(prediction) in _acceptable_answers(case)
        return ObservationCase(
            case_id=case.case_id,
            expected_answer=case.expected_answer,
            prediction=prediction,
            success=success,
            prompt_token_count=prompt_token_count,
            answer_token_spans=answer_token_spans,
            steps=tuple(steps),
        )

    def _forward_with_hidden_states_and_selected_attentions(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        position_ids: Any,
        capture_attention_layers: tuple[int, ...],
    ) -> tuple[Any, dict[str, list[float]]]:
        self._ensure_model()

        import torch
        from transformers.models.mistral import modeling_mistral

        assert self._model is not None

        captured_attentions: dict[str, list[float]] = {}
        original_forwards: dict[int, Any] = {}

        for layer_index in capture_attention_layers:
            attention_module = self._model.model.layers[layer_index].self_attn
            original_forward = attention_module.forward
            original_forwards[layer_index] = original_forward

            def _make_wrapped_forward(
                *,
                captured_layer_index: int,
                captured_original_forward: Any,
            ) -> Any:
                def _wrapped_forward(
                    module_self: Any,
                    hidden_states: Any,
                    position_embeddings: tuple[Any, Any],
                    attention_mask: Any,
                    past_key_values: Any = None,
                    cache_position: Any = None,
                    **kwargs: Any,
                ) -> tuple[Any, Any]:
                    attn_output, attn_weights = captured_original_forward(
                        hidden_states=hidden_states,
                        position_embeddings=position_embeddings,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        cache_position=cache_position,
                        **kwargs,
                    )

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, module_self.head_dim)
                    query_states = module_self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_states = module_self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    cos, sin = position_embeddings
                    query_states, key_states = modeling_mistral.apply_rotary_pos_emb(
                        query_states,
                        key_states,
                        cos,
                        sin,
                    )
                    key_states = modeling_mistral.repeat_kv(
                        key_states,
                        module_self.num_key_value_groups,
                    )

                    last_query = query_states[:, :, -1:, :]
                    scores = torch.matmul(
                        last_query,
                        key_states.transpose(2, 3),
                    ) * module_self.scaling
                    if attention_mask is not None:
                        scores = scores + attention_mask[:, :, -1:, : scores.shape[-1]]
                    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
                        query_states.dtype
                    )
                    layer_scores = weights[0, :, 0, :].mean(dim=0)
                    captured_attentions[str(captured_layer_index)] = round_float_list(
                        np.asarray(layer_scores.detach().cpu(), dtype=np.float32)
                    )
                    return attn_output, attn_weights

                return _wrapped_forward

            attention_module.forward = MethodType(
                _make_wrapped_forward(
                    captured_layer_index=layer_index,
                    captured_original_forward=original_forward,
                ),
                attention_module,
            )

        try:
            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    output_attentions=False,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for layer_index, original_forward in original_forwards.items():
                self._model.model.layers[layer_index].self_attn.forward = original_forward

        return outputs, captured_attentions

    def _last_n_layer_attentions(
        self,
        attentions: Any,
        *,
        last_n_layers: int,
    ) -> dict[str, list[float]]:
        if not attentions:
            raise ValueError("attentions must be non-empty")
        start = max(0, len(attentions) - last_n_layers)
        result: dict[str, list[float]] = {}
        for layer_index in range(start, len(attentions)):
            layer = attentions[layer_index]
            scores = layer[0, :, -1, :].mean(dim=0)
            result[str(layer_index)] = round_float_list(
                np.asarray(scores.detach().cpu(), dtype=np.float32)
            )
        return result


class TransformersScoreDistributionObservationGenerator(_TransformersGeneratorBase):
    """Live budgeted observer for token/word/sentence score-shape comparisons."""

    def observe_live_case(
        self,
        case: BenchmarkCase,
        *,
        config: LiveEvictionConfig,
    ) -> ScoreDistributionObservationCase:
        self._ensure_model()

        import torch

        assert self._tokenizer is not None

        prompt_input_ids, prompt_attention_mask = self._render_prompt(case.prompt)
        prompt_token_count = int(prompt_input_ids.shape[-1])
        rendered_prompt, offsets = self._render_prompt_text_with_offsets(case.prompt)
        if len(offsets) != prompt_token_count:
            raise ValueError(
                "prompt offsets do not match rendered prompt token count for score observation"
            )

        prompt_sentence_ids = assign_sentence_ids_from_offsets(rendered_prompt, offsets)
        prompt_word_ids = assign_word_ids_from_offsets(rendered_prompt, offsets)

        active_input_ids = prompt_input_ids
        active_attention_mask = prompt_attention_mask
        active_position_ids = torch.arange(
            prompt_token_count,
            device=prompt_input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)
        next_absolute_position = prompt_token_count
        generated_token_ids: list[int] = []
        steps: list[ScoreDistributionObservationStep] = []

        for _ in range(self.config.max_new_tokens):
            outputs = self._forward_with_hidden_states(
                input_ids=active_input_ids,
                attention_mask=active_attention_mask,
                position_ids=active_position_ids,
                output_attentions=False,
            )
            token_summaries = extract_canonical_token_summaries(
                outputs.hidden_states,
                canonical_layer=config.canonical_layer,
            )
            preserve_recent_window = (
                min(config.recent_token_window, token_summaries.shape[0])
                if config.preserve_recent_window
                else 0
            )

            if active_input_ids.shape[-1] > config.cache_budget_tokens:
                vorn_direction = compute_vorn_direction(
                    token_summaries,
                    recent_token_window=config.recent_token_window,
                )
                alignment_scores = np.asarray(
                    [
                        cosine_similarity(token_summaries[position], vorn_direction)
                        for position in range(token_summaries.shape[0])
                    ],
                    dtype=np.float32,
                )
                active_absolute_positions = tuple(
                    int(position)
                    for position in active_position_ids[0].detach().cpu().tolist()
                )
                word_unit_ids = build_word_unit_ids_for_active_positions(
                    prompt_word_ids=prompt_word_ids,
                    active_absolute_positions=active_absolute_positions,
                )
                sentence_unit_ids = build_sentence_unit_ids_for_active_positions(
                    prompt_sentence_ids=prompt_sentence_ids,
                    active_absolute_positions=active_absolute_positions,
                )
                word_scores = aggregate_unit_alignment_scores(
                    alignment_scores,
                    unit_ids=word_unit_ids,
                    pooling=config.sentence_pooling,
                    top_k=config.sentence_top_k,
                )
                sentence_scores = aggregate_unit_alignment_scores(
                    alignment_scores,
                    unit_ids=sentence_unit_ids,
                    pooling=config.sentence_pooling,
                    top_k=config.sentence_top_k,
                )
                steps.append(
                    ScoreDistributionObservationStep(
                        step_index=len(steps),
                        active_token_count=int(active_input_ids.shape[-1]),
                        granularity_stats={
                            "token": score_distribution_stats(alignment_scores),
                            "word": score_distribution_stats(word_scores),
                            "sentence": score_distribution_stats(sentence_scores),
                        },
                    )
                )

                if config.retention_policy == "vorn":
                    keep_positions = select_retained_positions(
                        token_summaries,
                        vorn_direction,
                        cache_budget_tokens=config.cache_budget_tokens,
                        always_keep_prefix_tokens=config.always_keep_prefix_tokens,
                        preserve_recent_window=preserve_recent_window,
                    )
                elif config.retention_policy == "sentence_vorn":
                    keep_positions = select_sentence_retained_positions(
                        alignment_scores,
                        unit_ids=sentence_unit_ids,
                        cache_budget_tokens=config.cache_budget_tokens,
                        always_keep_prefix_tokens=config.always_keep_prefix_tokens,
                        preserve_recent_window=preserve_recent_window,
                        pooling=config.sentence_pooling,
                        top_k=config.sentence_top_k,
                    )
                elif config.retention_policy == "word_vorn":
                    keep_positions = select_sentence_retained_positions(
                        alignment_scores,
                        unit_ids=word_unit_ids,
                        cache_budget_tokens=config.cache_budget_tokens,
                        always_keep_prefix_tokens=config.always_keep_prefix_tokens,
                        preserve_recent_window=preserve_recent_window,
                        pooling=config.sentence_pooling,
                        top_k=config.sentence_top_k,
                    )
                else:
                    raise ValueError(
                        "score-distribution observation only supports "
                        "vorn, sentence_vorn, and word_vorn"
                    )

                keep_tensor = torch.tensor(
                    keep_positions,
                    device=active_input_ids.device,
                )
                active_input_ids = active_input_ids.index_select(1, keep_tensor)
                active_attention_mask = active_attention_mask.index_select(1, keep_tensor)
                active_position_ids = active_position_ids.index_select(1, keep_tensor)
                outputs = self._forward_with_hidden_states(
                    input_ids=active_input_ids,
                    attention_mask=active_attention_mask,
                    position_ids=active_position_ids,
                    output_attentions=False,
                )

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_token_id = int(next_token.item())
            if next_token_id == int(self._tokenizer.eos_token_id):
                break

            generated_token_ids.append(next_token_id)
            active_input_ids = torch.cat([active_input_ids, next_token], dim=1)
            active_attention_mask = torch.cat(
                [
                    active_attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=active_attention_mask.dtype,
                        device=active_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            active_position_ids = torch.cat(
                [
                    active_position_ids,
                    torch.tensor(
                        [[next_absolute_position]],
                        dtype=active_position_ids.dtype,
                        device=active_position_ids.device,
                    ),
                ],
                dim=1,
            )
            next_absolute_position += 1

        prediction = self._tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        ).strip()
        success = normalize_answer(prediction) in _acceptable_answers(case)
        return ScoreDistributionObservationCase(
            case_id=case.case_id,
            expected_answer=case.expected_answer,
            prediction=prediction,
            success=success,
            observations=(
                CaseObservation(
                    fixture_id=case.case_id,
                    correct=success,
                    prediction=prediction,
                ),
            ),
            steps=tuple(steps),
        )


def split_context_and_question_for_summary(prompt: str) -> tuple[str, str]:
    marker = "\nWhat is "
    head, sep, tail = prompt.rpartition(marker)
    if not sep:
        return prompt, ""
    return head.rstrip(), f"What is {tail}".strip()


def build_summary_prompt(context: str) -> str:
    return f"{SUMMARY_PROMPT_PREFIX}{context}{SUMMARY_PROMPT_SUFFIX}"


def build_summary_answer_prompt(*, summary: str, question: str) -> str:
    return (
        f"{ANSWER_FROM_SUMMARY_PREFIX}{summary}"
        f"{ANSWER_FROM_SUMMARY_MIDDLE}{question}{ANSWER_FROM_SUMMARY_SUFFIX}"
    )


class TransformersVornGenerator(_TransformersGeneratorBase, VornTextGenerator):
    """Canonical-layer prompt-retention proxy for Step 1 comparison runs."""

    def generate_with_retention(
        self,
        prompt: str,
        config: VornRetentionConfig,
    ) -> tuple[str, RetentionStats]:
        input_ids, attention_mask = self._render_prompt(prompt)
        token_summaries = self._canonical_token_summaries(
            input_ids=input_ids,
            attention_mask=attention_mask,
            canonical_layer=config.canonical_layer,
        )

        preserve_recent_window = (
            min(config.recent_token_window, token_summaries.shape[0])
            if config.preserve_recent_window
            else 0
        )
        vorn_direction = compute_vorn_direction(
            token_summaries,
            recent_token_window=config.recent_token_window,
        )
        keep_positions = select_retained_positions(
            token_summaries,
            vorn_direction,
            cache_budget_tokens=config.cache_budget_tokens,
            always_keep_prefix_tokens=config.always_keep_prefix_tokens,
            preserve_recent_window=preserve_recent_window,
        )

        import torch

        keep_tensor = torch.tensor(keep_positions, device=input_ids.device)
        compressed_input_ids = input_ids.index_select(1, keep_tensor)
        compressed_attention_mask = attention_mask.index_select(1, keep_tensor)
        prediction = self._generate_from_tensors(
            compressed_input_ids,
            compressed_attention_mask,
        )

        original_token_count = int(input_ids.shape[-1])
        kept_token_count = int(compressed_input_ids.shape[-1])
        dropped_positions = tuple(
            position
            for position in range(original_token_count)
            if position not in set(keep_positions)
        )
        return prediction, RetentionStats(
            original_token_count=original_token_count,
            kept_token_count=kept_token_count,
            kept_positions=keep_positions,
            dropped_positions=dropped_positions,
        )


class TransformersLiveEvictionGenerator(
    _TransformersGeneratorBase, LiveEvictionTextGenerator
):
    """Live rotary-safe eviction arm using explicit absolute positions."""

    def generate_with_live_eviction(
        self,
        prompt: str,
        config: LiveEvictionConfig,
    ) -> tuple[str, LiveEvictionStats]:
        self._ensure_model()

        import torch

        assert self._tokenizer is not None

        prompt_input_ids, prompt_attention_mask = self._render_prompt(prompt)
        prompt_token_count = int(prompt_input_ids.shape[-1])
        prompt_unit_ids: tuple[int, ...] = ()
        sentence_level_policies = {
            "sentence_vorn",
            "sentence_tova",
            "sentence_h2o",
            "adaptive_vorn",
        }
        if config.retention_policy in sentence_level_policies | {"word_vorn"}:
            rendered_prompt, offsets = self._render_prompt_text_with_offsets(prompt)
            if len(offsets) != prompt_token_count:
                raise ValueError(
                    f"{config.retention_policy} prompt offsets do not match rendered prompt token count"
                )
            if config.retention_policy in sentence_level_policies:
                prompt_unit_ids = assign_sentence_ids_from_offsets(
                    rendered_prompt,
                    offsets,
                )
            else:
                prompt_unit_ids = assign_word_ids_from_offsets(
                    rendered_prompt,
                    offsets,
                )
        active_input_ids = prompt_input_ids
        active_attention_mask = prompt_attention_mask
        active_position_ids = torch.arange(
            prompt_token_count,
            device=prompt_input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0)
        next_absolute_position = prompt_token_count
        generated_token_ids: list[int] = []
        retained_token_counts: list[int] = []
        eviction_steps = 0
        first_summary_fingerprint: str | None = None
        first_policy_fingerprint: str | None = None
        accumulated_attention_scores: np.ndarray | None = None
        attention_score_policies = {
            "tova",
            "h2o",
            "sentence_tova",
            "sentence_h2o",
        }
        needs_attention_scores = config.retention_policy in attention_score_policies
        adaptive_token_steps = 0
        adaptive_sentence_steps = 0

        for _ in range(self.config.max_new_tokens):
            outputs = self._forward_with_hidden_states(
                input_ids=active_input_ids,
                attention_mask=active_attention_mask,
                position_ids=active_position_ids,
                output_attentions=needs_attention_scores,
            )
            token_summaries = extract_canonical_token_summaries(
                outputs.hidden_states,
                canonical_layer=config.canonical_layer,
            )
            if first_summary_fingerprint is None:
                first_summary_fingerprint = summary_fingerprint(token_summaries)

            preserve_recent_window = (
                min(config.recent_token_window, token_summaries.shape[0])
                if config.preserve_recent_window
                else 0
            )
            current_attention_scores: np.ndarray | None = None
            if config.retention_policy in attention_score_policies:
                current_attention_scores = extract_last_token_attention_scores(
                    outputs.attentions
                )
                if first_policy_fingerprint is None:
                    first_policy_fingerprint = summary_fingerprint(
                        current_attention_scores.reshape(-1, 1)
                    )
                if config.retention_policy in {"h2o", "sentence_h2o"}:
                    if accumulated_attention_scores is None:
                        accumulated_attention_scores = current_attention_scores
                    else:
                        accumulated_attention_scores = accumulate_attention_scores(
                            accumulated_attention_scores,
                            current_attention_scores,
                        )
            should_evict = active_input_ids.shape[-1] > config.cache_budget_tokens
            if (
                config.retention_policy == "sentence_vorn"
                and config.eviction_trigger == "sentence_boundary"
            ):
                rendered_active_text = self._tokenizer.decode(
                    active_input_ids[0],
                    skip_special_tokens=True,
                )
                should_evict = should_trigger_sentence_boundary_eviction(
                    rendered_text=rendered_active_text,
                    token_count=int(active_input_ids.shape[-1]),
                    cache_budget_tokens=config.cache_budget_tokens,
                    lookahead_tokens=config.sentence_boundary_lookahead_tokens,
                    force_overflow_ratio=config.force_eviction_overflow_ratio,
                )
            if should_evict:
                if config.retention_policy == "vorn":
                    keep_positions = select_retained_positions(
                        token_summaries,
                        compute_vorn_direction(
                            token_summaries,
                            recent_token_window=config.recent_token_window,
                        ),
                        cache_budget_tokens=config.cache_budget_tokens,
                        always_keep_prefix_tokens=config.always_keep_prefix_tokens,
                        preserve_recent_window=preserve_recent_window,
                    )
                elif config.retention_policy in {
                    "sentence_vorn",
                    "word_vorn",
                    "sentence_tova",
                    "sentence_h2o",
                }:
                    if config.retention_policy in {"sentence_tova", "sentence_h2o"}:
                        if config.retention_policy == "sentence_tova":
                            assert current_attention_scores is not None
                            unit_scores = current_attention_scores
                        else:
                            assert accumulated_attention_scores is not None
                            unit_scores = accumulated_attention_scores
                    else:
                        vorn_direction = compute_vorn_direction(
                            token_summaries,
                            recent_token_window=config.recent_token_window,
                        )
                        unit_scores = np.asarray(
                            [
                                cosine_similarity(
                                    token_summaries[position],
                                    vorn_direction,
                                )
                                for position in range(token_summaries.shape[0])
                            ],
                            dtype=np.float32,
                        )
                    unit_ids = build_unit_ids_for_active_positions(
                        prompt_unit_ids=prompt_unit_ids,
                        active_absolute_positions=tuple(
                            int(position)
                            for position in active_position_ids[0].detach().cpu().tolist()
                        ),
                    )
                    keep_positions = select_sentence_retained_positions(
                        unit_scores,
                        unit_ids=unit_ids,
                        cache_budget_tokens=config.cache_budget_tokens,
                        always_keep_prefix_tokens=config.always_keep_prefix_tokens,
                        preserve_recent_window=preserve_recent_window,
                        pooling=config.sentence_pooling,
                        top_k=config.sentence_top_k,
                    )
                elif config.retention_policy == "adaptive_vorn":
                    vorn_direction = compute_vorn_direction(
                        token_summaries,
                        recent_token_window=config.recent_token_window,
                    )
                    alignment_scores = np.asarray(
                        [
                            cosine_similarity(token_summaries[position], vorn_direction)
                            for position in range(token_summaries.shape[0])
                        ],
                        dtype=np.float32,
                    )
                    unit_ids = build_sentence_unit_ids_for_active_positions(
                        prompt_sentence_ids=prompt_unit_ids,
                        active_absolute_positions=tuple(
                            int(position)
                            for position in active_position_ids[0].detach().cpu().tolist()
                        ),
                    )
                    (
                        keep_positions,
                        selected_unit,
                        _token_signal,
                        _sentence_signal,
                    ) = select_adaptive_retained_positions(
                        alignment_scores,
                        unit_ids=unit_ids,
                        cache_budget_tokens=config.cache_budget_tokens,
                        always_keep_prefix_tokens=config.always_keep_prefix_tokens,
                        preserve_recent_window=preserve_recent_window,
                        pooling=config.sentence_pooling,
                        top_k=config.sentence_top_k,
                    )
                    if selected_unit == "sentence":
                        adaptive_sentence_steps += 1
                    else:
                        adaptive_token_steps += 1
                elif config.retention_policy == "tova":
                    assert current_attention_scores is not None
                    keep_positions = select_attention_score_retained_positions(
                        current_attention_scores,
                        cache_budget_tokens=config.cache_budget_tokens,
                        always_keep_prefix_tokens=config.always_keep_prefix_tokens,
                        preserve_recent_window=preserve_recent_window,
                    )
                elif config.retention_policy == "h2o":
                    assert accumulated_attention_scores is not None
                    keep_positions = select_attention_score_retained_positions(
                        accumulated_attention_scores,
                        cache_budget_tokens=config.cache_budget_tokens,
                        always_keep_prefix_tokens=config.always_keep_prefix_tokens,
                        preserve_recent_window=preserve_recent_window,
                    )
                elif config.retention_policy == "random":
                    keep_positions = select_random_retained_positions(
                        token_count=token_summaries.shape[0],
                        cache_budget_tokens=config.cache_budget_tokens,
                        always_keep_prefix_tokens=config.always_keep_prefix_tokens,
                        preserve_recent_window=preserve_recent_window,
                        rng=self._deterministic_live_rng(
                            prompt=prompt,
                            active_position_ids=active_position_ids,
                            random_seed=config.random_seed,
                        ),
                    )
                elif config.retention_policy == "sliding_window":
                    keep_positions = select_sliding_window_retained_positions(
                        token_count=token_summaries.shape[0],
                        cache_budget_tokens=config.cache_budget_tokens,
                    )
                elif config.retention_policy == "prefix_suffix":
                    keep_positions = select_prefix_suffix_retained_positions(
                        token_count=token_summaries.shape[0],
                        cache_budget_tokens=config.cache_budget_tokens,
                        prefix_token_count=config.prefix_suffix_prefix_tokens,
                    )
                elif config.retention_policy == "summarize":
                    context, question = split_context_and_question_for_summary(prompt)
                    summarize_start = time.perf_counter()
                    summary_prompt = build_summary_prompt(context)
                    summary_input_ids, summary_attention_mask = self._render_prompt(
                        summary_prompt
                    )
                    summary = self._generate_from_tensors(
                        summary_input_ids,
                        summary_attention_mask,
                        max_new_tokens=config.cache_budget_tokens,
                    )
                    summarize_elapsed_seconds = time.perf_counter() - summarize_start
                    fitted_summary = self._trim_summary_to_budget(
                        summary=summary,
                        question=question,
                        cache_budget_tokens=config.cache_budget_tokens,
                    )
                    answer_prompt = build_summary_answer_prompt(
                        summary=fitted_summary,
                        question=question,
                    )
                    answer_input_ids, answer_attention_mask = self._render_prompt(
                        answer_prompt
                    )
                    prediction = self._generate_from_tensors(
                        answer_input_ids,
                        answer_attention_mask,
                    )
                    answer_prompt_token_count = int(answer_input_ids.shape[-1])
                    return prediction, LiveEvictionStats(
                        prompt_token_count=prompt_token_count,
                        generated_token_count=0,
                        mean_kept_token_count=float(answer_prompt_token_count),
                        final_kept_token_count=answer_prompt_token_count,
                        eviction_steps=0,
                        mean_retention_ratio=(
                            answer_prompt_token_count / prompt_token_count
                            if prompt_token_count
                            else 0.0
                        ),
                        retention_policy=config.retention_policy,
                        summary_contract=SUMMARIZE_COMPACT_CONTRACT,
                        summary_fingerprint=hashlib.sha256(
                            fitted_summary.encode("utf-8")
                        ).hexdigest(),
                        preprocessing_elapsed_seconds=summarize_elapsed_seconds,
                    )
                else:
                    raise ValueError(
                        f"unknown retention_policy: {config.retention_policy}"
                    )
                keep_tensor = torch.tensor(
                    keep_positions,
                    device=active_input_ids.device,
                )
                active_input_ids = active_input_ids.index_select(1, keep_tensor)
                active_attention_mask = active_attention_mask.index_select(1, keep_tensor)
                active_position_ids = active_position_ids.index_select(1, keep_tensor)
                if accumulated_attention_scores is not None:
                    accumulated_attention_scores = accumulated_attention_scores[
                        list(keep_positions)
                    ]
                retained_token_counts.append(len(keep_positions))
                eviction_steps += 1
                outputs = self._forward_with_hidden_states(
                    input_ids=active_input_ids,
                    attention_mask=active_attention_mask,
                    position_ids=active_position_ids,
                    output_attentions=needs_attention_scores,
                )
            else:
                retained_token_counts.append(int(active_input_ids.shape[-1]))

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_token_id = int(next_token.item())
            if next_token_id == int(self._tokenizer.eos_token_id):
                break

            generated_token_ids.append(next_token_id)
            active_input_ids = torch.cat([active_input_ids, next_token], dim=1)
            active_attention_mask = torch.cat(
                [
                    active_attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=active_attention_mask.dtype,
                        device=active_attention_mask.device,
                    ),
                ],
                dim=1,
            )
            active_position_ids = torch.cat(
                [
                    active_position_ids,
                    torch.tensor(
                        [[next_absolute_position]],
                        dtype=active_position_ids.dtype,
                        device=active_position_ids.device,
                    ),
                ],
                dim=1,
            )
            if accumulated_attention_scores is not None:
                accumulated_attention_scores = np.concatenate(
                    [
                        accumulated_attention_scores,
                        np.zeros((1,), dtype=np.float32),
                    ]
                )
            next_absolute_position += 1

        generated_token_count = len(generated_token_ids)
        mean_kept_token_count = (
            float(sum(retained_token_counts)) / len(retained_token_counts)
            if retained_token_counts
            else float(prompt_token_count)
        )
        mean_retention_ratio = (
            mean_kept_token_count / prompt_token_count if prompt_token_count else 0.0
        )
        prediction = self._tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
        ).strip()
        if config.retention_policy in {"tova", "sentence_tova"}:
            policy_contract = TOVA_ATTENTION_CONTRACT
            policy_fingerprint = first_policy_fingerprint or ""
        elif config.retention_policy in {"h2o", "sentence_h2o"}:
            policy_contract = H2O_ATTENTION_CONTRACT
            policy_fingerprint = first_policy_fingerprint or ""
        elif config.retention_policy == "adaptive_vorn":
            policy_contract = SEMANTIC_SUMMARY_CONTRACT
            policy_fingerprint = first_summary_fingerprint or ""
        else:
            policy_contract = SEMANTIC_SUMMARY_CONTRACT
            policy_fingerprint = first_summary_fingerprint or ""
        return prediction, LiveEvictionStats(
            prompt_token_count=prompt_token_count,
            generated_token_count=generated_token_count,
            mean_kept_token_count=mean_kept_token_count,
            final_kept_token_count=int(active_input_ids.shape[-1]),
            eviction_steps=eviction_steps,
            mean_retention_ratio=mean_retention_ratio,
            retention_policy=config.retention_policy,
            summary_contract=policy_contract,
            summary_fingerprint=policy_fingerprint,
            adaptive_token_steps=adaptive_token_steps,
            adaptive_sentence_steps=adaptive_sentence_steps,
            adaptive_selector_contract=(
                ADAPTIVE_SELECTOR_CONTRACT
                if config.retention_policy == "adaptive_vorn"
                else ""
            ),
        )

    @staticmethod
    def _deterministic_live_rng(
        *,
        prompt: str,
        active_position_ids: Any,
        random_seed: int,
    ) -> np.random.Generator:
        digest = hashlib.sha256()
        digest.update(prompt.encode("utf-8"))
        digest.update(str(random_seed).encode("ascii"))
        digest.update(
            np.ascontiguousarray(
                active_position_ids.detach().cpu().numpy(),
                dtype=np.int64,
            ).tobytes()
        )
        seed = int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)
        return np.random.default_rng(seed)


def select_week1_plan(benchmark: str, baseline: str = "vanilla") -> ExecutionPlan:
    run_id = f"week1-{benchmark}-{baseline}"
    for plan in build_execution_plans():
        if plan.run.run_id == run_id:
            return plan
    raise ValueError(f"unknown Week 1 plan: {run_id}")


def select_step1_plans() -> tuple[ExecutionPlan, ExecutionPlan]:
    plans = build_execution_plans(build_step1_run_matrix())
    if len(plans) != 2:
        raise ValueError("expected exactly two Step 1 execution plans")
    return plans[0], plans[1]


def select_live_eviction_plan(
    *,
    cache_budget_tokens: int | None = None,
    retention_policy: str = "vorn",
    random_seed: int = 17,
    always_keep_prefix_tokens: int = 1,
    preserve_recent_window: bool = True,
    sentence_pooling: str = "max",
    sentence_top_k: int = 3,
    eviction_trigger: str = "budget_threshold",
    sentence_boundary_lookahead_tokens: int = 25,
    force_eviction_overflow_ratio: float = 1.2,
) -> ExecutionPlan:
    baseline_by_policy = {
        "vorn": "vorn_live",
        "sentence_vorn": "sentence_vorn_live",
        "word_vorn": "word_vorn_live",
        "adaptive_vorn": "adaptive_vorn_live",
        "tova": "tova_live",
        "h2o": "h2o_live",
        "sentence_tova": "sentence_tova_live",
        "sentence_h2o": "sentence_h2o_live",
        "random": "random_live",
        "sliding_window": "sliding_window_live",
        "prefix_suffix": "prefix_suffix_live",
        "summarize": "summarize_compact",
    }
    try:
        baseline = baseline_by_policy[retention_policy]
    except KeyError as exc:
        raise ValueError(f"unknown retention_policy: {retention_policy}") from exc

    if (
        cache_budget_tokens is None
        and retention_policy == "vorn"
        and random_seed == 17
        and always_keep_prefix_tokens == 1
        and preserve_recent_window is True
    ):
        run = build_live_eviction_run()
    else:
        compression_mode = f"live_eviction_only_{retention_policy}"
        if (
            retention_policy == "vorn"
            and always_keep_prefix_tokens == 0
            and preserve_recent_window is False
        ):
            compression_mode = "live_eviction_only_vorn_no_guardrails"
        elif retention_policy in {"sentence_vorn", "word_vorn"}:
            unit_name = "sentence" if retention_policy == "sentence_vorn" else "word"
            compression_mode = f"live_eviction_{unit_name}_vorn_{sentence_pooling}"
            if (
                always_keep_prefix_tokens == 0
                and preserve_recent_window is False
            ):
                compression_mode = (
                    f"live_eviction_{unit_name}_vorn_{sentence_pooling}_no_guardrails"
                )
            if eviction_trigger == "sentence_boundary":
                compression_mode = f"{compression_mode}_sentence_boundary"
        elif retention_policy in {"sentence_tova", "sentence_h2o"}:
            method_name = "tova" if retention_policy == "sentence_tova" else "h2o"
            compression_mode = (
                f"live_eviction_sentence_{method_name}_{sentence_pooling}"
            )
        elif retention_policy == "adaptive_vorn":
            compression_mode = (
                f"live_eviction_adaptive_vorn_{sentence_pooling}"
            )
            if (
                always_keep_prefix_tokens == 0
                and preserve_recent_window is False
            ):
                compression_mode = (
                    f"live_eviction_adaptive_vorn_{sentence_pooling}_no_guardrails"
                )
        run = build_live_eviction_run(
            live=LiveEvictionDefaults(
                cache_budget_tokens=cache_budget_tokens
                or DEFAULT_LIVE_EVICTION_CACHE_BUDGET,
                baseline=baseline,
                retention_policy=retention_policy,
                random_seed=random_seed,
                always_keep_prefix_tokens=always_keep_prefix_tokens,
                preserve_recent_window=preserve_recent_window,
                eviction_unit=(
                    "adaptive_token_or_sentence"
                    if retention_policy == "adaptive_vorn"
                    else "sentence"
                    if retention_policy in {
                        "sentence_vorn",
                        "sentence_tova",
                        "sentence_h2o",
                    }
                    else "word"
                    if retention_policy == "word_vorn"
                    else "token_position"
                ),
                sentence_pooling=(
                    sentence_pooling
                    if retention_policy in {
                        "sentence_vorn",
                        "sentence_tova",
                        "sentence_h2o",
                        "word_vorn",
                        "adaptive_vorn",
                    }
                    else None
                ),
                sentence_top_k=(
                    sentence_top_k
                    if retention_policy in {
                        "sentence_vorn",
                        "sentence_tova",
                        "sentence_h2o",
                        "word_vorn",
                        "adaptive_vorn",
                    }
                    else None
                ),
                eviction_trigger=eviction_trigger,
                sentence_boundary_lookahead_tokens=sentence_boundary_lookahead_tokens,
                force_eviction_overflow_ratio=force_eviction_overflow_ratio,
                compression_mode=compression_mode,
            )
        )
    plans = build_execution_plans((run,))
    if len(plans) != 1:
        raise ValueError("expected exactly one live-eviction execution plan")
    return plans[0]


def run_local_vanilla_smoke(
    *,
    benchmark: str,
    dataset_path: Path,
    output_path: Path,
    case_limit: int = 5,
    generator: TextGenerator | None = None,
    model_config: LocalModelConfig = LocalModelConfig(),
) -> tuple[RunResult, tuple[PredictionTrace, ...]]:
    if case_limit <= 0:
        raise ValueError("case_limit must be positive")

    plan = select_week1_plan(benchmark, baseline="vanilla")
    cases = load_cases(benchmark, dataset_path)[:case_limit]
    if generator is None:
        generator = TransformersTextGenerator(model_config)

    reset_runtime_telemetry()
    result, traces = run_vanilla(plan, cases, generator)
    result = attach_runtime_telemetry(result)
    append_result(output_path, result)
    return result, traces


def run_local_step1_pair(
    *,
    dataset_path: Path,
    output_dir: Path,
    case_limit: int = 5,
    vanilla_generator: TextGenerator | None = None,
    vorn_generator: VornTextGenerator | None = None,
    model_config: LocalModelConfig = LocalModelConfig(),
) -> tuple[
    tuple[RunResult, tuple[PredictionTrace, ...]],
    tuple[RunResult, tuple[VornPredictionTrace, ...]],
]:
    if case_limit <= 0:
        raise ValueError("case_limit must be positive")

    vanilla_plan, vorn_plan = select_step1_plans()
    cases = load_cases("niah", dataset_path)[:case_limit]
    if vanilla_generator is None:
        vanilla_generator = TransformersTextGenerator(model_config)
    if vorn_generator is None:
        vorn_generator = TransformersVornGenerator(model_config)

    reset_runtime_telemetry()
    vanilla_result, vanilla_traces = run_vanilla(
        vanilla_plan,
        cases,
        vanilla_generator,
    )
    vanilla_result = attach_runtime_telemetry(vanilla_result)

    reset_runtime_telemetry()
    vorn_result, vorn_traces = run_vorn(
        vorn_plan,
        cases,
        vorn_generator,
    )
    vorn_result = attach_runtime_telemetry(vorn_result)

    output_dir.mkdir(parents=True, exist_ok=True)
    append_result(output_dir / f"{vanilla_result.run_id}.jsonl", vanilla_result)
    append_result(output_dir / f"{vorn_result.run_id}.jsonl", vorn_result)
    return (vanilla_result, vanilla_traces), (vorn_result, vorn_traces)


def run_local_live_eviction_smoke(
    *,
    dataset_path: Path,
    output_path: Path,
    case_limit: int = 5,
    cache_budget_tokens: int = DEFAULT_LIVE_EVICTION_CACHE_BUDGET,
    retention_policy: str = "vorn",
    random_seed: int = 17,
    generator: LiveEvictionTextGenerator | None = None,
    model_config: LocalModelConfig = LocalModelConfig(),
) -> tuple[RunResult, tuple[Any, ...]]:
    if case_limit <= 0:
        raise ValueError("case_limit must be positive")

    plan = select_live_eviction_plan(
        cache_budget_tokens=cache_budget_tokens,
        retention_policy=retention_policy,
        random_seed=random_seed,
    )
    cases = load_cases("niah", dataset_path)[:case_limit]
    if generator is None:
        generator = TransformersLiveEvictionGenerator(model_config)

    reset_runtime_telemetry()
    result, traces = run_live_eviction(plan, cases, generator)
    result = attach_runtime_telemetry(result)
    append_result(output_path, result)
    return result, traces
