"""Baseline registry for Week 1 reproduction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    implementation_slug: str
    cache_strategy: str
    notes: str


BASELINES: dict[str, BaselineSpec] = {
    "vanilla": BaselineSpec(
        name="vanilla",
        implementation_slug="vanilla",
        cache_strategy="full_cache",
        notes="Unmodified model baseline.",
    ),
    "vorn": BaselineSpec(
        name="vorn",
        implementation_slug="vorn",
        cache_strategy="canonical_layer_prompt_retention_proxy",
        notes=(
            "Step 1 proxy for vorn-aligned eviction: rank token positions in one "
            "canonical residual space and retain the highest-alignment prompt tokens "
            "under a fixed budget."
        ),
    ),
    "vorn_live": BaselineSpec(
        name="vorn_live",
        implementation_slug="vorn_live",
        cache_strategy="live_token_position_eviction",
        notes=(
            "Rotary-safe live eviction arm: preserve absolute token positions, "
            "drop whole token-position families across layers, and re-score the "
            "active sequence at each decode step."
        ),
    ),
    "sentence_vorn_live": BaselineSpec(
        name="sentence_vorn_live",
        implementation_slug="sentence_vorn_live",
        cache_strategy="live_sentence_eviction",
        notes=(
            "Sentence-level vorn control: aggregate token alignment into semantic "
            "sentence units, then drop whole sentences under the same fixed budget "
            "instead of fragmenting individual token positions."
        ),
    ),
    "word_vorn_live": BaselineSpec(
        name="word_vorn_live",
        implementation_slug="word_vorn_live",
        cache_strategy="live_word_eviction",
        notes=(
            "Word-level vorn control: aggregate token alignment into semantic "
            "word units, then drop whole words under the same fixed budget "
            "instead of fragmenting individual token positions."
        ),
    ),
    "adaptive_vorn_live": BaselineSpec(
        name="adaptive_vorn_live",
        implementation_slug="adaptive_vorn_live",
        cache_strategy="online_token_or_sentence_eviction",
        notes=(
            "Adaptive v0 control: choose between token-position and sentence-unit "
            "eviction online at each eviction step using the current alignment "
            "score distribution, rather than locking one unit globally."
        ),
    ),
    "tova_live": BaselineSpec(
        name="tova_live",
        implementation_slug="tova_live",
        cache_strategy="last_token_attention_weight_eviction",
        notes=(
            "TOVA-style control arm: rank retained token positions by the most "
            "recent generated token's final-layer attention weights and evict the "
            "lowest-scoring positions under the same fixed budget."
        ),
    ),
    "sentence_tova_live": BaselineSpec(
        name="sentence_tova_live",
        implementation_slug="sentence_tova_live",
        cache_strategy="last_token_attention_weight_sentence_eviction",
        notes=(
            "Sentence-level TOVA-style control arm: aggregate final-layer "
            "last-token attention weights into sentence units, then evict the "
            "lowest-scoring sentences under the same fixed budget."
        ),
    ),
    "h2o_live": BaselineSpec(
        name="h2o_live",
        implementation_slug="h2o_live",
        cache_strategy="accumulated_attention_weight_eviction",
        notes=(
            "H2O-style control arm: rank retained token positions by accumulated "
            "attention mass over time, while preserving the same prefix and recent "
            "window guardrails under the same fixed budget."
        ),
    ),
    "sentence_h2o_live": BaselineSpec(
        name="sentence_h2o_live",
        implementation_slug="sentence_h2o_live",
        cache_strategy="accumulated_attention_weight_sentence_eviction",
        notes=(
            "Sentence-level H2O-style control arm: accumulate attention mass over "
            "time, aggregate it into sentence units, and evict the lowest-scoring "
            "sentences under the same fixed budget."
        ),
    ),
    "random_live": BaselineSpec(
        name="random_live",
        implementation_slug="random_live",
        cache_strategy="uniform_random_token_position_eviction",
        notes=(
            "Control arm for live eviction: preserve the same required prefix and "
            "recent-window positions, but fill the remaining retention budget by "
            "deterministic random selection instead of vorn-conditioned scoring."
        ),
    ),
    "sliding_window_live": BaselineSpec(
        name="sliding_window_live",
        implementation_slug="sliding_window_live",
        cache_strategy="recent_only_token_position_eviction",
        notes=(
            "Production-realistic truncation control: keep the most recent token "
            "positions under the same fixed budget and drop everything older."
        ),
    ),
    "prefix_suffix_live": BaselineSpec(
        name="prefix_suffix_live",
        implementation_slug="prefix_suffix_live",
        cache_strategy="prefix_and_recent_suffix_token_position_eviction",
        notes=(
            "Attention-sinks-style control: preserve a fixed prompt prefix plus "
            "the most recent token positions, and drop the middle under the same "
            "fixed budget."
        ),
    ),
    "summarize_compact": BaselineSpec(
        name="summarize_compact",
        implementation_slug="summarize_compact",
        cache_strategy="whole_context_summary_then_answer",
        notes=(
            "Two-pass compaction control: summarize the context without peeking at "
            "the final question, then answer from the summary under the same fixed "
            "budget."
        ),
    ),
    "h2o": BaselineSpec(
        name="h2o",
        implementation_slug="h2o",
        cache_strategy="heavy_hitter_oracle",
        notes="Retention by recent-plus-heavy-hitter heuristic.",
    ),
    "streamingllm": BaselineSpec(
        name="streamingllm",
        implementation_slug="streamingllm",
        cache_strategy="attention_sinks_plus_recent_window",
        notes="Streaming baseline with sink preservation.",
    ),
}


def get_baseline(name: str) -> BaselineSpec:
    try:
        return BASELINES[name]
    except KeyError as exc:
        raise ValueError(f"unknown baseline: {name}") from exc
