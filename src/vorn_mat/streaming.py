"""Continuous sentence-at-a-time streaming KV compression.

A fixed-capacity live cache of ``B`` tokens ingests a document one
sentence at a time. Each step scores the resident cache, evicts the
lowest-vorn unguarded sentence units until the incoming sentence fits
(pre-admission minimal eviction), then admits the incoming sentence
intact. The vorn trajectory across steps is the primary deliverable.

Three conditions:

- ``forward_sentence_vorn_continuous_v1`` -- stream the document
  earliest-first, admit at the back, then admit the question block.
- ``backward_sentence_vorn_question_guarded_continuous_v1`` -- warm-start
  with the question pinned at the back, stream earlier sentences in at
  the front; the question is never evicted.
- ``backward_sentence_vorn_question_unguarded_continuous_v1`` -- same
  geometry, the question competes with no retention privilege.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baselines.live_eviction import (
    assign_sentence_ids_from_offsets,
    extract_canonical_token_summaries,
)
from .baselines.vorn import compute_vorn_direction, cosine_similarity
from .benchmarks import BenchmarkCase
from .layout import (
    BackwardContinuousLayoutV1DocumentOrder,
    BackwardContinuousLayoutV1QuestionSinkBack,
    ForwardContinuousLayoutV1,
    LayoutPolicy,
    SentenceUnit,
)
from .local_exec import LocalModelConfig, _TransformersGeneratorBase
from .observation import find_subsequence_spans
from .trace import (
    CaseRawTrace,
    RawStepTrace,
    ReplayTrace,
    ROLE_FILLER,
    ROLE_FRESH_GUARD,
    ROLE_NEEDLE,
    ROLE_QUESTION,
    ROLE_SINK_GUARD,
    REPLAY_ROLE_FILLER,
    REPLAY_ROLE_NEEDLE,
    REPLAY_ROLE_QUESTION,
    REPLAY_ROLE_SINK,
    broadcast_sentence_scores,
    compute_unit_eviction_order,
    derive_replay_trace,
    primary_role,
    rank_descending,
    sentence_rank_descending,
)

DIRECTION_FORWARD = "forward"
DIRECTION_BACKWARD = "backward"

FORWARD_CONDITION_ID = "forward_sentence_vorn_continuous_v1"
BACKWARD_GUARDED_CONDITION_ID = (
    "backward_sentence_vorn_question_guarded_continuous_v1"
)
BACKWARD_UNGUARDED_CONDITION_ID = (
    "backward_sentence_vorn_question_unguarded_continuous_v1"
)

OVERSIZE_FLAG = "runtime_unsupported_oversize_sentence_unit"


def condition_id_for(direction: str, question_guarded: bool) -> str:
    """Resolve the locked continuous condition id from direction + guard."""
    if direction == DIRECTION_FORWARD:
        return FORWARD_CONDITION_ID
    if direction == DIRECTION_BACKWARD:
        return (
            BACKWARD_GUARDED_CONDITION_ID
            if question_guarded
            else BACKWARD_UNGUARDED_CONDITION_ID
        )
    raise ValueError(f"unknown compression direction: {direction}")


@dataclass(frozen=True)
class StreamingCompressionConfig:
    cache_budget_tokens: int = 1024
    canonical_layer: int = 16
    recent_token_window: int = 16
    sentence_pooling: str = "max"
    sentence_top_k: int = 3
    seed: int = 17
    direction: str = DIRECTION_FORWARD
    question_guarded: bool = True


@dataclass
class StreamingCaseOutcome:
    """Per-fixture mechanism + outcome bundle for one continuous run."""

    case_id: str
    condition_id: str
    budget: int
    direction: str
    question_guarded: bool
    prediction: str
    correct: bool
    n_steps: int
    needle_survives_step: list[bool]
    question_survives_step: list[bool]
    final_needle_survives: bool
    final_question_survives: bool
    layout_policy_id: str
    runtime_flag: str | None
    raw_trace: CaseRawTrace
    replay_trace: ReplayTrace


# Priority-ordered question markers. RULER NIAH 8k prompts open the
# question with "What is the special magic number ..." rather than a
# "Question:" label, so those variants are matched explicitly.
QUESTION_MARKERS = [
    "\n\nQuestion:",
    "\nQuestion:",
    "\n\nQ:",
    "\nWhat is the special magic",
    "\nWhat are the special magic",
    "What is the special magic",
    "What are the special magic",
    "Question:",
]


def _split_document_and_question(prompt: str) -> tuple[str, str]:
    """Split a RULER NIAH prompt into document and question portions."""
    for marker in QUESTION_MARKERS:
        pos = prompt.rfind(marker)
        if pos >= 0:
            return prompt[:pos], prompt[pos:]
    raise ValueError("could not find question marker in prompt")


def _needle_survives(
    active_token_ids: list[int],
    answer_variants: list[list[int]],
) -> bool:
    """True if any answer-token variant survives as a subsequence."""
    for variant in answer_variants:
        if variant and find_subsequence_spans(active_token_ids, variant):
            return True
    return False


def _collapse_sentence_role(
    sentence_id: int,
    needle_sentence_ids: set[int],
    question_sentence_ids: set[int],
    guarded_sentence_ids: set[int],
) -> str:
    """Collapse a sentence to the public replay role with precedence."""
    if sentence_id in question_sentence_ids:
        return REPLAY_ROLE_QUESTION
    if sentence_id in needle_sentence_ids:
        return REPLAY_ROLE_NEEDLE
    if sentence_id in guarded_sentence_ids:
        return REPLAY_ROLE_SINK
    return REPLAY_ROLE_FILLER


class StreamingCompressionGenerator(_TransformersGeneratorBase):
    """Continuous sentence-at-a-time streaming-compression generator."""

    def __init__(
        self,
        config: LocalModelConfig = LocalModelConfig(),
        forward_layout_policy: LayoutPolicy | None = None,
        backward_guarded_layout_policy: LayoutPolicy | None = None,
        backward_unguarded_layout_policy: LayoutPolicy | None = None,
    ):
        super().__init__(config)
        self.forward_layout_policy = (
            forward_layout_policy or ForwardContinuousLayoutV1()
        )
        self.backward_guarded_layout_policy = (
            backward_guarded_layout_policy
            or BackwardContinuousLayoutV1QuestionSinkBack()
        )
        self.backward_unguarded_layout_policy = (
            backward_unguarded_layout_policy
            or BackwardContinuousLayoutV1DocumentOrder()
        )

    # -- tokenization / segmentation helpers -----------------------------

    def _encode_with_sentences(
        self, text: str,
    ) -> tuple[list[int], tuple[int, ...]]:
        """Tokenize text and assign a stable sentence id to every token."""
        assert self._tokenizer is not None
        encoded = self._tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True,
        )
        token_ids = list(encoded["input_ids"])
        offsets = [(int(s), int(e)) for s, e in encoded["offset_mapping"]]
        sentence_ids = assign_sentence_ids_from_offsets(text, offsets)
        if len(sentence_ids) != len(token_ids):
            sentence_ids = tuple(range(len(token_ids)))
        return token_ids, sentence_ids

    def _render_question_text(self, question_text: str) -> str:
        """Render the question into its answer-frame chat-template form."""
        assert self._tokenizer is not None
        if getattr(self._tokenizer, "chat_template", None):
            return self._tokenizer.apply_chat_template(
                [{"role": "user", "content": question_text.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return question_text

    def _sentence_texts(
        self, full_seq: list[int], sentence_id_by_abs: list[int],
    ) -> tuple[dict[int, str], dict[int, int]]:
        """Decode per-sentence text and doc-order position."""
        assert self._tokenizer is not None
        texts: dict[int, str] = {}
        positions: dict[int, int] = {}
        if not full_seq:
            return texts, positions
        start = 0
        current = sentence_id_by_abs[0]
        for i in range(1, len(full_seq) + 1):
            if i == len(full_seq) or sentence_id_by_abs[i] != current:
                texts[current] = self._tokenizer.decode(
                    full_seq[start:i], skip_special_tokens=True,
                ).strip()
                positions[current] = current
                if i < len(full_seq):
                    current = sentence_id_by_abs[i]
                    start = i
        return texts, positions

    def _vorn_scores(
        self, active_ids: list[int], config: StreamingCompressionConfig,
    ) -> np.ndarray:
        """Run a forward pass and score every active token against vorn."""
        import torch

        tensor = torch.tensor(
            [active_ids], device=self._device, dtype=torch.long,
        )
        mask = torch.ones_like(tensor)
        positions = torch.tensor(
            [list(range(len(active_ids)))],
            device=self._device, dtype=torch.long,
        )
        outputs = self._forward_with_hidden_states(
            input_ids=tensor,
            attention_mask=mask,
            position_ids=positions,
        )
        summaries = extract_canonical_token_summaries(
            outputs.hidden_states,
            canonical_layer=config.canonical_layer,
        )
        direction = compute_vorn_direction(
            summaries, recent_token_window=config.recent_token_window,
        )
        return np.array(
            [
                cosine_similarity(summaries[i], direction)
                for i in range(summaries.shape[0])
            ],
            dtype=np.float32,
        )

    def _build_units(
        self,
        full_seq: list[int],
        sentence_id_by_abs: list[int],
        needle_abs: set[int],
        question_abs: set[int],
        guard_question: bool,
    ) -> list[SentenceUnit]:
        """Group the streamed sequence into whole-sentence cache units."""
        units: list[SentenceUnit] = []
        if not full_seq:
            return units
        start = 0
        current = sentence_id_by_abs[0]
        for i in range(1, len(full_seq) + 1):
            if i == len(full_seq) or sentence_id_by_abs[i] != current:
                abs_positions = tuple(range(start, i))
                is_question = any(a in question_abs for a in abs_positions)
                static_tags: list[tuple[str, ...]] = []
                for a in abs_positions:
                    tags: list[str] = []
                    if a in needle_abs:
                        tags.append(ROLE_NEEDLE)
                    if a in question_abs:
                        tags.append(ROLE_QUESTION)
                    if not tags:
                        tags = [ROLE_FILLER]
                    static_tags.append(tuple(tags))
                units.append(SentenceUnit(
                    sentence_id=current,
                    doc_position=current,
                    token_ids=tuple(full_seq[start:i]),
                    absolute_positions=abs_positions,
                    static_role_tags=tuple(static_tags),
                    is_question=is_question,
                    guarded=(guard_question and is_question),
                ))
                if i < len(full_seq):
                    current = sentence_id_by_abs[i]
                    start = i
        return units

    # -- main streaming entrypoint ---------------------------------------

    def generate_with_continuous_streaming(
        self,
        prompt: str,
        expected_answer: str,
        case_id: str,
        config: StreamingCompressionConfig,
    ) -> StreamingCaseOutcome:
        self._ensure_model()
        assert self._tokenizer is not None
        assert self._model is not None

        direction = config.direction
        guarded = config.question_guarded
        condition_id = condition_id_for(direction, guarded)
        B = config.cache_budget_tokens

        if direction == DIRECTION_FORWARD:
            layout_policy: LayoutPolicy = self.forward_layout_policy
        elif guarded:
            layout_policy = self.backward_guarded_layout_policy
        else:
            layout_policy = self.backward_unguarded_layout_policy

        document_text, question_text = _split_document_and_question(prompt)
        doc_ids, doc_sentence_ids = self._encode_with_sentences(document_text)
        q_ids, q_sentence_ids_local = self._encode_with_sentences(
            self._render_question_text(question_text)
        )

        answer_variants = [
            list(
                self._tokenizer(expected_answer, add_special_tokens=False)[
                    "input_ids"
                ]
            ),
            list(
                self._tokenizer(
                    " " + expected_answer, add_special_tokens=False,
                )["input_ids"]
            ),
        ]

        q_offset = (max(doc_sentence_ids) + 1) if doc_sentence_ids else 0
        full_seq = list(doc_ids) + list(q_ids)
        sentence_id_by_abs = list(doc_sentence_ids) + [
            q_offset + s for s in q_sentence_ids_local
        ]
        question_abs = set(range(len(doc_ids), len(full_seq)))

        sentence_texts, sentence_doc_positions = self._sentence_texts(
            full_seq, sentence_id_by_abs,
        )
        needle_abs: set[int] = set()
        for variant in answer_variants:
            for span_start, span_end in find_subsequence_spans(
                full_seq, variant,
            ):
                needle_abs.update(range(span_start, span_end))
        needle_sentence_ids = sorted(
            {sentence_id_by_abs[a] for a in needle_abs}
        )
        question_sentence_ids = sorted(
            {sentence_id_by_abs[a] for a in question_abs}
        )

        units = self._build_units(
            full_seq, sentence_id_by_abs, needle_abs, question_abs,
            guard_question=(direction == DIRECTION_BACKWARD and guarded),
        )
        guarded_sentence_ids = {
            u.sentence_id for u in units if u.guarded
        }

        return self._run_stream(
            units=units,
            config=config,
            direction=direction,
            guarded=guarded,
            condition_id=condition_id,
            layout_policy=layout_policy,
            budget=B,
            case_id=case_id,
            expected_answer=expected_answer,
            answer_variants=answer_variants,
            needle_sentence_ids=needle_sentence_ids,
            question_sentence_ids=question_sentence_ids,
            guarded_sentence_ids=guarded_sentence_ids,
            sentence_texts=sentence_texts,
            sentence_doc_positions=sentence_doc_positions,
        )

    def _warm_start(
        self,
        units: list[SentenceUnit],
        direction: str,
        budget: int,
    ) -> tuple[list[SentenceUnit], list[SentenceUnit]]:
        """Fill the cache to budget; return (cache, incoming_sequence)."""
        if direction == DIRECTION_FORWARD:
            cache: list[SentenceUnit] = []
            used = 0
            idx = 0
            while idx < len(units):
                unit = units[idx]
                if used + unit.token_count > budget:
                    break
                cache.append(unit)
                used += unit.token_count
                idx += 1
            return cache, units[idx:]

        # backward: question block pinned, fill from the latest doc unit
        question_units = [u for u in units if u.is_question]
        doc_units = [u for u in units if not u.is_question]
        used = sum(u.token_count for u in question_units)
        start_idx = len(doc_units)
        while start_idx > 0:
            unit = doc_units[start_idx - 1]
            if used + unit.token_count > budget:
                break
            used += unit.token_count
            start_idx -= 1
        cache = doc_units[start_idx:] + question_units
        incoming = list(reversed(doc_units[:start_idx]))
        return cache, incoming

    def _run_stream(
        self,
        *,
        units: list[SentenceUnit],
        config: StreamingCompressionConfig,
        direction: str,
        guarded: bool,
        condition_id: str,
        layout_policy: LayoutPolicy,
        budget: int,
        case_id: str,
        expected_answer: str,
        answer_variants: list[list[int]],
        needle_sentence_ids: list[int],
        question_sentence_ids: list[int],
        guarded_sentence_ids: set[int],
        sentence_texts: dict[int, str],
        sentence_doc_positions: dict[int, int],
    ) -> StreamingCaseOutcome:
        runtime_flag: str | None = None
        if any(u.token_count > budget for u in units):
            runtime_flag = OVERSIZE_FLAG
        guarded_total = sum(
            u.token_count for u in units if u.guarded
        )
        if guarded_total > budget:
            runtime_flag = OVERSIZE_FLAG

        steps: list[RawStepTrace] = []
        needle_survives_step: list[bool] = []
        question_survives_step: list[bool] = []
        cache: list[SentenceUnit] = []

        if runtime_flag is None:
            cache, incoming_sequence = self._warm_start(
                units, direction, budget,
            )
            last_admitted_id: int | None = None

            for step_index, incoming in enumerate(incoming_sequence):
                raw_step, evicted_ids, fatal = self._run_step(
                    step_index=step_index,
                    cache=cache,
                    incoming=incoming,
                    config=config,
                    direction=direction,
                    guarded=guarded,
                    condition_id=condition_id,
                    layout_policy_id=layout_policy.policy_id,
                    budget=budget,
                    last_admitted_id=last_admitted_id,
                )
                if fatal:
                    runtime_flag = OVERSIZE_FLAG
                    break
                placement = layout_policy.place_after_eviction(
                    active_units=cache,
                    evicted_sentence_ids=evicted_ids,
                    incoming_unit=incoming,
                )
                cache = list(placement.units)
                last_admitted_id = incoming.sentence_id

                cache_token_ids = [
                    t for u in cache for t in u.token_ids
                ]
                needle_survives_step.append(
                    _needle_survives(cache_token_ids, answer_variants)
                )
                if direction == DIRECTION_BACKWARD:
                    question_survives_step.append(
                        self._question_fully_resident(
                            cache, question_sentence_ids,
                        )
                    )
                steps.append(raw_step)

        final_token_ids = [t for u in cache for t in u.token_ids]
        final_needle_survives = _needle_survives(
            final_token_ids, answer_variants,
        )
        final_question_survives = self._question_fully_resident(
            cache, question_sentence_ids,
        )

        if runtime_flag is None and self.config.max_new_tokens > 0:
            prediction = self._generate_answer(final_token_ids)
            correct = expected_answer.lower() in prediction.lower()
        else:
            prediction = ""
            correct = runtime_flag is None and final_needle_survives

        raw_trace = CaseRawTrace(
            case_id=case_id,
            condition_id=condition_id,
            budget=budget,
            compression_direction=direction,
            steps=steps,
            needle_sentence_ids=needle_sentence_ids,
            question_sentence_ids=question_sentence_ids,
            needle_survives_step=needle_survives_step,
            question_survives_step=question_survives_step,
            final_needle_survives=final_needle_survives,
            final_question_survives=final_question_survives,
            runtime_flag=runtime_flag,
        )
        sentence_roles = {
            sid: _collapse_sentence_role(
                sid,
                set(needle_sentence_ids),
                set(question_sentence_ids),
                guarded_sentence_ids,
            )
            for sid in sentence_texts
        }
        replay_trace = derive_replay_trace(
            case_raw=raw_trace,
            sentence_texts=sentence_texts,
            sentence_doc_positions=sentence_doc_positions,
            sentence_roles=sentence_roles,
            model_answer=prediction,
            correct=correct,
            model_id=self.config.model_id,
            dataset_config="",
        )
        return StreamingCaseOutcome(
            case_id=case_id,
            condition_id=condition_id,
            budget=budget,
            direction=direction,
            question_guarded=guarded,
            prediction=prediction,
            correct=correct,
            n_steps=len(steps),
            needle_survives_step=needle_survives_step,
            question_survives_step=question_survives_step,
            final_needle_survives=final_needle_survives,
            final_question_survives=final_question_survives,
            layout_policy_id=layout_policy.policy_id,
            runtime_flag=runtime_flag,
            raw_trace=raw_trace,
            replay_trace=replay_trace,
        )

    def _run_step(
        self,
        *,
        step_index: int,
        cache: list[SentenceUnit],
        incoming: SentenceUnit,
        config: StreamingCompressionConfig,
        direction: str,
        guarded: bool,
        condition_id: str,
        layout_policy_id: str,
        budget: int,
        last_admitted_id: int | None,
    ) -> tuple[RawStepTrace, set[int], bool]:
        """Score the cache, decide eviction, build the raw step trace.

        Returns (raw_step, evicted_sentence_ids, fatal). ``fatal`` is set
        when the incoming sentence cannot fit even after evicting every
        unguarded unit.
        """
        cache_token_ids = [t for u in cache for t in u.token_ids]
        scores = self._vorn_scores(cache_token_ids, config)

        unit_ids: list[int] = []
        abs_positions: list[int] = []
        role_tags: list[list[str]] = []
        for unit in cache:
            for k, token in enumerate(unit.token_ids):
                unit_ids.append(unit.sentence_id)
                abs_positions.append(unit.absolute_positions[k])
                tags = list(unit.static_role_tags[k])
                if unit.guarded:
                    tags.append(ROLE_SINK_GUARD)
                if unit.sentence_id == last_admitted_id:
                    tags.append(ROLE_FRESH_GUARD)
                role_tags.append(tags)

        token_scores = [float(s) for s in scores.tolist()]
        sentence_scores = broadcast_sentence_scores(
            unit_ids, token_scores, pooling=config.sentence_pooling,
        )

        # per-unit score (max pool) for the eviction decision
        unit_score: dict[int, float] = {}
        for sid, score in zip(unit_ids, sentence_scores):
            unit_score[sid] = score

        used = sum(u.token_count for u in cache)
        need = incoming.token_count - (budget - used)
        evicted_ids: set[int] = set()
        fatal = False
        if need > 0:
            candidates = [
                (unit, idx)
                for idx, unit in enumerate(cache)
                if not unit.guarded
            ]
            candidates.sort(
                key=lambda c: (unit_score[c[0].sentence_id], -c[1])
            )
            freed = 0
            for unit, _idx in candidates:
                if freed >= need:
                    break
                evicted_ids.add(unit.sentence_id)
                freed += unit.token_count
            if freed < need:
                fatal = True

        kept_flags = [sid not in evicted_ids for sid in unit_ids]
        evicted_flags = [not f for f in kept_flags]
        retained_tokens = sum(
            u.token_count for u in cache if u.sentence_id not in evicted_ids
        )

        raw_step = RawStepTrace(
            step_index=step_index,
            condition_id=condition_id,
            budget=budget,
            incoming_sentence_id=incoming.sentence_id,
            incoming_sentence_token_count=incoming.token_count,
            active_token_count_pre_eviction=len(cache_token_ids),
            active_token_count_post_admission=(
                retained_tokens + incoming.token_count
            ),
            compression_direction=direction,
            question_present_in_active_window=any(
                u.is_question for u in cache
            ),
            question_guarded=guarded,
            layout_policy_id=layout_policy_id,
            window_positions=list(range(len(cache_token_ids))),
            absolute_positions=abs_positions,
            primary_role=[primary_role(t) for t in role_tags],
            role_tags=role_tags,
            sentence_unit_ids=unit_ids,
            token_vorn_scores=token_scores,
            token_vorn_rank_desc=rank_descending(token_scores),
            sentence_vorn_scores=sentence_scores,
            sentence_vorn_rank_desc=sentence_rank_descending(
                unit_ids, sentence_scores,
            ),
            unit_eviction_order=compute_unit_eviction_order(
                unit_ids, sentence_scores, kept_flags,
            ),
            evicted_this_step=evicted_flags,
            kept_this_step=kept_flags,
        )
        return raw_step, evicted_ids, fatal

    @staticmethod
    def _question_fully_resident(
        cache: list[SentenceUnit],
        question_sentence_ids: list[int],
    ) -> bool:
        """True if every question sentence unit is still in the cache."""
        if not question_sentence_ids:
            return True
        resident = {u.sentence_id for u in cache}
        return all(sid in resident for sid in question_sentence_ids)

    def _generate_answer(self, final_ids: list[int]) -> str:
        """Greedy-decode the answer from a fully-assembled readout frame."""
        import torch

        assert self._tokenizer is not None
        assert self._model is not None

        active_in = torch.tensor(
            [final_ids], device=self._device, dtype=torch.long,
        )
        active_m = torch.ones_like(active_in)
        active_p = torch.tensor(
            [list(range(len(final_ids)))],
            device=self._device, dtype=torch.long,
        )

        generated_ids: list[int] = []
        for _ in range(self.config.max_new_tokens):
            with torch.no_grad():
                out = self._model(
                    input_ids=active_in,
                    attention_mask=active_m,
                    position_ids=active_p,
                    use_cache=False,
                    return_dict=True,
                )
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            next_id = int(next_token.item())
            if self._is_terminal_token_id(next_id):
                break
            generated_ids.append(next_id)
            next_pos = active_p[0, -1].item() + 1
            active_in = torch.cat([active_in, next_token], dim=1)
            active_m = torch.cat(
                [
                    active_m,
                    torch.ones(
                        (1, 1), dtype=active_m.dtype, device=active_m.device,
                    ),
                ],
                dim=1,
            )
            active_p = torch.cat(
                [
                    active_p,
                    torch.tensor(
                        [[next_pos]],
                        dtype=active_p.dtype, device=active_p.device,
                    ),
                ],
                dim=1,
            )

        return self._tokenizer.decode(
            generated_ids, skip_special_tokens=True,
        ).strip()

    def _get_token_offsets(
        self, token_ids: list[int],
    ) -> tuple[tuple[int, int], ...]:
        """Get character-level offsets for token IDs."""
        assert self._tokenizer is not None
        text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        encoded = self._tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True,
        )
        return tuple(
            (int(s), int(e)) for s, e in encoded["offset_mapping"]
        )


def run_continuous_streaming(
    cases: tuple[BenchmarkCase, ...],
    config: StreamingCompressionConfig,
    generator: StreamingCompressionGenerator,
) -> list[StreamingCaseOutcome]:
    """Run one continuous streaming condition over a case slice."""
    outcomes: list[StreamingCaseOutcome] = []
    for case in cases:
        outcomes.append(
            generator.generate_with_continuous_streaming(
                prompt=case.prompt,
                expected_answer=case.expected_answer,
                case_id=case.case_id,
                config=config,
            )
        )
    return outcomes
