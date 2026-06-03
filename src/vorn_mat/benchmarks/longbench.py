"""LongBench PassageRetrieval-en adapter.

The loader intentionally reads the official Hugging Face `data.zip` payload
instead of using `datasets.load_dataset`: modern `datasets` releases reject
dataset scripts such as LongBench.py. The preregistered probe is tied to the
raw JSONL file and snapshot below.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import BenchmarkCase, score_prediction_text


LONGBENCH_DATASET_ID = "THUDM/LongBench"
LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
PASSAGE_RETRIEVAL_EN_CONFIG = "passage_retrieval_en"
PASSAGE_RETRIEVAL_EN_FILE = "data/passage_retrieval_en.jsonl"
PASSAGE_RETRIEVAL_EN_PROMPT_TEMPLATE_ID = "longbench_passage_retrieval_en_v1"
PASSAGE_RETRIEVAL_EN_MAX_NEW_TOKENS = 32
PASSAGE_RETRIEVAL_EN_SCORING_CONTRACT = "longbench_retrieval_score_v1"
PASSAGE_RETRIEVAL_EN_LICENSE_NOTE = (
    "No explicit license string found in the inspected THUDM/LongBench HF README "
    f"at revision {LONGBENCH_REVISION}."
)

PASSAGE_RETRIEVAL_EN_PROMPT = """Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.

{context}

The following is an abstract.

{input}

Please enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.

The answer is: """

_NUMBER_RE = re.compile(r"\d+")
_GROUND_TRUTH_RE = re.compile(r"Paragraph (\d+)")


@dataclass(frozen=True)
class PassageRetrievalScore:
    official_score: float
    binary_paragraph_hit: bool
    numbers_extracted: tuple[str, ...]
    ground_truth_id: str


@dataclass(frozen=True)
class PassageRetrievalCell:
    family: str
    model_id: str
    method_label: str
    retention_policy: str


PASSAGE_RETRIEVAL_EN_PREREGISTERED_CELLS = (
    PassageRetrievalCell(
        family="Mistral",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        method_label="sentence vorn",
        retention_policy="sentence_vorn",
    ),
    PassageRetrievalCell(
        family="Mistral",
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        method_label="sentence attention",
        retention_policy="sentence_tova",
    ),
    PassageRetrievalCell(
        family="Llama 3.1",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        method_label="sentence vorn",
        retention_policy="sentence_vorn",
    ),
    PassageRetrievalCell(
        family="Llama 3.1",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        method_label="sentence attention",
        retention_policy="sentence_tova",
    ),
    PassageRetrievalCell(
        family="Gemma 4",
        model_id="google/gemma-4-E4B-it",
        method_label="sentence vorn",
        retention_policy="sentence_vorn",
    ),
    PassageRetrievalCell(
        family="Gemma 4",
        model_id="google/gemma-4-E4B-it",
        method_label="sentence attention",
        retention_policy="sentence_tova",
    ),
    PassageRetrievalCell(
        family="Qwen 3-NT 8B",
        model_id="Qwen/Qwen3-8B",
        method_label="sentence vorn",
        retention_policy="sentence_vorn",
    ),
    PassageRetrievalCell(
        family="Qwen 3-NT 8B",
        model_id="Qwen/Qwen3-8B",
        method_label="sentence attention",
        retention_policy="sentence_tova",
    ),
)


def _slug(value: str) -> str:
    return (
        value.lower()
        .replace("/", "--")
        .replace(" ", "-")
        .replace(".", "")
        .replace("_", "-")
    )


def build_passage_retrieval_en_cell_specs(
    *,
    results_root: str,
    case_limit: int = 50,
    case_offset_start: int = 0,
    dataset_revision: str = LONGBENCH_REVISION,
    attempt_label: str = "config316",
) -> tuple[dict[str, object], ...]:
    """Return the exact 8 Modal request specs locked by config#316."""

    if case_limit != 50:
        raise ValueError("config#316 locks case_limit=50")
    if case_offset_start != 0:
        raise ValueError("config#316 locks case_offset_start=0")
    if not attempt_label:
        raise ValueError("attempt_label must be non-empty")

    specs: list[dict[str, object]] = []
    for cell in PASSAGE_RETRIEVAL_EN_PREREGISTERED_CELLS:
        family_slug = _slug(cell.family)
        model_slug = _slug(cell.model_id)
        specs.append(
            {
                "dataset_revision": dataset_revision,
                "case_limit": case_limit,
                "case_offset_start": case_offset_start,
                "output_path": (
                    f"{results_root}/longbench-passage-retrieval-en/"
                    f"{attempt_label}-{family_slug}-{model_slug}-"
                    f"{cell.retention_policy}-b1024-n50.jsonl"
                ),
                "max_new_tokens": PASSAGE_RETRIEVAL_EN_MAX_NEW_TOKENS,
                "cache_budget_tokens": 1024,
                "retention_policy": cell.retention_policy,
                "random_seed": 17,
                "always_keep_prefix_tokens": 1,
                "preserve_recent_window": True,
                "sentence_pooling": "max",
                "sentence_top_k": 3,
                "eviction_trigger": "budget_threshold",
                "sentence_boundary_lookahead_tokens": 25,
                "force_eviction_overflow_ratio": 1.2,
                "model_id": cell.model_id,
            }
        )
    return tuple(specs)


def render_passage_retrieval_en_prompt(*, context: str, input_text: str) -> str:
    return PASSAGE_RETRIEVAL_EN_PROMPT.format(context=context, input=input_text)


def extract_prediction_numbers(prediction: str) -> tuple[str, ...]:
    return tuple(_NUMBER_RE.findall(prediction))


def ground_truth_paragraph_id(ground_truth: str) -> str:
    matches = _GROUND_TRUTH_RE.findall(ground_truth)
    if not matches:
        raise ValueError(f"LongBench passage answer is malformed: {ground_truth!r}")
    return matches[0]


def official_retrieval_score(prediction: str, ground_truth: str) -> float:
    """Replicate LongBench.metrics.retrieval_score."""

    ground_truth_id = ground_truth_paragraph_id(ground_truth)
    numbers = extract_prediction_numbers(prediction)
    if not numbers:
        return 0.0
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth_id))
    return float(right_num / len(numbers))


def score_passage_retrieval_prediction(
    prediction: str,
    ground_truth: str,
) -> PassageRetrievalScore:
    scored_prediction = score_prediction_text(prediction)
    ground_truth_id = ground_truth_paragraph_id(ground_truth)
    numbers = extract_prediction_numbers(scored_prediction)
    official_score = official_retrieval_score(scored_prediction, ground_truth)
    binary_hit = (
        bool(numbers)
        and numbers[0] == ground_truth_id
        and all(number == ground_truth_id for number in numbers)
    )
    return PassageRetrievalScore(
        official_score=official_score,
        binary_paragraph_hit=binary_hit,
        numbers_extracted=numbers,
        ground_truth_id=ground_truth_id,
    )


def build_passage_retrieval_observation_fields(
    case: BenchmarkCase,
    prediction: str,
) -> dict[str, object]:
    score = score_passage_retrieval_prediction(prediction, case.expected_answer)
    return {
        "correct": score.binary_paragraph_hit,
        "expected_answer": case.expected_answer,
        "official_score": score.official_score,
        "binary_paragraph_hit": score.binary_paragraph_hit,
        "numbers_extracted": score.numbers_extracted,
        "case_metadata": dict(case.metadata),
    }


def score_longbench_passage_retrieval_predictions(
    cases: tuple[BenchmarkCase, ...],
    predictions: tuple[str, ...],
) -> dict[str, float]:
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must have the same length")
    if not cases:
        return {
            "mean_official_score": 0.0,
            "longbench_percent": 0.0,
            "binary_paragraph_hit_rate": 0.0,
        }

    scores = tuple(
        score_passage_retrieval_prediction(prediction, case.expected_answer)
        for case, prediction in zip(cases, predictions, strict=True)
    )
    mean_score = sum(score.official_score for score in scores) / len(scores)
    binary_hits = sum(1 for score in scores if score.binary_paragraph_hit)
    return {
        "mean_official_score": mean_score,
        "longbench_percent": mean_score * 100.0,
        "binary_paragraph_hit_rate": binary_hits / len(scores),
    }


def benchmark_case_from_longbench_passage_retrieval_record(
    record: dict[str, Any],
    *,
    source_index: int,
    dataset_id: str = LONGBENCH_DATASET_ID,
    revision: str = LONGBENCH_REVISION,
    split: str = "test[:50]",
    file_path: str = PASSAGE_RETRIEVAL_EN_FILE,
) -> BenchmarkCase:
    answers = record.get("answers") or []
    if not answers:
        raise ValueError("LongBench passage_retrieval_en record is missing answers")
    if record.get("dataset") != PASSAGE_RETRIEVAL_EN_CONFIG:
        raise ValueError(
            "LongBench record dataset mismatch: "
            f"{record.get('dataset')!r} != {PASSAGE_RETRIEVAL_EN_CONFIG!r}"
        )
    prompt = render_passage_retrieval_en_prompt(
        context=str(record["context"]),
        input_text=str(record["input"]),
    )
    return BenchmarkCase(
        case_id=str(record["_id"]),
        prompt=prompt,
        expected_answer=str(answers[0]),
        metadata={
            "dataset_id": dataset_id,
            "dataset_config": PASSAGE_RETRIEVAL_EN_CONFIG,
            "dataset": str(record["dataset"]),
            "split": split,
            "source_index": str(source_index),
            "longbench_id": str(record["_id"]),
            "length": str(record["length"]),
            "language": str(record.get("language", "")),
            "all_classes": json.dumps(record.get("all_classes")),
            "answers": json.dumps([str(answer) for answer in answers]),
            "input": str(record["input"]),
            "context": str(record["context"]),
            "file_path": file_path,
            "revision": revision,
            "prompt_template_id": PASSAGE_RETRIEVAL_EN_PROMPT_TEMPLATE_ID,
            "max_new_tokens": str(PASSAGE_RETRIEVAL_EN_MAX_NEW_TOKENS),
            "scoring_contract": PASSAGE_RETRIEVAL_EN_SCORING_CONTRACT,
            "license_note": PASSAGE_RETRIEVAL_EN_LICENSE_NOTE,
        },
    )


def load_longbench_passage_retrieval_en_slice(
    *,
    case_limit: int = 50,
    case_offset_start: int = 0,
    dataset_id: str = LONGBENCH_DATASET_ID,
    revision: str = LONGBENCH_REVISION,
) -> tuple[BenchmarkCase, ...]:
    if case_limit <= 0:
        raise ValueError("case_limit must be positive")
    if case_offset_start < 0:
        raise ValueError("case_offset_start must be non-negative")

    from huggingface_hub import hf_hub_download

    data_zip = Path(
        hf_hub_download(
            dataset_id,
            "data.zip",
            repo_type="dataset",
            revision=revision,
        )
    )
    rows = _load_jsonl_from_zip(data_zip, PASSAGE_RETRIEVAL_EN_FILE)
    end = case_offset_start + case_limit
    if end > len(rows):
        raise ValueError(
            "requested LongBench slice exceeds available rows: "
            f"{case_offset_start}:{end} > {len(rows)}"
        )
    split = (
        f"test[:{case_limit}]"
        if case_offset_start == 0
        else f"test[{case_offset_start}:{end}]"
    )
    return tuple(
        benchmark_case_from_longbench_passage_retrieval_record(
            row,
            source_index=index,
            dataset_id=dataset_id,
            revision=revision,
            split=split,
        )
        for index, row in enumerate(rows[case_offset_start:end], start=case_offset_start)
    )


def _load_jsonl_from_zip(path: Path, member: str) -> tuple[dict[str, Any], ...]:
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as handle:
            return tuple(json.loads(line) for line in handle if line.strip())
