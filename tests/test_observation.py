from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat.observation import (
    ObservationCase,
    ObservationReport,
    ObservationStep,
    analyze_observation_report,
    answer_token_indices,
    find_subsequence_spans,
    residual_l2_norms,
    select_top_alignment_positions,
    write_observation_artifacts,
)


def test_find_subsequence_spans_finds_all_matches():
    spans = find_subsequence_spans(
        [1, 2, 3, 2, 3, 4],
        [2, 3],
    )

    assert spans == ((1, 3), (3, 5))


def test_answer_token_indices_expands_spans():
    indices = answer_token_indices(((2, 4), (6, 7)))

    assert indices == (2, 3, 6)


def test_residual_l2_norms_returns_one_norm_per_position():
    norms = residual_l2_norms(
        np.array([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32)
    )

    assert np.allclose(norms, np.array([5.0, 13.0], dtype=np.float32))


def test_select_top_alignment_positions_orders_and_flags_answer_tokens():
    top = select_top_alignment_positions(
        alignment_scores=np.array([0.2, 0.9, 0.5, 0.7], dtype=np.float32),
        residual_norms=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        answer_token_spans=((1, 2),),
        top_k=3,
    )

    assert [item.position for item in top] == [1, 3, 2]
    assert [item.is_answer_token for item in top] == [True, False, False]
    assert top[0].alignment_score == pytest.approx(0.9)


def test_analyze_observation_report_splits_success_and_failure_patterns():
    report = ObservationReport(
        dataset_config="niah_multikey_1_4k",
        split="validation[:2]",
        case_count=2,
        elapsed_seconds=1.0,
        estimated_cost_usd=0.1,
        cases=(
            ObservationCase(
                case_id="success",
                expected_answer="needle",
                prediction="needle",
                success=True,
                prompt_token_count=4,
                answer_token_spans=((1, 2),),
                steps=(
                    ObservationStep(
                        step_index=0,
                        generated_token_id=11,
                        generated_token_text="n",
                        vorn_vector=[0.1, 0.2],
                        alignment_scores=[0.1, 0.8, 0.2, 0.3],
                        residual_norms=[1.0, 2.0, 1.2, 1.1],
                        attention_by_layer={"30": [0.1, 0.6, 0.2, 0.1]},
                        top_alignment_positions=(1, 3, 2),
                        top_alignment_scores=(0.8, 0.3, 0.2),
                        ranking_stability_with_prev=None,
                    ),
                    ObservationStep(
                        step_index=1,
                        generated_token_id=12,
                        generated_token_text="eedle",
                        vorn_vector=[0.2, 0.3],
                        alignment_scores=[0.1, 0.9, 0.2, 0.4],
                        residual_norms=[1.0, 2.2, 1.2, 1.1],
                        attention_by_layer={"30": [0.1, 0.7, 0.1, 0.1]},
                        top_alignment_positions=(1, 3, 2),
                        top_alignment_scores=(0.9, 0.4, 0.2),
                        ranking_stability_with_prev=1.0,
                    ),
                ),
            ),
            ObservationCase(
                case_id="failure",
                expected_answer="needle",
                prediction="wrong",
                success=False,
                prompt_token_count=4,
                answer_token_spans=((1, 2),),
                steps=(
                    ObservationStep(
                        step_index=0,
                        generated_token_id=21,
                        generated_token_text="x",
                        vorn_vector=[0.1, 0.2],
                        alignment_scores=[0.5, 0.1, 0.4, 0.3],
                        residual_norms=[1.0, 1.1, 1.2, 1.1],
                        attention_by_layer={"30": [0.5, 0.1, 0.3, 0.1]},
                        top_alignment_positions=(0, 2, 3),
                        top_alignment_scores=(0.5, 0.4, 0.3),
                        ranking_stability_with_prev=None,
                    ),
                    ObservationStep(
                        step_index=1,
                        generated_token_id=22,
                        generated_token_text="y",
                        vorn_vector=[0.2, 0.2],
                        alignment_scores=[0.4, 0.2, 0.3, 0.1],
                        residual_norms=[1.0, 1.0, 1.1, 1.0],
                        attention_by_layer={"30": [0.4, 0.2, 0.3, 0.1]},
                        top_alignment_positions=(0, 2, 1),
                        top_alignment_scores=(0.4, 0.3, 0.2),
                        ranking_stability_with_prev=0.5,
                    ),
                ),
            ),
        ),
    )

    summary = analyze_observation_report(report, top_k=3)

    assert summary["success_cases"] == 1
    assert summary["failure_cases"] == 1
    assert summary["success"]["cases_with_answer_in_top_k"] == 1
    assert summary["failure"]["cases_with_answer_in_top_k"] == 1
    assert summary["success"]["mean_first_answer_top_k_step"] == 0.0
    assert summary["failure"]["mean_first_answer_top_k_step"] == 1.0


def test_write_observation_artifacts_writes_manifest_and_gzip_case_shards(tmp_path):
    report = ObservationReport(
        dataset_config="niah_multikey_1_4k",
        split="validation[:1]",
        case_count=1,
        elapsed_seconds=1.0,
        estimated_cost_usd=0.1,
        cases=(
            ObservationCase(
                case_id="success",
                expected_answer="needle",
                prediction="needle",
                success=True,
                prompt_token_count=4,
                answer_token_spans=((1, 2),),
                steps=(
                    ObservationStep(
                        step_index=0,
                        generated_token_id=11,
                        generated_token_text="n",
                        vorn_vector=[0.1, 0.2],
                        alignment_scores=[0.1, 0.8, 0.2, 0.3],
                        residual_norms=[1.0, 2.0, 1.2, 1.1],
                        attention_by_layer={"30": [0.1, 0.6, 0.2, 0.1]},
                        top_alignment_positions=(1, 3, 2),
                        top_alignment_scores=(0.8, 0.3, 0.2),
                        ranking_stability_with_prev=None,
                    ),
                ),
            ),
        ),
    )

    json_path = tmp_path / "observation.json"
    markdown_path = tmp_path / "observation.md"
    figures_dir = tmp_path / "figures"

    summary = write_observation_artifacts(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
        figures_dir=figures_dir,
        top_k=3,
        cases_per_shard=1,
    )

    manifest = json.loads(json_path.read_text())
    assert manifest["format"] == "observation-report-sharded-v1"
    assert manifest["summary"]["success_cases"] == 1
    assert summary["success_cases"] == 1
    assert len(manifest["case_shards"]) == 1

    shard_path = json_path.parent / manifest["case_shards"][0]["path"]
    with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
        shard = json.load(handle)

    assert shard["format"] == "observation-case-shard-v1"
    assert shard["case_count"] == 1
    assert shard["cases"][0]["case_id"] == "success"
    assert markdown_path.exists()
    assert (figures_dir / "answer-topk-hit-rate-by-step.png").exists()
