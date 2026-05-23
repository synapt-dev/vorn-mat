"""Per-case incremental persistence: results.py sink layer.

Per Layne directive 2026-05-23: every completed case must be on disk BEFORE
the next case runs. These tests cover the lowest-altitude primitives
(append_observation + load_observations + observations_path) that the
baseline runners and wrappers compose on top of.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat.results import (
    CaseObservation,
    append_observation,
    load_observations,
    observations_path,
)


def test_append_observation_writes_single_jsonl_line(tmp_path: Path):
    path = tmp_path / "cell.observations.jsonl"
    observation = CaseObservation(
        fixture_id="case-1",
        correct=True,
        prediction="Paris",
    )

    append_observation(path, observation)

    raw = path.read_text().splitlines()
    assert len(raw) == 1
    payload = json.loads(raw[0])
    assert payload == {
        "fixture_id": "case-1",
        "correct": True,
        "prediction": "Paris",
    }


def test_append_observation_appends_subsequent_lines_in_order(tmp_path: Path):
    path = tmp_path / "cell.observations.jsonl"
    observations = (
        CaseObservation(fixture_id="case-1", correct=True, prediction="a"),
        CaseObservation(fixture_id="case-2", correct=False, prediction="b"),
        CaseObservation(fixture_id="case-3", correct=True, prediction="c"),
    )

    for observation in observations:
        append_observation(path, observation)

    lines = path.read_text().splitlines()
    assert [json.loads(line)["fixture_id"] for line in lines] == [
        "case-1",
        "case-2",
        "case-3",
    ]


def test_append_observation_creates_parent_directories(tmp_path: Path):
    path = tmp_path / "nested" / "subdir" / "cell.observations.jsonl"
    observation = CaseObservation(fixture_id="case-1", correct=True, prediction="x")

    append_observation(path, observation)

    assert path.exists()
    assert json.loads(path.read_text().strip())["fixture_id"] == "case-1"


def test_load_observations_round_trips_append_order(tmp_path: Path):
    path = tmp_path / "cell.observations.jsonl"
    expected = (
        CaseObservation(fixture_id="case-1", correct=True, prediction="alpha"),
        CaseObservation(fixture_id="case-2", correct=False, prediction="beta"),
    )

    for observation in expected:
        append_observation(path, observation)

    assert load_observations(path) == expected


def test_load_observations_returns_empty_tuple_for_missing_path(tmp_path: Path):
    assert load_observations(tmp_path / "does-not-exist.jsonl") == ()


def test_load_observations_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "cell.observations.jsonl"
    path.write_text(
        '{"fixture_id": "case-1", "correct": true, "prediction": "a"}\n'
        '\n'
        '{"fixture_id": "case-2", "correct": false, "prediction": "b"}\n'
    )

    loaded = load_observations(path)

    assert [obs.fixture_id for obs in loaded] == ["case-1", "case-2"]


def test_observations_path_derives_from_summary_path():
    summary = Path("/vol/results/vorn-mat/modal-mistral-niah-vorn-live-b512.jsonl")

    ledger = observations_path(summary)

    assert ledger == Path(
        "/vol/results/vorn-mat/modal-mistral-niah-vorn-live-b512.observations.jsonl"
    )


def test_observations_path_handles_summary_without_extension():
    summary = Path("/vol/results/vorn-mat/cell-1")

    ledger = observations_path(summary)

    assert ledger == Path("/vol/results/vorn-mat/cell-1.observations.jsonl")
