"""Result sink for Week 1 baseline runs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CaseObservation:
    fixture_id: str
    correct: bool
    prediction: str
    scored_prediction: str | None = None
    peak_memory_allocated_mb: float | None = None
    peak_memory_reserved_mb: float | None = None


@dataclass(frozen=True)
class RunResult:
    run_id: str
    benchmark: str
    baseline: str
    metrics: dict[str, float]
    metadata: dict[str, str]
    preprocessing_elapsed_seconds: float = 0.0
    preprocessing_cost_usd: float = 0.0
    observations: tuple[CaseObservation, ...] = ()
    # Reproducibility-substrate telemetry (added 2026-05-23). Optional fields
    # so existing serialized envelopes load unchanged; cells run on the pinned
    # substrate populate them from local_exec.capture_runtime_telemetry().
    peak_memory_allocated_gb: float | None = None
    peak_memory_reserved_gb: float | None = None
    oom_near_miss: bool = False
    env_versions: dict[str, str] = field(default_factory=dict)


def _case_observation_payload(observation: CaseObservation) -> dict[str, object]:
    payload = asdict(observation)
    return {key: value for key, value in payload.items() if value is not None}


def _run_result_payload(result: RunResult) -> dict[str, object]:
    payload = asdict(result)
    payload["observations"] = [
        _case_observation_payload(observation) for observation in result.observations
    ]
    return payload


def append_result(path: Path, result: RunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(_run_result_payload(result), sort_keys=True))
        handle.write("\n")


def load_results(path: Path) -> list[RunResult]:
    if not path.exists():
        return []
    results: list[RunResult] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        observations = tuple(
            CaseObservation(**item) for item in payload.pop("observations", [])
        )
        results.append(RunResult(observations=observations, **payload))
    return results


# Per-case incremental persistence (Layer 2, Layne directive 2026-05-23:
# "store each and every result before moving to next one"). The summary
# envelope still lands at output_path; the per-case ledger lands at
# observations_path(output_path) and is appended one line per completed case
# BEFORE the next case runs. Mid-run kills preserve every completed case.

def append_observation(path: Path, observation: CaseObservation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(_case_observation_payload(observation), sort_keys=True))
        handle.write("\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


def load_observations(path: Path) -> tuple[CaseObservation, ...]:
    if not path.exists():
        return ()
    items: list[CaseObservation] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        items.append(CaseObservation(**json.loads(line)))
    return tuple(items)


def observations_path(output_path: Path) -> Path:
    return output_path.with_suffix(".observations.jsonl")
