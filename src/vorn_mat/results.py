"""Result sink for Week 1 baseline runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CaseObservation:
    fixture_id: str
    correct: bool
    prediction: str


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


def append_result(path: Path, result: RunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(asdict(result), sort_keys=True))
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
