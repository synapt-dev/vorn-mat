"""Layer 3 cloud-side orchestrator: --detach protection survives local disconnect.

Layer 2 shipped Modal-native server-side fanout via binding.remote_fn.spawn()
called from a local entrypoint. Modal's launch warning surfaced a narrowed
protection story under --detach: "running a local entrypoint in detached mode
only keeps the last triggered Modal function alive after the parent process
has been killed or disconnected." Whether the warning literally narrows
multi-spawn protection or is more nuanced, the strictly-safer pattern is to
move the orchestration boundary one altitude up so the local entrypoint
makes ONE .remote() call to a cloud-side orchestrator, which then does the
.spawn() loop on the cloud side under the protection of that single call.

These tests cover:
- serialize_cell_wave_report: CellWaveReport -> JSON-safe dict for cross-
  container transport (the orchestrate_wave function returns via Modal's
  serialization layer, so the return shape must round-trip through json)
- run_wave_serialized: thin wrapper that orchestrate_wave delegates to,
  enabling unit-testable cloud-side orchestration without Modal-actual
- main_entrypoint_calls_orchestrate_wave_via_remote: the local entrypoint
  refactor moves from collect_cells_parallel(binding, ...) to
  orchestrate_wave.remote(specs) so --detach protects the single call
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat.orchestration import (
    CellFailure,
    CellWaveReport,
    run_wave_serialized,
    serialize_cell_wave_report,
)


@dataclass(frozen=True)
class _FakeRequest:
    cell_id: str
    cache_budget_tokens: int = 256


@dataclass(frozen=True)
class _FakeReport:
    run_id: str
    metric: float


class _FakeHandle:
    def __init__(self, result=None, raises: Exception | None = None):
        self._result = result
        self._raises = raises

    def get(self):
        if self._raises is not None:
            raise self._raises
        return self._result


class _FakeRemoteFn:
    def __init__(self, handles_by_cell_id: dict[str, _FakeHandle]):
        self.handles_by_cell_id = handles_by_cell_id
        self.spawn_log: list[str] = []

    def spawn(self, request):
        self.spawn_log.append(request.cell_id)
        return self.handles_by_cell_id[request.cell_id]


class _FakeBinding:
    def __init__(self, remote_fn):
        self.remote_fn = remote_fn


def test_serialize_cell_wave_report_emits_json_safe_dict():
    report = CellWaveReport(
        reports=(_FakeReport(run_id="cell-a", metric=0.30),),
        failures=(
            CellFailure(
                request=_FakeRequest(cell_id="cell-b", cache_budget_tokens=512),
                error="OOM on attention head",
            ),
        ),
    )

    serialized = serialize_cell_wave_report(report)

    assert serialized == {
        "reports": [{"run_id": "cell-a", "metric": 0.30}],
        "failures": [
            {
                "request": {"cell_id": "cell-b", "cache_budget_tokens": 512},
                "error": "OOM on attention head",
            }
        ],
    }


def test_serialize_cell_wave_report_round_trips_through_json():
    """Modal's serialization layer round-trips the orchestrate_wave return
    value through JSON-compatible types. Verify our serializer's output is
    actually json-loadable."""
    report = CellWaveReport(
        reports=(_FakeReport(run_id="cell-a", metric=0.96),),
        failures=(),
    )

    serialized = serialize_cell_wave_report(report)
    rehydrated = json.loads(json.dumps(serialized))

    assert rehydrated == serialized
    assert rehydrated["reports"][0]["run_id"] == "cell-a"
    assert rehydrated["failures"] == []


def test_serialize_handles_non_dataclass_report_values():
    """If a report is already a dict (e.g., a Modal-returned envelope),
    pass it through unchanged."""
    report = CellWaveReport(
        reports=({"run_id": "cell-a", "metric": 0.30},),
        failures=(),
    )

    serialized = serialize_cell_wave_report(report)

    assert serialized["reports"] == [{"run_id": "cell-a", "metric": 0.30}]


def test_run_wave_serialized_calls_collect_cells_parallel_then_serializes():
    handles = {
        "cell-a": _FakeHandle(result=_FakeReport(run_id="cell-a", metric=0.30)),
        "cell-b": _FakeHandle(result=_FakeReport(run_id="cell-b", metric=0.56)),
    }
    remote_fn = _FakeRemoteFn(handles)
    binding = _FakeBinding(remote_fn)
    cell_specs = [
        {"cell_id": "cell-a", "cache_budget_tokens": 256},
        {"cell_id": "cell-b", "cache_budget_tokens": 512},
    ]

    result = run_wave_serialized(binding, cell_specs, _FakeRequest)

    assert remote_fn.spawn_log == ["cell-a", "cell-b"]
    assert isinstance(result, dict)
    assert [r["run_id"] for r in result["reports"]] == ["cell-a", "cell-b"]
    assert result["failures"] == []


def test_run_wave_serialized_preserves_per_cell_failure_isolation():
    boom = RuntimeError("cell-b kernel OOM")
    handles = {
        "cell-a": _FakeHandle(result=_FakeReport(run_id="cell-a", metric=0.30)),
        "cell-b": _FakeHandle(raises=boom),
        "cell-c": _FakeHandle(result=_FakeReport(run_id="cell-c", metric=0.96)),
    }
    remote_fn = _FakeRemoteFn(handles)
    binding = _FakeBinding(remote_fn)
    cell_specs = [
        {"cell_id": "cell-a"},
        {"cell_id": "cell-b"},
        {"cell_id": "cell-c"},
    ]

    result = run_wave_serialized(binding, cell_specs, _FakeRequest)

    assert [r["run_id"] for r in result["reports"]] == ["cell-a", "cell-c"]
    assert len(result["failures"]) == 1
    assert result["failures"][0]["request"]["cell_id"] == "cell-b"
    assert "OOM" in result["failures"][0]["error"]
