"""Modal-native server-side fanout: .spawn() + .get() collection.

Layer 2 PR Modal-orchestration shape per Layne directive 2026-05-23:
- ONE local lifecycle (single `modal run --detach`)
- N server-side function invocations via binding.remote_fn.spawn(spec)
- Per-cell exception isolation via per-handle .get() with capture
- Modal scheduler handles GPU acquisition from the pool
- max_containers on the function binding caps server-side concurrency

This replaces user-side parallelism patterns (ThreadPoolExecutor wrapping
per-cell `modal run`) which create N independent local-client lifecycles =
N independent disconnect-class failure points. The new pattern collapses
that to one.

Reference doc: config/process/modal-async-patterns.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat.orchestration import (
    CellFailure,
    CellWaveReport,
    collect_cells_parallel,
)


class _FakeHandle:
    def __init__(self, result=None, raises: Exception | None = None):
        self._result = result
        self._raises = raises
        self.get_called = False

    def get(self):
        self.get_called = True
        if self._raises is not None:
            raise self._raises
        return self._result


class _FakeRemoteFn:
    def __init__(self, handles_by_request: dict[str, _FakeHandle]):
        self.handles_by_request = handles_by_request
        self.spawn_log: list[str] = []

    def spawn(self, request):
        key = request.cell_id
        self.spawn_log.append(key)
        return self.handles_by_request[key]


class _FakeBinding:
    def __init__(self, remote_fn):
        self.remote_fn = remote_fn


class _CellRequest:
    def __init__(self, cell_id: str):
        self.cell_id = cell_id


def test_collect_cells_parallel_spawns_one_per_request_then_collects_in_order():
    handles = {
        "cell-a": _FakeHandle(result={"cell_id": "cell-a", "metric": 0.30}),
        "cell-b": _FakeHandle(result={"cell_id": "cell-b", "metric": 0.56}),
        "cell-c": _FakeHandle(result={"cell_id": "cell-c", "metric": 0.96}),
    }
    remote_fn = _FakeRemoteFn(handles)
    binding = _FakeBinding(remote_fn)
    requests = [_CellRequest("cell-a"), _CellRequest("cell-b"), _CellRequest("cell-c")]

    report = collect_cells_parallel(binding, requests)

    assert isinstance(report, CellWaveReport)
    assert remote_fn.spawn_log == ["cell-a", "cell-b", "cell-c"]
    assert all(handle.get_called for handle in handles.values())
    assert [r["cell_id"] for r in report.reports] == ["cell-a", "cell-b", "cell-c"]
    assert report.failures == ()


def test_collect_cells_parallel_isolates_per_cell_failures():
    """One cell raising must NOT prevent other cells from being collected.

    This is the per-cell exception-isolation property that makes .spawn() +
    per-handle .get() preferable to .map() for our shape: partial-wave
    failures surface as CellFailure entries instead of killing the wave.
    """
    boom = RuntimeError("cell-b OOM'd on the GPU")
    handles = {
        "cell-a": _FakeHandle(result={"cell_id": "cell-a", "metric": 0.30}),
        "cell-b": _FakeHandle(raises=boom),
        "cell-c": _FakeHandle(result={"cell_id": "cell-c", "metric": 0.96}),
    }
    remote_fn = _FakeRemoteFn(handles)
    binding = _FakeBinding(remote_fn)
    requests = [_CellRequest("cell-a"), _CellRequest("cell-b"), _CellRequest("cell-c")]

    report = collect_cells_parallel(binding, requests)

    assert [r["cell_id"] for r in report.reports] == ["cell-a", "cell-c"]
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert isinstance(failure, CellFailure)
    assert failure.request is requests[1]
    assert "OOM" in failure.error


def test_collect_cells_parallel_handles_empty_request_list():
    binding = _FakeBinding(_FakeRemoteFn({}))

    report = collect_cells_parallel(binding, [])

    assert report.reports == ()
    assert report.failures == ()
