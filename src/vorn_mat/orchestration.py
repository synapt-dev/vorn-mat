"""Modal-native server-side fanout for parallel cell execution.

Per Layne directive 2026-05-23: the canonical pattern for firing multiple
cells in parallel is Modal-native via binding.remote_fn.spawn(spec) per cell
+ per-handle .get() collection. ONE local lifecycle replaces N-thread user-
side parallelism (ThreadPoolExecutor wrapping per-cell modal-run), which
created N independent disconnect-class failure points.

Per-cell exceptions are isolated via per-handle .get() with capture: a
partial-wave failure surfaces as a CellFailure entry instead of killing the
whole wave. Successful reports come back in spawn order.

Layer 3 (2026-05-23): the cloud-side orchestrator pattern moves the .spawn()
loop one altitude up so the local entrypoint makes ONE .remote() call to a
cloud-side orchestrator function. Under `modal run --detach`, the protection
guarantee for that single .remote() call is unambiguous; the spawned cells
become children of that protected call and inherit its protection. Use
run_wave_serialized + serialize_cell_wave_report inside an @app.function
orchestrator instead of calling collect_cells_parallel from local.

Reference: config/process/modal-async-patterns.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CellFailure:
    request: Any
    error: str


@dataclass(frozen=True)
class CellWaveReport:
    reports: tuple[Any, ...]
    failures: tuple[CellFailure, ...]


def collect_cells_parallel(binding: Any, requests: Iterable[Any]) -> CellWaveReport:
    requests_list = list(requests)
    handles = [binding.remote_fn.spawn(request) for request in requests_list]

    reports: list[Any] = []
    failures: list[CellFailure] = []
    for handle, request in zip(handles, requests_list, strict=True):
        try:
            reports.append(handle.get())
        except Exception as exc:  # noqa: BLE001 - per-cell isolation by design
            failures.append(CellFailure(request=request, error=str(exc)))
    return CellWaveReport(reports=tuple(reports), failures=tuple(failures))


def _to_json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def serialize_cell_wave_report(report: CellWaveReport) -> dict:
    """Convert CellWaveReport to a JSON-safe dict for cross-container transport.

    Modal's serialization layer round-trips function return values; this
    helper ensures the orchestrate_wave function's return shape is composed
    entirely of JSON-compatible types (dicts, lists, primitives) regardless
    of whether the underlying reports are dataclasses or pre-serialized
    dicts. Already-serialized dicts pass through unchanged.
    """
    return {
        "reports": [_to_json_safe(r) for r in report.reports],
        "failures": [
            {"request": _to_json_safe(f.request), "error": f.error}
            for f in report.failures
        ],
    }


def run_wave_serialized(
    binding: Any,
    cell_specs: list[dict],
    request_class: type,
) -> dict:
    """Cloud-side orchestrator body: hydrate cell specs into requests, run
    collect_cells_parallel, serialize the wave report for cross-container
    return transport.

    This is the function body an @app.function-decorated orchestrate_wave
    delegates to. Splitting the logic out makes the orchestration unit-
    testable without Modal-actual: tests pass a fake binding + fake request
    class + plain dict cell specs and assert on the serialized return shape.
    """
    requests = [request_class(**spec) for spec in cell_specs]
    report = collect_cells_parallel(binding, requests)
    return serialize_cell_wave_report(report)
