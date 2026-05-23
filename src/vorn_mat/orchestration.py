"""Modal-native server-side fanout for parallel cell execution.

Per Layne directive 2026-05-23: the canonical pattern for firing multiple
cells in parallel is Modal-native via binding.remote_fn.spawn(spec) per cell
+ per-handle .get() collection. ONE local lifecycle replaces N-thread user-
side parallelism (ThreadPoolExecutor wrapping per-cell modal-run), which
created N independent disconnect-class failure points.

Per-cell exceptions are isolated via per-handle .get() with capture: a
partial-wave failure surfaces as a CellFailure entry instead of killing the
whole wave. Successful reports come back in spawn order.

Reference: config/process/modal-async-patterns.md.
"""

from __future__ import annotations

from dataclasses import dataclass
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
