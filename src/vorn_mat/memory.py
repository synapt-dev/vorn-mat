"""Default-on per-case CUDA memory telemetry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseMemoryStats:
    peak_memory_allocated_mb: float | None = None
    peak_memory_reserved_mb: float | None = None


def reset_case_memory_stats() -> None:
    """Reset CUDA peak counters before a case, if CUDA is available."""
    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats()


def capture_case_memory_stats() -> CaseMemoryStats:
    """Capture peak CUDA memory for the just-completed case."""
    try:
        import torch
    except ImportError:
        return CaseMemoryStats()

    if not torch.cuda.is_available():
        return CaseMemoryStats()

    try:
        torch.cuda.synchronize()
    except RuntimeError:
        pass

    return CaseMemoryStats(
        peak_memory_allocated_mb=_bytes_to_mb(torch.cuda.max_memory_allocated()),
        peak_memory_reserved_mb=_bytes_to_mb(torch.cuda.max_memory_reserved()),
    )


def _bytes_to_mb(value: int | float) -> float:
    return round(float(value) / (1024 * 1024), 3)
