"""Fire-and-forget Modal dispatch + separate result collection.

Substrate-fix for the Modal client-disconnect class-of-failure (2nd occurrence
2026-05-26, first 2026-05-23). Layer 3's ONE-.remote()-call architecture
correctly protects the cloud function under --detach, but the synchronous
.remote() blocks the local entrypoint waiting on the wave_report return.
If local disconnects mid-wait, --detach keeps the function running but the
local artifact-writing step (reports.json/failures.json) is lost.

This module provides the canonical Modal fire-and-forget pattern:
- dispatch_wave_async: orchestrate_fn.spawn(specs) + persist call_id to
  wave-state.json + return immediately. Local exits within seconds.
- collect_wave_results: FunctionCall.from_id(call_id).get() retrieves the
  wave_report later. Writes reports.json + failures.json from the retrieved
  payload. Decoupled from the dispatching local entrypoint.

Tests use fake spawn/FunctionCall surfaces so the dispatch + collection
shape is unit-testable without Modal-actual.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vorn_mat.async_dispatch import (
    WaveStateFile,
    collect_wave_results,
    dispatch_wave_async,
)


class _FakeFunctionCall:
    """Fake `modal.FunctionCall` returned by orchestrate_fn.spawn()."""

    def __init__(self, object_id: str):
        self.object_id = object_id


class _FakeOrchestrateFn:
    """Fake `@app.function`-decorated orchestrate_wave with .spawn() method."""

    def __init__(self, call_id: str = "fc_test_001"):
        self.call_id = call_id
        self.spawn_calls: list[list[dict]] = []

    def spawn(self, cell_specs: list[dict]) -> _FakeFunctionCall:
        self.spawn_calls.append(cell_specs)
        return _FakeFunctionCall(object_id=self.call_id)


class _FakeModalModule:
    """Fake `modal` module with FunctionCall.from_id() lookup."""

    def __init__(self, results_by_call_id: dict[str, dict | Exception]):
        self.results_by_call_id = results_by_call_id
        self.from_id_calls: list[str] = []

    class FunctionCall:
        def __init__(self, call_id: str, result):
            self.call_id = call_id
            self._result = result

        def get(self):
            if isinstance(self._result, Exception):
                raise self._result
            return self._result

    def make_from_id(self):
        def from_id(call_id: str):
            self.from_id_calls.append(call_id)
            if call_id not in self.results_by_call_id:
                raise RuntimeError(f"FunctionCall {call_id} not found in fake")
            return self.FunctionCall(call_id, self.results_by_call_id[call_id])

        return from_id


def test_dispatch_wave_async_spawns_orchestrate_fn_once(tmp_path: Path):
    orchestrate_fn = _FakeOrchestrateFn(call_id="fc_dispatch_001")
    cell_specs = [
        {"cell_id": "cell-a", "cache_budget_tokens": 256},
        {"cell_id": "cell-b", "cache_budget_tokens": 512},
    ]

    state = dispatch_wave_async(orchestrate_fn, cell_specs, output_dir=tmp_path)

    assert len(orchestrate_fn.spawn_calls) == 1
    assert orchestrate_fn.spawn_calls[0] == cell_specs
    assert state["call_id"] == "fc_dispatch_001"
    assert state["spec_count"] == 2


def test_dispatch_wave_async_persists_state_file(tmp_path: Path):
    """wave-state.json must be written with the debugging-load-bearing fields.

    Per Opus 2026-05-26 refinement: includes modal_app name + cell-spec hash
    so a stale call_id can be matched back to its dispatch context for
    debugging ("which spec did this call_id correspond to?") and supports
    resume-verification later.
    """
    orchestrate_fn = _FakeOrchestrateFn(call_id="fc_persist_002")
    cell_specs = [
        {"cell_id": "cell-a"},
        {"cell_id": "cell-b"},
        {"cell_id": "cell-c"},
    ]

    state = dispatch_wave_async(
        orchestrate_fn,
        cell_specs,
        output_dir=tmp_path,
        modal_app_name="vorn-mat-live-eviction-niah",
    )

    state_path = tmp_path / "wave-state.json"
    assert state_path.exists()

    persisted = json.loads(state_path.read_text())
    assert persisted["call_id"] == "fc_persist_002"
    assert persisted["spec_count"] == 3
    assert "dispatched_at" in persisted
    assert persisted["dispatched_at"].endswith("Z") or "+" in persisted["dispatched_at"]
    assert persisted["output_dir"] == str(tmp_path)
    assert persisted["modal_app_name"] == "vorn-mat-live-eviction-niah"
    # Hash should be deterministic + non-empty so repeat-dispatch with same
    # specs produces the same hash (resume verification)
    assert isinstance(persisted["cell_specs_hash"], str)
    assert len(persisted["cell_specs_hash"]) >= 16

    assert state["state_path"] == str(state_path)


def test_dispatch_wave_async_hash_is_deterministic_and_order_sensitive(tmp_path: Path):
    """Same specs in same order → same hash. Reordered specs → different hash.

    Order-sensitivity is intentional: spawn order determines which spec maps
    to which cell-position in reports; reordering changes the dispatch
    contract and should be visible in the hash.
    """
    specs_a = [{"cell_id": "x"}, {"cell_id": "y"}]
    specs_b_same = [{"cell_id": "x"}, {"cell_id": "y"}]
    specs_c_reordered = [{"cell_id": "y"}, {"cell_id": "x"}]

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_c = tmp_path / "c"

    state_a = dispatch_wave_async(
        _FakeOrchestrateFn("fc_a"), specs_a, output_dir=dir_a, modal_app_name="app1"
    )
    state_b = dispatch_wave_async(
        _FakeOrchestrateFn("fc_b"), specs_b_same, output_dir=dir_b, modal_app_name="app1"
    )
    state_c = dispatch_wave_async(
        _FakeOrchestrateFn("fc_c"), specs_c_reordered, output_dir=dir_c, modal_app_name="app1"
    )

    hash_a = json.loads((dir_a / "wave-state.json").read_text())["cell_specs_hash"]
    hash_b = json.loads((dir_b / "wave-state.json").read_text())["cell_specs_hash"]
    hash_c = json.loads((dir_c / "wave-state.json").read_text())["cell_specs_hash"]

    assert hash_a == hash_b
    assert hash_a != hash_c


def test_dispatch_wave_async_creates_output_dir_if_missing(tmp_path: Path):
    """output_dir is mkdir-d on dispatch so collect step has a place to land."""
    output_dir = tmp_path / "deeply" / "nested" / "wave-output"
    orchestrate_fn = _FakeOrchestrateFn(call_id="fc_mkdir_003")

    dispatch_wave_async(orchestrate_fn, [{"cell_id": "x"}], output_dir=output_dir)

    assert output_dir.exists()
    assert (output_dir / "wave-state.json").exists()


def test_collect_wave_results_reads_state_and_retrieves(tmp_path: Path):
    """collect_wave_results must read wave-state.json, retrieve via from_id, write reports."""
    state_path = tmp_path / "wave-state.json"
    state_path.write_text(
        json.dumps(
            {
                "call_id": "fc_collect_004",
                "spec_count": 2,
                "dispatched_at": "2026-05-26T12:00:00Z",
                "output_dir": str(tmp_path),
                "modal_app_name": "vorn-mat-test",
                "cell_specs_hash": "abc123" * 6,
            }
        )
    )
    wave_report = {
        "reports": [
            {"run_id": "cell-a", "metric": 0.30},
            {"run_id": "cell-b", "metric": 0.96},
        ],
        "failures": [],
    }
    fake_modal = _FakeModalModule({"fc_collect_004": wave_report})

    summary = collect_wave_results(
        wave_state_path=state_path,
        output_dir=tmp_path,
        modal_from_id=fake_modal.make_from_id(),
    )

    assert fake_modal.from_id_calls == ["fc_collect_004"]
    assert summary["cells_succeeded"] == 2
    assert summary["cells_failed"] == 0

    reports = json.loads((tmp_path / "reports.json").read_text())
    assert reports == wave_report["reports"]
    assert not (tmp_path / "failures.json").exists()


def test_collect_wave_results_writes_failures_when_present(tmp_path: Path):
    state_path = tmp_path / "wave-state.json"
    state_path.write_text(
        json.dumps(
            {
                "call_id": "fc_failures_005",
                "spec_count": 3,
                "dispatched_at": "2026-05-26T12:00:00Z",
                "output_dir": str(tmp_path),
                "modal_app_name": "vorn-mat-test",
                "cell_specs_hash": "def456" * 6,
            }
        )
    )
    wave_report = {
        "reports": [{"run_id": "cell-a", "metric": 0.30}],
        "failures": [
            {
                "request": {"cell_id": "cell-b", "cache_budget_tokens": 512},
                "error": "OOM on Mistral B=1536 eager-attention",
            },
            {
                "request": {"cell_id": "cell-c", "cache_budget_tokens": 1024},
                "error": "Modal task timeout",
            },
        ],
    }
    fake_modal = _FakeModalModule({"fc_failures_005": wave_report})

    summary = collect_wave_results(
        wave_state_path=state_path,
        output_dir=tmp_path,
        modal_from_id=fake_modal.make_from_id(),
    )

    assert summary["cells_succeeded"] == 1
    assert summary["cells_failed"] == 2

    reports = json.loads((tmp_path / "reports.json").read_text())
    assert len(reports) == 1
    failures = json.loads((tmp_path / "failures.json").read_text())
    assert len(failures) == 2
    assert "OOM" in failures[0]["error"]


def test_collect_wave_results_propagates_get_exceptions(tmp_path: Path):
    """If the FunctionCall hasn't completed (or was canceled), surface the error
    with context, not a bare modal exception."""
    state_path = tmp_path / "wave-state.json"
    state_path.write_text(
        json.dumps(
            {
                "call_id": "fc_missing_006",
                "spec_count": 1,
                "dispatched_at": "2026-05-26T12:00:00Z",
                "output_dir": str(tmp_path),
                "modal_app_name": "vorn-mat-test",
                "cell_specs_hash": "0123abcd" * 4,
            }
        )
    )
    fake_modal = _FakeModalModule(
        {"fc_missing_006": RuntimeError("FunctionCall not yet complete")}
    )

    with pytest.raises(RuntimeError, match="FunctionCall not yet complete"):
        collect_wave_results(
            wave_state_path=state_path,
            output_dir=tmp_path,
            modal_from_id=fake_modal.make_from_id(),
        )


def test_wave_state_file_round_trips_via_helper(tmp_path: Path):
    """WaveStateFile is the typed wrapper for the JSON serialization contract.

    Round-trip via helper guarantees that dispatch + collect use the same
    schema; if one side drifts, the other side's load() fails loudly.
    """
    state_path = tmp_path / "wave-state.json"
    original = WaveStateFile(
        call_id="fc_roundtrip_007",
        spec_count=5,
        dispatched_at="2026-05-26T12:00:00Z",
        output_dir=str(tmp_path),
        modal_app_name="vorn-mat-live-eviction-niah",
        cell_specs_hash="abc123def456abc123def456",
    )

    original.save(state_path)
    loaded = WaveStateFile.load(state_path)

    assert loaded == original


def test_collect_wave_results_is_idempotent(tmp_path: Path):
    """Re-running collect against the same call_id must produce identical
    artifacts, not error or duplicate. Idempotency is required so an agent
    can re-run collect (e.g., after a transient retrieval failure) without
    losing or corrupting prior output. Per Opus 2026-05-26 refinement #2.
    """
    state_path = tmp_path / "wave-state.json"
    state_path.write_text(
        json.dumps(
            {
                "call_id": "fc_idempotent_009",
                "spec_count": 2,
                "dispatched_at": "2026-05-26T12:00:00Z",
                "output_dir": str(tmp_path),
                "modal_app_name": "vorn-mat-live-eviction-niah",
                "cell_specs_hash": "deadbeefdeadbeefdeadbeefdeadbeef",
            }
        )
    )
    wave_report = {
        "reports": [{"run_id": "cell-a", "metric": 0.30}],
        "failures": [
            {
                "request": {"cell_id": "cell-b"},
                "error": "OOM",
            }
        ],
    }
    fake_modal = _FakeModalModule({"fc_idempotent_009": wave_report})
    from_id = fake_modal.make_from_id()

    summary_1 = collect_wave_results(
        wave_state_path=state_path,
        output_dir=tmp_path,
        modal_from_id=from_id,
    )
    reports_after_first = (tmp_path / "reports.json").read_text()
    failures_after_first = (tmp_path / "failures.json").read_text()

    summary_2 = collect_wave_results(
        wave_state_path=state_path,
        output_dir=tmp_path,
        modal_from_id=from_id,
    )
    reports_after_second = (tmp_path / "reports.json").read_text()
    failures_after_second = (tmp_path / "failures.json").read_text()

    # Substantive fields match across invocations (call_id, counts, output_dir)
    for field in ("call_id", "cells_succeeded", "cells_failed", "output_dir"):
        assert summary_1[field] == summary_2[field]
    # from_cache flag indicates path-taken: False on first (fetched), True on
    # second (read from on-disk artifacts) — this IS the idempotency signal
    assert summary_1["from_cache"] is False
    assert summary_2["from_cache"] is True
    # Artifacts must be byte-identical between invocations
    assert reports_after_first == reports_after_second
    assert failures_after_first == failures_after_second
    # Second collect should NOT re-fetch from Modal if artifacts already exist
    # with matching call_id — exactly one from_id call across both invocations
    assert fake_modal.from_id_calls == ["fc_idempotent_009"]


def test_wave_state_file_rejects_missing_fields(tmp_path: Path):
    """If wave-state.json is corrupted or partial, load() must fail loudly
    rather than silently default fields."""
    state_path = tmp_path / "wave-state.json"
    state_path.write_text(json.dumps({"call_id": "fc_partial_008"}))

    with pytest.raises((KeyError, TypeError)):
        WaveStateFile.load(state_path)
