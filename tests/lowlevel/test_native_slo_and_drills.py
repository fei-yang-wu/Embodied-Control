"""Seqlock contention stress, forced-fault drills, and SLO grading."""

import os
import threading
import uuid

import numpy as np
import pytest

ec_native = pytest.importorskip("ec_native")

from embodied_control.lowlevel.slo import SLO_TARGETS, evaluate_report  # noqa: E402


def _name(prefix: str) -> str:
    return f"/{prefix}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def test_seqlock_survives_write_read_contention():
    """A hot writer must never let the reader observe a torn payload."""
    slot = ec_native.ShmCommandSlot(_name("stress"), True)
    width = 64
    iterations = 20000
    stop = threading.Event()
    write_error: list[BaseException] = []

    def writer():
        sequence = 0
        values = np.empty(width, dtype=np.float32)
        try:
            while not stop.is_set():
                sequence += 1
                values.fill(float(sequence % 65536))
                values[0] = float(sequence % 65536)
                slot.publish(sequence, 1, values, 0.0)
        except BaseException as exc:  # pragma: no cover - fail loud below
            write_error.append(exc)

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    try:
        seen = 0
        last_sequence = 0
        torn = 0
        while seen < iterations:
            raw = slot.snapshot()
            if raw is None:
                continue
            sequence, _tag, values, _recv, _sender = raw
            expected = float(sequence % 65536)
            if not np.all(np.asarray(values) == expected):
                torn += 1
            if sequence < last_sequence:
                torn += 1
            last_sequence = sequence
            seen += 1
        assert torn == 0
        assert last_sequence > 0
    finally:
        stop.set()
        thread.join(timeout=10)
    assert not write_error


def _fake_loop(latent_manifest, tmp_path, response_slot, **overrides):
    pytest.importorskip("torch")
    from embodied_control.lowlevel.native_core import NativeFakeLoop
    from test_native_core import _native_bundle

    bundle = _native_bundle(tmp_path, latent_manifest)
    defaults = dict(
        response_slot=response_slot,
        command_absent_ticks=8,
        command_stale_ms=10_000.0,
        command_source="vla",
    )
    defaults.update(overrides)
    return NativeFakeLoop(bundle, **defaults)


def test_drill_command_absence_transitions_to_damp(latent_manifest, tmp_path):
    """No command ever arrives: the loop must fault into DAMP, not wait forever."""
    response = _name("drill_absent")
    loop = _fake_loop(latent_manifest, tmp_path, response)
    loop.start(80, paced=False)
    loop.wait()
    stats = loop.stats()
    assert stats["fault"] != 0
    assert stats["damp_ticks"] > 0


def test_drill_wrong_sequence_response_faults(latent_manifest, tmp_path):
    """A response that violates the RPC pairing must DAMP the loop."""
    request = _name("drill_seq_req")
    response = _name("drill_seq_resp")
    loop = _fake_loop(
        latent_manifest, tmp_path, response,
        request_slot=request, lead_ticks=4, command_absent_ticks=1000,
    )
    request_reader = ec_native.ShmCommandSlot(request, False)
    responder = ec_native.ShmCommandSlot(response, False)
    loop.start(200, paced=False)
    try:
        replied = False
        for _ in range(200000):
            raw = request_reader.snapshot()
            if raw is None:
                continue
            sequence = raw[0]
            if sequence:
                responder.publish(
                    sequence + 7, 1, np.ones(8, dtype=np.float32), 0.0
                )
                replied = True
                break
        assert replied
        loop.wait()
    finally:
        loop.stop()
    assert loop.stats()["fault"] != 0


def test_slo_grading_passes_and_fails():
    report = {
        "control": {
            "tick_ns_max": 1_400_000,
            "wake_late_ns_max": 215_000,
            "backend_wake_late_ns_max": 223_000,
            "deadline_misses": 0,
            "backend_deadline_misses": 0,
            "scheduler_deadlines_missed": 0,
            "response_overruns": 0,
            "damp_ticks": 0,
            "fault": 0,
            "realtime_configured": True,
            "backend_realtime_configured": True,
        }
    }
    ticks = np.full(460, 500_000, dtype=np.int64)
    verdict = evaluate_report(report, ticks)
    assert verdict["pass"]
    assert verdict["measured"]["control_tick_compute_p99_ms"] == pytest.approx(0.5)
    assert verdict["realtime_configured"]

    report["control"]["deadline_misses"] = 3
    verdict = evaluate_report(report, ticks)
    assert not verdict["pass"]
    assert not verdict["checks"]["deadline_misses"]
    assert set(SLO_TARGETS) >= set(verdict["checks"])