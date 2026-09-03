"""The gated hoist-to-run lifecycle (docs/design/robot_lifecycle.md).

One state machine drives both axes of the robot: the vendor (sport service,
motion switcher) and our 500 Hz tracker. Every transition is a gate that
reads evidence from the writer's stats, the vendor's FSM and the robot state;
an operator key is a *request* and the machine decides.

The machine runs on the operator thread. Every collaborator call is either a
blocking RPC or a lock-free mode store into the writer; nothing here is ever
called from a control thread. `damp()` is the only verb that skips the gates.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from embodied_control.robot.base import RobotMode
from embodied_control.robot.gates import (
    GateResult,
    counter_advanced,
    max_abs_diff,
    pose_match,
    tilt_degrees,
    tilt_match,
    unchanged,
)

# The native writer's init-ramp speed limit (unitree_backend.cpp).
RAMP_SPEED_CAP_RAD_S = 0.5


class LifecycleState(StrEnum):
    IDLE = "IDLE"
    PRECHECK = "PRECHECK"
    VENDOR_DAMP_CONFIRMED = "VENDOR_DAMP_CONFIRMED"
    SAFE_EXTERNAL_COMMAND_PRESENT = "SAFE_EXTERNAL_COMMAND_PRESENT"
    USER_CONTROL_CONFIRMED = "USER_CONTROL_CONFIRMED"
    START_POSE_RAMP = "START_POSE_RAMP"
    POSE_SETTLED = "POSE_SETTLED"
    LOWERED = "LOWERED"
    POSE_MATCH_VERIFIED = "POSE_MATCH_VERIFIED"
    POLICY_COMMAND_FRESH = "POLICY_COMMAND_FRESH"
    PRIMED = "PRIMED"
    BLEND_IN = "BLEND_IN"
    RUNNING = "RUNNING"
    HOLD = "HOLD"
    DAMP = "DAMP"
    RELEASED = "RELEASED"
    VENDOR_RESTORED = "VENDOR_RESTORED"
    VENDOR_STAND = "VENDOR_STAND"
    STANDING = "STANDING"
    FAULT = "FAULT"


# The forward ladder `advance()` climbs one rung at a time. PRIMED is a resting
# state: the policy is verified and waiting for `go()`.
LADDER: tuple[LifecycleState, ...] = (
    LifecycleState.IDLE,
    LifecycleState.PRECHECK,
    LifecycleState.VENDOR_DAMP_CONFIRMED,
    LifecycleState.SAFE_EXTERNAL_COMMAND_PRESENT,
    LifecycleState.USER_CONTROL_CONFIRMED,
    LifecycleState.START_POSE_RAMP,
    LifecycleState.POSE_SETTLED,
    LifecycleState.LOWERED,
    LifecycleState.POSE_MATCH_VERIFIED,
    LifecycleState.POLICY_COMMAND_FRESH,
    LifecycleState.PRIMED,
)

# Where a finished episode rests; `advance()` from here starts the next one.
RESTART_STATES = frozenset(
    {
        LifecycleState.STANDING,
        LifecycleState.VENDOR_RESTORED,
        LifecycleState.RELEASED,
    }
)

# Where `recover()` has something left to hand back. Every other state
# either never took the joints or has already given them up.
RECOVERABLE_STATES = frozenset(
    {
        LifecycleState.HOLD,
        LifecycleState.FAULT,
        LifecycleState.DAMP,
        LifecycleState.RELEASED,
        LifecycleState.VENDOR_RESTORED,
        LifecycleState.VENDOR_STAND,
    }
)

# States in which our writer holds the joints and a fault must damp.
OWNING_STATES = frozenset(
    {
        LifecycleState.SAFE_EXTERNAL_COMMAND_PRESENT,
        LifecycleState.USER_CONTROL_CONFIRMED,
        LifecycleState.START_POSE_RAMP,
        LifecycleState.POSE_SETTLED,
        LifecycleState.LOWERED,
        LifecycleState.POSE_MATCH_VERIFIED,
        LifecycleState.POLICY_COMMAND_FRESH,
        LifecycleState.PRIMED,
        LifecycleState.BLEND_IN,
        LifecycleState.RUNNING,
        LifecycleState.HOLD,
    }
)

# Writer modes, mirrored from NativeUnitreeLoop so this module needs no numpy.
WRITER_DISABLED = 0
WRITER_INITIALIZE = 1
WRITER_WAIT = 2
WRITER_CONTROL = 3
WRITER_DAMP = 4
WRITER_HOLD = 5
WRITER_MODE_NAMES = {
    WRITER_DISABLED: "disabled",
    WRITER_INITIALIZE: "initialize",
    WRITER_WAIT: "wait",
    WRITER_CONTROL: "control",
    WRITER_DAMP: "damp",
    WRITER_HOLD: "hold",
}


class LifecycleError(RuntimeError):
    """A request that is illegal in the current state."""


class Tracker(Protocol):
    """What the lifecycle needs from the process that owns rt/lowcmd."""

    running: bool
    unitree_mode: int

    def wait_for_state(self, timeout_seconds: float) -> bool: ...

    def writer_stats(self) -> dict: ...

    def stats(self) -> dict: ...

    def vendor_mode(self) -> str: ...

    def open_damp_gate(self) -> None: ...

    def release_vendor(self) -> None: ...

    def begin_initialization(
        self,
        duration_seconds: float,
        *,
        target_position=None,
        hold_current: bool = False,
        skip_motion_switcher: bool = False,
    ) -> None: ...

    def wait_for_mode(self, expected: int, timeout_seconds: float) -> bool: ...

    def start(self, ticks: int, paced: bool = True) -> None: ...

    def engage_control(self, blend_ticks: int = 0) -> None: ...

    def hold(self) -> None: ...

    def force_damp(self) -> None: ...

    def stop(self) -> None: ...

    def wait(self) -> None: ...

    def close_gate(self) -> None: ...

    def restore_vendor(self, name: str) -> None: ...

    def joint_position(self) -> list[float]: ...

    def projected_gravity(self) -> list[float]: ...


class Vendor(Protocol):
    def mode(self) -> RobotMode: ...

    def damp(self) -> None: ...

    def stand(self, height: float | None = None) -> None: ...


class Hoist(Protocol):
    """The gantry, where one can be commanded: the simulated plant's.

    hoist: rigid hold. lower: the feet carry the weight, the strap only
    catches a drop. slack: strap paid out, the policy is on its own.
    """

    def hoist(self) -> None: ...

    def lower(self) -> None: ...

    def slack(self) -> None: ...


@dataclass
class LifecycleConfig:
    start_pose: list[float] | None = None
    reference_gravity: list[float] | None = None
    ramp_seconds: float = 3.0
    # Ramp guard: a joint this far behind the ramp for this long is blocked.
    ramp_fault_rad: float = 0.5
    ramp_fault_ms: float = 100.0
    # Hold gains relative to the policy's, for the ramp, the settle and HOLD.
    # Measured on the plant: at 1x the waist sagged 0.42 rad off the sim
    # frame and the policy fell on release; the blend-in ramps back to 1x.
    hold_gain_scale: float = 3.0
    # Pay out the strap when RUNNING begins (sim: the plant's gantry).
    slack_on_run: bool = True
    # Hold the reference at frame 0 through the probe and the blend-in, so
    # the motion starts the tick the policy fully owns the joints.
    pin_reference: bool = True
    ticks: int = 500
    blend_ticks: int = 250
    # Where a run rests. An episode ends with the robot limp under the
    # vendor's own damp, so the next one has to climb PRECHECK again; a
    # vendor stand is the explicit `s` request, never the default.
    end_state: str = "vendor_damp"
    # `damp` hands the joints back to the vendor after our kd-only frames
    # land, so DAMP is never a resting state with our writer still owning
    # rt/lowcmd. The kd frames are unconditional; only the hand-back waits
    # for the hoist acknowledgement.
    damp_hands_back: bool = True
    # A retake re-reads the link before it drives the robot again. It is a
    # subset of PRECHECK: the vendor is already released and the ladder's
    # vendor rows would fail by design.
    retake_precheck: bool = True
    vendor_name: str = ""
    require_vendor: bool = True
    allow_non_realtime: bool = False
    state_timeout_seconds: float = 5.0
    damp_publish_frames: int = 100
    settle_seconds: float = 0.5
    settle_timeout_seconds: float = 10.0
    # "Quiet" is position drift over the window, not joint speed: the wire
    # carries SONIC-scale velocity noise (0.5 rad/s half-range on the plant),
    # so a limp robot never reads as still by velocity, while encoder
    # position noise is 0.01 rad.
    drift_rad: float = 0.03
    # Optional tracking-error bound while settling. Off by default: under the
    # policy's own PD gains a held pose sags under gravity (0.4 rad on a
    # 14 N m/rad shoulder), and the pose-match gate is where that is judged
    # per joint.
    settle_position_rad: float | None = None
    hoist_release_seconds: float = 1.5
    pose_tolerance_rad: list[float] | float = 0.05
    tilt_tolerance_degrees: float = 10.0
    # Per-sample wire noise (0.01 rad joints, 0.05 rad IMU tilt on the plant)
    # is averaged out of the pose-match reading.
    pose_samples: int = 25
    # The policy's PD targets legitimately sit far from the pose they hold on
    # weak-gain joints (1.4 rad on the waist under gravity with SONIC gains),
    # so the radian bound is coarse and the real check is the torque the
    # first action would command, per joint, against the actuator's limit.
    # 3x, not 1x: the wrists' 5 N m limit saturates on any real command
    # (1.4x on the plant with the SONIC bundle) and the policy was trained
    # with that clipping. A wrong joint order or frame shows up far higher.
    first_action_rad: float = 2.5
    first_action_torque_ratio: float = 3.0
    stiffness: list[float] | None = None
    effort_limit: list[float] | None = None
    command_timeout_seconds: float = 10.0
    vendor_timeout_seconds: float = 5.0
    poll_seconds: float = 0.02

    def __post_init__(self) -> None:
        if self.end_state not in {"vendor_stand", "vendor_damp", "damp"}:
            raise ValueError(
                "end_state must be vendor_stand, vendor_damp or damp"
            )
        if self.ticks <= 0 or self.ramp_seconds <= 0.0:
            raise ValueError("ticks and ramp_seconds must be positive")
        if self.start_pose is not None and len(self.start_pose) != 29:
            raise ValueError("start_pose must have 29 joint values")


@dataclass
class Transition:
    at: float
    from_state: str
    to_state: str
    ok: bool
    detail: str
    values: dict = field(default_factory=dict)


class LifecycleLog:
    """Append-only evidence trail: one JSON line per attempted transition."""

    def __init__(self, directory: str | Path | None) -> None:
        self.directory = Path(directory) if directory else None
        self.transitions: list[Transition] = []
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / "lifecycle.jsonl").write_text("")

    def record(self, transition: Transition) -> None:
        self.transitions.append(transition)
        if self.directory is not None:
            with (self.directory / "lifecycle.jsonl").open("a") as stream:
                stream.write(json.dumps(transition.__dict__) + "\n")

    def write_json(self, name: str, payload: dict) -> None:
        if self.directory is not None:
            (self.directory / name).write_text(json.dumps(payload, indent=2) + "\n")

    def finish(self, summary: dict) -> None:
        self.write_json("lifecycle.json", summary)


class Lifecycle:
    def __init__(
        self,
        tracker: Tracker,
        vendor: Vendor | None,
        config: LifecycleConfig,
        *,
        hoist: Hoist | None = None,
        auto_ack: bool = False,
        log: LifecycleLog | None = None,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        note: Callable[[str], None] = lambda _msg: None,
    ) -> None:
        if vendor is None and config.require_vendor:
            raise ValueError("the lifecycle needs a vendor runtime")
        self.tracker = tracker
        self.vendor = vendor
        self.config = config
        self.hoist = hoist
        self.auto_ack = bool(auto_ack)
        self.log = log if log is not None else LifecycleLog(None)
        self._now = now
        self._sleep = sleep
        self._note = note
        self._lock = threading.RLock()
        self.state = LifecycleState.IDLE
        self.fault_reason = ""
        self.vendor_name = config.vendor_name
        self.hoisted_ack = False
        self.lowered_ack = False
        self.episode = 0
        self.last_result: GateResult | None = None
        # Called after every recorded transition; the session uses it to
        # write an episode's telemetry the moment RUNNING ends.
        self.on_transition: Callable[[Transition], None] | None = None

    # ------------------------------------------------------------ requests

    def advance(self) -> GateResult:
        """Climb one rung of the ladder."""
        with self._lock:
            if self.state in RESTART_STATES:
                self.state = LifecycleState.IDLE
            if self.state not in LADDER[:-1]:
                return self._refuse(f"cannot advance from {self.state}")
            target = LADDER[LADDER.index(self.state) + 1]
            return self._attempt(target)

    def auto(self, until: LifecycleState = LifecycleState.PRIMED) -> GateResult:
        """Advance until `until`, the first failing gate, or an operator ack."""
        with self._lock:
            result = GateResult(True, f"already at {self.state}")
            if self.state in RESTART_STATES:
                self.state = LifecycleState.IDLE
            while self.state != until:
                if self.state not in LADDER[:-1]:
                    return self._refuse(f"cannot auto-advance from {self.state}")
                result = self.advance()
                if not result.ok:
                    return result
            return result

    def go(self) -> GateResult:
        """PRIMED -> BLEND_IN -> RUNNING."""
        with self._lock:
            if self.state is not LifecycleState.PRIMED:
                return self._refuse(f"go needs PRIMED, not {self.state}")
            result = self._attempt(LifecycleState.BLEND_IN)
            if not result.ok:
                return result
            return self._attempt(LifecycleState.RUNNING)

    def hold(self) -> GateResult:
        with self._lock:
            if self.state not in {LifecycleState.RUNNING, LifecycleState.BLEND_IN}:
                return self._refuse(f"hold needs RUNNING, not {self.state}")
            return self._attempt(LifecycleState.HOLD)

    def damp(self, *, hand_back: bool | None = None) -> GateResult:
        """Always legal. Never goes through stop() first.

        The kd-only frames land first and unconditionally. Then, unless the
        caller says otherwise, the joints go back to the vendor's own damp
        (`SelectMode`, FSM 1) so the robot rests limp under the vendor and
        the next trajectory has to climb PRECHECK again. That hand-back
        needs the hoist: `SelectMode` restarts the vendor service in damp,
        and the robot is limp for about a second while it comes back.
        """
        with self._lock:
            result = self._attempt(LifecycleState.DAMP)
            if not result.ok:
                return result
            wants = self.config.damp_hands_back if hand_back is None else hand_back
            # `end_state: damp` is a deliberate "our frames keep the joints".
            wants = wants and self.config.end_state != "damp"
            if not wants or self.vendor is None or not self.vendor_name:
                return result
            if not self._hoist_ready():
                return GateResult(
                    True,
                    "damp frames on the wire; hook the hoist and press H, "
                    "then `d` to hand the joints back to the vendor",
                    result.values,
                )
            handed = self._hand_back_to_vendor()
            if not handed.ok:
                return handed
            return GateResult(
                True,
                f"damped, then handed back: {handed.detail}",
                {**result.values, **handed.values},
            )

    def _hand_back_to_vendor(self) -> GateResult:
        """DAMP -> RELEASED -> VENDOR_RESTORED. The caller holds the lock."""
        if self.state is LifecycleState.DAMP:
            result = self._attempt(LifecycleState.RELEASED)
            if not result.ok:
                return result
        if self.state is LifecycleState.RELEASED:
            return self._attempt(LifecycleState.VENDOR_RESTORED)
        return GateResult(True, f"already at {self.state}")

    def emergency_damp(self) -> None:
        """Damp without waiting for the lock.

        A gate can hold the lock for its whole timeout; the writer's mode
        store is lock-free, so the frame lands now and `damp()` records the
        transition once the gate returns.
        """
        self.tracker.force_damp()

    def snapshot(self) -> dict:
        """Lock-free view for a display thread; never blocks on a gate."""
        try:
            ws = dict(self.tracker.writer_stats())
        except Exception:
            ws = {}
        try:
            st = dict(self.tracker.stats())
        except Exception:
            st = {}
        hoist = None
        status = getattr(self.hoist, "status", None)
        if status is not None:
            try:
                hoist = status()
            except Exception:
                hoist = None
        last = self.last_result
        return {
            "state": str(self.state),
            "fault_reason": self.fault_reason,
            "vendor_name": self.vendor_name,
            "episode": self.episode,
            "hoisted_ack": self.hoisted_ack,
            "lowered_ack": self.lowered_ack,
            "writer": ws,
            "control": st,
            "hoist": hoist,
            "last_ok": None if last is None else last.ok,
            "last_detail": "" if last is None else last.detail,
        }

    def abort(self) -> GateResult:
        """Operator abort before the policy ever drove: damp, then recover."""
        with self._lock:
            if self.state not in LADDER:
                return self._refuse(f"abort is for the ladder, not {self.state}")
            if self.state in OWNING_STATES:
                result = self.damp()
                if not result.ok:
                    return result
            return self.recover()

    def recover(self, end_state: str | None = None) -> GateResult:
        """HOLD / DAMP / FAULT -> RELEASED -> vendor, per end_state.

        Stops at the hoist acknowledgement when one is needed, and at the
        lowering acknowledgement before STANDING.
        """
        with self._lock:
            end = end_state or self.config.end_state
            if self.state not in RECOVERABLE_STATES:
                return self._refuse(f"nothing to recover from {self.state}")
            if self.state in {LifecycleState.HOLD, LifecycleState.FAULT}:
                if self.state is LifecycleState.HOLD and not self._hoist_ready():
                    return self._refuse(
                        "hook the hoist and press H before leaving HOLD"
                    )
                result = self._attempt(LifecycleState.DAMP)
                if not result.ok:
                    return result
            if self.state is LifecycleState.DAMP:
                result = self._attempt(LifecycleState.RELEASED)
                if not result.ok:
                    return result
            if end == "damp":
                return GateResult(True, f"recovered to {self.state}")
            if self.state is LifecycleState.RELEASED:
                if not self._hoist_ready():
                    return self._refuse(
                        "the vendor restarts in damp: hook the hoist and "
                        "press H before restoring"
                    )
                result = self._attempt(LifecycleState.VENDOR_RESTORED)
                if not result.ok:
                    return result
            if end == "vendor_damp":
                return GateResult(True, f"recovered to {self.state}")
            if self.state is LifecycleState.VENDOR_RESTORED:
                result = self._attempt(LifecycleState.VENDOR_STAND)
                if not result.ok:
                    return result
            if self.state is LifecycleState.VENDOR_STAND:
                return self._attempt(LifecycleState.STANDING)
            return self._refuse(f"nothing to recover from {self.state}")

    def retake(self) -> GateResult:
        """HOLD -> START_POSE_RAMP with the vendor still released.

        A second episode drives the robot again, so it re-reads the link
        first: the PRECHECK rows that do not involve the vendor, which is
        already released and would fail those rows by design.
        """
        with self._lock:
            if self.state is not LifecycleState.HOLD:
                return self._refuse(f"retake needs HOLD, not {self.state}")
            if not self._hoist_ready():
                return self._refuse("hook the hoist and press H before retaking")
            if self.config.retake_precheck:
                link = self._link_evidence({})
                self._record(
                    Transition(
                        at=self._now(),
                        from_state=str(self.state),
                        to_state="RETAKE_PRECHECK",
                        ok=link.ok,
                        detail=link.detail,
                        values=dict(link.values),
                    )
                )
                if not link.ok:
                    return self._refuse(f"retake precheck failed: {link.detail}")
            self.lowered_ack = False
            self.episode += 1
            self.state = LifecycleState.USER_CONTROL_CONFIRMED
            return self._attempt(LifecycleState.START_POSE_RAMP)

    def ack_hoisted(self) -> None:
        with self._lock:
            if self.hoist is not None:
                self.hoist.hoist()
            self.hoisted_ack = True
            self.lowered_ack = False

    def ack_lowered(self) -> None:
        with self._lock:
            if self.hoist is not None:
                self.hoist.lower()
            self.lowered_ack = True

    def poll(self) -> None:
        """Watch a running episode: budget end -> HOLD, writer damp -> FAULT."""
        with self._lock:
            if self.state in {LifecycleState.RUNNING, LifecycleState.BLEND_IN}:
                writer_mode = int(self.tracker.unitree_mode)
                if writer_mode == WRITER_DAMP:
                    self._fault("writer damped during " + str(self.state))
                elif not self.tracker.running:
                    self._attempt(LifecycleState.HOLD)
            elif self.state in OWNING_STATES and self.state is not LifecycleState.HOLD:
                if int(self.tracker.unitree_mode) == WRITER_DAMP:
                    self._fault("writer damped during " + str(self.state))

    def shutdown(self) -> None:
        """Console exit: damp, silence the wire, give the vendor back."""
        with self._lock:
            # A fault already forced DAMP and stopped the tracker, but leaves
            # the write gate open until the operator chooses recovery. Exit
            # must still silence the wire before the process disappears.
            if self.state in OWNING_STATES or self.state is LifecycleState.FAULT:
                self._attempt(LifecycleState.DAMP)
            if self.state is LifecycleState.DAMP:
                self._attempt(LifecycleState.RELEASED)
            if self.state is LifecycleState.RELEASED and self.vendor_name:
                self._attempt(LifecycleState.VENDOR_RESTORED)
            self.log.finish(self.summary())

    # ------------------------------------------------------------- status

    def summary(self) -> dict:
        return {
            "state": str(self.state),
            "fault_reason": self.fault_reason,
            "vendor_name": self.vendor_name,
            "episode": self.episode,
            "transitions": len(self.log.transitions),
            "failed_transitions": sum(
                1 for t in self.log.transitions if not t.ok
            ),
        }

    def status(self) -> str:
        with self._lock:
            try:
                ws = self.tracker.writer_stats()
                st = self.tracker.stats()
            except Exception:  # status must render mid-teardown
                ws, st = {}, {}
            writer = WRITER_MODE_NAMES.get(int(ws.get("mode", -1)), "?")
            vendor = "released" if ws.get("vendor_released") else "owns"
            parts = [
                f"[{self.state}]",
                f"vendor: {vendor}",
                f"writer: {writer}",
                f"pub: {ws.get('publishes', 0)}",
                f"ticks: {st.get('control_ticks', 0)}",
                f"fault: {st.get('fault', 0)}/{ws.get('watchdog_faults', 0)}",
            ]
            if self.fault_reason:
                parts.append(f"reason: {self.fault_reason}")
            return "  ".join(parts)

    # ------------------------------------------------------------ internals

    def _refuse(self, detail: str) -> GateResult:
        result = GateResult(False, detail)
        self.last_result = result
        self._note(detail)
        return result

    def _hoist_ready(self) -> bool:
        if self.auto_ack:
            self.ack_hoisted()
        return self.hoisted_ack

    def _record(self, transition: Transition) -> None:
        self.log.record(transition)
        if self.on_transition is not None:
            try:
                self.on_transition(transition)
            except Exception as exc:  # a listener must never break a gate
                self._note(f"transition listener failed: {exc}")

    def _attempt(self, target: LifecycleState) -> GateResult:
        previous = self.state
        handler = getattr(self, f"_enter_{target.lower()}")
        try:
            result = handler()
        except LifecycleError as exc:
            result = GateResult(False, str(exc))
        except RuntimeError as exc:
            result = GateResult(False, f"{type(exc).__name__}: {exc}")
        if result.ok:
            self.state = target
        self.last_result = result
        self._record(
            Transition(
                at=self._now(),
                from_state=str(previous),
                to_state=str(target),
                ok=result.ok,
                detail=result.detail,
                values=dict(result.values),
            )
        )
        self._note(f"{previous} -> {target}: {'ok' if result.ok else 'FAILED'} ({result.detail})")
        return result

    def _fault(self, reason: str) -> None:
        self.fault_reason = reason
        self.tracker.force_damp()
        try:
            self.tracker.stop()
        except RuntimeError:
            pass
        previous = self.state
        self.state = LifecycleState.FAULT
        self._record(
            Transition(
                at=self._now(),
                from_state=str(previous),
                to_state=str(LifecycleState.FAULT),
                ok=True,
                detail=reason,
                values=self._writer_snapshot(),
            )
        )
        self._note(f"FAULT: {reason}")

    def _writer_snapshot(self) -> dict:
        try:
            ws = self.tracker.writer_stats()
        except Exception:
            return {}
        keys = (
            "mode", "publishes", "publish_failures", "hardware_faults",
            "watchdog_faults", "ramp_faults", "ramp_error_max", "ramp_error_joint",
            "state_frames", "state_gap_ns_max", "crc_errors", "state_fault_reason",
            "joint_speed_max", "tracking_error_max", "command_target_error_max",
        )
        return {k: ws[k] for k in keys if k in ws}

    def _wait_until(
        self, predicate: Callable[[], bool], timeout: float, what: str
    ) -> bool:
        deadline = self._now() + timeout
        while True:
            if predicate():
                return True
            if self._now() >= deadline:
                self._note(f"timed out waiting for {what}")
                return False
            self._sleep(self.config.poll_seconds)

    def _quiet_window(
        self,
        *,
        position_tol: float | None,
        window: float,
        timeout: float,
        damp_is_fault: bool = True,
    ) -> GateResult:
        """Hold for `window` seconds with the pose drifting less than drift_rad
        and, when position_tol is given, the writer's tracking error under it."""
        deadline = self._now() + timeout
        window_start: float | None = None
        anchor: list[float] | None = None
        worst_drift = 0.0
        worst_error = 0.0
        while True:
            ws = self.tracker.writer_stats()
            error = float(ws.get("tracking_error_max", 0.0))
            worst_error = max(worst_error, error)
            if damp_is_fault and int(ws.get("mode", -1)) == WRITER_DAMP:
                return GateResult(False, "writer damped while settling", self._writer_snapshot())
            current = [float(v) for v in self.tracker.joint_position()]
            now = self._now()
            if anchor is None:
                anchor, window_start = current, now
            drift = max_abs_diff(current, anchor)
            worst_drift = max(worst_drift, drift)
            quiet = drift <= self.config.drift_rad and (
                position_tol is None or error <= position_tol
            )
            if quiet:
                if now - window_start >= window:
                    return GateResult(
                        True,
                        f"quiet for {window:.2f}s (drift {drift:.3f} rad"
                        + (f", error {error:.3f} rad)" if position_tol is not None else ")"),
                        {"drift_rad": drift, "tracking_error_max": error,
                         "joint_speed_max": float(ws.get("joint_speed_max", 0.0))},
                    )
            else:
                anchor, window_start = current, now
            if now >= deadline:
                return GateResult(
                    False,
                    f"not quiet within {timeout:.1f}s (drift up to {worst_drift:.3f} rad"
                    + (f", error up to {worst_error:.3f} rad)" if position_tol is not None else ")"),
                    {"drift_rad": worst_drift, "tracking_error_max": worst_error},
                )
            self._sleep(self.config.poll_seconds)

    def _vendor_mode_is(self, mode: RobotMode, timeout: float) -> bool:
        assert self.vendor is not None
        return self._wait_until(
            lambda: self.vendor.mode() is mode, timeout, f"vendor {mode}"
        )

    # --------------------------------------------------------- the ladder

    def _link_evidence(self, values: dict) -> GateResult:
        """The link rows shared by PRECHECK and a retake: a fresh state
        stream, no CRC errors and no latched hardware fault."""
        if not self.tracker.wait_for_state(self.config.state_timeout_seconds):
            return GateResult(False, "no fresh rt/lowstate", values)
        # A previous episode's latch (a fall trips joint limits) must not
        # block the next; the next frame re-latches if the robot is still out.
        clear = getattr(self.tracker, "clear_latched_fault", None)
        if clear is not None:
            clear()
            self._sleep(self.config.poll_seconds * 5)
        set_scale = getattr(self.tracker, "set_hold_gain_scale", None)
        if set_scale is not None:
            set_scale(self.config.hold_gain_scale)
        ws = self.tracker.writer_stats()
        values.update(self._writer_snapshot())
        if int(ws.get("crc_errors", 0)) != 0:
            return GateResult(False, "rt/lowstate CRC errors", values)
        if int(ws.get("hardware_faults", 0)) != 0:
            return GateResult(
                False,
                f"hardware fault latched (reason {ws.get('state_fault_reason')})",
                values,
            )
        return GateResult(True, "link fresh, no CRC errors, no latched fault", values)

    def _enter_precheck(self) -> GateResult:
        values: dict = {}
        self.fault_reason = ""
        self.lowered_ack = False
        self.episode += 1
        link = self._link_evidence(values)
        if not link.ok:
            return link
        ws = self.tracker.writer_stats()
        if not ws.get("realtime_configured", False) and not self.config.allow_non_realtime:
            return GateResult(False, "real-time setup failed and non-RT was not acknowledged", values)
        if self.vendor is not None:
            name = self.tracker.vendor_mode()
            values["vendor_service"] = name
            if self.config.require_vendor and not name:
                return GateResult(False, "no active vendor motion service (CheckMode empty)", values)
            if self.config.vendor_name and name and name != self.config.vendor_name:
                return GateResult(False, f"vendor service is '{name}', expected '{self.config.vendor_name}'", values)
            self.vendor_name = name or self.config.vendor_name
            mode = self.vendor.mode()
            values["vendor_mode"] = str(mode)
            if mode is RobotMode.UNKNOWN:
                return GateResult(False, "sport service unreachable", values)
        if not self._hoist_ready():
            return GateResult(False, "confirm the robot is hoisted (press H)", values)
        values["hoisted_ack"] = True
        return GateResult(True, "link, vendor and hoist confirmed", values)

    def _enter_vendor_damp_confirmed(self) -> GateResult:
        if self.vendor is None:
            return GateResult(True, "no vendor: skipped")
        self.vendor.damp()
        if not self._vendor_mode_is(RobotMode.DAMP, self.config.vendor_timeout_seconds):
            return GateResult(False, f"vendor did not report damp (mode {self.vendor.mode()})")
        result = self._quiet_window(
            position_tol=None,
            window=self.config.settle_seconds,
            timeout=self.config.settle_timeout_seconds,
            damp_is_fault=False,
        )
        if not result.ok:
            return result
        return GateResult(True, "vendor damp read back; robot limp", result.values)

    def _enter_safe_external_command_present(self) -> GateResult:
        before = self.tracker.writer_stats()
        self.tracker.open_damp_gate()
        target = int(before.get("publishes", 0)) + self.config.damp_publish_frames
        ok = self._wait_until(
            lambda: int(self.tracker.writer_stats().get("publishes", 0)) >= target,
            timeout=max(2.0, self.config.damp_publish_frames * 0.004),
            what="damp frames on the wire",
        )
        after = self.tracker.writer_stats()
        result = counter_advanced(
            int(before.get("publishes", 0)), int(after.get("publishes", 0)),
            self.config.damp_publish_frames, "publishes",
        )
        if not ok or not result.ok:
            return result
        failures = unchanged(
            int(before.get("publish_failures", 0)),
            int(after.get("publish_failures", 0)), "publish_failures",
        )
        if not failures.ok:
            return failures
        if int(after.get("mode", -1)) != WRITER_DAMP:
            return GateResult(False, "writer left DAMP while publishing", self._writer_snapshot())
        return GateResult(True, f"{result.values['publishes']} damp frames on rt/lowcmd", {**result.values, **failures.values})

    def _enter_user_control_confirmed(self) -> GateResult:
        before = int(self.tracker.writer_stats().get("publishes", 0))
        self.tracker.release_vendor()
        after = self.tracker.writer_stats()
        if not after.get("vendor_released", False):
            return GateResult(False, "motion switcher still reports an active service")
        if not self._wait_until(
            lambda: int(self.tracker.writer_stats().get("publishes", 0)) > before,
            1.0,
            "damp frames after the release",
        ):
            return GateResult(False, "publisher stopped after the release", self._writer_snapshot())
        result = self._quiet_window(
            position_tol=None,
            window=self.config.settle_seconds,
            timeout=self.config.settle_timeout_seconds,
            damp_is_fault=False,
        )
        if not result.ok:
            return result
        return GateResult(True, "vendor released; our damp frames own the joints", result.values)

    def _enter_start_pose_ramp(self) -> GateResult:
        # A limp robot hangs far from its stance. ramp_seconds is the floor;
        # the ramp stretches so no joint exceeds the writer's speed cap, the
        # same "lengthen the duration" an operator would apply by hand.
        duration = self.config.ramp_seconds
        if self.config.start_pose is not None:
            current = [float(v) for v in self.tracker.joint_position()]
            travel = max_abs_diff(current, list(self.config.start_pose))
            duration = max(duration, travel / (RAMP_SPEED_CAP_RAD_S * 0.8))
        set_guard = getattr(self.tracker, "set_ramp_guard", None)
        if set_guard is not None:
            set_guard(self.config.ramp_fault_rad, max(1, int(self.config.ramp_fault_ms / 2.0)))
        self.tracker.begin_initialization(
            duration,
            target_position=self.config.start_pose,
            hold_current=False,
            skip_motion_switcher=True,
        )
        reached = self.tracker.wait_for_mode(WRITER_WAIT, duration + 5.0)
        ws = self.tracker.writer_stats()
        if int(ws.get("ramp_faults", 0)) > 0 or int(ws.get("mode", -1)) == WRITER_DAMP:
            self._fault(
                f"ramp guard tripped: joint {ws.get('ramp_error_joint')} lagged "
                f"the ramp by {float(ws.get('ramp_error_max', 0.0)):.3f} rad"
            )
            raise LifecycleError("ramp guard tripped")
        if not reached:
            return GateResult(False, "ramp did not reach WAIT", self._writer_snapshot())
        return GateResult(True, f"ramped to the start pose over {duration:.2f}s", {**self._writer_snapshot(), "ramp_seconds": duration})

    def _enter_pose_settled(self) -> GateResult:
        return self._quiet_window(
            position_tol=self.config.settle_position_rad,
            window=self.config.settle_seconds,
            timeout=self.config.settle_timeout_seconds,
        )

    def _enter_lowered(self) -> GateResult:
        if self.auto_ack and not self.lowered_ack:
            self.ack_lowered()
        if not self.lowered_ack:
            return GateResult(False, "lower the hoist until the feet carry the weight, then press l")
        self._sleep(self.config.hoist_release_seconds)
        result = self._quiet_window(
            position_tol=None,
            window=self.config.settle_seconds,
            timeout=self.config.settle_timeout_seconds,
        )
        if not result.ok:
            return result
        return GateResult(True, "feet loaded; robot settled again", result.values)

    def _sampled_state(self) -> tuple[list[float], list[float]]:
        """Mean joint position and projected gravity over pose_samples reads."""
        count = max(1, int(self.config.pose_samples))
        joints: list[float] | None = None
        gravity: list[float] | None = None
        for index in range(count):
            q = [float(v) for v in self.tracker.joint_position()]
            g = [float(v) for v in self.tracker.projected_gravity()]
            joints = q if joints is None else [a + b for a, b in zip(joints, q)]
            gravity = g if gravity is None else [a + b for a, b in zip(gravity, g)]
            if index + 1 < count:
                self._sleep(self.config.poll_seconds)
        assert joints is not None and gravity is not None
        return [v / count for v in joints], [v / count for v in gravity]

    def _enter_pose_match_verified(self) -> GateResult:
        if self.config.start_pose is None:
            return GateResult(True, "no start pose given: skipped")
        measured, gravity = self._sampled_state()
        tolerance = self.config.pose_tolerance_rad
        if isinstance(tolerance, (int, float)):
            tolerance = [float(tolerance)] * len(measured)
        result = pose_match(measured, list(self.config.start_pose), list(tolerance))
        report = {
            "measured": measured,
            "target": list(self.config.start_pose),
            "tolerance": list(tolerance),
            "pose": result.values,
            "ok": result.ok,
            "detail": result.detail,
        }
        if self.config.reference_gravity is not None:
            tilt = tilt_match(
                gravity, list(self.config.reference_gravity),
                self.config.tilt_tolerance_degrees,
            )
            report["tilt"] = tilt.values
            report["measured_gravity"] = gravity
            if result.ok and not tilt.ok:
                result = tilt
            elif result.ok:
                result = GateResult(True, f"{result.detail}; {tilt.detail}", {**result.values, **tilt.values})
            report["ok"] = result.ok
            report["detail"] = result.detail
        self.log.write_json("pose_match.json", report)
        return result

    def _pin_reference(self, paused: bool) -> None:
        if not self.config.pin_reference:
            return
        pin = getattr(self.tracker, "set_reference_paused", None)
        if pin is not None:
            pin(paused)

    def _enter_policy_command_fresh(self) -> GateResult:
        self._pin_reference(True)
        self.tracker.start(self.config.ticks, paced=True)
        ok = self._wait_until(
            lambda: int(self.tracker.stats().get("control_ticks", 0)) > 0
            or int(self.tracker.stats().get("fault", 0)) != 0,
            self.config.command_timeout_seconds,
            "the first control tick",
        )
        st = self.tracker.stats()
        ws = self.tracker.writer_stats()
        values = {
            "control_ticks": int(st.get("control_ticks", 0)),
            "planner_requests": int(st.get("planner_requests", 0)),
            "planner_responses": int(st.get("planner_responses", 0)),
            "runtime_fault": int(st.get("fault", 0)),
            "command_target_error_max": float(ws.get("command_target_error_max", 0.0)),
        }
        if not ok or int(st.get("fault", 0)) != 0 or int(st.get("control_ticks", 0)) == 0:
            self.tracker.force_damp()
            self.tracker.stop()
            self.tracker.wait()
            self._fault("no fresh policy command: " + json.dumps(values))
            raise LifecycleError("policy command never became fresh")
        if int(ws.get("mode", -1)) != WRITER_WAIT:
            self.tracker.stop()
            self.tracker.wait()
            return GateResult(False, "writer left WAIT while the policy warmed", values)
        # The writer measures the control thread's unapplied target against
        # the held pose: the policy's first action, before any torque.
        self._sleep(self.config.poll_seconds * 5)
        values["command_target_error_max"] = float(
            self.tracker.writer_stats().get("command_target_error_max", 0.0)
        )
        if values["command_target_error_max"] > self.config.first_action_rad:
            self.tracker.stop()
            self.tracker.wait()
            return GateResult(
                False,
                f"first policy action is {values['command_target_error_max']:.3f} rad "
                f"from the held pose (limit {self.config.first_action_rad:.3f})",
                values,
            )
        torque = self._first_action_torque()
        if torque is not None:
            values.update(torque.values)
            if not torque.ok:
                self.tracker.stop()
                self.tracker.wait()
                return GateResult(False, torque.detail, values)
        # The probe is over: stop the control thread so the policy does not
        # keep integrating actions the writer never applies. go() starts it
        # again and engages on its first tick, the way a sim episode begins.
        self.tracker.stop()
        self.tracker.wait()
        return GateResult(True, "planner answering; first action consistent", values)

    def _first_action_torque(self) -> GateResult | None:
        """kp * |target - held pose| per joint against the effort limit."""
        errors_fn = getattr(self.tracker, "command_target_error", None)
        if errors_fn is None or self.config.stiffness is None or self.config.effort_limit is None:
            return None
        errors = [float(v) for v in errors_fn()]
        ratios = [
            k * e / limit if limit > 0.0 else 0.0
            for k, e, limit in zip(self.config.stiffness, errors, self.config.effort_limit)
        ]
        worst = max(range(len(ratios)), key=lambda i: ratios[i])
        values = {
            "first_action_torque_ratio_max": ratios[worst],
            "first_action_torque_joint": worst,
            "first_action_error_rad": errors[worst],
        }
        if ratios[worst] > self.config.first_action_torque_ratio:
            return GateResult(
                False,
                f"first policy action would demand {ratios[worst]:.2f}x the effort "
                f"limit on joint {worst} ({errors[worst]:.3f} rad off the held pose)",
                values,
            )
        return GateResult(True, f"first action torque {ratios[worst]:.2f}x limit on joint {worst}", values)

    def _enter_primed(self) -> GateResult:
        st = self.tracker.stats()
        ws = self.tracker.writer_stats()
        if int(ws.get("mode", -1)) != WRITER_WAIT:
            return GateResult(False, "writer must be in WAIT to engage", self._writer_snapshot())
        if int(st.get("fault", 0)) != 0:
            return GateResult(False, f"runtime fault {st.get('fault')}", self._writer_snapshot())
        return GateResult(True, "verified; waiting for go", {"control_ticks": int(st.get("control_ticks", 0))})

    def _enter_blend_in(self) -> GateResult:
        self._pin_reference(True)
        self.tracker.start(self.config.ticks, paced=True)
        fresh = self._wait_until(
            lambda: int(self.tracker.stats().get("control_ticks", 0)) > 0
            or int(self.tracker.stats().get("fault", 0)) != 0,
            self.config.command_timeout_seconds,
            "the first control tick",
        )
        st = self.tracker.stats()
        if not fresh or int(st.get("fault", 0)) != 0:
            self.tracker.force_damp()
            self.tracker.stop()
            self.tracker.wait()
            self._fault(f"policy did not produce a fresh command at go (fault {st.get('fault')})")
            raise LifecycleError("no fresh command at go")
        self.tracker.engage_control(self.config.blend_ticks)
        ok = self._wait_until(
            lambda: int(self.tracker.writer_stats().get("blend_ticks_remaining", 0)) == 0
            or int(self.tracker.unitree_mode) != WRITER_CONTROL,
            timeout=self.config.blend_ticks * 0.002 + 2.0,
            what="the blend to finish",
        )
        ws = self.tracker.writer_stats()
        if int(ws.get("mode", -1)) != WRITER_CONTROL:
            self._fault("writer left CONTROL during blend-in")
            raise LifecycleError("blend-in interrupted")
        if not ok:
            return GateResult(False, "blend did not complete", self._writer_snapshot())
        self._pin_reference(False)
        return GateResult(True, f"blended in over {self.config.blend_ticks} ticks; reference released", self._writer_snapshot())

    def _enter_running(self) -> GateResult:
        if int(self.tracker.unitree_mode) != WRITER_CONTROL:
            return GateResult(False, "writer is not in CONTROL", self._writer_snapshot())
        # The policy is balancing now: pay out the strap. On hardware this is
        # the operator's hand; in sim it is the plant's gantry.
        if self.hoist is not None and self.auto_ack and self.config.slack_on_run:
            self.hoist.slack()
        return GateResult(True, f"policy driving, budget {self.config.ticks} ticks"
                          + ("; strap slack" if self.config.slack_on_run else "; strap kept"))

    def _enter_hold(self) -> GateResult:
        writer_mode = int(self.tracker.unitree_mode)
        if writer_mode == WRITER_DAMP:
            self._fault("writer damped before hold")
            raise LifecycleError("cannot hold a damped robot")
        self.tracker.hold()
        self.tracker.stop()
        self.tracker.wait()
        self.hoisted_ack = False
        st = self.tracker.stats()
        values = {
            "control_ticks": int(st.get("control_ticks", 0)),
            "runtime_fault": int(st.get("fault", 0)),
        }
        values.update(self._upright_evidence())
        return GateResult(
            True,
            "frozen on the last target; hook the hoist and press H",
            values,
        )

    def _upright_evidence(self) -> dict:
        """Pelvis tilt at HOLD, recorded for the untethered question.

        An untethered end would skip the hoist and hand a robot that is
        already on its feet straight to the vendor's stand. Whether the
        vendor accepts `SelectMode` on a loaded robot is rung H2 of the
        hardware ladder and is not answered here, so this is evidence in
        `lifecycle.jsonl`, not a gate: the tilt says only that the pelvis is
        near vertical, never that both feet carry the weight.
        """
        try:
            gravity = [float(v) for v in self.tracker.projected_gravity()]
        except Exception:
            return {}
        if len(gravity) != 3:
            return {}
        # Upright is the body frame reading world down, the same convention
        # projected_gravity_from_xyzw produces.
        tilt = tilt_degrees(gravity, [0.0, 0.0, -1.0])
        return {
            "pelvis_tilt_degrees": tilt,
            "pelvis_upright": tilt <= self.config.tilt_tolerance_degrees,
        }

    def _enter_damp(self) -> GateResult:
        self.tracker.force_damp()
        if self.tracker.running:
            self.tracker.stop()
            self.tracker.wait()
        self.hoisted_ack = False if self.state in {LifecycleState.RUNNING, LifecycleState.BLEND_IN} else self.hoisted_ack
        return GateResult(True, "damp frame on the wire", self._writer_snapshot())

    def _enter_released(self) -> GateResult:
        before = self.tracker.writer_stats()
        self.tracker.close_gate()
        self._sleep(0.2)
        after = self.tracker.writer_stats()
        if int(after.get("mode", -1)) != WRITER_DISABLED or after.get("gate_open", False):
            return GateResult(False, "writer did not disable", self._writer_snapshot())
        silent = unchanged(int(before.get("publishes", 0)), int(after.get("publishes", 0)), "publishes")
        if not silent.ok:
            return GateResult(False, "publisher still writing after close_gate", silent.values)
        return GateResult(True, "rt/lowcmd silent", silent.values)

    def _enter_vendor_restored(self) -> GateResult:
        if not self.vendor_name:
            return GateResult(False, "no vendor service name to restore")
        self.tracker.restore_vendor(self.vendor_name)
        if self.vendor is not None and not self._vendor_mode_is(RobotMode.DAMP, self.config.vendor_timeout_seconds):
            return GateResult(False, f"vendor restored but not in damp (mode {self.vendor.mode()})")
        return GateResult(True, f"vendor service '{self.vendor_name}' back, in damp")

    def _enter_vendor_stand(self) -> GateResult:
        if self.vendor is None:
            return GateResult(False, "no vendor to stand")
        self.vendor.stand()
        if not self._vendor_mode_is(RobotMode.READY, self.config.vendor_timeout_seconds):
            return GateResult(False, f"vendor did not report standing (mode {self.vendor.mode()})")
        self.lowered_ack = False
        return GateResult(True, "vendor standing, hoisted; lower and press l")

    def _enter_standing(self) -> GateResult:
        if self.auto_ack and not self.lowered_ack:
            self.ack_lowered()
        if not self.lowered_ack:
            return GateResult(False, "lower the robot onto its feet, then press l")
        self._sleep(self.config.hoist_release_seconds)
        if self.vendor is not None and self.vendor.mode() is not RobotMode.READY:
            return GateResult(False, f"vendor left standing (mode {self.vendor.mode()})")
        return GateResult(True, "standing still under the vendor")
