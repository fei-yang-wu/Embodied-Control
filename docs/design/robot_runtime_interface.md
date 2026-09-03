# Robot runtime interface

Status: v1 adopted (takeover path decided 2026-09-02, see §9)
Audience: embodied-control developers, G1 rig operators
Last updated: 2026-09-02
Primary goal: drive a physical robot's *lifecycle and mode* from Python — start,
damp, hand control over and take it back — through a contract that the Unitree
G1 realizes first and a second robot can realize without touching callers.

## 1. Scope

This is the **high-level command** axis: the things a human currently does with
the vendor's joystick. Bring the robot up, put it in a compliant damp state,
hand the joints to our 500 Hz tracker, take them back, ask what mode it is in.

**Non-goal: per-tick actuation.** Joint targets stay where they are — in
`NativeUnitreeBackend` on the native hot path. That boundary is the single most
important constraint in this design, because the two axes have incompatible
execution characteristics:

| | High-level (this doc) | Low-level (existing) |
|---|---|---|
| Rate | ~1 Hz, operator-driven | 500 Hz, deadline-bound |
| Call style | blocking RPC, seconds of latency | lock-free, `noexcept` |
| Failure | raises, operator retries | damps, never throws |
| Thread | any Python thread | the writer thread only |

A `LocoClient` RPC blocks for its whole timeout. Putting one anywhere near the
control thread starves the 500 Hz writer and trips the watchdog into damp. The
interface below is therefore explicitly documented as *never callable from the
control thread*, and it shares no object with the native backend.

## 2. Where it lives

```
src/embodied_control/robot/
  __init__.py     # open_robot() factory
  base.py         # RobotRuntime, RobotMode, capability protocols, errors
  fake.py         # FakeRobotRuntime — hardware-free, for CLI and gate tests
  g1.py           # G1Runtime — the Unitree realization
```

A new top-level `robot/` package, deliberately separate from the three
neighbours it could be confused with:

- `lowlevel/` is the 50 Hz tracker runtime — the *control* path.
- `embodiments/` maps a policy's normalized action onto simulator actuation.
- `runtime/` supervises *processes* (docker/local), not robots.

The G1's `LocoClient` is C++-only in the vendored SDK, so `g1.py` sits on a thin
pybind wrapper, `native/ec_native/src/g1_loco_client.{hpp,cpp}`, exposed as
`ec_native.G1LocoClient`. That native piece is a G1 implementation detail: a
robot reachable over ROS or HTTP implements `RobotRuntime` in pure Python and
compiles nothing.

## 3. The generic contract

Two layers, because robots genuinely differ in what they can do.

### 3.1 Core — every robot implements this

```python
class RobotMode(StrEnum):
    UNKNOWN = "unknown"
    ZERO_TORQUE = "zero_torque"   # motors energized, no torque
    DAMP = "damp"                 # compliant braking, the universal safe state
    READY = "ready"               # vendor controller holding the robot stable
    VENDOR_CONTROL = "vendor"     # vendor's own controller is driving
    USER_CONTROL = "user"         # our low-level runtime owns rt/lowcmd
    FAULT = "fault"

class RobotRuntime(Protocol):
    def info(self) -> RobotInfo: ...          # id, capabilities, transport
    def mode(self) -> RobotMode: ...
    def health(self) -> RobotHealth: ...

    def damp(self) -> None: ...               # always legal, from any mode
    def zero_torque(self) -> None: ...
    def ready(self) -> None: ...              # vendor's nominal holding state

    def close(self, *, damp: bool = True) -> None: ...
```

Takeover is deliberately *not* a verb here. Handing the joints to our controller
is `MotionSwitcherClient::ReleaseMode`, and it belongs to the process that owns
the 500 Hz `rt/lowcmd` writer, because the only safe order is "our damp frames
are already on the wire, then the vendor lets go". The lifecycle that sequences
this is `docs/design/robot_lifecycle.md`.

```python
```

`damp()` is the one verb required to work from every mode including `FAULT`. It
is the interface's safety floor, and the reason the enum's vocabulary is
generic rather than a passthrough of vendor FSM ids.

### 3.2 Capabilities — declared, not assumed

```python
@runtime_checkable
class PostureControl(Protocol):
    def sit(self) -> None: ...
    def squat(self) -> None: ...
    def stand(self, height: float | None = None) -> None: ...

@runtime_checkable
class VelocityControl(Protocol):
    def move(self, vx: float, vy: float, vyaw: float, *, continuous: bool = False) -> None: ...
    def stop(self) -> None: ...

@runtime_checkable
class GestureControl(Protocol):
    def wave_hand(self, *, turn: bool = False) -> None: ...
    def shake_hand(self, stage: int = -1) -> None: ...
```

Callers branch on `isinstance(rt, VelocityControl)`, and `info().capabilities`
carries the same set for display and for the CLI to hide inapplicable verbs. A
fixed-base arm implements the core and none of these; nothing in the core
contract assumes legs, a floating base, or locomotion.

### 3.3 Transitions are declared, not hardcoded in callers

Each realization exposes a transition table the generic layer validates against,
so illegal sequences fail with an actionable message instead of a vendor error
code:

```python
TRANSITIONS: dict[RobotMode, frozenset[RobotMode]]
```

For the G1 this encodes the real constraint that releasing the vendor's motion
service is only safe from the damp FSM state — the robot must be limp (and
therefore hoisted) at handover. There is no "stand up, then take over" path, and the
table is where that fact lives, rather than in a comment in the CLI.

## 4. The G1 realization

`G1Runtime` wraps `ec_native.G1LocoClient` over the SDK's `LOCO_SERVICE_NAME`
("sport") service. Mapping verified against
`native/thirdparty/unitree_sdk2/include/unitree/robot/g1/loco/g1_loco_client.hpp`:

| Contract verb | LocoClient call | FSM id |
|---|---|---|
| `zero_torque()` | `ZeroTorque()` | 0 |
| `damp()` | `Damp()` | 1 |
| `ready()` | `StandUp()` then `BalanceStand()` | 4 |
| `sit()` / `squat()` | `Sit()` / `Squat()` | 3 / 2 |
| `stand(h)` | `SetStandHeight(h)`, `HighStand`, `LowStand` | — |
| `move(...)` / `stop()` | `Move(...)` / `StopMove()` | — |
| `mode()` | `GetFsmId()` + `GetFsmMode()` → `RobotMode` | — |

Every call returns the SDK's `int32_t` status; the wrapper raises
`RobotCommandError` on non-zero rather than returning codes, so a failed
takeover cannot be silently ignored by a caller that forgot to check.

Capabilities: `PostureControl`, `VelocityControl`, `GestureControl` — all three.

## 5. Safety

- **Write gate.** Constructing a `G1Runtime` with mutating verbs enabled
  requires the existing token convention (`--enable-writes --confirm
  ENABLE_G1_LOWLEVEL`, `cli.py:_unitree_write_gate_error`). Getters —
  `info()`, `mode()`, `health()` — are ungated and always available, so
  "what is the robot doing" is never a privileged question.
- **`damp()` is exempt from nothing but is refused by nothing.** It is
  reachable whenever the transport is up.
- **Single owner.** The tracker process is the only `rt/lowcmd` writer and the
  only caller of `ReleaseMode`; this runtime never touches the motion
  switcher. `close()` damps by default; one-shot CLI verbs pass `damp=False`
  so `ec robot ready` does not stand the robot up and drop it in one command.

## 6. CLI surface

```
ec robot status                     # ungated: mode, capabilities, health
ec robot damp   --enable-writes --confirm ENABLE_G1_LOWLEVEL
ec robot ready  --enable-writes --confirm ENABLE_G1_LOWLEVEL
ec robot acquire / release
ec robot move --vx 0.2 --vyaw 0.1   # only if VelocityControl
```

`--robot g1` selects the realization and defaults to `g1`. Per this repo's
standing convention ("don't build the proper abstraction for one instance"),
`open_robot()` is an `if/else`, **not** a plugin registry — the registry arrives
with the third robot, if ever.

## 7. Extensibility check

The design is only worth its cost if a second robot is cheap. Worked example, a
fixed-base arm over ROS:

1. `robot/arm_ros.py` implements the seven core methods and no capability
   protocol.
2. `ready()` maps to the arm's home pose; `move()` does not exist and the CLI
   hides it automatically from `info().capabilities`.
3. `TRANSITIONS` omits `USER_CONTROL` if the arm has no torque handover.
4. `open_robot()` gains one `elif`. No caller, no CLI code, no test changes.

Nothing in the core mentions DDS, FSM ids, legs, or the G1.

## 8. Testing

Per the repo's "prove new architecture with something cheap" rule,
`FakeRobotRuntime` lands first: an in-memory mode machine honouring the same
transition table, with no DDS. It covers the gate, the CLI, the transition
validation and the capability branching in the light `pixi run test` env.

The simulator gap is closed in the lifecycle implementation: the MuJoCo DDS
plant now serves minimal `sport` and `motion_switcher` RPCs through the vendored
SDK, so `G1Runtime` and `ReleaseMode` are exercised in the same loopback
rehearsal as `rt/lowcmd`. Hardware remains the next exercise for vendor-specific
FSM behavior and the `SelectMode` round trip.

## 9. Decisions

1. **Takeover path: `MotionSwitcherClient::ReleaseMode` + `rt/lowcmd`.**
   Decided 2026-09-02. Every reference deployment (unitree_rl_gym
   `deploy_real`, unitree_rl_lab, the SDK's own low-level examples) takes over
   this way; `SwitchToUserCtrl` / `rt/user_lowcmd` is an SDK example added in
   March 2026 with no documentation and no public users, and it would have
   forced a second write topic into the native backend. The wrapper, the
   `acquire_control`/`release_control` verbs and their console keys were
   removed. "Return to vendor control" is `MotionSwitcherClient::SelectMode`
   (unverified on hardware, scheduled in the lifecycle plan).
2. **Plant support**: the plant grows a vendor-side mode machine so the
   lifecycle rehearses in sim, per `docs/design/robot_lifecycle.md`.
