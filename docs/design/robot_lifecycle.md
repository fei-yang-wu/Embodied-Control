# Robot lifecycle: hoist to run, from the control PC

Status: software v1 implemented; hardware ladder pending
Audience: embodied-control developers, G1 rig operators
Last updated: 2026-09-02
Depends on: `docs/design/robot_runtime_interface.md` (vendor axis, decided:
`MotionSwitcherClient::ReleaseMode` + `rt/lowcmd`)

## 1. Goal

One managed lifecycle that takes the G1 from "hoisted, vendor damp" to "our
policy is tracking" and back, driven entirely from the control PC with no
joystick, gated at every step by evidence, and rehearsed in MuJoCo on the
**same code path** before it is run on hardware.

The shape is what a SONIC-style operator console does (one key to start, one to
stop, restart a motion at frame 0), but every transition is a gate the runtime
checks rather than a key the operator trusts. The three things the current
code does not manage, and this design adds:

1. **Takeover is a sequence, not a call.** Today `ReleaseMode` runs inside
   `begin_initialization` and the write gate opens afterwards, so there is a
   window where nobody commands the motors. The lifecycle puts our damp frames
   on the wire first, then lets the vendor go, then proves it let go.
2. **The start pose is ramped and verified against the sim.** Sim teleports
   the robot onto the reference frame; hardware has to be ramped there, lowered
   onto its feet, and checked against the sim start qpos before any torque from
   the policy is allowed.
3. **Sim rehearses the whole thing.** The plant grows a vendor: it answers the
   sport-service and motion-switcher RPCs, ignores `rt/lowcmd` while it owns
   the joints, and hoists the robot until the operator lowers it. A command sent
   in the wrong mode is rejected in sim exactly as it would be on the robot.

## 2. What exists today

| Piece | Owns | Modes / state | Gap |
|---|---|---|---|
| `robot/` (`G1Runtime`) | sport service RPCs | reads FSM id; damp / zero torque / stand / sit / move | no notion of takeover; `ready()` ends in FSM 4, may not accept `move` |
| `NativeUnitreeBackend` | `rt/lowcmd` writer, motion switcher | DISABLED → INITIALIZE → WAIT → CONTROL → DAMP | release buried in init; no settle, no pose match, no hold, no restore |
| `MujocoDdsPlant` | `rt/lowstate` / `rt/lowcmd` on `lo` | freeze-until-command gantry, initial-pose teleport | no vendor, accepts any `lowcmd` any time, no hoist |
| consoles | keys | robot shell (vendor verbs), tracker shell (ramp / engage / serve / damp) | two consoles, no shared state, nothing gates the keys |

## 3. The lifecycle

Reviewed against the proposed list (PRECHECK → DAMP_CONFIRMED →
SAFE_EXTERNAL_COMMAND_PRESENT → USER_CONTROL_CONFIRMED → START_POSE_RAMP →
POSE_SETTLED → POLICY_COMMAND_FRESH → PRIMED → RUNNING → DAMP → vendor). The
spine is right. Five states are added, marked **new**, each because a real
failure hides between two of the original steps.

Every state below lists: what happens on entry, the gate that must hold to
leave, who produces the evidence, and how sim rehearses it. Thresholds are
starting values, tuned during the hardware ladder in §6.

| # | State | Entry action | Gate to leave | Evidence source | Sim parity |
|---|---|---|---|---|---|
| 0 | `PRECHECK` | open sport client, motion switcher (read-only), tracker in DISABLED | `rt/lowstate` ≥ 900 Hz, CRC 0, no motor errors, max temperature < 70 °C; sport `GetFsmId` answers; `CheckMode` returns a service name (recorded for restore); bundle joint limits loaded; RT configured or non-RT explicitly acknowledged; **operator acks hoist** (unsensable) | probe + sport + switcher | plant serves both RPC services; hoist is a weld |
| 1 | `VENDOR_DAMP_CONFIRMED` | our `Damp()` over the sport service (FSM 1), from the PC | `GetFsmId == 1` read back; joint speed max < 0.05 rad/s for 500 ms | sport + lowstate | plant FSM 1 = kd-only |
| 2 | `SAFE_EXTERNAL_COMMAND_PRESENT` | tracker opens the write gate in DAMP: kp 0, kd 8 on `rt/lowcmd` | ≥ 100 consecutive publishes, 0 publish failures, lowstate still fresh, robot still limp | writer stats | plant counts and **rejects** these frames (vendor still owns) |
| 3 | `USER_CONTROL_CONFIRMED` | `ReleaseMode` | `CheckMode` name empty on 3 polls 100 ms apart; lowstate fresh; publishes continuing; joint speed still < 0.05 rad/s | switcher + writer | plant releases, starts applying `rt/lowcmd` |
| 4 | `START_POSE_RAMP` | ramp current → start pose over T s (0.5 rad/s cap, existing) | ramp complete; **new guard**: any joint with tracking error > 0.3 rad for > 50 ms → `FAULT` (blocked joint / collision) | writer + lowstate | identical |
| 5 | `POSE_SETTLED` | hold start pose (WAIT) | ‖q − q*‖∞ < 0.05 rad and ‖q̇‖∞ < 0.1 rad/s for 500 ms | lowstate | identical |
| 6 | `LOWERED` **new** | operator lowers the hoist while the tracker holds the pose | operator ack on hardware; joint speed settles again (< 0.1 rad/s for 500 ms) after the load change | operator + lowstate | plant releases the pelvis weld on `lower` |
| 7 | `POSE_MATCH_VERIFIED` **new** | capture the fixed anchor, write `pose_match.json` | ‖q_hw − q_sim_start‖∞ < per-joint tolerance (0.05 rad legs, 0.1 rad arms); projected-gravity tilt vs reference frame 0 < 5°; base height plausible (IMU-only, so tilt is the real check) | lowstate + reference | identical; sim additionally has ground-truth root pose to grade the IMU check |
| 8 | `POLICY_COMMAND_FRESH` | planner warm-up: N ≥ 20 inferences, p99 latency < 60 % of the tick | command age < `command_stale_ms`; sequence advancing; **first-action consistency**: ‖a₀ − q_hold‖∞ < 0.2 rad | command slot + planner stats | identical |
| 9 | `PRIMED` | `engage_control` | immediate | backend | identical |
| 10 | `BLEND_IN` **new** | target and gains blend held → policy, weight 0 → 1 over 250 writer ticks (0.5 s); `last_action` reports the blended action; the reference clock stays on frame 0 until the blend is done | blend complete, no watchdog trip | writer + control thread | identical |
| 11 | `RUNNING` | serve | operator `hold` / `damp`, tick budget, or any watchdog | existing | identical |
| 12 | `HOLD` **new** | freeze on the last commanded target | operator takes the load on the hoist, then `damp` | operator | plant re-welds on `hoist` |
| 13 | `DAMP` | kd-only frames | always reachable; `FAULT_DAMP` is the same frame with a recorded reason | writer | identical |
| 14 | `RELEASED` **new** | write gate closed, publisher silent | writer publishes 0 for 200 ms | writer stats | identical |
| 15 | `VENDOR_RESTORED` (optional) | `SelectMode(name from PRECHECK)` | `CheckMode == name`; `GetFsmId == 1`; robot damp under the vendor again, so `ec robot ready` is available | switcher + sport | plant re-takes ownership, rejects `rt/lowcmd` again |

Two sinks:

- **`FAULT`**: from any state ≥ 2. Action is the DAMP frame; the reason is the
  backend's latched fault code or the gate that failed. Leaving `FAULT` means
  `RELEASED` and a fresh `PRECHECK`, never a resume.
- **`ABORT`**: operator, from any state < 9. Same path: DAMP → RELEASED →
  optional restore.

Why the five additions:

- **`LOWERED`** because the ramp and the settle both happen with the feet in
  the air; the loaded pose differs, and the policy was trained standing on the
  ground. Without it the pose-match check verifies the wrong configuration.
- **`POSE_MATCH_VERIFIED`** is the actual "match the sim setup" requirement.
  Settled is not the same as correct: a joint can settle on a limit, and the
  IMU tilt is the only observable that ties the robot's frame to the reference.
- **`BLEND_IN`** because the first policy action is a step function against a
  held pose. A weight ramp of half a second removes the one discontinuity the
  policy never saw in training.
- **`HOLD`** because damping a standing robot is a fall. Freezing on the last
  target gives the operator a stable robot to take the load from.
- **`RELEASED`** because `SelectMode` with our writer still publishing is two
  controllers on one topic, the exact thing the takeover order avoided.

### 3.1 End of run: recover to standing

The default end state is **the robot standing still under the vendor's
balance controller**, not hanging damped. Job setting `end_state:
vendor_stand | vendor_damp | damp`, default `vendor_stand`. The chain after
`RUNNING`, and the same chain out of `FAULT`:

```
RUNNING ─h─▶ HOLD ──(operator hooks the hoist, `H`)──▶ DAMP ▶ RELEASED
        ▶ VENDOR_RESTORED (vendor FSM 1) ▶ VENDOR_STAND (StandUp, FSM 4, hoisted)
        ──(operator lowers, `l`)──▶ STANDING (idle; next episode starts at PRECHECK)
```

Rules that shape it:

- **Our controller never stands the robot on its own.** `HOLD` is a stiff PD
  hold, stable only while the hoist can catch it; the vendor's balance stand
  is the only proven stand. Recovery therefore always hands back to the vendor.
- **The hoist ack (`H`) is a gate, not a courtesy.** `SelectMode` restarts the
  vendor service in damp, so the robot is limp for at least a second during
  restore. Standing on its feet, that is a fall.
- **`FAULT` recovers the same way**, after `DAMP` and `RELEASED`, so the
  operator has exactly one recovery procedure to learn.
- **`HOLD` target is the last commanded pose**, not the start pose: dragging a
  mid-stride robot back to frame 0 is itself a motion. For the stationary
  references this plan covers, the two are the same pose.
- **Retake without releasing.** `HOLD ─e─▶ START_POSE_RAMP` is a legal edge:
  a second episode ramps back to the start pose, re-verifies (rows 4–9) and
  re-arms with the vendor still released and the planner still up. This is the
  SONIC "restart at frame 0" move, but gated.

## 4. Where it lives

```
src/embodied_control/robot/
  lifecycle.py     # Lifecycle: states, gates, advance(), damp(), abort(), restore()
  gates.py         # one Gate per row of §3: check() -> GateResult(ok, detail, values)
  shell.py         # existing vendor keys, plus lifecycle keys (§5)
```

`Lifecycle` takes three collaborators and nothing else: a `RobotRuntime`
(vendor axis, existing), a `Tracker` protocol (the surface `NativeUnitreeLoop`
already has plus the new calls below, satisfied by a stub in tests), and a
clock. It is pure Python and runs in the light env against
`FakeRobotRuntime` + a stub tracker. Every transition is appended to
`lifecycle.jsonl` with the gate's measured values, and the final state goes to
`lifecycle.json`, so a rehearsal or a hardware run leaves the same artifact
trail as an eval.

**Native backend** (`NativeUnitreeBackend`), splitting what is one call today:

- `vendor_mode() -> str`: read-only `CheckMode`, for `PRECHECK`.
- `open_damp_gate()`: DISABLED → DAMP with the write gate open (row 2).
- `release_vendor()`: `ReleaseMode` + confirmation, legal only from DAMP with
  the gate open (row 3). `begin_initialization` then takes
  `skip_motion_switcher=True` from the lifecycle and additionally accepts DAMP
  as its starting mode.
- ramp tracking-error guard (row 4), settle metrics in `writer_stats`
  (joint speed max, tracking error max, per-tick) so gates 5–7 read numbers
  rather than recomputing them in Python.
- `blend_in_ticks` on `engage_control` (row 10), `hold()` (row 12),
  `close_gate()` (row 14), `restore_vendor(name)` (row 15, legal only from
  DISABLED with the gate closed).

**Plant** (`MujocoDdsPlant`), the sim side of parity:

- `PlantVendor`: two `unitree::robot::Server` instances, `"sport"` answering
  `GetFsmId` / `GetFsmMode` / `SetFsmId` for ids 0, 1, 2, 3, 4, 500, and
  `"motion_switcher"` answering `CheckMode` / `ReleaseMode` / `SelectMode`.
  The SDK's client classes talk to these unchanged, so `G1Runtime` and the
  backend's motion-switcher code run against the plant with zero branches.
- Ownership gate: while the vendor owns the joints the plant applies its own
  behaviour (FSM 0 zero torque, 1 kd-only, 4/500 hold default stance at the
  hold gains) and **rejects** `rt/lowcmd`, counting `rejected_commands` in
  `PlantStats`. `ReleaseMode` is refused unless the vendor FSM is 1, matching
  the hardware rule. `rt/lowcmd` with the wrong `mode_machine` is rejected
  in every state.
- Hoist: a mocap weld on the pelvis, on by default for rehearsal jobs,
  released by `lower` and re-applied by `hoist`, sent on a plant-only DDS topic
  (`rt/ec_plant_cmd`). The existing `freeze_until_command` stays for pure
  policy evals that still want to teleport.

### 4.1 Planner service

The planner is the other async service, and its lifecycle is deliberately
simpler because of one existing fact: **the tracker is the requester, the
planner only answers.** The oracle request is `(generation, reference_frame)`
and the VLA request is the state vector; the tracker owns the frame cursor and
the generation counter. So "pause at frame 0 until the policy starts" and
"restart the motion" need no control channel to the planner. The tracker
requests frame 0 while it holds, advances the cursor only in `RUNNING`, and
bumps `generation` on retake.

Service states, in the planner terminal:

| State | What | Exit |
|---|---|---|
| `BOOT` | load bundle / checkpoint, hash the reference, **create** the slots | slots exist |
| `WARM` | VLA: one inference on a synthetic request so the first real one is not cold; oracle: nothing | warm report printed |
| `READY` | answering probes; stays here between episodes | tracker connects |
| `SERVING` | answering real requests, per-request latency logged | tracker damps or stops |
| `DRAIN` | requests stop; worker keeps the slots and returns to `READY` | next episode or Ctrl-C |
| `EXIT` | Ctrl-C: report written (`requests`, latency mean/max, last error) | |

The lifecycle's `POLICY_COMMAND_FRESH` gate is the planner's readiness check:
the tracker sends N probe requests for frame 0 and requires every reply within
budget (p99 < 60 % of the tick) before engaging. Nothing else is needed for the
planner to be "ready", and the planner never learns which lifecycle state the
robot is in. Guards on the planner side: refuse a request whose `generation`
goes backwards, refuse a frame past the horizon (exists), and keep the slots
across tracker restarts so one planner terminal serves a whole session.

## 5. Operator surface

One console, `ec lifecycle console <job.yaml> --network <nic|lo>`, replacing
the split between the robot shell and the tracker shell for lifecycle work
(both stay for ad-hoc use). Keys are requests; the state machine decides:

| Key | Request | Notes |
|---|---|---|
| `SPACE` | damp | from anywhere, never gated, never goes through `stop()` |
| `n` | advance one state | runs the next gate, prints its evidence, stops on failure |
| `a` | auto-advance to `PRIMED` | stops at the first failing gate or at an operator-ack state |
| `g` | go | `PRIMED` → `BLEND_IN` → `RUNNING` (SONIC `]`) |
| `h` | hold | `RUNNING` → `HOLD` |
| `l` / `H` | lowered / hoisted acks | hardware: operator; sim: also sends `lower` / `hoist` to the plant |
| `s` | recover to stand (default end) | `HOLD`/`DAMP` → `RELEASED` → `VENDOR_RESTORED` → `VENDOR_STAND`, stops at the `H` ack |
| `d` | release to vendor damp only | same chain without the stand (`end_state: vendor_damp`) |
| `e` | retake episode | `HOLD` → `START_POSE_RAMP`, generation + 1 (SONIC `R`) |
| `q` | quit | damps first (SONIC `O`) |

The console is an experiment session (`robot/session.py`), not one
lifecycle: it owns the planner process (oracle worker or VLA planner worker,
started with the slots), the selection (command source, motion, start
frame), and rebuilds the tracker for a new selection while nothing owns the
joints. Every episode that ends in `HOLD` writes `episodes/epNNN_.../`
with `telemetry.npz` and `summary.json` (joint MAE; MPJPE via forward
kinematics when the job names an `mjcf`), so grading never depends on
remembering to save.

On a terminal the console is full-screen (`robot/tui.py`): the ladder with
the current rung marked, the writer's live numbers, the key legend, the last
gate result and a log. Keys run on one worker thread so the screen never
freezes behind a gate; `SPACE` bypasses the lifecycle lock and stores DAMP
straight into the writer, then records the transition once the gate returns.
`--plain` keeps the line-mode console for pipes and tests.

`ec lifecycle run <job.yaml> --until <STATE>` is the scripted form for tests
and for CI-style loopback rehearsals, exit code 0 only if the target state
was reached and no fault was recorded. The job YAML carries the start pose
(`motion@frame`, bundle default, or explicit qpos), ramp seconds, tolerances,
`end_state`, and whether the run is `rehearsal` (hoist + ramp) or `eval`
(teleport).

### 5.1 How a session runs

Three terminals on the control PC; the same three against the plant with
`--network lo` plus a fourth for `ec lowlevel plant --hoist`.

```
T1  planner service (non-RT, stays up all session, owns the slots)
    pixi run -e native ec lowlevel oracle-worker <bundle> --reference-root ... \
      --motion hurry_idle_001_A277 --request-slot /ec_g1_request \
      --response-slot /ec_g1_response --create-slots
    # or: ec lowlevel planner-worker ... --create-slots -- <VLA service command>

T2  lifecycle + tracker + console (RT, owns rt/lowcmd, one process)
    pixi run -e native ec lifecycle console <job.yaml> --network enp128s31f6 \
      --connect-slots --enable-writes --confirm ENABLE_G1_LOWLEVEL

T3  watch (optional, read-only): ec robot status, telemetry tail
```

The console and the tracker share one process on purpose: `SPACE` reaches the
writer through a lock-free mode store, which no socket can promise. A control
socket so T2 can be split (or scripted from another host) is a later step,
after M3.

The operator's script for one episode, hoisted, robot under vendor damp:

1. T1: start the planner, wait for the `READY` line.
2. T2: start the console. It opens in `PRECHECK`; the status line shows the
   lifecycle state, vendor FSM id, writer mode, publish count, fault code and
   planner probe latency.
3. `n` runs `PRECHECK` and prints its table (rates, temperatures, vendor
   service name, planner probe). `a` then auto-advances through rows 1–5 and
   stops at `POSE_SETTLED` because the next state needs a person.
4. Lower the hoist until the feet carry the weight. Press `l`.
5. `n` → `POSE_MATCH_VERIFIED` prints the per-joint table and writes
   `pose_match.json`. `n` → `POLICY_COMMAND_FRESH` prints probe latencies.
   `n` → `PRIMED`.
6. `g` → `BLEND_IN` → `RUNNING`. The status line shows ticks and reference
   frame. `SPACE` at any moment damps.
7. The tick budget ends, or `h`: `HOLD`. Hook the hoist, take the load, press
   `H`.
8. `s`: `DAMP` → `RELEASED` → `VENDOR_RESTORED` → `VENDOR_STAND`. Lower the
   robot onto its feet; it is standing still under the vendor. Press `q`, or
   `n` to begin the next episode at `PRECHECK` with the planner still up.

For a second take of the same motion without handing back: at `HOLD`, hoisted,
press `e`; the lifecycle ramps to the start pose and re-runs rows 4–9.

## 6. Milestones

Per the repo rule, each is proven with the cheap thing before the real one.

**M1 — lifecycle in Python (light env). Implemented.** `lifecycle.py`,
`gates.py`, the console keys, artifacts. Tests drive every row of §3 with `FakeRobotRuntime`
and a stub tracker: happy path to `VENDOR_STAND`, every gate failing once,
`FAULT` from `RUNNING` and its recovery to `VENDOR_STAND`, `ABORT` before
`PRIMED`, retake from `HOLD`, keys refused in the wrong state. Planner side:
the warm-up probe and the generation guard, tested with the existing stub
service. Done when `pixi run test` covers the whole table.

**M2 — plant vendor and hoist (native env). Implemented and
loopback-validated.** `PlantVendor` servers, ownership gate,
`rejected_commands`, mode-machine check, pelvis weld. Loopback tests on
`lo`: `G1Runtime` reads FSM from the plant; `lowcmd` during vendor ownership
is rejected and counted; `ReleaseMode` from FSM 4 is refused; the full
lifecycle runs `PRECHECK → VENDOR_RESTORED` with the stationary
`hurry_idle_001_A277` oracle and the existing MPJPE grade still passes.

**M3 — backend split and guards (native env). Implemented and
loopback-validated.** The `NativeUnitreeBackend` calls in §4, ramp tracking
guard, settle metrics, blend-in, hold, restore.
Verified in the same loopback lifecycle test; a deliberately blocked joint
in the plant (a weld on one link) must trip the ramp guard into `FAULT`.

**M4 — hardware ladder (next).** Hoisted throughout until the last rung. Each
rung is a recorded `lifecycle.jsonl` and is a pass/fail with the criteria
written down before the session.

| Rung | What | Pass |
|---|---|---|
| H0 | read-only: cycle the vendor through 0 / 1 / 4 / 500 / 3 from the PC with `ec robot`, log `GetFsmId`/`GetFsmMode` | FSM table confirmed, `_FSM_TO_MODE` corrected from evidence |
| H1 | `ec robot ready` then `move`: does FSM 4 accept velocity, or does `ready()` need `Start` (500)? | decision recorded, `ready()` fixed |
| H2 | rows 1–3 only, then `DAMP → RELEASED → VENDOR_RESTORED` | vendor tolerates our damp frames before `ReleaseMode`; `SelectMode` name and round trip confirmed; if the vendor refuses the early publisher, fall back to release-then-publish and record it |
| H3 | rows 1–5 with the bundle default stance, then hold, damp | ramp guard never trips on a free robot; settle thresholds hold |
| H4 | rows 1–7 with `hurry_idle_001_A277@0`, feet lowered | `pose_match.json` inside tolerance |
| H5 | full lifecycle with the stationary oracle, hoist slack | MPJPE within the sim rehearsal band; watchdog drills: kill the planner (command-stale damp), pull the NIC (state-absent damp), Ctrl-C (damp before join) |
| H6 | same with the VLA command source | as H5 |

## 7. Decisions and unknowns to close on hardware

1. Publish-before-release (row 2 before row 3) is the SDK user-control
   example's order, not something the `ReleaseMode` path documents. H2
   decides; the lifecycle keeps both orders behind one flag until then.
2. `SelectMode` argument: the name `CheckMode` returned at `PRECHECK`, not a
   hardcoded alias.
3. Vendor damp vs ours: the vendor's FSM 1 gains are unknown; ours is kd 8.
   If the hand-off in row 3 twitches, match the vendor's value.
4. Hold gains: decided on the plant (§9). Ramp, settle and HOLD use the
   policy's stiffness times 3 (damping times √3); the blend-in ramps both
   the target and the gains down to the policy's. H3/H4 confirm the scale
   on the robot.
5. Start pose source for a moving reference: this design covers stationary
   references (fixed anchor). Moving references still need external
   localization, unchanged.
6. Anchor orientation feedback rides on the IMU quaternion (WXYZ, world
   frame) relative to its first frame. H3 confirms the convention on the
   robot before H4 trusts the tilt term.

## 8. Non-goals

- No joystick path, no `SwitchToUserCtrl`, no second write topic.
- No RPC of any kind from the control or writer thread; the lifecycle runs
  on the operator thread and talks to the backend through lock-free mode
  stores, as `force_damp` does today.
- No plugin registry for lifecycles or vendors: one lifecycle, one G1
  realization, one plant vendor.

## 9. What the sim rehearsal found (2026-09-02)

The plant rehearsal with the SONIC bundle and the stationary `hurry_idle`
oracle now completes every row: 504 ticks, no runtime or writer fault, pelvis
tilt under 3° throughout, pose match within 0.075 rad, ending in `STANDING`.
Getting there changed the design in these ways, each recorded because it
would have reached hardware otherwise.

- **Projected gravity was sign-flipped on the hardware path.** The Unitree
  backend carried its own copy of the quaternion-to-gravity formula with roll
  and pitch inverted; the policy leaned into every tilt and fell four seconds
  after engaging. Every backend now calls the one `projected_gravity_from_xyzw`,
  and the loopback test hangs the plant 10° nose-down to check the sign.
- **A timestamp race tripped the watchdogs.** Ages were unsigned differences
  against a stamp another thread could store just after `now` was sampled;
  the wrap read as 1.8e10 ms and damped the robot about once per 20 s of
  loopback. Ages are signed and clamped now.
- **The fixed anchor needs the robot's own tilt.** An IMU-only anchor told
  the policy its target was rotated by the start-heading offset (88° on the
  plant: it turned, then fell); a reference-only anchor left it blind to its
  tilt and it drifted over. The anchor orientation is now the reference's
  frame-0 orientation rotated by the IMU's change since the first frame;
  the position stays frozen at frame 0.
- **The policy's own PD gains cannot hold the start pose.** Under gravity the
  28 N m/rad waist sagged 0.42 rad and the robot toppled when lowered. Ramp,
  settle and HOLD use 3x stiffness (√3x damping); the blend-in walks gains
  and target down together. The `ec lowlevel unitree` legacy ramp (1x gains,
  no hoist) falls on the plant for the same reason.
- **Quiet means position drift, not joint speed.** The wire carries
  ±0.5 rad/s velocity noise, so a limp robot never reads still by velocity;
  the gates use 0.03 rad of drift over the window instead.
- **Tolerances measured, not guessed.** Legs settle within 0.08 rad of the
  sim frame (tolerance 0.1); waist and arms sag under their gains (0.5);
  the tilt reading is averaged over 25 samples against ±0.05 rad IMU noise.
  The first-action gate is a per-joint torque ratio against the effort limit
  at 3x (the 5 N m wrists saturate at 1.4x on any real command) plus a coarse
  2.5 rad bound. The ramp guard is 0.5 rad for 100 ms, configurable.
- **The strap has three states.** Hoisted (rigid), lowered (the feet carry
  the weight, the rope only catches a 5 cm drop and steadies the tilt at half
  gain), slack (paid out over 3 s when RUNNING begins). Lowering with the
  strap fully released toppled the held robot.
- **Probe, then restart.** A policy whose actions the writer does not apply
  integrates open loop; POLICY_COMMAND_FRESH probes the planner and stops the
  control thread, and `go()` starts it again and engages on the first tick.
  Planner replies to requests from before a stop are counted as stale, not
  as a broken contract.
- **The reference waits for the policy.** The runtime keeps a reference
  clock separate from the loop tick; it is pinned at frame 0 through the
  probe and the blend-in and released the tick the policy fully owns the
  joints, so a moving motion starts then and not 25 ticks earlier. During
  the blend the control thread writes the blended target and overwrites the
  tracker's `last_action` with the blended action, the way Isaac's
  `last_action` is the action that ran.
- **Soft joint limits are enforced only while we hold or drive.** They are
  the policy's training limits, not the actuators': a limp robot hanging in
  vendor damp folds its torso past the waist's 0.52 rad and used to latch a
  fault before the next episode could start. The latch now applies in WAIT,
  CONTROL and HOLD; the init ramp starts from wherever the joints are and
  its tracking guard covers it.
- **The sim strap hangs the robot level.** Re-hooking a leaning robot used to
  carry its 8° tilt into the next episode's pose check; the strap keeps the
  heading and drops roll and pitch, as a real strap does. The tilt gate is
  10° (a lowered robot on a flexed frame sat at 6°), ankles get 0.2 rad.
- **The planner restarts with the tracker.** A rebuilt tracker numbers its
  requests from 1 again and an oracle worker serves one start frame, so the
  session stops the worker, drops the mailboxes, starts it for the new
  selection and only then connects. Episode budgets follow the start frame
  (`min(ticks, frames left)`).
- **A fallen robot must not pin the writer.** Joint-limit reports from a
  robot on the floor re-damped a DISABLED writer and blocked `close_gate`;
  the latch no longer touches a disabled writer, and PRECHECK clears it so
  the next episode can start.

