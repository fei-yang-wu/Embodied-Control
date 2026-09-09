# Robot lifecycle: hoist to run, from the control PC

Status: software v1 implemented; hardware ladder pending
Audience: embodied-control developers, G1 rig operators
Last updated: 2026-09-09
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
| 0 | `PRECHECK` | open sport client, motion switcher (read-only), tracker in DISABLED | `rt/lowstate` ≥ 900 Hz, CRC 0, no motor errors, max temperature < 70 °C; sport `GetFsmId` answers; `CheckMode` returns a service name (recorded for restore); bundle joint limits loaded; RT configured or non-RT explicitly acknowledged; **operator acks hoist** — feet clear of the floor by ~10 cm (unsensable on hardware; the plant measures it) | probe + sport + switcher | plant serves both RPC services; hoist is a weld, and the gate reads the plant's measured floor clearance |
| 1 | `VENDOR_DAMP_CONFIRMED` | our `Damp()` over the sport service (FSM 1), from the PC | `GetFsmId == 1` read back; joint speed max < 0.05 rad/s for 500 ms | sport + lowstate | plant FSM 1 = kd-only |
| 2 | `SAFE_EXTERNAL_COMMAND_PRESENT` | tracker opens the write gate in DAMP: kp 0, kd 8 on `rt/lowcmd` | ≥ 100 consecutive publishes, 0 publish failures, lowstate still fresh, robot still limp | writer stats | plant counts and **rejects** these frames (vendor still owns) |
| 3 | `USER_CONTROL_CONFIRMED` | `ReleaseMode` | `CheckMode` name empty on 3 polls 100 ms apart; lowstate fresh; publishes continuing; joint speed still < 0.05 rad/s | switcher + writer | plant releases, starts applying `rt/lowcmd` |
| 4 | `START_POSE_RAMP` | ramp current → start pose over T s (0.5 rad/s cap, existing) | ramp complete; **new guard**: any joint with tracking error > 0.3 rad for > 50 ms → `FAULT` (blocked joint / collision) | writer + lowstate | identical |
| 5 | `POSE_SETTLED` | hold start pose (WAIT) | ‖q − q*‖∞ < 0.05 rad and ‖q̇‖∞ < 0.1 rad/s for 500 ms | lowstate | identical |
| 6 | `LOWERED` **new** | operator lowers the hoist while the tracker holds the pose | operator ack on hardware; joint speed settles again (< 0.1 rad/s for 500 ms) after the load change | operator + lowstate | plant pays the strap out at 0.05 m/s to past the hang height; the gate waits for measured clearance ≤ 2 mm before it grades the settle |
| 7 | `POSE_MATCH_VERIFIED` **new** | capture the fixed anchor, write `pose_match.json` | valid finite joint/IMU state; pose and reference-relative tilt tolerances are diagnostics, not refusals | lowstate + reference | identical; sim additionally has ground-truth root pose to grade the IMU check |
| 8 | `POLICY_COMMAND_FRESH` | planner warm-up: N ≥ 20 inferences, p99 latency < 60 % of the tick | command age < `command_stale_ms`; sequence advancing; **first-action consistency**: ‖a₀ − q_hold‖∞ < 0.2 rad | command slot + planner stats | identical |
| 9 | `PRIMED` | `engage_control` | immediate | backend | identical |
| 10 | `BLEND_IN` **new** | target and gains blend held → policy, weight 0 → 1 over 250 writer ticks (0.5 s); `last_action` reports the blended action; the reference clock stays on frame 0 until the blend is done | blend complete, no watchdog trip | writer + control thread | identical |
| 10a | `ARMED` **new (2026-09-09)** | nothing: the policy owns the joints with the reference clock pinned; a composed stance prefix is required for a stationary encoder window | writer in CONTROL, no runtime fault, valid joint/IMU state; tilt is diagnostic; then an explicit `play` | writer + lowstate | identical; the strap is still `lowered` here |
| 11 | `RUNNING` | strap paid out, countdown announced, reference clock released | operator `hold` / `damp`, reference frames played out, tick budget, or any watchdog | existing | identical |
| 11a | `STAND_HOLD` | composed reference reaches its stationary suffix; pin the clock and keep policy control | operator hoists and damps, configured timeout enters HOLD, or runtime fault damps | reference progress + writer | identical |
| 12 | `HOLD` **new** | freeze on the last commanded target; in sim the strap takes the load on entry | operator takes the load on the hoist, then `damp` | operator | plant re-welds on `hoist` |
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
- **`POSE_MATCH_VERIFIED`** retains its historical state name for artifact
  compatibility. It records pose error and reference-relative tilt, but only
  invalid state blocks this step. Matching the reference's first pose is not
  a prerequisite for a tracking policy to start.
- **`BLEND_IN`** because the first policy action is a step function against a
  held pose. A weight ramp of half a second removes the one discontinuity the
  policy never saw in training.
- **`ARMED`** because engaging the policy and playing a motion are two
  decisions, and one key used to do both: `go` blended in and released the
  reference clock in the same call, so on hardware (where there is no gantry
  status to wait for) the motion began about half a second after the keypress,
  with the operator still within reach of the robot. Arming leaves the robot
  tracking a paused reference. A raw clip may still encode future movement;
  use a rehearsed stance prefix when a stationary encoder window is needed.
  SONIC's prerecorded-motion console uses `]` to start the policy and `T` to
  play the selected motion; it does not guarantee a separate standing reference;
  their `R` pauses at the first frame without terminating the policy, which is
  the same pinned-reference state. `play` displays a countdown with terminal bell cues before it
  releases the clock. The armed stand is bounded by `arm_timeout_seconds`
  (60 s): past it the run goes to `HOLD`, never into a motion nobody is
  standing by for. The tick budget is padded by the same allowance, so an
  armed stand is not paid out of the frames the operator asked for, and the
  episode ends on reference exhaustion instead.
- **`HOLD`** because damping a standing robot is a fall. Freezing on the last
  target gives the operator a stable robot to take the load from.
- **`RELEASED`** because `SelectMode` with our writer still publishing is two
  controllers on one topic, the exact thing the takeover order avoided.

### 3.1 End of run: damp, under the vendor

**A run ends with the robot limp under the vendor's own damp**, not standing
and not hanging on our frames. Job setting `end_state: vendor_damp |
vendor_stand | damp`, default `vendor_damp` (changed 2026-09-03; it was
`vendor_stand`). The chain after `RUNNING`, and the same chain out of
`FAULT`:

```
RUNNING ─h─▶ HOLD ──(operator hooks the hoist, `H`)──▶ DAMP ▶ RELEASED
        ▶ VENDOR_RESTORED (vendor FSM 1: the robot is limp, the vendor owns it)
        ══ next trajectory starts at PRECHECK ══
                                 ─s─▶ VENDOR_STAND (StandUp, FSM 4, hoisted)
                                 ──(operator lowers, `l`)──▶ STANDING
```

`damp` is one request, not three. `Ctrl-D` and `/damp` put the kd-only frames
on the wire immediately and unconditionally, then close the gate and hand the
joints back with `SelectMode`, so `DAMP` is never a resting state with our
writer still owning `rt/lowcmd`. Only the hand-back waits for the hoist; the
frames never do.

Rules that shape it:

- **Our controller never stands the robot on its own.** `HOLD` is a stiff PD
  hold, stable only while the hoist can catch it; the vendor's balance stand
  is the only proven stand. Recovery therefore always hands back to the vendor.
- **The hoist ack (`H`) is a gate, not a courtesy.** `SelectMode` restarts the
  vendor service in damp, so the robot is limp for at least a second during
  restore. Standing on its feet, that is a fall. Without the ack, `damp`
  stops after the kd frames and says so.
- **A second trajectory re-runs the checks.** `VENDOR_RESTORED` is a restart
  state: `advance()` rewinds to `PRECHECK` and climbs the whole ladder. The
  one shortcut, `e` from `HOLD`, now re-reads the link first (`RETAKE_PRECHECK`
  in `lifecycle.jsonl`): fresh `rt/lowstate`, zero CRC errors, no latched
  hardware fault. It skips only the vendor rows, which would fail by design
  with the vendor already released.
- **`FAULT` recovers the same way**, after `DAMP` and `RELEASED`, so the
  operator has exactly one recovery procedure to learn.
- **`HOLD` target is the last commanded pose**, not the start pose: dragging a
  mid-stride robot back to frame 0 is itself a motion. For the stationary
  references this plan covers, the two are the same pose.

**Untethered, later.** An untethered end would skip the hoist and hand a
robot that is already on its feet straight to a vendor stand. Two things
block it and neither is answered in sim: `SelectMode` on a loaded robot has
never been tried (rung H2), and "standing-ish" needs to mean both feet carry
the weight, which an IMU cannot see. `HOLD` therefore records what it can —
`pelvis_tilt_degrees` and `pelvis_upright` in `lifecycle.jsonl` — as evidence
for that decision, never as a gate.

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
| `Ctrl-D` | damp | from anywhere, never gated, never goes through `stop()` |
| `n` | advance one state | runs the next gate, prints its evidence, stops on failure |
| `a` | auto-advance to `PRIMED` | stops at the first failing gate or at an operator-ack state |
| `g` | go | `PRIMED` → `BLEND_IN` → `RUNNING` (SONIC `]`) |
| `h` | hold | `RUNNING` → `HOLD` |
| `l` / `H` | lowered / hoisted acks | hardware: operator; sim: also sends `lower` / `hoist` to the plant |
| `s` | recover to stand (default end) | `HOLD`/`DAMP` → `RELEASED` → `VENDOR_RESTORED` → `VENDOR_STAND`, stops at the `H` ack |
| `d` | release to vendor damp only | same chain without the stand (`end_state: vendor_damp`) |
| `e` | retake episode | `HOLD` → `START_POSE_RAMP`, generation + 1 (SONIC `R`) |
| `q` | quit | damps first (SONIC `O`) |
| `t` / `T` | next / previous tracker | selects a named bundle while joints are unowned |
| `R` | reset sim | closes tracker/planner and resets plant to nominal, vendor damp, hoisted |

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
freezes behind a gate; `Ctrl-D` bypasses the lifecycle lock and stores DAMP
straight into the writer, then records the transition once the gate returns.
`--plain` keeps the line-mode console for pipes and tests.

### Who runs where

Nothing the operator touches shares a thread, a core or a process with the
robot.

| Runs | Where | Priority |
|---|---|---|
| 500 Hz writer | C++ thread, this process, pinned `writer_cpu` | SCHED_FIFO 90 |
| 50 Hz control + policy | C++ thread, this process, pinned `control_cpu` | SCHED_FIFO 80 |
| planner (oracle or VLA) | its own process, own session, off the robot cores | normal |
| lifecycle gates and keys | console worker thread, off the robot cores | normal |
| display and key reader | console main thread, off the robot cores | normal |
| `/diagnose` agent | its own process, off the robot cores, `nice +10` | idle-ish |

Three rules hold it together. The C++ threads never take the GIL, so Python
cannot delay a control tick and a control tick cannot delay the display.
Every console thread pins itself away from `control_cpu` and `writer_cpu` on
first use (`robot/isolation.py`); a SCHED_OTHER thread that lands on a
SCHED_FIFO 90 core is not slow, it is starved, and the thread in question is
the one reading the damp key. And the render path performs no blocking call:
`snapshot()` reads native counters and a cached copy of the plant's hoist
status, which the watcher thread refreshes at 2 Hz, because that status is a
DDS RPC with a timeout and a hung plant must never freeze the display.

The planner is a child process and is never renice'd: the control loop waits
on its replies inside a deadline. The diagnostic agent is renice'd hard,
because it is an LLM CLI that would otherwise take every core it can find
while a robot is standing on the floor.

Damp is `Ctrl-D`, not `SPACE`. The space bar is the key an operator rests a
hand on, scrolls with and taps while thinking, and an unwanted damp drops a
standing robot; a control chord cannot be typed by accident, is never inserted
into the prompt, and Ctrl-D collides with no terminal signal (Ctrl-C, Ctrl-Z,
Ctrl-S/Q all do). `SPACE` is inert and says so once when pressed.

The prompt is a real line editor (`CommandLine`), because the console is a
harness and behaves like one: a cursor with arrows and Home/End, Backspace and
Delete, `Ctrl-U`/`Ctrl-W`/`Ctrl-K`, `TAB` completion to the longest shared
prefix, `↑`/`↓` history that outlives the prompt, the single remaining match
shown as ghost text, and the matching commands listed under the line. Curses
key codes are decoded to tokens first: with `keypad(True)` Backspace arrives as
263, and `chr(263)` is a letter, which is why the first version inserted a
character instead of deleting one.

`render_rows` returns rows of styled spans and `render` flattens the same
rows to plain text, which is what `--plain`, a pipe and every test read.
Colour therefore lands on a word rather than a line: a label stays grey while
its value is white, and one link's dot turns red without taking its
neighbours with it. The state is a reverse-video badge toned by urgency, each
link carries a sparkline of its own rate, and joint speed and tracking error
draw gauges against the writer's own guards. The palette is xterm-256 with an
8-colour fallback that keeps the same meanings; a terminal without colour
still gets bold and dim. The layout reflows rather than truncates: above 118
columns the three links share a row, below 96 the ladder is one column and
the episode summary gives up its row, and under about fifteen rows the ladder
leaves rather than showing a header with nothing under it.

The prompt is a fixed two rows — the line, plus the row that lists what
matches — open or closed, matching or not. It grew by a row when a command
matched at first, and ncurses realised that shift with an insert-line: on a
24-row terminal the bottom row fell off and the prompt was drawn inside the
key legend. A frame whose shape never changes costs one line and cannot do
that; `render` is asserted to keep the same row count and the same `CONTROLS`
row index across every prompt state.

The full-screen view uses a command palette modeled after coding-agent CLIs.
`/help` lists named forms of every key (`/next`, `/auto`, `/go`, `/hold`,
`/stand`, `/damp`, and the selection commands). Commands intentionally accept
no free-form arguments: `Ctrl-D` remains the emergency action during command
entry. `/diagnose` is the sole agent command. It runs Codex or Claude Code as
a bounded, non-persistent subprocess with read-only repository access and
gives it the current snapshot plus recent console errors. It may explain and
suggest safe checks; it cannot edit files, execute robot commands, or dispatch
lifecycle actions. Agent use is explicit per diagnosis and can be disabled
with `--diagnostic-agent off` when runtime telemetry must not leave the host.

`ec lifecycle run <job.yaml> --until <STATE>` is the scripted form for tests
and for CI-style loopback rehearsals, exit code 0 only if the target state
was reached and no fault was recorded. The job YAML carries the start pose
(`motion@frame`, bundle default, or explicit qpos), ramp seconds, tolerances,
`end_state`, and whether the run is `rehearsal` (hoist + ramp) or `eval`
(teleport).

### 5.1 How a session runs

The plant starts from `examples/g1_plant.yaml` and an MJCF, independently of
the session's tracker bundle:

```bash
pixi run -e native ec lowlevel plant examples/g1_plant.yaml \
  --model assets/latent_playkit/model/g1_29dof_rev_1_0.xml \
  --network lo --vendor --hoist --dds-domain 51
```

`--hoist` hangs the robot with its feet `--hoist-clearance` metres (0.10 by
default) off the floor, the same thing the operator is asked to do with the
real strap before `PRECHECK`.

Its nominal pose and simulated vendor gains belong to the robot config.
The session still selects a policy bundle for the tracker and a reference
for the planner. Those choices do not reconfigure the plant. Initial-pose
arrays use the plant config's joint order; recorded state includes names
for explicit remapping when scoring against a reference.

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

The console and the tracker share one process on purpose: `Ctrl-D` reaches the
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
4. Lower the hoist until the feet carry the weight. Press `l`. (In sim the
   `l` key pays the plant's strap out for you and the gate waits for the
   measured clearance to reach the floor.)
5. `n` → `POSE_MATCH_VERIFIED` prints the per-joint table and writes
   `pose_match.json`. `n` → `POLICY_COMMAND_FRESH` prints probe latencies.
   `n` → `PRIMED`.
6. `g` → `BLEND_IN` → `RUNNING`. The status line shows ticks and reference
   frame. `Ctrl-D` at any moment damps.
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

### SONIC v1.1 compatibility

The native runtime supports the exported G1 reference contract used for
SONIC v1.1: `encoder_state_interface: joint_qpos_qvel_anchor_ori`,
`macro_anchor_mode: robot_heading`, `state_dim: 64`, ten frames at
`macro_frame_stride: 5`, and `encoder_trigger: every_control_tick`.
Reference arrays must include joint velocities. The oracle worker streams
raw poses and velocities; C++ selects the requested frames and packs the
640-value encoder input. Synthetic native tests cover packing, stride,
per-tick encoding, and independence from the initial IMU heading.

**Rehearsed on the plant, 2026-09-03.** The exported v1.1 bundle
(`assets/models/controller/sonic_v1_1`, encoder parity 0.0, policy parity
3.34e-6 against the torch actor) completed the whole ladder with
`examples/lifecycle_sim_sonic_v1_1.yaml`: 504 control ticks, no runtime or
writer fault, pose match within 0.080 rad, IMU tilt 3.05 deg at the match and
1.15 deg at HOLD, ending in `VENDOR_RESTORED`. The boot heading the fixed
anchor absorbed was -88.5 deg. Two things had to change first, and both were
contract errors rather than tuning:

- **The oracle horizon was fixed at 30 frames.** The control thread encodes
  from offset `o` out to `o + (window_frames - 1) * stride` strictly inside
  the chunk in hand, and `o` reaches `hold_steps` on the tick a new chunk is
  due. Stride 5 spans 45 frames, so 30 could never work and even the exact
  sum is one frame short. The worker now sizes a reply from the bundle
  (`--horizon 0`) and adds one hold of slack, so a late reply is a deadline
  miss instead of a `command contract` fault mid-run.
- **The reference tree must carry `qvel`.** The playkit copy on disk declared
  it in its manifest and did not ship the array. It is pinned now.

This is closed-loop behaviour on the plant, not numerical parity with
NVIDIA's release weights in their own stack.

This is support for a matching **exported EC bundle**, not direct loading of
NVIDIA's release directory. The local `fsq64_sonic_4500m` bundle is a
lab-trained checkpoint with a different reference contract. It is not the
official SONIC v1.1 checkpoint.

The official release is `nvidia/GEAR-SONIC/sonic_v1_1`, checked against Hub
revision `6733128a3d8a523b1418b06bca3cdf61c8b0987f`. Its
[observation configuration](https://huggingface.co/nvidia/GEAR-SONIC/blob/6733128a3d8a523b1418b06bca3cdf61c8b0987f/sonic_v1_1/observation_config.yaml)
declares a 1,751-value multimode encoder input, a 994-value decoder input,
and a 64-value token. The G1 reference uses ten frames at step 5; other
modalities have their own cadence. Its encoder, decoder, and observation
configuration must remain matched; see the
[NVIDIA model card](https://nvlabs.github.io/GR00T-WholeBodyControl/model_card.html).

Making the official release selectable here still requires an export or
adapter for the G1 encoder branch, exact decoder observation ordering and
history, the matching action/joint/gain contract, provenance hashes, and
parity traces. The current tests do not establish numerical parity or
closed-loop performance for NVIDIA's release weights. The Python
`ec lowlevel run` path is also separate from this native encoder support.

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
  tilt and it drifted over. A full initial-orientation alignment also erased
  the initial tilt error. The anchor now uses a constant yaw-only offset,
  `heading(q_ref_start) * inverse(heading(q_imu_start)) * q_imu_current`.
  This maps the robot into the reference world while preserving absolute
  tilt and subsequent turns. The heading is recaptured at every runtime start
  (probe, go, and retake); position stays at the selected reference start.
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
  the weight, the rope only catches a drop and steadies the tilt at half
  gain), slack (paid out over 3 s). Lowering with the strap fully released
  toppled the held robot.
- **A held robot falls if nothing holds the strap (2026-09-05).** `HOLD`
  freezes the joints on the last commanded target, which is a stiff PD hold
  and not balance, and the strap was paid out at `BLEND_IN` for the run. In a
  console session the robot stood for a few seconds after the motion ended,
  toppled, and the writer's fall guard (`state_fault_reason` 64, pelvis past
  60°) damped it — which reads as "it went limp at the end of the motion"
  rather than as a fall. On hardware the operator hooks the hoist here, which
  is what the rung asks for; in sim nothing did, because the `H` ack is
  auto-acknowledged and never reached the gantry. `HOLD` now takes the load
  itself when a gantry is present and the acks are automatic, and the
  loopback test sits in `HOLD` for five seconds and requires the fall guard
  to stay clear.
- **Hoisted has to mean off the floor (2026-09-05).** The plant reset the
  pelvis to a fixed 0.76 m and welded the strap there. The nominal crouch
  touches the floor at 0.757 m, so "hoisted" was 3 mm of clearance, and a
  start pose from the reference stands 0.795 m tall: the ramp drove the feet
  3.5 cm through the floor, `POSE_SETTLED` and `POSE_MATCH_VERIFIED` graded a
  loaded robot (tracking error pinned at 0.27 rad on the waist through the
  whole ladder, and the pose match failed on it), and `LOWERED` was a no-op
  because the 5 cm drop landed on feet that were already carrying. The plant
  now measures the smallest distance from any robot geom to the floor
  (`mj_geomDistance`, every 10 ms) and hangs the robot at
  `--hoist-clearance` (default 0.10 m) on reset, whatever the pose; `lower`
  pays the strap out past that height at 0.05 m/s, because a target that
  drops 15 cm in one step is a 1.4 m/s free fall onto the ankles; and `hoist`
  from a robot standing on the floor winches it back to the same clearance,
  so a retake ramps with the feet clear exactly as the first episode did. The
  measurement is served in the plant's status RPC as `foot_clearance`, and
  `PRECHECK` and `LOWERED` gate on it — 5 cm clear to start the ladder (the
  ramp needs 4 cm of leg extension; the strap is still winching up when the
  first centimetre arrives) and 2 mm to call the feet loaded. In sim the
  plant proves what the operator confirms by eye on hardware, where no such
  number exists and the `H` / `l` acks remain the evidence.
- **The reference waits for the strap, too (2026-09-03).** Slack used to
  begin at RUNNING, the same tick the reference clock started. The lateral
  spring is 20 kN/m, so for the first three seconds of every episode the
  pelvis was tied to its start: the stationary `hurry_idle` never noticed,
  and `injured_R_leg_turn_walk_360_001_A069` faulted at 2.4 s on an ankle
  roll with root xy never having moved. The strap is now paid out as the
  policy blends in, and `BLEND_IN` holds frame 0 until the plant's status
  reports `hoist_gain` 0, measured rather than timed, before it releases the
  reference. The wait is recorded in `lifecycle.jsonl`; on hardware there is
  no status and nothing to wait for. The tracker's tick budget grows by the
  release window (`hoist_release_seconds * control_hz`), so `ticks` in a job
  stays the number of reference frames the operator asked for: before that,
  a 350-tick job played 198 frames.
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

## Reference deployment update (2026-09-09)

See [reference_deployment.md](reference_deployment.md) for the implemented
unified BONES collection, endpoint screening, derived stance/bridge references,
`arm`/`play` split, final `STAND_HOLD`, content-based rehearsal matching, and
measured simulation results. A pinned raw frame is not proof of balance; the
new examples use a stationary prefix covering the encoder window.


### Startup gate revision (2026-09-09)

The pose and tilt tolerances now label diagnostics, rather than refusing
policy activation or playback. Active-policy playback no longer waits for
all joints to remain within 0.03 rad for 0.5 seconds. PD settling during
preparation is unchanged, as are vendor ownership, state validity, runtime
faults, first-action checks, hoist handling, and operator countdown.

This follows inspection of upstream SONIC commit
`087f9ac01d46f6d8e4d0b73c01ae64799f292a38`: `InitControl` ramps to default angles
for three seconds; `WAIT_FOR_CONTROL` checks state freshness and waits for
operator start; the keyboard `play_motion` handler enables playback without
checking first-frame pose error or joint quietness. This is a comparison of
startup behavior, not a claim that 50b has SONIC's recovery capability.

The deployment identity includes `diagnostic_pose_operator_play_v1`, so old
rehearsals cannot qualify the revised contract. New experiments use default
stance and remain separate from the previous motion-pose initialization runs.
Lifecycle completion and tracking quality must be reported separately.

The rehearsal identity also includes planner lookahead (`lead_ticks`). A
retained runtime fault disqualifies a rehearsal even if recovery restored the
vendor without any failed transition. Arming rechecks active control after
its sampling window, so a fault during sampling cannot be reported as ARMED.


### G1 shoulder-strap hoist (2026-09-09)

The previous pelvis spring and pelvis orientation servo are replaced by two
tension-only straps attached to `torso_link`, beside the neck at the upper
shoulder/back buckles. Unitree's G1 user manual describes two shoulder
suspension buckles; deployment photographs show two straps and a spreader
above the head. This does not model a rope around the neck.

The attachment coordinates are approximate model-frame positions in metres:
`(-0.035, +0.08, 0.30)` and `(-0.035, -0.08, 0.30)`. They are not manufacturer
measurements. Each strap has a 0.50 m unloaded length. A level spreader moves
at the existing winch speed. Only positive extension produces tension; the
damper uses attachment-point velocity relative to the moving spreader. Forces
act on the torso with their physical moments about its centre of mass. There
is no pelvis wrench or independent orientation servo. The articulated waist
and legs respond to the suspension naturally.

Lowering pays the spreader down so the feet take the load. Slack still releases
tension over three seconds. Plant state recordings now include spreader
positions, release gain, strap tensions, and attachment locations, allowing
videos to draw the support at the actual simulated attachment points.
The straight strap segments in replay represent the suspension line of action;
unloaded straps are not a cloth simulation.

The rehearsal identity includes `g1_shoulder_straps_v1`; old pelvis-supported
campaigns cannot qualify this revised setup. They remain historical results.

Sources: [Unitree G1 user manual](https://www.manualslib.com/manual/3693046/Unitree-G1.html)
and [photograph of the shoulder attachments](https://qiita.com/ShibataRyoichi/items/94c1948dbf91959c359e).
