# Operator manual: one episode, hoist to damp

Generated from the console's own bindings by
`scripts/build_operator_manual.py`. Do not edit by hand; run
`pixi run build-manual` after changing a key or a state.

The design and the reasoning behind each gate are in
[docs/design/robot_lifecycle.md](design/robot_lifecycle.md). This page is
what to do, in order, in front of the robot.

## Before the robot

**Hoist it clear.** Every rung up to `LOWERED` assumes the feet are
off the floor: the ramp to the start pose extends the legs by about
4 cm, and a robot whose toes are already down has that ramp pushing
against the floor instead of hanging free, which makes the pose match
grade a loaded robot. Hang it about **10 cm clear** before `PRECHECK`,
and press `H` only once it is. The plant rehearses the same 10 cm
(`--hoist-clearance`) and refuses `PRECHECK` when its measured
clearance says the robot is on the floor.

**Rehearse first.** A hardware run refuses to start until the same
bundle and motion have reached the end of the ladder against the
MuJoCo plant. The plant serves the vendor's RPCs and rejects
`rt/lowcmd` in the wrong mode, so the rehearsal exercises the code
path the robot will run; only the network interface differs.

```bash
# 1. the plant, owning the joints until ReleaseMode, robot hoisted
pixi run -e native ec lowlevel plant examples/g1_plant.yaml \
  --model assets/latent_playkit/model/g1_29dof_rev_1_0.xml \
  --network lo --vendor --hoist --hoist-clearance 0.10 --dds-domain 51
# 2. the console, against the plant
pixi run -e native ec lifecycle console examples/lifecycle_sim_sonic_v1_1.yaml \
  --enable-writes --confirm ENABLE_G1_LOWLEVEL_NON_REALTIME --allow-non-realtime
```

The rehearsal has to end cleanly: no failed gate, and a final state of
`VENDOR_RESTORED`, `VENDOR_STAND` or `STANDING`. It expires after two
weeks. Check what is on hand with `ec models list` and pull anything
missing before you start, so a fetch never happens with the robot
hanging.

## The planner

A planner is a separate process, and the tracker connects to the
mailboxes it owns, so one has to be running before `r` can build.
In oracle mode `r` starts it for you: that worker memory-maps the
reference and loads no weights, about 58 MB and no GPU. In VLA mode
it is yours to start with `p`, because that worker loads its own
checkpoint, several gigabytes on the GPU, and a lifecycle key is not
how that should begin. `--planner-autostart` starts either one.

## The ladder

Each rung is a gate the runtime checks, not a key you are trusted
with. `/next` runs the next one and prints its evidence; `/auto`
climbs until a gate fails or a rung needs you.

Rungs are numbered in the console; a tick marks one behind you and
`▸` the one you are on. The states below `PRIMED` are where a run can
end up, not rungs to climb.

| # | State | What it proves |
|---|---|---|
| 1 | `PRECHECK` | the link, the vendor and the hoist are all confirmed (feet clear of the floor) |
| 2 | `VENDOR_DAMP_CONFIRMED` | the vendor has the robot limp, read back from its own FSM |
| 3 | `SAFE_EXTERNAL_COMMAND_PRESENT` | our damp frames are on the wire, before anyone lets go |
| 4 | `USER_CONTROL_CONFIRMED` | the vendor released and our frames own the joints |
| 5 | `START_POSE_RAMP` | the joints reached the start pose without one lagging behind |
| 6 | `POSE_SETTLED` | the robot stopped moving there |
| 7 | `LOWERED` | you lowered it and it settled again with the feet loaded |
| 8 | `POSE_MATCH_VERIFIED` | the pose and the pelvis tilt match the sim start frame |
| 9 | `POLICY_COMMAND_FRESH` | the planner answers in time and its first action is sane |
| 10 | `PRIMED` | everything is verified and the robot is waiting for you |

Then `/go` blends the policy in over half a second and runs the
episode. `/hold` freezes on the last target when you want to stop
early; the budget ends it otherwise.

## Ending a run

A run ends with the robot **limp under the vendor's own damp**, so the
next trajectory climbs the whole ladder again. `Ctrl-D` puts our
kd-only frames on the wire immediately and unconditionally, then hands
the joints back once you have acknowledged the hoist. Hook the hoist
and take the load before that hand-back: `SelectMode` restarts the
vendor service in damp, and the robot is limp for about a second while
it comes back.

Recovery is available from DAMP, FAULT, HOLD, RELEASED, VENDOR_RESTORED, VENDOR_STAND.
`/stand` takes it further, to the vendor's balance stand, when you want
the robot on its feet. Our controller never stands the robot itself.

## Every key

### Safety

| Key | Command | What it does |
|---|---|---|
| Ctrl-D | `/damp` | DAMP (safety stop, then vendor damp) |
| `R` | `/reset-sim` | reset simulated robot to nominal pose |

### Choose what to run

| Key | Command | What it does |
|---|---|---|
| `o` | `/mode` | mode: oracle / vla |
| `m` | `/motion` | motion / VLA goal: next |
| `M` | `/motion-prev` | motion / VLA goal: previous |
| `t` | `/tracker` | tracker: next |
| `T` | `/tracker-prev` | tracker: previous |
| `f` | `/frame-next` | start frame +25 |
| `F` | `/frame-prev` | start frame -25 |
| `p` | `/planner` | planner: start / stop |
| `r` | `/rebuild` | rebuild tracker for selection |

### Climb the ladder

| Key | Command | What it does |
|---|---|---|
| `n` | `/next` | next: advance one state |
| `a` | `/auto` | auto-advance to PRIMED |

### Run an episode

| Key | Command | What it does |
|---|---|---|
| `g` | `/go` | go: PRIMED -> BLEND_IN -> RUNNING |
| `h` | `/hold` | hold: freeze on the last target |
| `e` | `/retake` | retake: HOLD -> link precheck -> ramp |

### Tell the console what you did

| Key | Command | What it does |
|---|---|---|
| `l` | `/lowered` | ack: lowered onto the feet |
| `H` | `/hoisted` | ack: hoist hooked, load taken |

### End the run

| Key | Command | What it does |
|---|---|---|
| `s` | `/stand` | recover to vendor stand |
| `d` | `/release` | hand back to vendor damp |
| `x` | `/abort` | abort the ladder (damp, hand back) |

### Console

| Key | Command | What it does |
|---|---|---|
| `q` | `/quit` | quit (damps, silences, restores) |

## What the console shows you

The header carries the episode, the state, whether the vendor still
owns the joints, the writer's mode, the boot heading the fixed anchor
absorbed, and the next command to type. The log colours what it says:
a passed gate reads green, a damp amber, a failure or refusal red.

`Ctrl-D` always damps, ahead of every gate and every queued key, and
works while the palette is open or the prompt has text. `SPACE` does
nothing on purpose: it is the key a hand rests on.

On a light terminal, start the console with `--theme light` (or set
`EC_TUI_THEME=light`); the default reads `COLORFGBG` and falls back
to the dark palette.

## When a gate fails

The gate prints what it measured and the state does not move. Nothing
is bypassable from the console; fix the cause or change the job. A run
that faults records the runtime's own fault code alongside the writer
counters in `lifecycle.jsonl`, so `writer damped during RUNNING` is
followed by the reason. `/diagnose` sends that snapshot and the recent
log to a read-only coding agent, which can explain but cannot act.

## Next state at a glance

| State | Type this |
|---|---|
| `NO TRACKER` | /rebuild |
| `IDLE` | /next |
| `PRECHECK` | /next |
| `VENDOR_DAMP_CONFIRMED` | /next |
| `SAFE_EXTERNAL_COMMAND_PRESENT` | /next |
| `USER_CONTROL_CONFIRMED` | /next |
| `START_POSE_RAMP` | /next |
| `POSE_SETTLED` | /lowered |
| `LOWERED` | /next |
| `POSE_MATCH_VERIFIED` | /next |
| `POLICY_COMMAND_FRESH` | /next |
| `PRIMED` | /go |
| `BLEND_IN` | wait |
| `RUNNING` | /hold |
| `HOLD` | /hoisted, then /damp |
| `DAMP` | /hoisted, then /damp |
| `FAULT` | /hoisted, then /damp |
| `RELEASED` | /next for the next episode |
| `VENDOR_RESTORED` | /next for the next episode |
| `VENDOR_STAND` | /lowered |
| `STANDING` | /next for the next episode |
