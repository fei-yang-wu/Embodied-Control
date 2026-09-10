# Action01 hardware test card

Prepared against `b8c93a2` on September 10, 2026. This card prepares an
operator-supervised G1 session; it is not evidence of a hardware test.
The robot interface and target test host still need to be identified.

## What changed

The latest merge adds the immutable `action01rate05_60b` bundle pin, a
simulation job and its release report. It changes no controller runtime or
hardware safety gates. The preceding `action01_55b` release is the useful
comparison: both use the frozen encoder, ten-frame actor history, a one-tick
command hold and `lead_ticks: 0`, without an action-output EMA.

The stronger action-rate penalty in 60B improved smoothness on the reported
Isaac benchmark but reduced success on the release's 13-motion asynchronous
tracking rehearsal from 9/13 for 55B to 6/13. Those runs started at the
reference pose; they did not exercise DDS takeover, the shoulder hoist or
hardware. See the [55B release](../examples/action01_55b_sonic_deployment.md)
and [60B release](../examples/action01rate05_60b_sonic_deployment.md).

## Prepared profiles and evidence

- [55B hardware job](../examples/lifecycle_hardware_action01_55b.yaml).
- [60B hardware job](../examples/lifecycle_hardware_action01rate05_60b.yaml).
- [Preparation report](../artifacts/hardware_prep_20260910/REPORT.md), including
  repeated DDS outcomes, true-root tracking, horizontal drift and exact
  simulation/hardware identity checks.

The jobs start with the composed `hurry_idle_001_A277__stand_f42c00b9b413`
reference. Its stationary prefix covers the encoder's future window; its
stationary suffix supports policy control while the operator takes the hoist
load. Preparation ramps to the bundle's default stance, rather than copying
the newly merged simulation examples' `start_pose: motion` setting. The
paired rehearsals use this same default-stance profile.

Each job uses its own slots, no alternate tracker selections, explicit
operator arm/play, strict real-time settings and the matching-rehearsal gate.
`stand_hold_seconds: 60` is a recovery deadline, not a tested 60-second balance
guarantee. The campaign tests a two-second final hold before rehoisting.
The full rehearsal directory must accompany the jobs on another host, as
must the pinned models, reference arrays and MJCF assets.

## Before any robot command

The September 10 host check could not enable FIFO priority 80: the native
benchmark returned `realtime_configured: false`, fault 5. The login's
real-time priority limit was zero. Its tiny failed-run timing numbers are
not inference measurements. Separate best-effort benchmarks are diagnostic
only. Keep the jobs strict; resolve target-host scheduling permissions and
repeat the benchmark before attempting the hardware launch below.

```bash
pixi run -e native ec lowlevel bench-native assets/models/controller/action01_55b \
  --ticks 1000 --policy-threads 1 --lead-ticks 0 --paced \
  --cpu 2 --fifo-priority 80 --lock-memory --require-realtime
```

Require fault 0 and successful real-time configuration, and inspect actual
compute overruns and wake latency. Also verify that the DDS writer can obtain
its configured priority 90; lifecycle PRECHECK checks its setup. Use CPU
indices valid on the target host. A synthetic benchmark alone does not
qualify the complete DDS control and writer path.

On the robot host, identify the G1 interface rather than selecting the first
active Ethernet device. Set `G1_TEST_NIC` to its actual name. The following
commands subscribe to state and render a snapshot; they do not command motors:

```bash
: "${G1_TEST_NIC:?Set G1_TEST_NIC to the interface connected to the G1}"
ip link show dev "$G1_TEST_NIC"
pixi run -e native ec lowlevel check-unitree \
  --network "$G1_TEST_NIC" --dds-domain 0 --samples 1000 --timeout 10 \
  --report artifacts/hardware_link.json
pixi run -e native ec lowlevel compare-unitree-pose \
  assets/models/controller/action01_55b \
  --model assets/latent_playkit/model/g1_29dof_rev_1_0.xml \
  --network "$G1_TEST_NIC" --dds-domain 0 \
  --output artifacts/hardware_pose_action01_55b
```

Inspect CRCs, motor errors, state rate/gaps and the actual robot-to-rendered
joint correspondence. The pose rendering uses an assumed root height;
LowState does not provide measured world translation. Do not interpret it
as a root-height or balance measurement. No live probe was run during this
preparation because the robot interface was not identified.

## Operator session

Use the established SONIC session as a baseline if takeover and recovery
have not yet been verified on this robot. Test 55B before comparing 60B.
The upper-back/shoulder hoist must be attached at the robot's actual lifting
points, with an operator managing the load. Simulation attachment coordinates
are model approximations, not installation measurements.

After the host and link checks pass, this command opens the first 55B session
with writes enabled. It does not automatically advance the lifecycle:

```bash
pixi run -e native ec lifecycle console \
  examples/lifecycle_hardware_action01_55b.yaml \
  --network "$G1_TEST_NIC" --offline \
  --enable-writes --confirm ENABLE_G1_LOWLEVEL
```

For the subsequent 60B comparison, close and fully recover the 55B session,
then launch the same command with
`examples/lifecycle_hardware_action01rate05_60b.yaml`.
Use neither `--auto-ack` nor scripted `--go` for the initial hardware ladder.
The console's `/rebuild` (`r`) starts its oracle worker; a second manually
started worker is unnecessary.

| Stage | Operator action and evidence |
|---|---|
| Hoisted takeover drill | Keep the hoist carrying the robot, with feet clear. Build, acknowledge `/hoisted`, and use `/next` one state at a time through `USER_CONTROL_CONFIRMED`. Request `/damp`; verify `VENDOR_RESTORED`, no refused transition or state fault, and a silent external writer after release. |
| Hoisted PD drill | Rebuild and repeat the climb through `POSE_SETTLED` with the default stance. Inspect the ramp and settle evidence, then damp and restore while hoisted. Do not acknowledge lowered while the robot is suspended. |
| Load transfer and preparation | On a fresh climb, physically lower onto the feet, then acknowledge `/lowered`. Advance to `PRIMED`. Pose/tilt differences are recorded diagnostics; finite state, runtime guards and operator observation still matter. |
| Arm | Request `/arm` (`g`). The reference remains at frame zero while the policy is active. Confirm the robot is supported on its feet, the straps are slack and the work area is clear before play. |
| Play | Request `/play` (`G`), allow the three-second countdown, and observe the single composed clip. Stop on unexpected movement or instability. |
| Recovery | At `STAND_HOLD`, physically take the hoist load, acknowledge `/hoisted`, then `/damp`. Confirm `VENDOR_RESTORED` and a clean log before another attempt. |

`Ctrl-D` sends DAMP immediately. During ordinary recovery, take the load
before damping; a floor-standing robot can collapse under DAMP. Perform
planner-loss, state-loss and interrupt/DAMP drills separately with the robot
supported by the hoist, not during unsupported playback. Vendor FSM changes,
`ready` and `move` are writes even when the test also reads status.

Record operator observations and video with the lifecycle artifacts. The
matching rehearsal identity covers bundle, reference content, start frame,
duration and deployment settings; changing the start pose, blend, timing,
reference or thresholds requires corresponding new rehearsal evidence.
Hardware joint/IMU logs support joint and orientation checks. Physical root
tracking and MPJPE require external root measurement; never substitute the
controller's fixed-translation anchor.
