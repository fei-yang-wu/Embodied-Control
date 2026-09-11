# Isaac tick replay through the EC pipeline (2026-09-10)

Question: when the same robot state and reference cursor go into the Isaac
evaluator and into the Embodied-Control deployment pipeline, do they produce
the same encoder window, latent command, observation vector, and action?

Method: `evaluate_checkpoint.py --dump_trace` (IsaacLab-Imitation) records one
environment per control step; `isaac_replay.py` here feeds only the recorded
robot state (joint positions and velocities, projected gravity, base angular
velocity, pelvis and torso world poses) and the reference cursor into EC's
`fill_robot_anchored_window`, `ObservationAssembler`, and the bundle's
TorchScript encoder and policy, then compares each stage with what Isaac
produced.

Inputs: `action01rate04_66500m` (checkpoint 66.5B, `combo_v3` export, clone
with `encoder_trigger: every_control_tick`), motion
`big_heavy_one_hand_front_high_to_front_low_R_001_A524` (rank 34945), clean
row (`--randomization none`), Newton backend, 419 steps, one episode. Trace:
`logs/isaac_replay/rate04_66500m_big_heavy_high_low.npz` in the
IsaacLab-Imitation tree. Result JSON: `rate04_big_heavy_high_low.json`.

## Result

| stage | max abs error | signal RMS |
|---|---:|---:|
| macro window, pelvis anchor (EC today) vs Isaac | 3.4e-7 | 0.547 |
| macro window, torso anchor vs Isaac | 0.49 | 0.547 |
| z: EC encoder on Isaac's window vs Isaac z | 9.5e-7 | 0.755 |
| z: EC encoder on EC pelvis window vs Isaac z | 5.0e-6 | 0.755 |
| observation vector, all six terms | 0 | 0.939 |
| action: EC policy on Isaac obs vs Isaac action | 1.4e-6 | 1.52 |
| action: EC policy on EC obs vs Isaac action | 1.4e-6 | 1.52 |

The observation comparison needs the raw joint arrays re-ordered from the
physics backend's joint order into the Isaac order recorded in the action
contract (`asset.data.joint_pos` is permuted under Newton); the replay script
does that once. Before the permutation the joint terms differed by up to
4 rad and the bundle's `default_joint_pos` by 1.03 rad; after it every term
matches exactly.

## What this settles

- The pelvis is the right anchor. `expert_anchor_body_name` defaults to
  `torso_link` in `tracking_env.py`, but the G1 configuration sets it to
  `pelvis` (`tracking_env.py:1154`) and the reference channel anchors on
  `pelvis` (`mdp/commands/reference.py:898`); EC's pelvis-anchored window is
  Isaac's window to float precision. A torso-anchored window is wrong by up to
  0.49 in rot6d and 0.044 m in anchor position.
- The bundle's actuator table (`_SONIC_ACTUATOR_TABLE` in the exporter) is
  the training contract: `Isaac-Imitation-G1-v2` uses `G1SonicRobotCfg` and
  `G1SonicActionsCfg` (hip pitch on the 7520-22 actuator, 139 N m), which is
  what the table encodes.
- EC's encoder, observation assembly (history order, reset fill, joint
  offsets, raw last action), and policy reproduce Isaac to 1e-6.

So the divergence observed on the MuJoCo plant (planar drift of 1.8 m with a
ground-truth anchor and 3.5 m with the fixed anchor, against 0.022 m
`root_pos_xy_error_m` in Isaac on the same motion) is not in the command or
observation pipeline. What remains different between the two closed loops:

1. the plant itself (MuJoCo model and contacts versus the Newton/USD asset;
   see `wiki` sim2sim notes in IsaacLab-Imitation);
2. the episode start: Isaac writes reference frame 0 with its velocity,
   `mujoco-native --mpjpe` writes the frame-0 pose at rest, the lifecycle
   rehearsal ramps from a crouch under the hoist;
3. request cadence at hold 1: a fresh reference chunk arrives every other
   tick (`deadline_misses == ticks/2` on every rehearsal and hardware run);
   with `encoder_trigger: every_control_tick` the window is re-encoded every
   tick from the chunk in hand, which removes the stale latent but not the
   one-tick-late chunk boundary;
4. the anchor translation on hardware and on the DDS plant is fixed at the
   start anchor (`fixed_initial_anchor`), while training and Isaac evaluation
   feed the true reference-minus-robot offset.

## Reproduce

```bash
# IsaacLab-Imitation root: one-environment trace of one rank
pixi run -e isaaclab python -m imitation_experiments.lowlevel.evaluate_checkpoint \
    ... --num_envs 1 --trajectory_ranks 34945 --dump_trace logs/isaac_replay/<name>.npz

# Embodied-Control root: replay it through EC
pixi run -e native python docs/evidence/isaac_replay_20260910/isaac_replay.py \
    <trace.npz> assets/models/controller/<bundle> assets/models/reference/bones \
    --motion <motion name> --json docs/evidence/isaac_replay_20260910/<name>.json
```

## Plant-side follow-up (same day)

Two more runs, both on the same motion and checkpoint, isolate the plant.

**Synchronous demo run** (`LatentPlayground.rollout`: python MuJoCo backend in
lockstep, robot placed on reference frame 0 at rest, live pelvis-anchored
window re-encoded every tick, no sensor noise, no transport). Script:
`artifacts/sync_demo_test/run_sync.py`, video
`artifacts/sync_demo_test/rate04_big_heavy_sync.mp4`.

- MPJPE-L 47.2 mm, MPJPE-G 1241 mm, joint MAE 0.150 rad; the pelvis leaves the
  reference by 0.44 m at tick 100, 1.31 m at tick 150, 1.90 m at tick 200,
  then holds 1.75 m. Joint MAE every 25 ticks: 0, 0.081, 0.128, 0.213, 0.215,
  0.205 ... — the same profile as the asynchronous `mujoco-native` run
  (0, 0.081, 0.128, 0.212, 0.213, 0.200). Timing and transport are not the
  cause.
- Isaac on the same motion: joint MAE 0, 0.081, 0.123, 0.158, 0.137, 0.122 ...
  and `root_pos_xy_error_m` 0.022 over the clip.

**Open-loop replay** (`artifacts/sync_demo_test/openloop_replay.py`): Isaac's
recorded per-tick actions decoded to joint targets with the bundle's action
contract and written to the MuJoCo plant from the same frame-0 pose.

- Mean joint response difference against Isaac: 0.0003 rad after 1 tick,
  0.0015 after 5, 0.005 after 10, 0.018 after 50, 0.088 after 100 (the
  open-loop robot then falls, as expected without feedback).
- The divergence starts in the ankle pitch joints (0.025 rad at tick 10,
  0.119 at tick 50) and waist pitch (0.078 at tick 50); every other joint is
  an order of magnitude closer. The ankle targets sit 0.4-0.6 rad away from
  the joint in both plants (the foot is flat on the ground, so the ankle
  command acts as a torque through the contact), which makes the ankle
  response a direct read-out of the contact model.

Plant differences on record: the MuJoCo model's feet are four 5 mm spheres
per foot (`g1_29dof_rev_1_0.xml`, `left_ankle_roll_link`), MuJoCo default
friction, `implicitfast` integrator, passive joint damping zeroed and armature
from the action contract; Isaac uses the USD collision geometry with the
terrain material (static/dynamic friction 1.0, event-randomized to 0.8/0.6
when domain randomization is on) on Newton/MJWarp.

Conclusion: with the command and observation pipeline shown identical, the
MuJoCo-side drift is a plant response difference that shows up first at the
ankles and waist pitch within 10-50 ticks. Closing it is plant modelling
(foot contact geometry and friction, integrator, joint passive terms), not
runtime engineering.

## Plant-side follow-up, part 2 (same day, later)

Same motion and checkpoint throughout (`big_heavy_one_hand_front_high_to_front_low`, rate04 66.5B).

| run | engine / model | result |
|---|---|---|
| Isaac eval, `physics=newton_mjwarp` | MJWarp, USD-converted G1 | MPJPE-L 15.9, xy error 0.019 m |
| Isaac eval, `physics=physx` | PhysX, same USD | MPJPE-L 17.3, xy error 0.020 m |
| EC teleport rollout | MuJoCo CPU, Unitree `g1_29dof_rev_1_0.xml` | MPJPE-L 47.2, drift 1.90 m |
| EC teleport rollout | MuJoCo CPU, Newton's own dumped MJCF (`save_to_mjcf`) | MPJPE-L 42.4, drift 1.63 m |
| EC teleport rollout, reference frame-0 velocities written in | MuJoCo CPU, Unitree XML | MPJPE-L 47.3, drift 1.91 m |

Verified equal between EC's plant and Isaac before these runs: state read-back at
the same pose (projected gravity, body-frame angular velocity, joint order —
all to 4 decimals); the 29 actuator gains, effort ranges and armature written
into the MuJoCo model against the action contract and against Newton's dumped
`<general>` actuators (0 mismatches); physics 0.005 s x 4 substeps,
`implicitfast`, gravity, joint limits enabled, passive damping and friction
loss zero; the reference cursor advancing one frame per tick.

Newton's dumped MJCF and the Unitree XML agree on total mass (33.341 kg,
per-body within 1 g), foot contact geometry (four 5 mm spheres per foot),
friction (1, 0.005, 0.0001), `solref 0.02 1`, `condim 3`, `cone pyramidal`,
`iterations 100`, `ls_iterations 50`, `impratio 1`. They differ in joint-limit
`solref` (`-10000 -10` against `0.02 1`) and collision-mesh set; neither
difference changes the EC result.

So two different Isaac engines hold the reference on this motion and MuJoCo CPU
does not, with the model, actuators, observation conventions, command pipeline
and start state matched. This is the contact-side backend gap recorded in
`presets.py` (2026-08-03): MJWarp equals stock MuJoCo in free flight and leaves
it once the feet load. The one config-unreachable residual named there
(`multiccd` disabled in MJWarp) is present in the dumped MJCF and does not
explain it either. What remains is the contact solve itself (MJWarp/PhysX
against MuJoCo CPU on sphere-plane contacts with this policy in the loop).
SONIC on the same MuJoCo plant, DDS rehearsal: 37/43 clean, planar speed
<= 0.35 m/s on this motion.

## Part 3: the difference, found and removed (same day, later)

MuJoCo-Warp stepping EC's own loop (`logs/isaac_replay/mjwarp_port/ec_loop_mjwarp.py`
in IsaacLab-Imitation) reproduced the MuJoCo-CPU drift to three decimals
(1.882 m against 1.904 m at tick 200), so the engine was never the cause. A
tick-by-tick diff of EC's closed loop against the Isaac trace from the same
start pose showed the observation and action already differing at tick 0-1
beyond the reset-velocity difference, and a second Isaac trace that records
the action term's `processed_actions` and the actuator torques gave the cause:

- Isaac writes the joint position target verbatim:
  `processed_actions = default_joint_pos + action_scale * action` to 1.3e-7,
  with the bundle's own scale and offset (inferred per joint from the trace,
  equal to the contract).
- EC clamped that target to the contract's `joint_limits_lower/upper`
  (`ActionContract.decode`, `native_tracker_core.cpp`), which the exporter
  writes as 0.9 x the joint range. On this motion Isaac commands right ankle
  pitch up to 1.271 rad where EC capped it at 0.454; 2.45% of (tick, joint)
  pairs were clipped, all ankle pitch/roll, waist pitch and one shoulder --
  the joints that led every divergence measured above.
- The applied torque in Isaac equals the implicit PD law on the unclipped
  target (max 6 N m, mean 0.06 N m residual across substep timing).

With the clamp disabled in EC's loop, MuJoCo CPU, same start: joint MAE
0 / .081 / .126 / .156 / .137 / .137 (Isaac 0 / .081 / .123 / .158 / .137 /
.122), pelvis drift max 0.11 m (was 1.91; Isaac 0.02-0.05).

Teleport eval over the 43 bones motions, clamp off, against Isaac on the 38
shared motions: EC MPJPE-L mean 20.0 mm (clamp on 27.5; Isaac 17.9), drift
> 0.5 m on 3 motions (clamp on 16; Isaac 0), per-motion |L_EC - L_Isaac| mean
2.6 mm. The three residual motions are panic_run_away (Isaac fails it too),
walk_big_dog_ff_225_stop (82.7 vs 37.5 mm) and jump_around (29.5 vs 25.6 mm).
Tables: `artifacts/sync_demo_test/teleport_rate04{,_noclip}.json`.

Decision (user, 2026-09-10): remove the clamp. "Sometimes the policy will use
a large joint target to exploit the PD control law; it does not mean the robot
will actually go to that joint limit." Changed: `ActionContract.decode` no
longer clamps; `native_tracker_core.cpp` no longer clamps; the
`joint_limits_*` fields stay in the contract for the Unitree writer's
measured-position fault and the initialization ramp check. Every DDS
rehearsal and hardware run before this change carried the clamp.

## Part 4: why the async plant / hardware path dithers more than the sync loop (2026-09-11)

Commanded-target dither (per-tick second difference, rad, mean over joints)
on rate04 66.5B, plant sensor noise OFF: the DDS lifecycle rehearsal gives
0.051 (injured turn walk) and 0.060 (big_heavy) on the ankles, the sync
teleport loop on the same plant 0.033 and 0.022, Isaac's clean row 0.025 and
0.016. Ruled out in the sync loop, each within +-10-20%: command latency
(constant 4-8 ms, jittered 2-12 ms), control-period jitter (12-28 ms), stale
state reads (p 0.1/0.3), physics step (2 vs 5 ms) and integrator (Euler vs
implicitfast), a lifecycle-style start (reference pinned 300 ticks, default
stance). The native C++ tracker reproduces Isaac per tick to 8e-6; the
controller's joint state equals plant truth to 4e-5 rad at zero phase.

The difference is the anchor the encoder window is built around. Sync and
Isaac use the live robot pelvis pose, so the window's anchor term is the true
tracking error. The DDS plant and hardware run `fixed_initial_anchor`: the
translation is frozen at the start anchor (no localization) and only the
heading is live, so the window carries `ref_anchor - START` = true error +
the robot's own drift. Freezing the anchor translation in the otherwise
identical sync loop reproduces the async numbers: injured 0.0502 (DDS
0.0514), big_heavy 0.0624 (DDS 0.0595), with MPJPE-L 27.6 / 19.8 and drift
3.0 m / 0.28 m; walking_quip 0.0346 with 4.98 m drift. The policy chases a
phantom offset, re-plans every tick, and walks off on translating motions.

The robot's measured sensor noise does not change the dither (off vs
measured within noise); the plant's training-level noise raises it to
0.077-0.082, and Isaac with its observation noise kept on
(`--keep_obs_noise`) shows the same rise (0.025 -> 0.067). The half-rate
chunk cadence is real (`deadline_misses = ticks/2`) but did not move the
dither once `encoder_trigger: every_control_tick` was on.

Consequence: to run this encoder interface on hardware as trained, the
anchor translation needs an onboard estimate (odometry), or the encoder must
be trained without the anchor translation term (SONIC's tokenizer takes joint
qpos/qvel and anchor orientation only, which is why it tolerates the frozen
anchor). Diagram: claude.ai/code/artifact/1f514408-88dd-4425-82f1-0f907f3e221b.

## Part 5: live anchor on the DDS path (2026-09-11)

The frozen anchor translation (Part 4) is replaced by a live estimate. The
lifecycle job gains `live_anchor` (default on) and
`anchor_position_source` (`auto`, `odometry`, `leg_kinematics`,
`fixed_start`). The start alignment (`fixed_initial_anchor`) and the IMU
heading are unchanged; only the translation now follows the robot:

- `odometry`: the G1 state estimator's `rt/odommodestate` (unitree_go
  `SportModeState_.position`), subscribed in `NativeUnitreeBackend`. The
  plant can serve the same message from its true pelvis
  (`ec lowlevel plant --odometry-topic`, `ec lifecycle rehearse
  --plant-odometry`). `ec lowlevel odometry-probe` reports rate, gaps, and
  motion on the wire before a hardware run.
- `leg_kinematics`: `LegOdometry` (native/ec_native/src/leg_odometry.cpp),
  forward kinematics of the G1 MJCF from the joints and the IMU
  orientation; the contact set is every sole corner within 5 mm of the
  lowest one in two consecutive ticks, the pelvis step is minus the
  per-axis median of their displacements. Against the plant's true pelvis
  on three sync rollouts (injured turn walk, walking quip, big_heavy) the
  final error is 3.3 / 4.0 / 2.3 cm on 3.3 / 5.3 / 1.3 m of path
  (0.8-1.8 %). A single stance foot picked by height with the sole mean
  drifted 5-9 % of the path; the lowest-point variant 17-58 %.
- `auto` (default): odometry when the topic is alive at capture, else leg
  kinematics when the job carries an `mjcf`, else the frozen start. The
  source is settled once per episode and reported in `writer_stats` /
  lifecycle.jsonl (`anchor_position_source`, `odometry_frames`,
  `odometry_stale_ticks`, `anchor_displacement_max`).

`live_anchor: false` anchors the window at its own first frame
(`NativePlannerConfig::AnchorSource::kExpertHeading`), training's
`expert_heading` mode: no localization at all, but a policy trained with
`robot_heading` then reads "on track" forever.

DDS lifecycle rehearsal, measured sensor noise, rate04 66.5B, injured turn
walk, one seed (artifacts/anchor_modes_*, summary in
artifacts/anchor_modes_injured_summary.json). Dither is the commanded
target's per-tick second difference over the ticks with a reference frame;
MPJPE from the plant's true state with frame-0 heading alignment;
anchor_error is the controller's anchor displacement against the plant truth:

    mode             ankle    all   MPJPE-L  MPJPE-G  drift m  anchor_err m
    fixed_start     0.0282  0.0172    17.9    416.7    1.44      4.45
    leg_kinematics  0.0194  0.0103    13.2    111.3    0.26      0.089
    plant_odometry  0.0180  0.0096    12.6     58.6    0.19      0.015
    expert_heading  0.0173  0.0092    37.8    418.1    2.02      -

Leg odometry reaches the perfect-odometry row on dither and MPJPE-L and
brings the async path to the sync / Isaac clean level (0.016-0.025 ankle).
The expert-heading row is the anchor-blind failure of Part 4's slot-0 test:
quiet feet, no drift correction. Videos: artifacts/anchor_modes_injured_4up.mp4.

Open on hardware: whether `rt/odommodestate` keeps publishing after the
lifecycle releases the vendor motion service (the unitree_mujoco README says
the go2's sport state does not); `auto` falls back to leg kinematics if it
does not. Run `ec lowlevel odometry-probe --network <NIC>` in PRECHECK.
