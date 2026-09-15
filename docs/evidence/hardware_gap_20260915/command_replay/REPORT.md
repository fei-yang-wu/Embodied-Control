# Command replay: hardware PD targets through the plant model (2026-09-15)

Tool: `ec lowlevel replay-commands <telemetry.npz> --bundle <bundle> --plant
examples/g1_plant.yaml [--states plant.states.npz]`
(`src/embodied_control/lowlevel/command_replay.py`). It rebuilds the
`MujocoDdsPlant` actuation model (position servo `gainprm=kp`,
`biasprm=(0,-kp,-kd)`, `forcerange` = effort, armature, stiff joint
`solref`/`solimp`, profile `frictionloss` / `damping` / `actuator_lag_ms`,
0.002 s timestep, 10 substeps per 50 Hz tick) and feeds it the exact PD
targets a run commanded, in 25-tick windows re-synchronised to the
recording: joints from the measured state, joint velocity by 50 Hz finite
difference, root orientation and velocities from the anchor pose log, root
height placed so the lowest sole sphere touches the floor (`--states` uses
the plant's true root instead). The first 3 ticks of every window settle
and are not scored. Metrics: leg MAE over the hip/knee/ankle joints
between the model's response and the measured response, per-joint bias,
and `amp_ratio_sim_over_hw` (std of the model response / std of the
measured response).

## Floor: a plant recording replayed through itself

| replay of post f845 plant run | leg MAE rad | leg p95 | ankle pitch amp ratio |
|---|---|---|---|
| root from `--states` (true root) | 0.0043 | 0.0114 | 1.34 / 1.34 |
| root from the anchor pose log | 0.0104 | 0.0308 | 1.38 / 1.37 |

The ankle amp ratio above 1 on the self-replay comes from the window
re-sync: the model's ankle starts each window from the measured pose and
velocity while the plant's foot was already loaded, so the first scored
ticks overshoot. Hardware runs are read against the 0.010 anchor-init
floor.

## Hardware recordings, stock profile (`examples/g1_plant.yaml`)

| recording | ticks | leg MAE | leg p95 | hip pitch amp | knee amp | ankle pitch amp | ankle bias rad |
|---|---|---|---|---|---|---|---|
| post f845, leg kinematics anchor | 1824 | 0.025 | 0.083 | 1.09 / 1.04 | 0.99 / 1.02 | 1.64 / 1.51 | +0.048 / +0.050 |
| post f845, odometry anchor | 698 | 0.039 | 0.125 | 1.02 / 0.93 | 1.04 / 1.01 | 1.97 / 2.07 | +0.076 / +0.075 |
| post f885, odometry anchor | 799 | 0.040 | 0.130 | 0.91 / 0.91 | 1.04 / 1.02 | 1.67 / 2.00 | +0.064 / +0.075 |
| c1 f80 | 727 | 0.039 | 0.152 | 0.93 / 0.96 | 1.04 / 1.00 | 1.71 / 1.98 | +0.063 / +0.062 |
| a1 f80 | 774 | 0.045 | 0.123 | 0.87 / 0.94 | 1.03 / 1.06 | 1.84 / 1.91 | +0.076 / +0.080 |
| sonic_v1_1 | 715 | 0.065 | 0.239 | 0.87 / 0.92 | 1.10 / 1.09 | 2.69 / 1.51 | +0.190 / +0.139 |

Pairs are left / right. Every hardware run sits 2.5-6.5x above the floor,
and the excess is on the ankle pitch: the model's ankle moves 1.5-2.7x more
than the hardware's for the same targets and settles 0.05-0.19 rad more
plantar-flexed. Hip and knee are within 10 % of the measured amplitude.
This is the same commanded-vs-measured gap the trip analysis measured
(hardware ankle amplitude 0.15-0.34 of command vs the plant's 0.35), now
attributed to the plant's ankle response and not to the policy's targets.

## Profile fit (`examples/g1_plant_hwfit_20260915.yaml`)

Grid over per-group `frictionloss`, `damping`, `actuator_lag_ms`, gain and
armature scales (scratch sweep, post f845 odometry as the fitting
recording, the other five as holdouts). Ankle damping 40 N m s/rad plus
hip/knee frictionloss 3 N m plus 15 ms actuator lag:

| recording | stock leg MAE | fitted leg MAE | fitted ankle amp | fitted hip amp |
|---|---|---|---|---|
| post f845, leg kinematics | 0.025 | 0.016 | 0.88 | 1.03 |
| post f845, odometry (fit set) | 0.039 | 0.029 | 1.13 | 0.99 |
| post f885, odometry | 0.040 | 0.030 | 1.13 | 0.92 |
| c1 f80 | 0.039 | 0.031 | 0.96 | 0.94 |
| a1 f80 | 0.045 | 0.033 | 1.11 | 0.92 |
| sonic_v1_1 | 0.065 | 0.038 | 1.21 | 0.89 |

Per-joint on post f845 odometry, fitted: hip pitch 0.024 / 0.021, knee
0.024 / 0.027, ankle pitch 0.042 / 0.038 rad (stock 0.087 / 0.085), ankle
bias +0.024 / +0.014 (stock +0.076 / +0.075). Adding hip/knee damping 2,
raising friction to 5 with ankle friction 3, or dropping the lag each
change the fit-set MAE by under 0.002; the ankle damping carries the cut.

## What this does and does not say

- Identical PD targets produce a 1.5-2.7x larger ankle-pitch excursion in
  the stock plant than on the robot, for every policy including SONIC.
  The fitted profile removes most of that (ankle amp 0.9-1.2, leg MAE cut
  25-40 % on all six recordings, holdouts included).
- The residual 0.016-0.038 rad against a 0.010 floor is not explained by
  this actuation model; contact, the 500 Hz servo, and the estimator state
  fed to the policy are outside the replay.
- Open-loop replay cannot say whether the fitted plant trips the policy:
  that is the closed-loop rehearsal on the fitted profile
  (`artifacts/hwfit_20260915/`).

## Closed loop on the fitted profile (`artifacts/hwfit_20260915/`, echost, 1 seed, measured noise)

| run | clean | L mm (all) | measured excess max | faults |
|---|---|---|---|---|
| post f845, walking_quip_360 | 1/1 | 19.39 | 0.000 | none |
| sonic_v1_1, walking_quip_360 | 0/1 | 22.42 | 0.088 | joint-limit guard on waist_pitch |
| post f845, all 43 | 28/43 | 30.31 | 0.096 | 10 tilt (falls), 5 joint-limit guard (hip roll x3, waist pitch, shoulder roll), 0 ankle |

Stock profile, same bundle: 39-42/43 clean, failures all ankle-pitch guard, 0-1
falls. The fitted profile moves the failure mode from the ankle stop to
falls and to hip-roll / waist limits, but the hardware trip motion itself
walks clean on it. So the fitted actuation does not reproduce the
walking_quip_360 trip; it does change which motions fail, and SONIC is
not immune (waist_pitch guard on the same motion). One seed each; the
fault set needs repeats before it says anything about a specific motion.

## Observation replay floor (`artifacts/obsreplay_20260915/`, echost, plant, stock profile)

`ec lowlevel replay-observations` on a post f845 walking_quip_360 plant
run recorded with the per-tick observation record (EC 2a6ae50), cuda
provider, blend 250 ticks:

| check | ticks | max abs | clean from tick |
|---|---|---|---|
| actor (observation -> policy -> decode vs written target) | 657 | 1.3e-7 | 250 |
| encoder (window -> encoder vs consumed latent) | 907 | 0 | 0 |
| assembly (state + command + last action -> observation) | 647 | 0 | 260 |

Tick period 20.00 ms median / 21.13 max; sensor age (tick start minus
LowState receive) 1.55 ms median / 1.72 p95 on the plant. A hardware
recording is read against these: any actor/encoder residual is inference,
any assembly residual is the observation pipeline, and the sensor-age
distribution is the state delay the policy actually saw.

## Lip lattice (`artifacts/lip_20260915/`, echost, stock profile, measured noise)

0.25 m lattice of 20 mm wide box lips clear of the start pose
(`g1_29dof_rev_1_0_lip3.xml` / `_lip5.xml`, 3 and 5 mm), walking_quip_360:

| lip | bundle | runs | clean | stumble (ends > 1 m short) |
|---|---|---|---|---|
| 3 mm | post f845 | 6 | 6 | 2 (x end 3.17, 3.41 of 4.96; ankle dither 0.035 vs 0.012) |
| 3 mm | sonic_v1_1 | 6 | 5 + 1 plant fault before ARMED | 0 |
| 5 mm | post f845 | 1 | 1 | 0 |
| 5 mm | sonic_v1_1 | 1 | 1 | 0 |

Nobody falls; the 3 mm lattice makes post stumble in 2 of 6 runs where
SONIC keeps pace in 5 of 5 walks. Stumble without a fall is the nearest
plant event to the hardware trip so far. The seed-0 run stumbled once and
walked clean once (async timing differs between 1-lane and 4-lane runs),
so the rate is over runs, not seeds.
