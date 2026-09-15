# Independent study: G1 walking hesitation and the simulation gap

Date: 2026-09-15. Motion: `walking_quip_360_R_002__A428`.

## Recommendation

Treat this as a **closed-loop transfer problem with multiple contributors**, not as an identified knee motor fault. Fix and explicitly version the known encoder scheduling mismatch; record the missing sensor/action timing and torque channels; then identify the contact/actuator response before fine-tuning against it. For the current post policy, also train against the deployed pelvis estimator instead of true position with generic additive noise.

**Training-repository follow-up:** [the source audit](TRAINING_AUDIT.md) confirms that the current post campaign already includes SONIC's base observation noise and physical randomization, plus 0–5 ms command delay, ±10% actuator gains, and additive anchor drift/jitter. Claude's earlier claim that SONIC uses 13–130 times more training noise is incorrect; its newer 19:39 UTC recommendation acknowledges the shared DR. Those ratios compare the deployment plant's injected noise with the robot's resting sensor measurements. The remaining question is whether the actual sensor/contact/actuator processes are represented and whether the resolved training backend applies the declared settings.

The specific hardware trip is **not yet reproduced**. This study establishes several useful negatives and one causal vulnerability, but cannot identify a unique physical cause from the available recordings.

The operator confirmed bare feet, a rubber floor that was not slippery, and a fully slack hoist. This makes loaded sole geometry, contact compliance, and toe catching more relevant than a low-friction slip model. It does not supply a measured friction coefficient or contact stiffness.

## What was independently checked

- Read the latest project Claude session, `9095fcf2-6877-4277-b730-6df4a1284dec`, including its code changes and calibration experiments, and the prior deployment session.
- Traced the native state subscriber, command writer, shared-memory slots, observation history, reference cursor, encoder scheduling, and simulated PD implementation.
- Recomputed four hardware/simulation comparisons from the original `telemetry.npz` files. Compared frames **1–255** across all runs, excluding paused startup and terminal rows. Every selected reference frame advances by exactly one.
- Replayed leg odometry on the **same hardware samples** used by vendor odometry, rather than comparing two different trials.
- Independently recomputed metrics from Claude's five completed calibration runs.
- Ran **22 new synchronous MuJoCo experiments**: two policies, eleven conditions, changing one factor at a time. These use the current native ONNX tracker and encoder, the same motion and action contract, and a reference-pose start.

Files: [raw-data comparison](summary.json), [counterfactual results](counterfactual.json), [source provenance](provenance.json), [event figure](event.png). Numerical results below come from these files. The original runtime had uncommitted Claude changes before this study; this study adds diagnostic scripts and evidence, with no production control changes.

## 1. The knee event is not a clean actuator-stall experiment

At the same reference frame, the hardware and simulation policies request different knee positions:

| Frame | Hardware knee measured | Hardware knee target | Old sim knee measured | Old sim knee target |
|---|---:|---:|---:|---:|
| 176 | 1.132 | 0.808 | 1.028 | 0.581 |
| 180 | 1.083 | 0.831 | 0.554 | 0.321 |
| 182 | 1.010 | 0.482 | 0.387 | 0.430 |
| 184 | 0.757 | 0.255 | 0.350 | 0.521 |

Units: radians; larger knee angle means more flexion. Hardware extension hesitates around frames 176–180, but the target also remains more flexed. The previous report's description of a knee staying at approximately 1.1 rad throughout frames 176–184 is too broad: by frame 184 it is already at 0.757 rad.

At frame 180, hardware and sim target-minus-position errors have similar magnitudes, approximately 0.253 and 0.233 rad. Their targets differ by 0.510 rad. A conclusion that the actuator cannot follow *the same request* is therefore unsupported. The policy may be reacting to earlier contact, attitude, or velocity differences. An actuator mismatch can still initiate that reaction; it has not been isolated.

This distinction matters for fine-tuning: injecting a 100–150 ms knee torque cap would manufacture one possible cause, not demonstrate the cause present on this robot.

### What the position-only torque reconstruction can and cannot tell us

`kp * (target - q) - kd * dq` estimates the requested servo torque if the gains, timing, derivative, and firmware law are correct. Here `dq` was reconstructed from 50 Hz position samples, and the target is the controller's logged target rather than a synchronized motor-side command.

- Larger reconstructed torque is evidence of larger control effort under those assumptions.
- It is not proof of larger delivered torque, friction, inertia, or a torque-speed limit.
- `tau_est - requested_torque` alone would not directly identify friction either. Friction acts in the joint dynamics; identification also requires acceleration, gravity/inertia, contact loading, and knowledge of what `tau_est` measures.
- `std(q) / std(target)` is not an actuator transfer-function estimate in a feedback-controlled, contact-changing gait. The policy changes the target spectrum and its amplitude in response to the plant.
- Cross-correlation peaks from a short periodic walk do not rule out additional sensor or actuator delay. At 50 Hz, sub-tick delay is particularly hard to resolve. A peak at the search limit is not a reliable delay estimate.

## 2. Position estimation is a real discrepancy; ground truth is still missing

I replayed the recorded joints and aligned IMU orientation through the existing native `LegOdometry`, preserving all pre-play history. The leg-kinematics hardware trial reproduces its recorded estimate to approximately 1.3 mm of endpoint displacement, which checks the replay setup.

| Hardware recording | Recorded anchor displacement | Kinematic replay of the same recording | Reference displacement |
|---|---:|---:|---:|
| f845, leg kinematics | 2.622 m | 2.624 m | 2.740 m |
| f845, vendor odometry | 1.590 m | 2.760 m | 2.887 m |
| f885, vendor odometry | 1.514 m | 2.598 m | 4.597 m |
| SONIC, vendor odometry | 1.379 m | 3.184 m | 3.855 m |

Each row uses that recording's own full advancing window; different rows end at different frames. These are endpoint displacement magnitudes, not path lengths.

**Confirmed:** the estimators disagree substantially on the same physical motion. **Unconfirmed:** which estimate is accurate. Reference agreement is not ground truth. The kinematic estimator assumes the lowest sole points are planted; it cannot independently detect shared slip, real sole deformation, or incorrect contact selection. Its geometry is the same geometry used by the simulator, which makes its simulation validation favorable by construction.

Keeping leg kinematics explicit is a reasonable current experimental choice and gives the same estimator algorithm in sim and hardware. It does not eliminate IMU use: heading and tilt still depend on the IMU, and contact estimates still depend on the physical feet.

### Why SONIC is less exposed to this particular error

The current manifests describe different encoders:

- Post: joint position plus root position/orientation, ten frames at stride 1, approximately 0.18 s from first to last reference sample; 64 latent values plus a two-value phase.
- SONIC: joint position, joint velocity, and anchor orientation, ten frames at stride 5, approximately 0.90 s from first to last sample; 64 latent values. No pelvis translation input.

Both actors run at 50 Hz and have ten frames of proprioceptive history. Their reference information is nevertheless different. SONIC's longer preview and explicit reference velocities are plausible advantages for contact timing, but have not been isolated by an architecture ablation.

The translation difference *was* isolated. In a synchronous simulation, reporting only 60% of true planar displacement increased post's endpoint position error from **0.033 m to 0.486 m**. SONIC's entire recorded trajectory remained numerically identical to its baseline because the changed position is not an input to its encoder. That establishes a causal vulnerability in post. The post run still completed; this perturbation alone did not recreate the hardware trip.

## 3. Buffer capacity is not the observed transport problem

The native transport is a latest-value shared-memory slot, not a FIFO of historical actions. There is at most one outstanding request, one pending response, and one active reference chunk. The hardware state subscriber requests queue depth 1 and writes into a latest-state snapshot. Ten-frame actor history is an input feature containing the current sample and nine earlier samples, spanning 180 ms; it does not withhold the current state for 180 ms.

Relevant source: `native/ec_native/src/shm_command_slot.hpp`, `unitree_backend.cpp`, `native_fake_runtime.cpp`, and `native_tracker_core.cpp`.

### A real scheduling bug exists separately

The old post configuration used `hold_steps=1`, `lead_ticks=0`, and `encoder_trigger=on_acceptance`. The runtime reads replies before publishing a new request. With no lookahead, it requests the next chunk only when needed, discovers that no reply exists yet, and reuses the previous latent. The reply is consumed on a later tick. The old f845 simulation records **454 encoder calls for 908 control ticks**.

Claude's current working-tree change re-encodes the active reference chunk every control tick. Its new baseline records **907 encoder calls for 907 control ticks**, while still recording **453 deadline misses**. Those misses now describe chunk renewal timing, not necessarily stale latent use. The source records the same last chunk offset of one frame. A robust fix should separate these counters.

The old bug was present in both hardware and sim. In a new synchronous ablation, holding the encoder output for two ticks changed post's leg MAE from 0.0600 to 0.0630 rad and did not reproduce the trip. This is worth fixing for training/deployment parity, but is not a complete explanation.

### Remaining timing blind spot

The historical hardware telemetry stores `tick_durations_ns`, which measures computation time, **not the control period or sensor-to-actuator latency**. Hardware f845 leg-kinematics computation p99 is 0.484 ms. LowState receive gaps were also small, approximately 2.05 ms maximum in the lifecycle log. Neither number measures delay inside sensor firmware, source timestamp age, velocity filtering, or the motor servo.

The callback stamps receipt with the current host clock. Frequent arrival of old measurements would still look fresh. Thus a large accumulating application FIFO is unsupported, but sensor/servo delay remains open.

## 4. New experiments: noise amplitude is a weak explanation; timing matters

These are exploratory synchronous simulations, not hardware-qualified lifecycle runs. Both policies start from reference frame 0 with zero velocity. Each condition changes one factor from the clean baseline. Normal simulations complete 455 steps; comparisons below use frames 1–255. Noise runs use seed 42. The 40 ms delay failures are listed separately because they terminate earlier.

| Single changed factor | Post leg MAE, rad | Post left-knee target step p99, rad | SONIC leg MAE, rad | SONIC knee target step p99, rad |
|---|---:|---:|---:|---:|
| Clean baseline | 0.0600 | 0.170 | 0.0779 | 0.189 |
| Encoder held for two ticks | 0.0630 | 0.175 | 0.0781 | 0.209 |
| Position estimate gain 0.6 | 0.0724 | 0.180 | 0.0779 | 0.189 |
| 5 cm x-position bias, ramped after frame 100 | 0.0604 | 0.174 | 0.0779 | 0.189 |
| Constant +2 degree IMU pitch bias | 0.0649 | 0.166 | 0.0769 | 0.180 |
| Gyro delayed 20 ms | 0.0677 | 0.201 | 0.0775 | 0.201 |
| Entire observed state delayed 20 ms | 0.0687 | 0.227 | 0.0834 | 0.180 |
| Measured-envelope white noise | 0.0598 | 0.170 | 0.0782 | 0.185 |
| Training-envelope white noise | 0.0596 | 0.195 | 0.0764 | 0.174 |
| Position command delayed 20 ms | 0.0739 | 0.228 | 0.0875 | 0.188 |

All rows above completed. Entire-state delay of **40 ms** caused post to cross the 0.4 m fall threshold after **113 steps**, and SONIC after **112 steps**. This establishes sensitivity to delayed feedback, not evidence of a 40 ms hardware delay. It is a different failure from the recorded late-swing hesitation.

The white-noise envelopes are `(joint position, joint velocity, gyro, orientation)` half-ranges: measured `(0.0001 rad, 0.025 rad/s, 0.015 rad/s, 0.0015 rad)`; training `(0.01 rad, 0.5 rad/s, 0.2 rad/s, 0.05 rad)`. Samples are applied at the actor's observation step, with the plant otherwise unchanged. They do not model bias drift, impact-conditioned noise, filtering, timestamp skew, or noise fed through the kinematic estimator. The label “training-envelope” refers to the deployment plant's profile: both training repositories add ±0.05 noise to gravity-vector components, whereas this harness perturbs orientation. These are different observation models; the sweep is not an exact reproduction of training noise.

**Interpretation:** larger independent noise is not equivalent to realistic sensors. The resting capture used to derive the measured envelope does not characterize dynamic gyro/velocity estimation during rubber-floor contact. Lowering training noise alone is not the recommended fix for this trip.

### The five latest asynchronous plant calibration runs

All completed. Recomputed on the common 1–255 window:

| f845 run | Leg MAE, rad | Left knee target step p99, rad |
|---|---:|---:|
| Recorded hardware, leg kinematics | 0.0901 | 0.371 |
| Old sim runtime | 0.0652 | 0.182 |
| New runtime baseline | 0.0615 | 0.163 |
| Low passive friction/damping | 0.0642 | 0.169 |
| High passive friction/damping | 0.0669 | 0.169 |
| 15 ms target low-pass time constant | 0.0697 | 0.176 |
| Low friction/damping plus 10 ms target low-pass | 0.0707 | 0.167 |

The friction profiles jointly change Coulomb friction and viscous damping, so those are not separated from each other. The sweep isolates each profile against its new baseline, but comparison to old hardware includes a cadence change. None reproduces the event in the checked frame range.

The new `actuator_lag_ms` filters the **position target**. It is not a pure delay or motor torque-response model: immediate feedback terms `-kp*q` and `-kd*dq` still act on the current simulated state. Separate command delay, sensor delay, target filtering, and torque dynamics during identification. MuJoCo's [actuation model](https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model) explicitly separates activation dynamics from force generation.

## 5. Why the current sim does not reproduce hardware

The current model is favorable in dimensions that matter at foot contact:

1. **Contact geometry and deformation:** four 5 mm contact spheres per foot are not a calibrated model of the bare foot on rubber. Changing sliding friction from 1.0 to 0.3 does not test toe catching, pad deformation, contact-patch migration, or torsional compliance. MuJoCo already has numerically soft contacts; the missing piece is *calibrated physical compliance*, not simply whether contacts are called rigid or soft.
2. **Sensor dynamics:** synthetic instantaneous samples plus independent noise do not represent separately filtered encoder velocity, gyro, and quaternion streams or impact-dependent error. The current recordings omit the channels needed to quantify these differences.
3. **Actuator dynamics:** fixed effort limits and ideal position servos do not identify real breakaway friction, load-dependent response, torque-speed/current limits, or servo measurement filtering. Nominally equal `kp`/`kd` does not imply equal closed-loop joint response.
4. **Estimated-state dependency:** post reacts to position error in its encoder. Simulation leg odometry uses the same geometry and idealized contact assumptions as the simulated plant; its accuracy does not establish hardware accuracy.
5. **Metrics:** completion, whole-motion MAE, and ankle-center height can miss a brief toe contact or knee hesitation.

For example, in the left swing around frames 160–180, ankle-center height relative to the other ankle peaks at **14.0 cm hardware / 15.3 cm sim**. The lowest modeled contact-sphere surface relative to the other foot peaks at only **5.28 / 6.16 cm**. These are still kinematic ground proxies, not measured toe clearance. A 12–14 cm ankle-height statistic should not be read as a 12–14 cm margin against tripping.

Working hypothesis: a contact/actuator/sensor mismatch changes the observed state near stance-to-swing transitions; post's feedback response amplifies it into different ankle/hip commands and a more flexed knee request. Estimator disagreement can add another disturbance. The data support this chain as a hypothesis, but do not establish which physical mismatch initiates it.

## 6. Proposed fixes, in order

### A. Finish the runtime correction and make it auditable

1. Retain actor and encoder execution at 50 Hz for the hold-1 post bundle. Fetch immutable reference data in advance, independently of how often it is encoded. A larger cache of **future reference frames** does not create the latency of a FIFO of stale actions.
2. Refill based on remaining reference coverage and measured response latency, with explicit margin. Keep one latest response and generation/frame checks.
3. Log encoder input frame, last encoded frame, actual latent age, request/response timing, and genuine uncovered-frame counts separately from chunk-renewal misses.
4. Record the effective encoder trigger and runtime contract version in the deployment identity. The old and new f845 leg-kinematics rehearsals currently have the same `deployment_sha` (`fb40cc4e…`) despite different encoder cadence: the manifest is hashed, but the new runtime overrides it. Rehearse the corrected implementation with its new identity.
5. Verify a hold-1 trace across delayed/missing replies, pause/resume, and episode restart. Existing native tests pass, but do not establish the new hardware outcome.

### B. Record enough to distinguish causes

At the next operator-controlled trial, record simultaneously:

- Raw LowState at its actual arrival rate: robot tick, host receive time, `q`, `dq`, `tau_est`, IMU quaternion, gyro, acceleration, motor flags, and available temperature/battery channels.
- The exact emitted LowCmd at writer rate: time, `q_des`, `dq_des`, `kp`, `kd`, feed-forward torque, and command sequence.
- At each actor tick: acquisition/start/end times, consumed state sequence, full observation, raw action, emitted target, encoder window/latent, reference cursor, both position estimates, and stance/contact decisions.
- A synchronized side view or external position reference, plus rubber/sole dimensions. Use actual contact evidence to label toe strike and touchdown; kinematic clearance alone cannot do this.

Keep both estimators recording on the **same trial**, even when only one drives the actor. Record an external displacement reference rather than choosing whichever estimator agrees with the motion target.

Then do two complementary offline replays:

1. **Same state into the policy:** replay the exact consumed observations through native/exported and training implementations. Replace one channel at a time with a delayed, corrected, or filtered counterpart. This locates whether the knee request follows gyro, velocity, tilt, anchor error, or history.
2. **Same commands into the plant:** initialize short 20–200 ms windows from matched hardware state and apply the actual timestamped command sequence. Fit and validate joint/contact responses on separate windows, including free swing and loaded stance. Use a separate standing-motion trial to validate parameters fitted on walking. Long open-loop replay is not an identification metric because divergence accumulates without feedback.

A simulator reproduces this issue only when it matches the **joint/contact event sequence, phase, target changes, and recovery**, with credible parameters that also predict held-out data. Making it fall under an arbitrary disturbance is insufficient.

### C. Fine-tune against the deployment loop

Use the current f845 checkpoint as the fixed diagnostic baseline; do not reuse the September 10 55B plan as though it described the present post checkpoint. The follow-up [training audit](TRAINING_AUDIT.md) identifies the September 14 `posture-c1l1j1-10b` campaign and traces its noise, action-delay, and anchor-jitter code. It already adds 0–5 ms command delay, ±10% PD gains, and bounded additive anchor drift/jitter. It also already has action-rate, energy, joint-limit, posture, and joint-tracking reward terms. The bundle's checkpoint metadata is empty, and its recorded export commit was unavailable from the remote, so the archived resolved run configuration and engine parameter readback remain needed to attest the exact checkpoint's training conditions.

Suggested ablations, equal training budget and fixed validation motions:

| Arm | Change | Question answered |
|---|---|---|
| F0 | Current checkpoint, corrected runtime | Deployment baseline |
| F1 | Replace the existing generic anchor-noise model with the deployed estimator; include contact mistakes, scale error, tilt bias, and measured channel filtering/delays | Does realistic estimated state close the gap beyond the current DR? |
| F2 | Identified actuator/contact dynamics, randomized around fitted uncertainty | Does the physical model close the gap? |
| F3 | F1 + F2 | Is the failure an interaction? |
| F4 | F3 plus small torque/target-acceleration regularization, normalized by joint capability | Can abrupt recovery commands be reduced without losing tracking? |
| F5 | Translation-independent or locally relative motion encoder, retrained/distilled with the actor | Can global localization be removed from the fast motion-tracking dependency? |

The privileged critic may still use true state; the actor must see the state available on the real robot. Distinguish white noise, correlated drift, channel filtering, occasional stale samples, and actuation delay. Initial **simulation stress tests**, not claimed hardware values, can use 0/5/10/20 ms delay and 1–3 degree attitude offsets; fit final distributions from measurements. Do not confuse policy ticks with physics substeps. Isaac Lab's [DelayedPDActuator configuration](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/actuators/actuator_pd_cfg.html) expresses delay in physics steps, and its [DCMotor implementation](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/actuators/actuator_pd.html) provides velocity-dependent effort clipping. Verify the training backend actually applies the selected model.

F5 should preserve useful relative future motion and heading information while reducing dependence on accumulated global position. Simply zeroing the current encoder's translation channels at deployment is a distribution change, not this ablation. Likewise, any action filter belongs in both training and deployment with matching history semantics and an explicit action contract.

Do not start by increasing `kd`, removing actor history, lowering all noise, or strongly penalizing action rate. Each changes the controller without identifying the missing dynamics. Earlier 55B results also showed that heavier action-rate penalties could sacrifice difficult motions. First preserve tracking, then penalize unnecessary command acceleration/torque variation, using physical joint scaling rather than the same raw-action weight for every motor.

### D. Acceptance criteria

- Full 43-motion catalog with baseline and candidate under the same runtime, plus repeated seeds for stochastic conditions and held-out motions.
- Grade actual or modeled toe clearance during mid-swing, touchdown phase, knee-extension timing, torque/saturation duty, pelvis pitch excursions, command step tails, and recovery, as well as completion/MAE.
- Test clean, measured-sensor, identified-contact, identified-actuator, and combined uncertainty conditions. A visually pleasant nominal sim is not the release criterion.
- Require event reproduction on held-out hardware traces before calling the model calibrated, and staged hardware comparison before calling the fine-tune a hardware improvement.

## Reproduction and validation

From the repository root:

```bash
pixi run -e native python docs/evidence/hardware_gap_20260915/analyze.py
python3 docs/evidence/hardware_gap_20260915/plot.py
pixi run -e native python docs/evidence/hardware_gap_20260915/counterfactual.py
```

The plot command uses this workstation's system matplotlib; the native Pixi environment does not contain it. The analysis scripts need the referenced gitignored hardware/simulation artifacts. `summary.json` stores their paths and content hashes. Counterfactual traces are included beside their summary.

Validation performed:

- All four hardware pairs and five calibration runs processed with asserted contiguous frame selection and explicit joint-name agreement.
- All 22 counterfactual simulations executed; their expected observed failures are recorded rather than discarded. The clean synchronous post baseline is close to the independent new DDS baseline (leg MAE 0.0600 vs 0.0615 rad), supporting the diagnostic harness without claiming identical initialization.
- `tests/lowlevel/test_native_core.py`, `test_native_shm.py`, and `test_tracker_and_loop.py`: **69 passed**.
- `pixi run test`: **249 passed, 7 skipped, 1 failed**. Existing `test_first_action_far_from_hold_is_refused_without_fault` expects a 3.0 rad mismatch to fail although the committed default threshold is 6.2832 rad. Both the expectation and threshold are present at `HEAD`; this study did not change them.

Limitations: one walking motion and one seed per noisy counterfactual; no new raw walking capture; no independent hardware contact/position ground truth; no training run; no hardware validation of the runtime change. These results rank experiments and fixes, not certify a unique root cause or a deployment-ready replacement policy.
