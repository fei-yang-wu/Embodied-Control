# Proposed Isaac fine-tuning campaign

Date: 2026-09-15. Based on the [source audit](TRAINING_AUDIT.md) and [hardware/simulation experiments](REPORT.md). This is a proposed campaign, not an implemented or submitted training job. Numerical ranges below are initial experimental choices unless explicitly identified as existing settings; they are not measured hardware parameters.

## Decision

Continue the current post policy with controlled experiments on terrain, sensor timing, and position estimation. Keep the existing reward recipe and base domain randomization fixed initially. Use the results to select a combined fine-tune. Add calibrated actuator/contact changes when identification data become available; test larger encoder changes separately.

The immediate training objective is to retain motion tracking while improving recovery from errors in sensed state and foot contact. Exact reproduction of the hardware trip remains a separate model-identification objective; a robustness improvement does not itself identify the original cause.

## 1. Freeze a common starting point

Use `post_c1l1j1p1_f84500152320` as the first baseline because it has the most useful matched hardware evidence. Pin its original resumable training checkpoint, encoder hash, resolved environment/agent configuration, simulator/container version, and observation/action contracts. An exported inference bundle is not a complete training-resume checkpoint. If a later post checkpoint wins a fresh baseline comparison, choose it before branching and rerun every arm from that same checkpoint.

For the first campaign:

- Resume actor, critic, learned exploration parameters, and optimizer state consistently across arms. Record any state that cannot be restored.
- Keep the frozen encoder, existing actor/critic network sizes, observation normalization policy, ten-frame actor history, 50 Hz actor and encoder, and 5 ms physics step unchanged.
- Keep SONIC-matched observation noise and existing physical events.
- Keep existing command delay at 0–1 physics substeps, ±10% actuator gains, and the current reward weights. The estimator arm explicitly replaces the generic anchor perturbation path rather than applying it twice.
- Keep the existing motion sampler, data, reset distribution, and termination settings. Do not mix new rewards or optimizer changes into the first comparison.
- Export explicit every-control-tick encoder semantics and include effective runtime behavior in deployment identity before comparing EC runs.

Run a short Newton/MJWarp audit before long jobs. Record per-environment realized gains, mass, COM, contact coefficients, and delays after startup/reset and after a physics step. Verify corruption is active on actor observations, and ensure declared randomizations reach the solver. This is verification, not an assertion that current randomization is broken.

## 2. Initial four-arm comparison

| Arm | Only new mechanism | Proposed initial setup |
|---|---|---|
| B: continuation control | No new mechanism | Continue the current recipe for the same additional environment steps as each treatment. Also evaluate the untouched starting checkpoint. |
| T: terrain | Small ground-height variations | 50% flat ground, 25% small grid boxes, 25% rough patches. Start perturbed heights at 1–2 mm, then increase to at most 5 mm over the first quarter of training. |
| S: sensor timing | Delayed body gyro and joint-velocity observations | Keep position/orientation observations and action delay at baseline settings. Initially draw a shared additional gyro/dq delay per environment at reset: 0, 5, or 10 ms with probabilities 0.5, 0.25, 0.25. Hold it fixed for that episode. |
| E: estimated position | Deployed pelvis-position estimate in the live encoder | Replace true position plus generic anchor noise with a batched equivalent of the deployed leg estimator, driven by the same measured joints and orientation available to the actor. Keep command delay and other sensor timing at baseline. |

These arms isolate mechanism classes. S does not identify gyro versus joint-velocity delay individually; if it helps, split those channels in the next ablation. E tests the estimator substitution first. Add calibrated estimator residuals as a separate subsequent treatment, so an estimator-only result remains interpretable.

### T: terrain implementation details

SONIC's [public terrain definition](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/087f9ac01d46f6d8e4d0b73c01ae64799f292a38/gear_sonic/envs/manager_env/mdp/terrain.py) configures small boxes and roughness at 1–5 mm. Use that as a reference for the scale, not as evidence that this robot encountered a floor lip. The proposed 50/25/25 mixture is our experimental choice, not SONIC's exact mixture.

Measure the generated collision mesh: height quantization and the generator's noise step must actually produce the intended millimeter variations. Log how often feet encounter perturbed terrain; a robot remaining on a flat spawn platform does not exercise the treatment. Keep initial pose/reference ground alignment consistent and avoid resets that start feet embedded in a box.

Retain flat episodes for the full motion set. Use perturbed terrain on locomotion clips identified by reference motion/contact information, and record the selection rule. Do not add a foot-height observation or terrain scan that deployment lacks.

The test asks whether learning to handle small touchdown-height changes improves clearance margin and recovery. It does not model rubber compliance, which needs a separate contact-model change.

### S: sensor-timing implementation details

At the current 5 ms physics step, 0/5/10 ms are 0/1/2 physics samples. Update the sensor buffers at that rate and select their output at each 20 ms policy step. A history buffer updated only at 50 Hz cannot represent 5 ms delays.

Construct each simulated measured sample once, then retain its noise and acquisition time when delayed. Append one delivered observation per control step to the actor's existing ten-frame history. Do not redraw noise on every historical sample or advance a delay buffer multiple times because actor and critic request observations separately. Reset only the affected environment's buffers and use initialized current measurements instead of zero/stale prior-episode samples.

Isaac Lab's documented [observation manager](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/managers/observation_manager.html) applies modifiers, noise, clipping/scaling, then history updates. A modifier that delays truth before built-in noise is not necessarily equivalent to delaying an already measured sensor sample. Implement and test the intended order in the installed training version. The critic/reward/termination paths may retain privileged true state.

Do not add an invented velocity low-pass at the same time. Once hardware channel filtering is measured, test that filter with the correct acquisition rate as a separate change. Keep 20 ms extra delay as an initial held-out stress condition, not a claim that the robot has that delay. The earlier 40 ms whole-state failure should not become the default training environment.

If implementation capacity is temporarily limited, a separate command-delay-only comparison can use the existing `env.actions.joint_pos.delay_substeps_max=2` override. This expands command delay to 0/5/10 ms; it does not replace S or simulate sensor lag. Do not combine it silently with S.

### E: estimator implementation details

Port the estimator mathematically into a batched training implementation rather than calling one CPU estimator separately for every environment. Match FK geometry, lowest-contact selection, stance switching, heading alignment, and reset behavior against recorded native-estimator traces. Feed measured q and orientation consistently to both the estimator and actor. Keep reference progression unchanged; only the live robot anchor used by the encoder changes.

Retain true state for rewards and privileged critic inputs. The actor/encoder should have access to the estimates that deployment can provide. Reset reference and estimator origins together, without leaking true translation during a rollout.

Simulation's estimator can still be overly accurate when its geometry matches the simulated feet. Measure its residuals instead of assuming the substitution creates realistic errors. As a later estimator-residual ablation, model persistent displacement-scale error and contact-conditioned drift, using measurements to set distributions. Before those measurements, diagnostic sweeps may test nominal scale, 0.8/1.2 displacement scale, and the previously tested 0.6 stress case. Apply scale to displacement from episode start, not global environment coordinates. These are stress conditions, not calibrated deployment distributions.

## 3. Training budget and selection

Proposed screening budget: **2 billion additional environment steps per arm**, one matched seed, with snapshots at +0.5B, +1B, and +2B. Four arms cost 8B steps total. The control must receive the same budget so continued training alone is not mistaken for a treatment effect. A short negative result is not proof the mechanism cannot help; inspect learning curves and failed motions.

Evaluate every snapshot against:

- The full 43-motion EC catalog under the same corrected runtime and nominal plant.
- Disjoint reference clips for training-side generalization; keep these separate from the catalog used for hardware diagnosis.
- Fixed terrain layouts, sensor-delay draws, and estimator perturbations, including held-out settings. Use ten fixed evaluation seeds per stochastic condition for screening, and more if outcomes are ambiguous.
- Both native MuJoCo and Isaac evaluation. A Newton-only improvement can exploit a backend-specific behavior.

Keep each treatment's evaluation conditions fixed while it trains. Log completion/falls, modeled sole/toe clearance during reference swing, unexpected swing contact, touchdown timing, knee-extension error, pelvis tilt, action/target step p95/p99, and torque saturation. Align comparisons by reference frame, not wall-clock time. Treat torque-model diagnostics as simulator outputs until hardware calibration exists.

Proposed promotion guardrails:

- No new nominal catalog failures on motions the continuation control completes.
- Nominal leg tracking MAE within 5% of the continuation control, unless a separately reviewed tradeoff is clearly worthwhile.
- Improved completion or reduced scuff/hesitation events under the relevant held-out perturbations, without larger command spikes or persistent crouching.
- Improvement repeated across seeds; a smoother video or lower aggregate reward loss is insufficient.

Continue the best single arm and the best combined treatment to **10B total additional steps**, with at least three training seeds and a matched continuation control. Only combine mechanisms supported by the screening results; this avoids assuming all treatments help. New hardware results remain necessary before calling a candidate a hardware improvement.

## 4. Second-stage additions

### Identified actuator and contact dynamics

When synchronized hardware state, target, torque estimate, and timing become available, fit and validate joint response under known loading. Add friction, viscous damping, torque-speed limits, servo dynamics, and contact compliance only through verified Newton/MJWarp mappings. Randomize around identified uncertainty and evaluate held-out traces. The current data do not justify claiming 6 Nm hip/knee friction or a 140 ms motor stall as the real hardware model.

### Reference-aware anti-scuff reward

If T improves recovery but toe scuffs remain, test a small additional reward term on locomotion clips during reference mid-swing only. Penalize insufficient lowest-sole/toe clearance or unexpected contact there. Taper it off before intended touchdown, and keep intended ground contact, crouching, kicking, and nonwalking motions unaffected. Begin with a normalized contribution below roughly 1–2% of the typical tracking reward and record actual magnitudes.

Evaluate clearance and natural gait together: a policy that raises every foot excessively, keeps knees bent, or stops following the motion has not solved the task. Do not increase all action-rate, posture, and joint-tracking penalties together; those terms already exist.

### Longer or translation-independent encoder

If estimated-position sensitivity persists, train/distill an encoder based on relative future motion and heading that does not depend on accumulated global translation. Separately test a longer reference preview and reference joint velocity. The current frozen encoder is tied to its input window and semantics; changing frame stride or zeroing position channels only at deployment is not a valid fine-tune.

## First implementation priorities

1. B/T campaign configuration, terrain realization checks, and event-based evaluation metrics.
2. Physics-rate simulated sensor buffer and S campaign, with history/reset validation.
3. Batched deployed estimator, native-trace parity check, and E campaign.
4. Measured contact/actuator identification; then reward and encoder ablations if needed.

This sequencing allows training improvements to begin while the hardware model is being identified. No additional blanket noise increase, exploration reset, PD-gain change on hardware, or arbitrary long joint-command freeze is part of the initial campaign.
