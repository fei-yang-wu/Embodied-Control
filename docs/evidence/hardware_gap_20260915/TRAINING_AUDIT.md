# Training audit: post campaign versus SONIC

Date: 2026-09-15. Follow-up to the [independent hardware-gap study](REPORT.md).

## Conclusion

The user's recollection is correct for the core observation-noise and physical-event settings. The current post campaign inherits SONIC's ranges and adds actuator-gain, command-delay, and anchor-noise randomization. Claude's earlier statement that SONIC learned greater robustness because its training noise is 13–130 times larger than ours is unsupported and conflicts with these sources. Its newer 19:39 UTC recommendation corrects the DR inventory but still overstates evidence for an identified knee stall, friction ranges, and exploration settings.

This does not establish identical training distributions, physical behavior, or learned robustness. The actual checkpoint's resolved training configuration and the simulator's applied parameters are separate evidence requirements.

## Sources and provenance

- Training repository `dev`, checked at `3059d2835cdf10d25eaebb09a70d745a59ac1b7e`.
- Official SONIC source, checked at `087f9ac01d46f6d8e4d0b73c01ae64799f292a38`.
- Current campaign: [September 14 posture-c1l1j1-10b](https://github.com/fei-yang-wu/IsaacLab-Imitation/blob/3059d2835cdf10d25eaebb09a70d745a59ac1b7e/experiments/campaigns/2026-09-14-posture-c1l1j1-10b/campaign.yaml). It continues from the 82,000,084,992-step c1l1j1 checkpoint, targeting another 10 billion environment steps. Both stage argument lists concatenate the base, c1, and posture overrides.
- Hardware f845 bundle: checkpoint SHA-256 `37118eaf48ebafb5a08c9014796c90a65ceb0e5371b6df5406c8667996ebd686`. Its manifest records export commit `6f474d20de6c942f9d8be292675e89fe1e895647`, but `git fetch` returned “not our ref.” Its checkpoint metadata is empty. The current branch is therefore evidence of the declared recipe and current implementation, not a complete historical attestation of the exported model.
- Claude session `9095fcf2-6877-4277-b730-6df4a1284dec`: both the 19:15:56 UTC analysis and the subsequent 19:39:34 UTC training recommendations were reviewed.

Audited source hashes and bundle provenance are recorded in [training_audit.json](training_audit.json). No training run or backend instantiation was performed in this follow-up.

## What matches

The active v2 task instantiates `G1V2ObservationCfg` and `G1SonicEventCfg`. Its policy observation group enables corruption. The post campaign requests ten-frame history for all five proprioceptive/action terms and does not disable that corruption.

| Setting | Post recipe | SONIC public v1.1 recipe |
|---|---|---|
| Joint-position additive uniform noise | ±0.01 rad | Same |
| Joint-velocity additive uniform noise | ±0.5 rad/s | Same |
| Body angular-velocity additive uniform noise | ±0.2 rad/s | Same |
| Gravity-vector component additive uniform noise | ±0.05, dimensionless | Same |
| Material coefficients | Static friction 0.3–1.6; dynamic 0.3–1.2; restitution 0–0.5; 64 buckets | Same |
| Joint default-position offset | ±0.01 rad, startup | Same |
| Torso center-of-mass offset | x ±2.5 cm; y/z ±5 cm, startup | Same |
| Torso and wrist-yaw mass scale | 0.8–2.5, startup | Same |
| Velocity pushes | Every 4–6 s; x/y ±0.5 m/s, z ±0.2 m/s; roll/pitch ±0.52 rad/s, yaw ±0.78 rad/s | Same |

Sources: training [observations](https://github.com/fei-yang-wu/IsaacLab-Imitation/blob/3059d2835cdf10d25eaebb09a70d745a59ac1b7e/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/common/observations.py) and [events](https://github.com/fei-yang-wu/IsaacLab-Imitation/blob/3059d2835cdf10d25eaebb09a70d745a59ac1b7e/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/common/events.py); SONIC [actor observations](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/087f9ac01d46f6d8e4d0b73c01ae64799f292a38/gear_sonic/config/manager_env/observations/policy/local_dir_hist.yaml) and [level0_4 event composition](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/087f9ac01d46f6d8e4d0b73c01ae64799f292a38/gear_sonic/config/manager_env/events/tracking/level0_4.yaml), including its referenced term files.

The 13–130 multiplier comes from the local [September 10 noise analysis](../../../artifacts/noise_analysis_20260910/REPORT.md), comparing the plant's injected noise against a resting LowState capture. It does not compare our training noise against SONIC's. Resting measurements also do not establish dynamic sensor fidelity under impact.

## What the latest post campaign already adds

### Command delay

`delay_substeps_min=0`, `delay_substeps_max=1`. At 5 ms physics steps, each environment draws either 0 or 5 ms delay at reset and holds that delay through the episode. `EMAJointPositionAction.apply_actions` applies the previous target for the selected number of physics substeps, then the new target.

This is implemented command delay. It is not sensor delay, torque-response lag, or time-varying packet jitter. The name of the action class does not mean an EMA is enabled: its default `ema_alpha=1.0` is the identity, and this campaign does not override it.

SONIC's selected G1 model uses ordinary `ImplicitActuatorCfg` and its action term is `JointPositionActionCfg`. A delayed actuator class exists in the public source, but the inspected v1.1 configuration does not select it. Its mere presence does not establish that SONIC trained with delays.

### PD gains

Stiffness and damping scale independently over 0.9–1.1 through the startup event. This is a declared ±10% gain perturbation, not a model of joint friction, torque-speed saturation, drivetrain compliance, or motor-side filtering. The event calls the installed Isaac Lab implementation; this audit has not read back the applied gains from the campaign's actual Newton/MJWarp training process.

### Anchor estimates

The live encoder-window path calls `_jittered_live_anchor` before constructing the `robot_heading` reference frame:

- Planar Gaussian random-walk increments: 2 mm standard deviation per build, clamped to ±10 cm per coordinate.
- Independent planar uniform jitter: ±5 mm.
- On 2% of builds, an extra planar uniform offset of up to ±3 cm. This extra offset lasts that build; it does not persist as a jump in the random-walk state.
- Walk state resets at episode reset. There is no z perturbation in this method.

The input remains simulated true body position plus these offsets. It does not run the deployed contact-based estimator. Persistent displacement-scale error, contact-selection error, correlated orientation/position error, and delayed sensor channels therefore remain distinct hypotheses. This is a reason to improve the observation model, not to add anchor jitter a second time.

Implementation sources: [action delay](https://github.com/fei-yang-wu/IsaacLab-Imitation/blob/3059d2835cdf10d25eaebb09a70d745a59ac1b7e/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/mdp/actions.py), [live anchor construction](https://github.com/fei-yang-wu/IsaacLab-Imitation/blob/3059d2835cdf10d25eaebb09a70d745a59ac1b7e/source/isaaclab_imitation/isaaclab_imitation/envs/expert_data_plane.py), and [SONIC G1 actuators](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/087f9ac01d46f6d8e4d0b73c01ae64799f292a38/gear_sonic/envs/manager_env/robots/g1.py).

## Differences worth isolating

1. **Physics and contact.** The post campaign explicitly selects Newton/MJWarp; SONIC's released environment uses PhysX. Matching nominal friction and PD numbers does not establish identical effective contacts, friction combination, solver response, or joint torque. The training repository's `common/presets.py` itself records a historical contact-parity investigation, including residual multiple-contact behavior. That historical comment is a lead to remeasure in the actual installed backend, not proof of the present failure.
2. **Terrain exposure.** Post explicitly selects a plane. SONIC selects generated terrain with small boxes and roughness configured at 1–5 mm. This is a real additional source of contact variation, unlike the alleged 13–130-fold noise difference. The generator's actual realized height distribution must be measured; declared ranges alone are insufficient. The user's rubber floor is not evidence of terrain unevenness, but millimeter-scale perturbations are a useful clearance-margin ablation.
3. **Reference information and localization dependence.** The deployed post encoder uses joint positions and root pose over approximately 0.18 s; SONIC includes reference joint velocities and approximately 0.90 s preview, without pelvis translation. The independent study's position-scale ablation affected post but left SONIC's trajectory exactly unchanged. This proves a post vulnerability, not the cause of the specific trip. Longer preview or a translation-independent encoder needs matched training/distillation; changing deployment inputs alone is not a valid comparison.
4. **Policy and learning differences.** A frozen encoder, normalization, rewards, data composition, and optimizer differ too. Identical DR does not force two learned policies to respond identically to the same observation error. These should be controlled during the initial estimator/contact ablations instead of changing everything together.

Terrain sources: SONIC [v1.1 experiment](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/087f9ac01d46f6d8e4d0b73c01ae64799f292a38/gear_sonic/config/exp/manager/universal_token/all_modes/sonic_v1_1.yaml) and [terrain generator](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/087f9ac01d46f6d8e4d0b73c01ae64799f292a38/gear_sonic/envs/manager_env/mdp/terrain.py). Training [backend preset](https://github.com/fei-yang-wu/IsaacLab-Imitation/blob/3059d2835cdf10d25eaebb09a70d745a59ac1b7e/source/isaaclab_imitation/isaaclab_imitation/tasks/manager_based/imitation/config/g1/common/presets.py).

## Revised fix and fine-tuning plan

1. **Attest the existing recipe before widening it.** Recover the checkpoint's archived resolved `env`/agent config and container identity. For sampled environments, record the actual simulator mass, COM, contact material, stiffness/damping, realized delay, and actor noise. Check before and after reset and after a simulation step so a setter that fails to reach the physics engine is detected. This is an audit requirement, not a finding that a randomization is currently broken.
2. **Close the missing hardware measurements.** Synchronize raw q, dq, gyro, quaternion, torque estimate, command targets, source/receive/publish timing, and full actor/encoder inputs. Preserve action and observation histories. Replay captured inputs offline through the exported policy to determine where hardware and simulated commands first diverge; identify actuator response with measured input-output data under known loading. The existing position-only gait logs cannot separate motor friction from a changed policy command.
3. **Fine-tune four matched arms:** corrected-runtime baseline; deployed estimator plus measured channel filtering/delays; identified contact/actuator dynamics; and their combination. Keep the current base DR and reward recipe fixed initially. Add small terrain/contact-geometry perturbations as a separate clearance-margin test, with ranges justified by measurements. Evaluate the hardware event's phase, knee extension, modeled toe clearance, command-step tails, and recovery, alongside the full motion catalog.
4. **Then test structural robustness.** If localization dependence remains limiting, train or distill a translation-independent/local-relative encoder with preserved useful future motion. If preview is limiting, separately test a longer reference window and reference velocities. These are bigger changes than continuing actor fine-tuning.

The current campaign already uses action-rate weight −0.5, energy weight −1e−4, joint-limit weight −50, posture deviation −0.001, joint tracking +0.05, and a near-limit termination. Adding these again or increasing all penalties is not a new intervention. Any target-acceleration or torque-variation term should be a separate ablation after the model discrepancy is constrained.

## Judgment on Claude's conclusions

- **Supported:** the 25 Hz versus 50 Hz encoder scheduling mismatch existed, and correcting it is appropriate. It existed in both sim and hardware, so it is not a complete explanation.
- **Corrected by Claude's newer message:** the current post recipe already has SONIC's core physical DR, plus small gain and delay additions. The earlier 13–130-fold training-noise explanation should be discarded.
- **Not identified:** hardware joint friction is 2–3 times the tested value, true servo delay is the same as simulation, or raising PD gains is the only physical fix. The available tests do not isolate those causes.
- **Still open:** the particular physical disturbance that starts the hardware hesitation. Clean nominal simulation cannot exclude inadequate policy robustness to a disturbance missing from that simulation.

### Review of the newer proposed training arms

| Claude proposal | Assessment and correction |
|---|---|
| `d1`: wider gains, armature, joint friction, all-link mass, and delay together | Reasonable stress-test directions, but these ranges are not identified hardware parameters. Changing all together can test a robustness package, but cannot identify which mechanism explains the gap. First measure applied backend parameters and short controlled responses; keep pure delay, target filtering, and torque-response dynamics distinct. |
| `t1`: SONIC's small terrain variations | A justified ablation because the training recipes actually differ here. It does not prove a floor lip caused the hardware event, or that a plane-trained policy has never encountered foot scuffing under its existing pushes and state perturbations. |
| `s1`: hold one joint target for 2–7 control ticks | This induces a 40–140 ms stale command. It does not reproduce an identified motor stall, and it cannot be called an exact reproduction of the observed event. The hardware knee target itself is more flexed than sim around the hesitation. Keep such injection as a labeled failure-mode robustness test only. |
| `x1`: std initialization 0.05, std clamp, action clip 20 | The older parity wiki's initialization comparison is insufficient for this continued checkpoint. September 13 campaign notes report the e5 ancestor's learned std already at 0.043–0.092, inside [0.001, 0.5], and warn that the umbrella optimizer override is inert through Hydra. Recover current checkpoint std and resume behavior before choosing the arm. Resetting learned std, bounding it, and clipping actions are separate interventions. |
| Explicit `encoder_trigger=every_control_tick` in the bundle | Agree. Also include effective scheduling semantics in the deployment identity and validate the exported observation/action contract; a changed preset must not silently alter history or encoder inputs. |

For the exploration proposal, the ancestor measurement is a dated campaign record, not a freshly measured current checkpoint statistic. The relevant [September 13 campaign notes](https://github.com/fei-yang-wu/IsaacLab-Imitation/blob/3059d2835cdf10d25eaebb09a70d745a59ac1b7e/experiments/campaigns/2026-09-13-e5-hardware-gap-10b/campaign.yaml) supersede the September 10 wiki's use of initialization as a diagnosis. They also report short Newton smokes of c1/h1; these establish that the path ran, not that every randomized quantity reached the solver with the intended distribution.

I additionally reconstructed target-equivalent raw actions using each bundle's declared offset and scale, on frames 1–255 of the three post hardware recordings. Maximum absolute values were **8.876** (f845 leg kinematics), **5.157** (f845 vendor odometry), and **6.156** (f885 vendor odometry); none of their 22,185 joint samples exceeded ±20. Therefore an action clip at ±20 would not alter these recorded decoded targets. This does not rule out a training-time benefit from clipping unseen exploratory actions. Values are recorded in `training_audit.json`.

Passing nominal and `fric_hi` plant runs is useful regression evidence, but neither model currently reproduces the hardware event. Those two passes alone cannot establish that a new policy fixes it. Use held-out uncertainty conditions and a staged hardware comparison as well.

The follow-up corrects the proposed training explanation. It does not claim that the hardware trip has now been reproduced or that a fine-tuned replacement has been validated.
