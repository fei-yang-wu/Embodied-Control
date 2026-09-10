# Foot dither: where it comes from (2026-09-10)

All sim numbers are the G1 plant with the robot's measured sensor noise, SONIC PD gains
(identical in every bundle), 50 Hz control. "Jitter" is the second-difference residual
per control tick, rad. `command_target_log` is the target handed to the writer.

## The policy output is the source

Four motions (walking_quip, casual_greeting, hurry_idle, big_heavy high->low):

| Tracker | commanded ankle target jitter | p99 target step | plant ankle jitter (truth) | mean |target - joint| |
|---|---:|---:|---:|---:|
| sonic_v1_1 (all 37 clean motions) | 0.011 | 0.16 | 0.0015 | 0.20 |
| action01rate05_60b | 0.022 | 0.26 | 0.0034 | 0.26 |
| action01_55b | 0.039 | 0.42 | 0.0042 | 0.24 |
| combo_50b | 0.053 | 0.50 | 0.0050 | 0.27 |

The plant's PD (ankle kp 29, kd 1.8, ~6 Hz bandwidth) attenuates the target dither about
10:1; correlation between target step and joint step is ~0.05. The real ankle tracks more
of it: hardware sonic showed 0.0018 (big_heavy) and 0.0050 (walking_quip) against 0.0012
in sim. Our lab trackers command 2-5x SONIC's output jitter, so on the robot they dance.

## Deployment gains are not the lever

kd x1.5 (stiffness unchanged), measured noise, same motions clean under both scales:

| Tracker | clean kd1.0 | clean kd1.5 | true ankle jitter kd1.0 -> kd1.5 |
|---|---:|---:|---:|
| sonic_v1_1 | 37/43 | 37/43 | 0.0018 -> 0.0014 (-22%) |
| action01_55b | 35/43 | 27/43 (8 lost, 0 gained) | 0.0059 -> 0.0049 (-17%) |

A fifth less visible dither, and 55B loses eight motions it survived at its training gains.
The policy is tuned to its closed loop; do not change kp/kd at deployment.

## Hardware sonic, this morning

| motion | hardware ankle jitter | sim truth (measured noise) |
|---|---:|---:|
| big_heavy high->low | 0.0018 | 0.0008 |
| walking_quip | 0.0050 | 0.0012 |

Reproduce: `artifacts/noise_analysis_20260910/dither.py <sweep dir...>`; sweeps under
`artifacts/rehearsal_dither_20260910`, `rehearsal_kd15_measured_20260910`,
`rehearsal_measured_noise_20260910`.
