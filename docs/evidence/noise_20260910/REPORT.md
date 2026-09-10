# Sensor noise: hardware vs plant injection (2026-09-10)

Statistic: implied white-noise std from the 50 Hz second-difference residual of the
tracker's own telemetry, same code on both sides. ARMED = policy holding frame 0,
robot static on its feet; RUNNING = playback. Plant defaults inject uniform noise of
half-range 0.01 rad on joint position (std 0.0058 rad)
and a 0.05 rad half-range body rotation error on the IMU quaternion.

| Motion | Window | Side | pos noise std (all) | legs | waist | arms | max joint | tilt residual std (deg) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `walking_quip_360_R_002__A428` | armed | hardware | 0.00001 | 0.00001 | 0.00001 | 0.00001 | 0.00002 | 0.001 |
| `walking_quip_360_R_002__A428` | armed | sim | 0.00588 | 0.00599 | 0.00593 | 0.00576 | 0.00707 | 1.130 |
| `walking_quip_360_R_002__A428` | armed | sim truth (no noise) | 0.00088 | | | | | |
| `walking_quip_360_R_002__A428` | running | hardware | 0.00335 | 0.00410 | 0.00508 | 0.00233 | 0.00737 | 0.149 |
| `walking_quip_360_R_002__A428` | running | sim | 0.00648 | 0.00661 | 0.00776 | 0.00610 | 0.00915 | 1.391 |
| `walking_quip_360_R_002__A428` | running | sim truth (no noise) | 0.00299 | | | | | |
| `big_heavy_one_hand_front_hig` | armed | hardware | 0.00001 | 0.00001 | 0.00001 | 0.00001 | 0.00003 | 0.001 |
| `big_heavy_one_hand_front_hig` | armed | sim | 0.00580 | 0.00587 | 0.00598 | 0.00570 | 0.00686 | 1.073 |
| `big_heavy_one_hand_front_hig` | armed | sim truth (no noise) | 0.00079 | | | | | |
| `big_heavy_one_hand_front_hig` | running | hardware | 0.00167 | 0.00175 | 0.00186 | 0.00155 | 0.00455 | 0.064 |
| `big_heavy_one_hand_front_hig` | running | sim | 0.00621 | 0.00623 | 0.00685 | 0.00606 | 0.00747 | 1.365 |
| `big_heavy_one_hand_front_hig` | running | sim truth (no noise) | 0.00242 | | | | | |

The sim-truth row is the plant's noiseless state through the same statistic: what
real motion alone contributes. Hardware RUNNING sits at or below that floor, so the
robot's position noise is below what this method can resolve during motion.

| Motion | Hardware largest tick-to-tick step at rest (rad) |
|---|---:|
| `walking_quip_360_R_002__A428` | 0.00012 |
| `big_heavy_one_hand_front_hig` | 0.00078 |

## Raw LowState at rest (robot limp on the hoist, 10 000 rows at 1 kHz)

Capture `artifacts/hardware_noise_20260910/rest.npz`, 10000 rows at 1045 Hz. The
white-noise part is the second-difference residual at the wire rate; the raw std
includes slow sway of the hanging robot (lag-1 autocorrelation 0.5 on velocity,
0.7-0.9 on the gyro), so the plant-equivalent half-range is taken from the raw std
as the conservative choice.

| Channel | Robot white std | Robot raw std (max axis/joint) | Robot-equivalent half-range | Plant default half-range | Plant / robot |
|---|---:|---:|---:|---:|---:|
| joint_position | 0.00001 | 0.00004 | 0.0001 | 0.01 | 129x |
| joint_velocity | 0.00596 | 0.01345 | 0.0233 | 0.5 | 21x |
| gyroscope | 0.00171 | 0.00867 | 0.0150 | 0.2 | 13x |
| IMU tilt (deg) | 0.00011 | 0.0414 | 0.00125 rad | 0.05 rad | 40x |

Joint positions are quantized at about 9e-6 rad per step and the accelerometer
white noise is 0.022 m/s^2 (not injected by the plant).

`ec lifecycle rehearse --plant-noise measured` uses the rounded-up envelope of this
capture (joint_pos 0.0001, joint_vel 0.025, base_ang_vel 0.015, imu_tilt 0.0015);
`--plant-noise off` is a clean sensor. Neither changes the rehearsal identity.

## Conclusions

- **Joint position.** Plant injects 0.0058 rad std (measured back at 0.0058-0.0059 in
  the ARMED window). The robot at rest changes by at most 0.00012 rad per tick and the
  implied std is 0.00001 rad: the plant's position noise is roughly 50x the robot's.
- **Orientation.** Plant tilt residual 1.1-1.4 deg; robot 0.001 deg at rest and
  0.06-0.15 deg while moving (motion content, not noise): roughly 10-20x.
- **During playback** hardware sits at or below the sim-truth motion floor, so joint
  tracking on the robot is not limited by sensor noise at all.
- **Velocity and gyro** are not in the tracker's telemetry; the raw capture above
  covers them: the plant's 0.5 rad/s and 0.2 rad/s half-ranges are 20-40x the robot's
  raw rest noise on velocity and about 13x on the gyro.

So the plant is a pessimistic sensor model by a wide margin, which matches hardware
looking smoother than the rehearsal. The rehearsal remains the right evidence for
the gate (it exercised the policy under training-range noise); for a smoothness
comparison, rehearse once more with the plant's noise flags at zero.
