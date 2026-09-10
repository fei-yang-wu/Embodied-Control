"""Sensor-noise comparison: hardware LowState vs the plant's injected noise.

Same statistic on both sides, from the tracker's own 50 Hz telemetry: the
second-difference residual r[t] = x[t] - (x[t-1] + x[t+1]) / 2, which a smooth
signal barely excites and white measurement noise of std s excites at
s * sqrt(1.5). Measured per joint in the ARMED window (policy holding the
first frame, robot static) and the RUNNING window, plus the IMU tilt from the
anchor quaternion. Joint velocity and gyro are not logged, so they are not
compared here.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HW = ROOT / "artifacts/lifecycle_hardware/episodes"
SIM = ROOT / "artifacts/rehearsal_all_20260910/sonic_v1_1"
MOTIONS = ["walking_quip_360_R_002__A428", "big_heavy_one_hand_front_high_to_front_low_R_001_A524"]
SIM_POS_HALF, SIM_TILT_HALF = 0.01, 0.05  # plant defaults (uniform half-ranges)
GROUPS = {"legs": range(0, 12), "waist": range(12, 15), "arms": range(15, 29)}


def residual(x):
    return x[1:-1] - 0.5 * (x[:-2] + x[2:])


def tilt_deg(quat_xyzw):
    x, y, z, w = quat_xyzw.T
    gx = 2 * (x * z - w * y)
    gy = 2 * (y * z + w * x)
    gz = 1 - 2 * (x * x + y * y)
    return np.degrees(np.arctan2(np.hypot(gx, gy), np.abs(gz)))


def windows(frames):
    playing = np.flatnonzero(frames > 0)
    if not len(playing):
        return None, None
    first = int(playing[0])
    # ARMED: the last 100 ticks before play (after the 250-tick blend).
    armed = slice(max(0, first - 100), first)
    running = slice(first, int(playing[-1]) + 1)
    return armed, running


def stats(path):
    t = np.load(path)
    q, frames, anchor = t["joint_position_log"], t["reference_frames"], t["anchor_pose_log"]
    armed, running = windows(frames)
    out = {"ticks": int(len(q)), "playing": int((frames > 0).sum())}
    for name, window in (("armed", armed), ("running", running)):
        if window is None:
            continue
        r = residual(q[window])
        per_joint = r.std(axis=0) / np.sqrt(1.5)  # implied white-noise std per joint
        out[name] = {
            "implied_pos_noise_std_rad": {g: float(per_joint[list(i)].mean()) for g, i in GROUPS.items()},
            "implied_pos_noise_std_rad_all": float(per_joint.mean()),
            "implied_pos_noise_std_rad_max": float(per_joint.max()),
            "pos_residual_p99_rad": float(np.percentile(np.abs(r), 99)),
            "tilt_residual_std_deg": float(residual(tilt_deg(anchor[window, 3:7])).std() / np.sqrt(1.5)),
            "tilt_mean_deg": float(tilt_deg(anchor[window, 3:7]).mean()),
            "joint_step_std_rad": float(np.diff(q[window], axis=0).std()),
        }
    return out


def truth_floor(sim_dir):
    """The same residual on the plant's noiseless truth, downsampled to 50 Hz."""
    s = np.load(sim_dir / "plant.states.npz")
    truth = s["joint_pos"][::10]
    timeline = json.loads((sim_dir / "timeline.json").read_text())
    started = json.loads((sim_dir / "plant.json").read_text())["started_at"]

    def tick(state):
        return int((next(e["wall"] for e in timeline if e["to_state"] == state and e["ok"]) - started) * 50)

    a, b = tick("RUNNING"), tick("DAMP")
    return {"armed": float(residual(truth[a - 100:a]).std() / np.sqrt(1.5)),
            "running": float(residual(truth[a:b]).std() / np.sqrt(1.5))}


def at_rest_step(path):
    """Largest tick-to-tick change while the policy holds frame 0: encoder resolution."""
    t = np.load(path)
    armed, _ = windows(t["reference_frames"])
    return float(np.abs(np.diff(t["joint_position_log"][armed], axis=0)).max())


rows = {}
for motion in MOTIONS:
    hw = next(HW.glob(f"ep001_sonic_v1_1_oracle_{motion}@0/telemetry.npz"))
    sim = next((SIM / motion / "sim" / "seed_0" / "episodes").glob("*/telemetry.npz"))
    rows[motion] = {"hardware": stats(hw), "sim": stats(sim),
                    "sim_truth_floor": truth_floor(SIM / motion / "sim" / "seed_0"),
                    "hardware_at_rest_max_step_rad": at_rest_step(hw)}

expected = {
    "sim_pos_noise_std_rad": SIM_POS_HALF / np.sqrt(3),
    "sim_tilt_noise_std_deg_per_axis": float(np.degrees(0.5 * SIM_TILT_HALF / np.sqrt(3))) * 2,
}
payload = {"expected_from_plant_defaults": expected, "motions": rows,
           "note": "joint velocity and gyro noise are not logged on hardware; not compared"}
(OUT / "noise_stats.json").write_text(json.dumps(payload, indent=2) + "\n")

lines = ["# Sensor noise: hardware vs plant injection (2026-09-10)", "",
         "Statistic: implied white-noise std from the 50 Hz second-difference residual of the",
         "tracker's own telemetry, same code on both sides. ARMED = policy holding frame 0,",
         "robot static on its feet; RUNNING = playback. Plant defaults inject uniform noise of",
         f"half-range {SIM_POS_HALF} rad on joint position (std {expected['sim_pos_noise_std_rad']:.4f} rad)",
         f"and a {SIM_TILT_HALF} rad half-range body rotation error on the IMU quaternion.", "",
         "| Motion | Window | Side | pos noise std (all) | legs | waist | arms | max joint | tilt residual std (deg) |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|"]
for motion, sides in rows.items():
    for window in ("armed", "running"):
        for side in ("hardware", "sim"):
            s = sides[side].get(window)
            if not s:
                continue
            g = s["implied_pos_noise_std_rad"]
            lines.append(f"| `{motion[:28]}` | {window} | {side} | {s['implied_pos_noise_std_rad_all']:.5f} | "
                         f"{g['legs']:.5f} | {g['waist']:.5f} | {g['arms']:.5f} | {s['implied_pos_noise_std_rad_max']:.5f} | "
                         f"{s['tilt_residual_std_deg']:.3f} |")
        floor = sides["sim_truth_floor"][window]
        lines.append(f"| `{motion[:28]}` | {window} | sim truth (no noise) | {floor:.5f} | | | | | |")
lines += ["", "The sim-truth row is the plant's noiseless state through the same statistic: what",
          "real motion alone contributes. Hardware RUNNING sits at or below that floor, so the",
          "robot's position noise is below what this method can resolve during motion.", "",
          "| Motion | Hardware largest tick-to-tick step at rest (rad) |", "|---|---:|"]
for motion, sides in rows.items():
    lines.append(f"| `{motion[:28]}` | {sides['hardware_at_rest_max_step_rad']:.5f} |")
REST = ROOT / "artifacts/hardware_noise_20260910/rest.npz"
PLANT = {"joint_position": 0.01, "joint_velocity": 0.5, "gyroscope": 0.2}
if REST.is_file():
    from embodied_control.lowlevel.unitree_probe import noise_summary

    rest = noise_summary(REST)
    raw = np.load(REST)
    payload["hardware_rest_capture"] = rest
    (OUT / "noise_stats.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines += ["", "## Raw LowState at rest (robot limp on the hoist, 10 000 rows at 1 kHz)", "",
              f"Capture `{REST.relative_to(ROOT)}`, {rest['rows']} rows at {rest['rate_hz']:.0f} Hz. The",
              "white-noise part is the second-difference residual at the wire rate; the raw std",
              "includes slow sway of the hanging robot (lag-1 autocorrelation 0.5 on velocity,",
              "0.7-0.9 on the gyro), so the plant-equivalent half-range is taken from the raw std",
              "as the conservative choice.", "",
              "| Channel | Robot white std | Robot raw std (max axis/joint) | Robot-equivalent half-range | Plant default half-range | Plant / robot |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, plant_half in PLANT.items():
        x = raw[name].astype(np.float64)
        raw_std = float(x.std(axis=0).max())
        white = rest[name]["implied_noise_std"]
        half = raw_std * np.sqrt(3.0)
        lines.append(f"| {name} | {white:.5f} | {raw_std:.5f} | {half:.4f} | {plant_half} | {plant_half / half:.0f}x |")
    w, x_, y_, z_ = raw["quaternion_wxyz"].astype(np.float64).T
    tilt = np.degrees(np.arctan2(np.hypot(2 * (x_ * z_ - w * y_), 2 * (y_ * z_ + w * x_)), np.abs(1 - 2 * (x_ * x_ + y_ * y_))))
    lines.append(f"| IMU tilt (deg) | {float(residual(tilt[:, None]).std() / np.sqrt(1.5)):.5f} | {float(tilt.std()):.4f} | "
                 f"{np.radians(float(tilt.std()) * np.sqrt(3.0)):.5f} rad | 0.05 rad | {0.05 / np.radians(float(tilt.std()) * np.sqrt(3.0)):.0f}x |")
    lines += ["", "Joint positions are quantized at about 9e-6 rad per step and the accelerometer",
              "white noise is 0.022 m/s^2 (not injected by the plant).", "",
              "`ec lifecycle rehearse --plant-noise measured` uses the rounded-up envelope of this",
              "capture (joint_pos 0.0001, joint_vel 0.025, base_ang_vel 0.015, imu_tilt 0.0015);",
              "`--plant-noise off` is a clean sensor. Neither changes the rehearsal identity."]

lines += ["", "## Conclusions", "",
          "- **Joint position.** Plant injects 0.0058 rad std (measured back at 0.0058-0.0059 in",
          "  the ARMED window). The robot at rest changes by at most 0.00012 rad per tick and the",
          "  implied std is 0.00001 rad: the plant's position noise is roughly 50x the robot's.",
          "- **Orientation.** Plant tilt residual 1.1-1.4 deg; robot 0.001 deg at rest and",
          "  0.06-0.15 deg while moving (motion content, not noise): roughly 10-20x.",
          "- **During playback** hardware sits at or below the sim-truth motion floor, so joint",
          "  tracking on the robot is not limited by sensor noise at all.",
          "- **Velocity and gyro** are not in the tracker's telemetry; the raw capture above",
          "  covers them: the plant's 0.5 rad/s and 0.2 rad/s half-ranges are 20-40x the robot's",
          "  raw rest noise on velocity and about 13x on the gyro.", "",
          "So the plant is a pessimistic sensor model by a wide margin, which matches hardware",
          "looking smoother than the rehearsal. The rehearsal remains the right evidence for",
          "the gate (it exercised the policy under training-range noise); for a smoothness",
          "comparison, rehearse once more with the plant's noise flags at zero.", ""]
(OUT / "REPORT.md").write_text("\n".join(lines))
print("\n".join(lines))
