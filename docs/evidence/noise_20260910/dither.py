"""Attribute dither: commanded target vs measured joint vs plant truth, per episode."""
import json, sys
from pathlib import Path
import numpy as np

names = json.load(open("assets/models/controller/sonic_v1_1/manifest.json"))["action"]["isaac_joint_names"]
ANK = [i for i, n in enumerate(names) if "ankle" in n]
LEG = [i for i, n in enumerate(names) if any(k in n for k in ("hip", "knee", "ankle"))]

def res(x):
    r = x[1:-1] - 0.5 * (x[:-2] + x[2:]); return r.std(0) / np.sqrt(1.5)

def episode(run):
    t = np.load(next((run / "episodes").glob("*/telemetry.npz")))
    f = t["reference_frames"]; i = np.flatnonzero(f > 0)
    if not len(i) or "command_target_log" not in t: return None
    w = slice(i[0], i[-1] + 1)
    q, c = t["joint_position_log"][w], t["command_target_log"][w]
    states = np.load(run / "plant.states.npz")
    # Plant states are in plant-config order; telemetry is Isaac order.
    order = [list(states["joint_names"]).index(n) for n in names]
    truth = states["joint_pos"][::10][:, order]
    tl = json.load(open(run / "timeline.json")); t0 = json.load(open(run / "plant.json"))["started_at"]
    at = lambda s: int((next(e["wall"] for e in tl if e["to_state"] == s and e["ok"]) - t0) * 50)
    tr = truth[at("RUNNING"):at("DAMP")]
    rc, rq, rt = res(c), res(q), res(tr)
    n = min(len(c), len(q))
    return {"target_ankle": rc[ANK].mean(), "measured_ankle": rq[ANK].mean(), "truth_ankle": rt[ANK].mean(),
            "target_leg": rc[LEG].mean(), "truth_leg": rt[LEG].mean(),
            "target_step_p99_ankle": float(np.percentile(np.abs(np.diff(c[:, ANK], axis=0)), 99)),
            "target_offset_ankle": float(np.abs(c[:n, ANK] - q[:n, ANK]).mean())}

for root in sys.argv[1:]:
    rows = []
    for r in sorted(Path(root).glob("*/sim/seed_0/result.json")):
        if json.loads(r.read_text()).get("passed"):
            e = episode(r.parent)
            if e: rows.append(e)
    if not rows: print(root, "no rows"); continue
    m = {k: float(np.mean([e[k] for e in rows])) for k in rows[0]}
    print(f"{root}: n={len(rows)}  ankle jitter: target {m['target_ankle']:.4f}  truth {m['truth_ankle']:.4f}  measured {m['measured_ankle']:.4f} | leg: target {m['target_leg']:.4f} truth {m['truth_leg']:.4f} | ankle target step p99 {m['target_step_p99_ankle']:.3f} rad, mean |target-joint| {m['target_offset_ankle']:.3f} rad")
