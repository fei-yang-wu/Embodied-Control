# Action01rate05 60B asynchronous rehearsal

EXPERIMENTAL smoothness ablation, not a recommended default. Pinned under
`assets/models/controller/action01rate05_60b` at Hugging Face revision
`8145990683ec09a5d3eb0fd9940d3414aa4bc6a0` of `fei-yang-wu/ec-g1-gr00t-eval-kit`.
The source checkpoint continues `action01` (action-rate L2 reward weight -0.1)
from 58,000,146,432 frames, raising the weight to -0.5, trained to
60,000,043,008 cumulative environment frames. No action-output EMA.

From the EC root:

```bash
pixi run ec models pull assets/models/controller/action01rate05_60b
pixi run -e native ec lowlevel verify-bundle assets/models/controller/action01rate05_60b
MUJOCO_GL=egl pixi run -e native python scripts/oracle_mpjpe_eval.py \
  --bundle assets/models/controller/action01rate05_60b \
  --model assets/latent_playkit/model/g1_29dof_rev_1_0.xml \
  --reference-root assets/models/reference/sonic_deployment \
  --output artifacts/action01rate05_60b_sonic/results.json \
  --artifact-root artifacts/action01rate05_60b_sonic/telemetry
```

The reference pin requires access to its private dataset. `dev` has since
merged a larger reference dataset than the 13-motion `sonic_deployment` pin
used for the measured pass below; re-running against the newer set is future
work, not done here. `lifecycle_sim_action01rate05_60b.yaml` selects this
tracker for simulated DDS lifecycle use; the measured run below uses the
native asynchronous MuJoCo oracle worker, not the DDS lifecycle or real
hardware.

**Isaac canonical clean 4096 board (matched to action01's own boards):**
SR 0.9775, successful-motion micro MPJPE-L 22.29 mm / MPJPE-G 85.62 mm.
Against action01 at the matching frame region: body jerk is roughly 25-30%
lower across every testbed4096/capability124 clean and robust row, at the
cost of ~1-3 mm higher MPJPE-L and 0.5-1.5 SR points. The action-rate penalty
trades tracking accuracy for smoothness fairly cleanly on this board.

**September 9-10 rehearsal (13-motion pinned set, single deterministic pass):**
6/13 post-hoc SONIC successes, 4/13 actual falls (both `tired_forward_lunge`
and `tired_one_leg_jumping` variants, both `macarena` variants and both
`neutral_kick` variants missed the SONIC criterion). Successful-motion
frame-weighted MPJPE-L 30.75 mm; all-motion MPJPE-L/G were 98.65/1497.64 mm
including drift and falls. No runtime faults or control-scheduler misses;
3800 zero-lead response-slot misses over the pass.

This is WORSE than the `action01_55b` precedent on the identical rig (9/13
success, 27.85 mm successful MPJPE-L): the stronger action-rate penalty
appears to cost robustness on aggressive motions specifically (lunges, jumps,
kicks), not just accuracy on the Isaac board. Treat the Isaac-board smoothness
win and the deployment-rehearsal robustness loss as two separate findings,
not a net verdict either way.

Full results are published under `evaluation/action01rate05_60b/` in the
model repository. Local telemetry and 13 videos are in the parent checkout
at `logs/rate05_60b_release_20260910/`. Videos replay every recorded control
frame at 50 fps, reference left and controller right, with independently
centered cameras. Read global metrics separately because follow cameras
conceal translation drift.

This single-pass deployment rehearsal differs from the randomized Isaac 4096
benchmark and does not establish hardware readiness. Treat this checkpoint as
one candidate to try, not a recommended default.
