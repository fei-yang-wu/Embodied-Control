# Action01 11.5B (61.5B absolute) asynchronous rehearsal

EXPERIMENTAL: one candidate to try, not a recommended default. Pinned under
`assets/models/controller/action01_11_5b` at Hugging Face revision
`4a83c6ba6fa83e815f7574da59ce68a96f5864af` of
`fei-yang-wu/ec-g1-gr00t-eval-kit`. The source checkpoint continues `action01`
(action-rate L2 reward weight -0.1, unchanged from the `action01_55b`
precedent) to 61,500,162,048 cumulative frames — 6.5B further training, same
recipe, no reward change. No action-output EMA.

From the EC root:

```bash
pixi run ec models pull assets/models/controller/action01_11_5b
pixi run -e native ec lowlevel verify-bundle assets/models/controller/action01_11_5b
MUJOCO_GL=egl pixi run -e native python scripts/oracle_mpjpe_eval.py \
  --bundle assets/models/controller/action01_11_5b \
  --model assets/latent_playkit/model/g1_29dof_rev_1_0.xml \
  --reference-root assets/models/reference/sonic_deployment \
  --output artifacts/action01_11_5b_sonic/results.json \
  --artifact-root artifacts/action01_11_5b_sonic/telemetry
```

The reference pin requires access to its private dataset; this pass used the
same 13-motion `sonic_deployment` pin as the `action01_55b` and
`action01rate05_60b` precedents. `dev` has since merged a larger reference
dataset; re-running against it is future work, not done here.
`lifecycle_sim_action01_11_5b.yaml` selects this tracker for simulated DDS
lifecycle use; the measured run below uses the native asynchronous MuJoCo
oracle worker, not the DDS lifecycle or real hardware.

**Isaac canonical clean 4096 board:** SR 0.9844, successful-motion micro
MPJPE-L 21.02 mm / body jerk 159.67 m/s^3. Full raw result is under
`evaluation/action01_11_5b/canonical_clean.json`.

**September 10 rehearsal (13-motion pinned set, single deterministic pass):**
11/13 post-hoc SONIC successes, only both `macarena` variants missed the
criterion. This is BETTER than the `action01_55b` precedent on the identical
rig (9/13 success) and substantially better than the `action01rate05_60b`
ablation (6/13 success, 4/13 actual falls): continued training under the
unchanged -0.1 recipe improved deployment robustness on its own, separate
from the smoothness/accuracy tradeoff the rate-sweep arms introduce. No
runtime faults or control-scheduler misses; 3800 zero-lead response-slot
misses over the pass.

Full results are published under `evaluation/action01_11_5b/` in the model
repository. Local telemetry and 13 videos are in the parent checkout at
`logs/action01_11_5b_release_20260910/`. Videos replay every recorded control
frame at 50 fps, reference left and controller right, with independently
centered cameras. Read global metrics separately because follow cameras
conceal translation drift.

This single-pass deployment rehearsal differs from the randomized Isaac 4096
benchmark and does not establish hardware readiness.
