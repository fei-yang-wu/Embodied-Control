# Action01 55B asynchronous rehearsal

The completed action-rate -0.1 fine-tune is pinned under
`assets/models/controller/action01_55b` at Hugging Face revision
`832f85896c08494d9a229dca0bcc828182a188ac` of
`fei-yang-wu/ec-g1-gr00t-eval-kit`. The source checkpoint has 55,000,301,568
cumulative environment frames; the subsequent 10B continuation is separate.
The bundle includes the original frozen affine encoder and ten-frame actor
history, without action-output EMA. All 512 export parity observations passed.

From the EC root:

```bash
pixi run ec models pull assets/models/controller/action01_55b
pixi run -e native ec lowlevel verify-bundle assets/models/controller/action01_55b
MUJOCO_GL=egl pixi run -e native python scripts/oracle_mpjpe_eval.py \
  --bundle assets/models/controller/action01_55b \
  --model assets/latent_playkit/model/g1_29dof_rev_1_0.xml \
  --reference-root assets/models/reference/sonic_deployment \
  --output artifacts/action01_55b_sonic/results.json \
  --artifact-root artifacts/action01_55b_sonic/telemetry
```

The reference pin requires access to its private dataset. The optional
`--video-motions MOTION...` flag renders selected telemetry comparisons.
`lifecycle_sim_action01_55b.yaml` selects this tracker for simulated DDS
lifecycle use; the measured run above uses the native asynchronous MuJoCo
oracle worker, not the DDS lifecycle or real hardware.

September 9 rehearsal: 9/13 post-hoc SONIC successes, successful-motion
frame-weighted MPJPE-L 27.85 mm. All-motion MPJPE-L/G were 62.08/1949.72 mm,
including drift and failures. Both Macarena variants, mirrored neutral kick,
and mirrored one-leg jump failed. No runtime faults or control-scheduler
misses; 3800 zero-lead response-slot misses over 7604 ticks. These slot misses
are distinct from scheduler overruns. The height-only `no_fall` field labels
the deep squat as a fall despite successful tracking.

Full results are published under `evaluation/action01_55b/` in the model
repository. Local telemetry and all 13 videos are in the parent checkout at
`logs/action01_55b_release_20260909/`; `index.html` is the video gallery.
Videos replay every recorded control frame at 50 fps, reference left and
controller right, with independently centered cameras. Read global metrics
separately because follow cameras conceal translation drift.

This single-pass deployment rehearsal differs from the randomized Isaac
4096 benchmark and does not establish hardware readiness.
