# Combo 50B deployment-example rehearsal

Immutable controller and reference pins live in
`assets/models/controller/combo_50b/model.pin.json` and
`assets/models/reference/sonic_deployment/model.pin.json`.
The motion dataset is private and requires authorized HF credentials.

From EC root:

```bash
MUJOCO_GL=egl pixi run -e native python scripts/oracle_mpjpe_eval.py --bundle assets/models/controller/combo_50b --model assets/latent_playkit/model/g1_29dof_rev_1_0.xml --reference-root assets/models/reference/sonic_deployment --output artifacts/combo_50b_sonic/results.json --artifact-root artifacts/combo_50b_sonic/telemetry
```

The lifecycle_sim_combo_50b.yaml simulation example selects the walking
motion, initialized at frame 0. This is not hardware qualification.

Recorded one-pass asynchronous sweep: 7/13 post-hoc SONIC successes;
success-only MPJPE-L/G 27.90/266.00 mm. All 7604 control ticks encode;
zero faults or missed control deadlines. Zero-lead requests missed 3800
response deadlines, using buffered current references in those ticks.
The legacy no_fall field is a 0.4m height check and mislabels deep squat.

References are exact upstream deployment examples, not verified identities
of the website video motions. Private dataset evaluation/ includes all
results, telemetry and 13 reference-left/policy-right videos. Each robot
is independently centered in replay: use MPJPE-G for translation drift.
