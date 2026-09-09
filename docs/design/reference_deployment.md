# BONES references and staged playback

Implemented 2026-09-09. The two original reference exports are one BONES
collection with different preparation histories, not separate motion domains.
The local merged collection preserves all 43 names, every source array, source
pins, mirror flags, sample rates, and content hashes. Original exports remain
available so older experiments can still be reproduced.

## Build the unified collection

From the repository root, with both pinned source exports already downloaded:

```bash
pixi run -e lowlevel python scripts/merge_reference_trees.py \
  assets/models/reference/root_qpos_v1 \
  assets/models/reference/sonic_deployment \
  --output assets/models/reference/bones

pixi run -e lowlevel ec lowlevel verify-bundle \
  assets/models/controller/combo_50b \
  --reference-root assets/models/reference/bones
```

The builder refuses an existing destination, duplicate names, mismatched array
layouts, joint/body order, anchor body, or sample rate. It writes a temporary
tree and renames it only after successful completion. Files under
`assets/models/reference/` remain generated artifacts, outside Git.

`reference_arrays_manifest.json` has a `motions` entry per name. `deployable`
means both endpoints passed velocity, pelvis height, tilt, and ankle-height
symmetry checks. Missing joint velocities or foot positions, non-finite data,
and invalid quaternions fail screening. This is a kinematic label, not proof
of balance or tracking quality. Thirty of the 43 motions pass. Both macarena
variants fail end-velocity checks despite belonging to the curated export.
Training motions remain available in simulation; hardware refuses failed
screening and nonzero start frames before constructing the writer.

The existing memory-map reader and `reference_root` / `motion` job fields stay
in use. Lifecycle examples now use the unified root and `ticks: auto`, which
resolves to the number of remaining reference transitions. Integer `ticks`
still explicitly caps a run. A locally available manifest is checked when the
job loads, after relative paths are resolved, so a wrong motion/tree pair or
an exhausted start frame is a schema error.

Bundle verification optionally checks the reference's joint order, 50 Hz
sample rate, required arrays, and encoder interface/width/anchor/stride.
Training distribution is reported as **unknown**: current exported bundles do
not record their trained reference dataset IDs. No provenance is invented.

## Build a stance/bridge candidate

A paused clock alone does not make a raw motion stationary: the encoder still
looks ahead into the reference. A composed reference gives it a full stationary
window before either bridge. The bridges are generated offline; the control
loop continues consuming the existing reference protocol.

```bash
pixi run -e native python scripts/compose_reference.py \
  --reference-root assets/models/reference/bones \
  --motion hurry_idle_001_A277 \
  --bundle assets/models/controller/sonic_v1_1 \
  --model assets/latent_playkit/model/g1_29dof_rev_1_0.xml \
  --output assets/models/reference/bones_hurry_stand
```

For the pinned inputs this produces
`hurry_idle_001_A277__stand_f42c00b9b413`: 100 stance frames, a 150-frame bridge,
the 505-frame source motion, a 150-frame return bridge, and 100 final stance
frames. The name hashes the parent content, stance, durations, sample rate,
model XML, and composition recipe version. The manifest also hashes the
actual generated arrays, which is what rehearsal matching uses.

Positional bridges use cubic Hermite curves with endpoint velocities;
quaternion curves use normalized Hermite interpolation with quaternion tangent
derivatives. MuJoCo reconstructs tracked body positions and differentiates
poses into velocities in the correct free-joint convention. The original
joint/anchor poses remain inside the derived motion, while velocities are
recomputed consistently over the composed sequence. Holds must exceed the
selected encoder's lookahead.

These are **kinematic candidates**, not contact-aware plans. Interpolation
can create foot sliding, collisions, or infeasible motion. Composition does not
automatically make a training clip deployable. Each derived reference and
checkpoint must pass simulation rehearsal before hardware use.

## Operator sequence

Use `examples/lifecycle_sim_stance_hurry.yaml` against the plant on DDS domain
51. The hardware counterpart is
`examples/lifecycle_hardware_stance_hurry.yaml`; set the actual robot network
and DDS domain when using it. Both start from the bundle's default stance.
The existing raw-motion examples remain useful for comparative simulation.

1. Build the tracker and advance through the existing hoist/ownership/pose
   gates to `PRIMED`.
2. Press `g` / `/arm`. The writer blends into policy control, with the reference
   clock pinned, and enters `ARMED`.
3. After the operator is clear, press `G` / `/play`. This is the explicit
   playback acknowledgement. The simulated strap releases; the lifecycle
   rechecks feet-down evidence, joint drift and upright tilt, then counts down.
   Displayed cues include terminal bells where enabled. Faults and emergency
   damp interrupt the countdown without releasing the reference.
4. At the final stationary suffix, the lifecycle pins the clock and enters
   `STAND_HOLD`. The policy keeps running while the hoist takes the load.
   Hoist, acknowledge, then damp/hand back to the vendor.

`stand_hold_seconds: 60` bounds the final policy hold. Its timeout enters the
existing PD `HOLD`; it does not certify that PD hold as balanced. An unplayed
`ARMED` state similarly expires after `arm_timeout_seconds`. Runtime tick
budgets include waiting allowances, while actual reference progress governs
completion. Pausing at the **start** of the verified final suffix is deliberate:
waiting until reference exhaustion lets a prefetched request kill the oracle
worker and produces a stale-command fault during recovery.

Hardware has no gantry foot-clearance measurement in this integration. Its
lowering/hoist acknowledgements remain operator evidence, not foot-force
measurements. Reference screening and a quiet IMU do not replace that evidence.

## Rehearsal identity and validation

Matching now includes checkpoint identity, reference array content, selected
motion/mode/start frame, resolved tick count, and deployment settings plus the
bundle manifest. Changing a reference under the same name, switching the
tracker, or changing the deployment setup requires matching new evidence.
Repacking identical arrays under the unified root preserves their content
identity. Legacy records without these hashes cannot vouch for a new run.

On this host, independent native MuJoCo runs completed the full trajectories:

| Reference | Checkpoint | Frames | MPJPE-L | Result |
|---|---|---:|---:|---|
| raw hurry-idle | combo_50b | 504 | 6.82 mm | no fall, no runtime fault |
| stance/bridge hurry-idle | sonic_v1_1 | 1004 | 11.11 mm | no fall, no runtime fault |
| stance/bridge hurry-idle | combo_50b | 1004 | 7.64 mm | no fall, no runtime fault |

These are single deterministic simulation samples, not hardware results.
The 50b runs reported planner deadline misses (252 raw, 502 composed), with
zero scheduler deadline misses; they are not timing certification.
Reports and telemetry are under `artifacts/design_continuation/`.

The real DDS lifecycle rehearsal separately exercises default-stance ramp,
arm, an operator pause, countdown, playback, final policy hold, hoist, and
vendor restoration. Its lifecycle artifacts, rather than the direct MuJoCo
tracking reports above, are the evidence consumed by the hardware gate.

Remote Hugging Face repositories have not been merged or deleted. Local
consolidation retains their pins. A future publication can package the merged
collection without changing these source repositories. General training-data
provenance in exported bundles and contact-aware bridge planning remain
separate upstream work.

The SONIC DDS rehearsal completed all 17 transitions with zero failures,
including an armed pause, playback, final policy hold and vendor restoration
(`artifacts/design_continuation/lifecycle_sonic_v1_1/lifecycle.json`). The
50b DDS rehearsal reached `ARMED` but refused playback: joint drift did not
settle within the existing 0.03 rad / 0.5 s gate over its 15 s timeout
(maximum observed drift 0.071 rad). That failed rehearsal remains recorded
in `artifacts/design_continuation/lifecycle_combo_50b/`; its clean shutdown
does not erase the failed transition. The gate therefore does not clear 50b
for this hardware deployment. Standing-reference/controller tuning is still
needed; the threshold was not relaxed to obtain a pass.
