# Latent playground (z256)

A notebook-first surface for asking what a DiffSR latent means to the frozen
low-level tracker: encode reference motions into `z`, perturb `z`, run the real
tracker in MuJoCo, and watch the result.

- notebook: `notebooks/z256_latent_perturbation.ipynb`
- library: `embodied_control.lowlevel.latent`,
  `embodied_control.lowlevel.publishers.latent_perturbation`
- environment: `pixi run -e latent-lab latent-lab`

## What it is

The tracker's command is a latent: 256 continuous values from the skill
encoder plus two phase values, republished every `hold` ticks (10 ticks =
0.2 s for the reference bundle). `LatentPublisher` keeps that schedule and
takes `z` from a pluggable source:

| source | z comes from |
| --- | --- |
| `ReferenceEncoderSource` | the encoder over the reference window — the oracle z, bit-identical to the deployment path |
| `TransformedLatentSource(inner, fn)` | any source, then `fn(z, renewal_index)` |
| `ConstantLatentSource` | one fixed z, held forever, no reference at all |
| `SequenceLatentSource` | a precomputed `[K, z_dim]` list, one per renewal |

`LatentPlayground` wires a bundle, a MuJoCo plant, and a reference tree
together: `encode_motion` builds a latent bank, `rollout` runs one episode with
video, `summary` scores survival and MPJPE-L/G against the reference.

Nothing about the policy is re-implemented. The playground loads an exported
policy bundle (TorchScript tracker, TorchScript encoder, contracts, normalizer,
golden trace, provenance) and `verify_bundle` replays the golden trace before
any experiment.

Parity is a test, not a claim: `tests/lowlevel/test_latent_playground.py`
asserts `LatentPublisher(ReferenceEncoderSource(...))` publishes byte-identical
packets to the certified `OnboardEncoderPublisher`, and feeds the encoder
identical windows.

## Getting the inputs

The notebook reads one **playkit** directory: bundle, reference-array tree, and
the G1 MJCF with meshes (~50 MB unpacked, ~37 MB as a tarball). Build one from
the training repository:

```bash
external/Embodied-Control/scripts/make_latent_playkit.sh \
  --bundle    logs/policy_bundles/rollout24_gamma097_3500m \
  --reference data/bones_seed_language10_v1/reference_arrays/root_qpos_v1 \
  --mjcf      source/isaaclab_imitation/isaaclab_imitation/assets/unitree/g1_description/g1_29dof_rev_1_0.xml \
  --output    /tmp/z256_latent_playkit
```

Then, from this repository:

```bash
export EC_LATENT_PLAYKIT=/path/to/z256_latent_playkit
pixi run -e latent-lab latent-lab
```

Unpacking the kit into `assets/z256_latent_playkit` works without the variable.
Everything runs on CPU: one 300-tick episode takes about a second, plus video
rendering.

## Reference bundle

`rollout24_gamma097_3500m` — continuous z256, hold 10, `root_qpos` encoder
state (10 frames x 38 values), robot anchor, mish + layer norm, 351-value
tracker input, checkpoint `23fdd62a...` at 3.5B frames. Continuous means
unquantized; the FSQ tracker family snaps z onto a lattice and the playground
refuses those bundles rather than silently ignoring the quantizer.

## What one pass looks like

Numbers below come from a single execution of the shipped notebook on
`walk_arc_cw_start_R_slow_001_A443` (300 ticks) and the ten-motion kit. Every
one is **one deterministic MuJoCo episode, no randomization, no repeats** —
preliminary signals about the interface, not measurements of it, and MuJoCo
actuator dynamics are not the ones the policy trained in.

- Golden trace: policy 1.9e-6, encoder 1.4e-6 max abs error.
- Bank: 518 latents over 10 motions; 17 principal components carry 90% of the
  variance; no dimension is dead (all std > 1e-3).
- Baseline oracle z: survives, MPJPE-L 18.0 mm, joint MAE 0.080 rad.
- Gaussian noise at renewal, scaled by per-dimension population std: survives
  at sigma 0.25 (20.2 mm) and 0.5 (29.1 mm), falls at 1.0 and 2.0.
- Principal directions at +/-2 sd: effects are asymmetric — PC0 falls at +2 sd
  but survives at -2 sd, PC1 the other way round, PC2 survives both.
- Frozen latent (constant z, started on the pose it was encoded at): 5 of 10
  motions hold a stable attractor for 200 ticks, 5 fall — including the
  mid-clip walking latent (fell at 92 ticks).
- Blending two motions' latents: alpha 0, 0.25 and 0.5 fell; 0.75 and 1.0
  survived. A convex blend of two valid latents is not itself reliably valid.
- Rescaling z: gains 0.0 and 0.5 stay upright but track badly (85.8 and
  34.3 mm), gain 1.5 falls.

Before any of this becomes a claim, repeat across seeds and start frames, and
confirm the direction in the training simulator.

## Extending it

An experiment is one function: `TransformedLatentSource(inner, fn)` with
`fn(z, renewal_index) -> z`. Ideas the notebook leaves open: single-dimension
sweeps, republishing a stale z to measure latency tolerance, cross-motion
latent swaps with `SequenceLatentSource`, and optimising a z against a target
pose (the playground is the forward map and it is cheap).
