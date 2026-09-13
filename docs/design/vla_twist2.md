# VLA evaluation and hardware inference with TWIST2

Status: base-GR00T body-only TWIST2 integration smoke passed through two Docker
runtimes. Task evaluation and hardware execution remain unvalidated.
Branch: `feat/vla-twist2`, based on dev `6ae6c827001c3059178c390c3ad29005bc944f34`.

## Ownership and runtime boundaries

Embodied-Control owns open VLA evaluation, the inference-facing observation/action
contract, rollout artifacts, and hardware lifecycle integration. DexVLA owns data
preparation, training and campaign provenance. VLA inference stays in a separate
process with its model-specific dependencies. TWIST2 owns whole-body control in
its own runtime; do not install Isaac Gym or controller dependencies into EC's
light orchestration environment.

DexVLA provides TWIST2 at `external/TWIST2`, pinned to
`d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a` from
https://github.com/amazon-far/TWIST2. Resolve its path explicitly in future jobs;
do not assume a sibling path from a standalone EC checkout.

## Existing surfaces to extend

- `transport/gr00t_client.py`, `gr00t_translate.py`, `gr00t_msgpack.py`:
  existing GR00T transport; audit G1 payloads separately from LIBERO adapters.
- `transport/chunking.py` and `lowlevel/command_buffer.py`: action chunk timing.
- `lowlevel/publishers/gr00t_service.py`: inspect existing controller command
  semantics before selecting it for this dataset.
- `robot/lifecycle_job.py`, `robot/build.py`, `robot/lifecycle.py`: shared
  simulation/hardware job construction and lifecycle.
- TWIST2 `deploy_real/server_low_level_g1_sim.py` and
  `deploy_real/server_low_level_g1_real.py`: controller-side integration points.

## First integration contract

Use the G1 pipette GR00T N1.7 campaign in DexVLA as the initial checkpoint/data
source. It selects 46 state dimensions, 47 action dimensions, three images and a
15-frame action horizon. Data was collected with TWIST2 according to the user.
The declared 60 Hz recording rate is not a verified controller/inference rate.
Do not infer wire mapping from dimensions alone: verify joint order, root-command
units/frame, hand order, action normalization and timestamp semantics against the
collector and deployed controller. Select and hash the exact TWIST2 checkpoint.

## Implementation sequence

1. Specify and test the G1 observation/action mapping using recorded samples.
2. Serve the base native GR00T checkpoint in its isolated runtime; validate
   actual wire requests and decoded action chunks through EC.
3. Add a TWIST2 command bridge with explicit timestamps, command freshness and
   reset semantics. Prove it with a fake controller before simulation.
4. Run a TWIST2 simulation evaluation through EC's existing artifact contract;
   record task success definitions, latency and controller tracking metrics.
5. Reuse the inference surface for hardware through EC's existing lifecycle,
   including its prechecks and stop handling. Hardware execution is a later stage.

Native and RLinf one-step SFT runs validate training plumbing only. They do not
establish policy quality, controller compatibility or hardware readiness.

## Implementation evidence (2026-09-13)

`Gr00tObservationMapping` now supports named `state_keys` with explicit
`state_dims`, ordered `action_dims`, and `batched` camera/state inputs. Camera
history accepts (T,H,W,3); split actions must have matching (1,H,D) shapes and
finite values. Existing LIBERO mappings remain covered by transport tests.
The client decoder accepts both legacy NPY and current numeric msgpack-numpy
responses without invoking pickle. Bidirectional arrays and modality metadata
were verified against the pinned Isaac-GR00T serializer, not just EC fixtures.

The requested checkpoint is the untouched base `nvidia/GR00T-N1.7-3B`. Its
processor supports `real_g1_relative_eef_relative_joints` with wrist EEF poses,
relative arm joints, one camera with two history frames, and navigation commands.
That differs from the custom pipette full-body contract used for SFT. Replacing
its processor with the SFT processor is not a validated zero-shot policy mapping.
The dataset's six-value hands also differ from TWIST2's seven-motor Dex3 driver.
These mappings must be resolved before controller execution. The user selected an integration smoke. The existing TWIST2 scene does not
establish a pipette success task.

## Body-only smoke implementation

`sim/twist2_eval.py` is the delegated headless MuJoCo/ONNX evaluator;
`examples/twist2_gr00t_smoke.yaml` uses the existing EC runner and normalizer.
`containers/twist2_eval/Dockerfile` isolates its CPU simulator/controller;
`containers/gr00t_runtime/Dockerfile` wraps a read-only, locally provisioned native
GR00T environment in a separate GPU container. DexVLA's dated campaign contains
build/launch commands and the exact input/output limitations.

Verified run: `20260913_184314_g1_twist2_base_body_smoke_42`, two real base-model
requests, 80 command ticks plus 100 neutral ticks, finite state, final base height
0.78298 m. Artifact validation passed. Integration success is not task success.
The simulated body bridge does not execute hands, EEF or navigation outputs;
arm/waist commands are bounded and locomotion stays nominal. Synthetic observations
and paused simulation during inference are intentional smoke-test limitations.
The three bridge tests run with `python tests/test_twist2_smoke.py` in the image.
