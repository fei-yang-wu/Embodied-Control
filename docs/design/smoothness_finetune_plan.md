# Smoothness fine-tune: ablation plan for the G1 tracker

Status: plan, ready to hand to a training session
Written: 2026-09-10
Audience: the agent or person running fine-tunes in `IsaacLab-Imitation`, and the
EC session that grades the results
Evidence: `docs/evidence/noise_20260910/{REPORT,DITHER}.md` (committed copies of
`artifacts/noise_analysis_20260910/`, which is gitignored)

## 1. What we know

Measured on 2026-09-10 with the G1 plant, the robot's own sensor-noise
envelope, and the commanded target logged per control tick. All trackers use
the same PD gains (SONIC's: legs 99/6.3, ankles 29/1.8, arms 14/0.9).

| Tracker | commanded ankle target jitter per tick | p99 target step | plant ankle motion | mean \|target − joint\| |
|---|---:|---:|---:|---:|
| sonic_v1_1 | 0.011 rad | 0.16 rad | 0.0015 | 0.20 |
| action01rate05_60b | 0.022 | 0.26 | 0.0034 | 0.26 |
| action01_55b | 0.039 | 0.42 | 0.0042 | 0.24 |
| combo_50b | 0.053 | 0.50 | 0.0050 | 0.27 |

Jitter is the second-difference residual `x[t] − (x[t−1] + x[t+1]) / 2` at
50 Hz, std over the RUNNING window, mean over the four ankle joints.

Conclusions that shape this plan:

- The dither is in the policy output. The plant's PD attenuates it 10:1; the
  real ankle attenuates it less (hardware sonic showed 2–4× the plant's
  motion), so on the robot it reads as feet vibrating.
- The lab trackers command 2–5× SONIC's output jitter. 60B's action-rate
  L2 weight −0.5 (from 55B's −0.1) cut it 45% and cost success on aggressive
  clips (13-motion release rehearsal 9/13 → 6/13).
- Deployment gains are not the lever: kd × 1.5 removes a fifth of the visible
  jitter and costs action01_55b eight motions (35/43 → 27/43). A filter
  bolted on at deployment is the same kind of change and is out.
- Training observation noise is 20–130× the robot's: ±0.5 rad/s on joint
  velocity against a measured ±0.023, ±0.2 on gyro against ±0.015, ±0.01 rad
  on joint position against ±0.0001. Under the measured envelope every
  tracker's output jitter dropped 20–40% in sim without retraining.

## 2. Baseline and budget

- Source checkpoint: `action01_model_step_55000301568.pt` from
  `logs/combo50b_action01_vs_hub_20260909/` (repo commit `4771c3c`), preset
  `combo64_history10_v1`, frozen affine encoder, ten-frame actor history,
  hold 1, no action EMA. This is `action01_55b` in EC.
- Continuation budget per arm: 2.0 B environment frames (60B used exactly
  this from 58B). Same optimizer state handling as the 60B continuation.
- Export with the same command the 55B manifest records
  (`export_policy_bundle.py --preset combo64_history10_v1 --macro-frame-stride 1
  --macro-anchor-mode robot_heading --hold-steps 1`, verify parity), so the
  reference contract stays identical and EC's rehearsal identity only changes
  through the checkpoint hash.

## 3. Arms

Every arm starts from the 55B checkpoint and changes exactly what its row
says. Weights are starting points; the sweep in §5 tunes them.

| Arm | Change | Why |
|---|---|---|
| A0 | none (55B as is) | reference row, already measured |
| A1 | action-rate L2 weight −0.1 → −0.5 (this is 60B) | reference row, already measured |
| A2 | add action-acceleration penalty `‖a_t − 2a_{t−1} + a_{t−2}‖²`, weight −0.5, rate stays −0.1 | taxes tick-to-tick alternation, leaves smooth ramps free; it is the metric we grade |
| A3 | observation noise to 4× the measured envelope: joint pos ±0.0005 rad, joint vel ±0.1 rad/s, gyro ±0.06 rad/s, orientation ±0.005 rad; rewards as 55B | stop training against noise the robot never produces |
| A4 | bound the output step: clip the per-tick change of the scaled target to ±0.15 rad per joint inside the env (before the PD, and what `last_action` reports); rewards as 55B | remove the 0.4–0.5 rad step tail the plant hides and the robot follows |
| A5 | in-loop action EMA, α = 0.4 at 50 Hz, applied before the PD and fed back as `last_action`; rewards as 55B | the policy learns against the filtered plant; deployment applies the same filter, so nothing changes between sim and robot |
| C1 | A2 + A3 | the two cheapest, orthogonal levers |
| C2 | A2 + A3 + A4 | C1 plus the tail bound |
| C3 | A2 + A3 + A5 | C1 plus in-loop filtering (A4 and A5 overlap; do not stack them) |

Order of execution: A3, A2, C1 first (they answer the main question), then
A4, A5, C2, C3. A1 and A0 need no training.

A4 and A5 change the action contract. The EC bundle manifest must carry the
filter so the deployment applies it identically: add `action.step_clip_rad`
(A4) or `action.ema_alpha` (A5) to the export, and EC's native tracker core
needs the matching one-line filter before `write_target` with `last_action`
set to the filtered value. Do not run A4/A5 on the robot until that lands in
EC and a loopback test shows the exported filter and the training filter
agree on a recorded action trace.

## 4. Metrics and gates

Grade every arm in EC, on the plant, from the exported bundle:

```bash
# on the EC workstation, bundle copied under assets/models/controller/<arm>
pixi run -e native ec lowlevel verify-bundle assets/models/controller/<arm> \
  --reference-root assets/models/reference/bones
pixi run -e native ec lifecycle rehearse assets/models/controller/<arm> all \
  --output artifacts/smooth_<arm>_measured --lanes 4 --dds-domain-base 100 --plant-noise measured
pixi run -e native ec lifecycle rehearse assets/models/controller/<arm> all \
  --output artifacts/smooth_<arm>_training --lanes 4 --dds-domain-base 100
pixi run -e native python docs/evidence/noise_20260910/dither.py artifacts/smooth_<arm>_measured
```

Three numbers per arm, from those runs:

| Metric | Source | 55B today | target |
|---|---|---:|---|
| commanded ankle jitter (measured noise) | `dither.py` | 0.039 | ≤ 0.015 (SONIC is 0.011) |
| clean motions, measured noise | `summary.json` | 35/43 | ≥ 35/43 |
| clean motions, training noise | `summary.json` | 30/43 | ≥ 28/43 |

Secondary, report but do not gate: true ankle jitter from `plant.states.npz`,
joint MAE from `episodes/*/summary.json`, and the Isaac 4096-board MPJPE-L
and success rate the training repo already produces.

Decision rule: an arm is a candidate if it meets all three gates; among
candidates prefer the lowest commanded jitter; break ties by training-noise
clean count. An arm that meets the jitter target and loses more than two
motions is over-weighted: halve its penalty weight (or raise its clip) and
rerun that one arm before comparing.

## 5. Weight sweep for the winner

For the best candidate among A2 / C1 / C2 / C3, run three weights on the
penalty (0.25×, 1×, 2× of the arm's value) at 1.0 B frames each, grade the
same way, and pick the largest weight that still meets the clean-count
gates. That is the checkpoint to pin and rehearse for hardware.

## 6. Deliverables

From the training session, per arm:

- The exported bundle directory with `manifest.json`, parity report, and the
  pin (`model.pin.json`) pushed to the model repository, named
  `smooth_<arm>_<frames>b`.
- A one-page note: exact config diff against the 55B run, frames trained,
  Isaac-board MPJPE-L and success, wall-clock.

From the EC session, one table with the three gated metrics and the two
secondary ones for A0, A1 and every trained arm, and the per-motion matrix
(`ec lifecycle rehearsal-matrix`) across the arms. Recommendation at the top:
which checkpoint to take to the robot, and which arm to sweep further.

## 7. Prompt for the training session

Paste this, with the two documents attached, into a session on the training
workstation:

```
You are running a smoothness fine-tune ablation for the G1 motion tracker in
IsaacLab-Imitation. Read docs/design/smoothness_finetune_plan.md (attached)
and the two evidence files (REPORT.md, DITHER.md) before touching anything.

Facts you must not change: the source checkpoint is
logs/combo50b_action01_vs_hub_20260909/checkpoints/action01_model_step_55000301568.pt
at repo commit 4771c3c, preset combo64_history10_v1, frozen encoder,
ten-frame history, hold 1. Each arm continues from it for 2.0 B environment
frames and is exported with the command recorded in the 55B bundle manifest
(export_policy_bundle.py --preset combo64_history10_v1 --macro-frame-stride 1
--macro-anchor-mode robot_heading --hold-steps 1), with parity verified.

Do, in order: A3, A2, C1, then A4, A5, C2, C3 from section 3 of the plan.
For each arm: locate the reward-term and observation-noise configuration
that the 55B run used, make exactly the change the arm's row describes as a
config diff (no code edits unless the term does not exist; then add it as a
reward term with a clear name and a unit test), train, export, push the pin
as smooth_<arm>_<frames>b, and write the one-page note from section 6.
Before A4 or A5, add action.step_clip_rad / action.ema_alpha to the export
manifest and stop to report, because the deployment side has to implement
the same filter first.

Report after each arm with: config diff, frames, Isaac-board MPJPE-L and
success against the 55B numbers, the pin name and revision. Never modify the
55B or 60B runs. If an arm diverges or its success drops below 50% of the
55B board, stop that arm and report instead of tuning on your own.
```
