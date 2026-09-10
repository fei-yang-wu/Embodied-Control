---
name: rehearse-measured
description: Rehearse one or more G1 tracker bundles on the MuJoCo plant with the robot's measured sensor noise (not SONIC's training ranges), with a video per motion and the cross-tracker matrix. Use for "rehearse with measured noise", "matched-noise sweep", "rerun trackers with the DR from the capture", or /rehearse-measured.
---

# Rehearse with measured sensor noise

`ec lifecycle rehearse` rehearses a tracker on every motion unattended: per
motion a private vendored + hoisted plant, the console's session driven
through build / prepare / arm / play / hoist / damp, per-episode evidence for
the hardware gate, and `video.mp4` of the plant's true state. By default the
plant injects SONIC's training-range sensor noise, which is 13-130x harsher
than the G1's own sensors (`artifacts/noise_analysis_20260910/REPORT.md`).
`--plant-noise measured` uses the robot's own envelope instead.

Neither profile changes the rehearsal identity. Keep the training-noise run
as the hardware gate evidence; the measured run is the smoothness comparison.

## Arguments

`$ARGUMENTS`: bundle names under `assets/models/controller/` (space
separated), or `all` for every compatible bundle. Optional trailing motion
names; default `all` motions. Ask nothing when the bundle exists; the
compatible set is whatever `PolicyBundle.load` + `reference_compatibility`
accept against `assets/models/reference/bones`.

## Steps

1. Check nothing else owns the cores or DDS domains:
   `pgrep -af "[r]ehearse-epis|[l]owlevel pla" | wc -l` must print 0. If not,
   wait; do not kill another sweep.
2. Confirm the measured profile still matches the newest capture. Values live
   in `PLANT_NOISE["measured"]` in `src/embodied_control/robot/rehearse.py`;
   the capture is `artifacts/hardware_noise_*/rest.npz` (newest). If a newer
   capture exists than the one named in the profile's comment, run
   `pixi run -e native python -c "from embodied_control.lowlevel.unitree_probe import noise_summary; print(noise_summary('<npz>'))"`
   and update the profile to the raw-std envelope, rounded up, before the sweep.
3. Run one sweep per bundle, sequentially, four lanes, on a domain base that
   leaves room: `base + motions - 1 <= 232`.

   ```bash
   OUT=artifacts/rehearsal_measured_noise_$(date +%Y%m%d)
   mkdir -p "$OUT"
   for bundle in <bundles>; do
     pixi run -e native ec lifecycle rehearse "assets/models/controller/$bundle" all \
       --output "$OUT/$bundle" --lanes 4 --dds-domain-base 100 --plant-noise measured \
       > "$OUT/$bundle.log" 2>&1
   done
   pixi run -e native ec lifecycle rehearsal-matrix "$OUT"
   ```

   Run it detached (`nohup setsid script.sh &`) and watch `$OUT/run_all.log`
   or the per-bundle logs; 43 motions take about 8 minutes per bundle on
   4 lanes. Exit code 1 per bundle means some motions failed, not a broken
   sweep; exit 2 is a real error, read the log.
4. A sweep is resumable: rerunning the same command skips every episode that
   already has `result.json`. To retry a motion, delete its `sim/seed_0/`.
5. When it ends, read `$OUT/REPORT.md` (motion x tracker matrix, videos
   linked) and compare with the training-noise sweep under
   `artifacts/rehearsal_all_*/REPORT.md`. Report per tracker: clean count
   under training noise vs measured noise, and which motions moved.

## Failure signatures

- `writer damped during RUNNING` is a fall (fall guard). Policy failure.
- `... (state absent)` is the plant starving under host load: rerun that
  motion with `--lanes 1` before blaming the policy.
- `plant did not become ready` with `Failed to create domain explicitly` in
  `plant.log`: DDS domain above 232; lower `--dds-domain-base`.
- `armed pause ended at FAULT` / `writer is not in active CONTROL`: the policy
  left control while holding frame 0; look at the video around ARMED.

## Do not

- Do not pass `--plant-noise off` for gate evidence; it hides real drops.
- Do not run two sweeps at once; they share cores and the DDS domain range.
- Do not `pkill -f rehearse` from a command whose own text contains the
  pattern; kill by pid list from `pgrep -f "[r]ehearse-epis"`.
