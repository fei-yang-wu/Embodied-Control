"""One job, two targets; endpoint screening policy; the rehearsal sweep's plumbing."""

import json
import time

import pytest
import yaml

from embodied_control.robot.build import screening_split
from embodied_control.robot.lifecycle_job import LifecycleJob, apply_target, load_lifecycle_job
from embodied_control.robot.rehearsal import explain_mismatch, run_identity


def _job(tmp_path, **overrides):
    values = {
        "bundle": str(tmp_path / "bundle"),
        "network": "",
        "dds_domain": 0,
        "command_source": "oracle",
        "reference_root": str(tmp_path / "bones"),
        "motion": "hurry_idle_001_A277",
        "fixed_initial_anchor": True,
        "ticks": "auto",
        "artifacts_dir": str(tmp_path / "campaign" / "hurry"),
    }
    values.update(overrides)
    return LifecycleJob.model_validate(values)


def test_sim_and_hardware_targets_share_everything_but_the_place(tmp_path):
    job = _job(tmp_path)
    sim = apply_target(job, "sim")
    hardware = apply_target(job, "hardware", network="enp1s0")

    assert sim.network == "lo" and sim.dds_domain == 51 and sim.sim_hoist
    assert sim.realtime.control_priority == 0 and not sim.realtime.lock_memory
    assert sim.artifacts_dir.endswith("/hurry/sim")
    assert hardware.network == "enp1s0" and hardware.dds_domain == 0 and not hardware.sim_hoist
    assert hardware.realtime.control_priority == 80 and hardware.realtime.lock_memory
    assert hardware.artifacts_dir.endswith("/hurry/hardware")
    # The hardware run looks for its rehearsal where the sim run wrote it.
    assert hardware.rehearsal_root == job.artifacts_dir
    assert sim.request_slot != hardware.request_slot
    for key in ("start_pose", "blend_ticks", "ramp_seconds", "lead_ticks", "thresholds", "motion"):
        assert getattr(sim, key) == getattr(hardware, key) == getattr(job, key)


def test_hardware_target_needs_an_interface_and_a_place_to_write(tmp_path):
    with pytest.raises(ValueError, match="--network"):
        apply_target(_job(tmp_path, network="lo"), "hardware")
    with pytest.raises(ValueError, match="artifacts_dir"):
        apply_target(_job(tmp_path, artifacts_dir=""), "sim")
    with pytest.raises(ValueError, match="target"):
        apply_target(_job(tmp_path), "robot")


def test_endpoint_screening_policy_only_blocks_the_start_by_default():
    reasons = [
        "start root speed 0.22 m/s exceeds 0.15",
        "end root speed 0.56 m/s exceeds 0.15",
        "missing joint velocities",
    ]
    blocking, reported = screening_split(reasons, "start")
    assert blocking == [reasons[0], reasons[2]] and reported == [reasons[1]]
    assert screening_split(reasons, "both") == (reasons, [])
    assert screening_split(reasons, "off") == ([], reasons)
    assert screening_split(["end joint speed 2.88 rad/s exceeds 2.00"], "start") == (
        [], ["end joint speed 2.88 rad/s exceeds 2.00"],
    )


def test_job_schema_accepts_the_screening_policy(tmp_path):
    assert _job(tmp_path).endpoint_screening == "start"
    assert _job(tmp_path, endpoint_screening="off").endpoint_screening == "off"
    with pytest.raises(ValueError):
        _job(tmp_path, endpoint_screening="end")


def _write_run(directory, identity, *, state="VENDOR_RESTORED", age_days=0.0):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "lifecycle.json").write_text(json.dumps({
        "state": state, "fault_reason": "", "episode": 1, "transitions": 16,
        "failed_transitions": 0, "rehearsal": identity,
        "finished_at": time.time() - age_days * 86400.0,
    }))


def test_explain_mismatch_names_the_keys_that_differ(tmp_path):
    base = dict(
        bundle_sha="a" * 64, bundle_name="combo", motion="m", command_source="oracle",
        network="lo", reference_sha="r" * 64, deployment_sha="d" * 64, ticks=500,
    )
    _write_run(tmp_path / "old_thresholds", run_identity(**{**base, "deployment_sha": "e" * 64}), age_days=1)
    _write_run(tmp_path / "other_motion", run_identity(**{**base, "motion": "n", "ticks": 300}))
    _write_run(tmp_path / "hardware_attempt", run_identity(**{**base, "network": "enp1s0"}), age_days=2)

    want = run_identity(**{**base, "network": "enp1s0"})
    rows = explain_mismatch(tmp_path, want)
    by_path = {row["path"].split("/")[-2]: row for row in rows}
    assert by_path["old_thresholds"]["differs"] == ["deployment_sha"]
    assert by_path["other_motion"]["differs"] == ["motion", "ticks"]
    assert by_path["hardware_attempt"]["differs"] == ["network"]
    assert rows[0]["path"].endswith("other_motion/lifecycle.json")


def test_canonical_job_is_hardware_shaped_and_its_sim_twin_matches(tmp_path):
    from embodied_control.robot.rehearse import RehearsePlan, canonical_job, episode_job

    class Command:
        horizon_steps, macro_frame_stride, hold_steps = 10, 5, 10

    class Manifest:
        command = Command()

    class OneTickHold:
        command = type("C", (), {"horizon_steps": 10, "macro_frame_stride": 1, "hold_steps": 1})()

    bones = tmp_path / "bones"
    bones.mkdir()
    (bones / "reference_arrays_manifest.json").write_text(json.dumps({
        "traj_info": {
            "ordered_traj_list": [["bones", "walk"], ["bones", "walk__stand_abc"]],
            "start_index": [0, 100], "end_index": [100, 500],
        },
        "motions": {"walk__stand_abc": {"hold_frames": 100}},
    }))
    plan = RehearsePlan(
        bundle=str(tmp_path / "combo_50b"), motions=["walk", "walk__stand_abc"],
        output=tmp_path / "out", reference_root=str(bones), model=str(tmp_path / "g1.xml"),
        template={"blend_ticks": 100, "bundle": "ignored", "lead_ticks": 4},
    )
    raw = canonical_job(plan, "walk", Manifest())
    composed = canonical_job(plan, "walk__stand_abc", Manifest())
    assert raw["network"] == "" and raw["dds_domain"] == 0 and not raw["sim_hoist"]
    assert raw["realtime"]["control_priority"] == 80 and raw["require_rehearsal"]
    assert raw["stand_hold_seconds"] == 0.0 and composed["stand_hold_seconds"] == 60.0
    assert raw["blend_ticks"] == 100 and raw["lead_ticks"] == 4
    assert raw["bundle"].endswith("combo_50b")
    plan.template = {}
    assert canonical_job(plan, "walk", Manifest())["lead_ticks"] == 4
    assert canonical_job(plan, "walk", OneTickHold())["lead_ticks"] == 0

    path = tmp_path / "out" / "walk" / "job.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    sim = episode_job(path, seed=2, domain=197, cores=[4, 5, 6, 7])
    assert sim.network == "lo" and sim.dds_domain == 197 and sim.sim_hoist
    assert sim.artifacts_dir.endswith("/walk/sim/seed_2")
    assert sim.realtime.control_cpu == 4 and sim.realtime.writer_cpu == 5
    assert sim.request_slot == "/ec_rehearse_197_request"
    hardware = apply_target(load_lifecycle_job(path), "hardware", network="enp1s0")
    assert hardware.rehearsal_root == raw["artifacts_dir"]
    assert hardware.blend_ticks == sim.blend_ticks == 100


def test_state_at_reads_the_newest_successful_transition():
    pytest.importorskip("numpy")  # the lowlevel package imports it on load
    from embodied_control.lowlevel.plant_render import state_at

    timeline = [
        {"wall": 10.0, "to_state": "PRECHECK", "ok": True},
        {"wall": 12.0, "to_state": "VENDOR_DAMP_CONFIRMED", "ok": True},
        {"wall": 13.0, "to_state": "RUNNING", "ok": False},
        {"wall": 15.0, "to_state": "RUNNING", "ok": True},
    ]
    assert state_at(timeline, 9.0) == ""
    assert state_at(timeline, 12.5) == "VENDOR_DAMP_CONFIRMED"
    assert state_at(timeline, 13.5) == "VENDOR_DAMP_CONFIRMED"
    assert state_at(timeline, 20.0) == "RUNNING"


def test_summarize_reports_every_seed(tmp_path):
    from embodied_control.robot.rehearse import summarize

    for motion, seed, passed in (("a", 0, True), ("a", 1, False), ("b", 0, True)):
        run = tmp_path / motion / "sim" / f"seed_{seed}"
        run.mkdir(parents=True)
        (run / "result.json").write_text(json.dumps({
            "motion": motion, "bundle": "combo", "seed": seed, "passed": passed,
            "error": "" if passed else "armed pause ended at FAULT",
            "episode": {"joint_mae_mean_rad": 0.05}, "video": str(run / "video.mp4"),
        }))
    summary = summarize(tmp_path)
    assert summary["episodes"] == 3 and summary["episodes_passed"] == 2
    assert summary["motions_all_passed"] == 1
    report = (tmp_path / "REPORT.md").read_text()
    assert "| `a` | 1/2 |" in report and "armed pause ended at FAULT" in report
    assert "seed 0](a/sim/seed_0/video.mp4)" in report


def test_matrix_report_crosses_trackers_and_motions(tmp_path):
    from embodied_control.robot.rehearse import matrix_report, summarize

    for bundle, outcomes in (("a", {"m1": True, "m2": False}), ("b", {"m1": True, "m2": True})):
        for motion, passed in outcomes.items():
            run = tmp_path / bundle / motion / "sim" / "seed_0"
            run.mkdir(parents=True)
            (run / "result.json").write_text(json.dumps({
                "motion": motion, "bundle": bundle, "seed": 0, "passed": passed,
                "video": str(run / "video.mp4"),
            }))
        summarize(tmp_path / bundle)
    payload = matrix_report(tmp_path)
    assert payload["bundles"] == ["b", "a"] and payload["clean"] == {"b": 2, "a": 1}
    assert payload["clean_everywhere"] == ["m1"] and payload["clean_nowhere"] == []
    report = (tmp_path / "REPORT.md").read_text()
    assert "| `m2` | [1/1](b/m2/sim/seed_0/video.mp4) | [0/1](a/m2/sim/seed_0/video.mp4) |" in report


def test_plant_noise_profiles_map_to_plant_flags():
    from embodied_control.robot.rehearse import plant_noise_argv

    assert plant_noise_argv("training") == []
    assert plant_noise_argv("off") == [
        "--noise-joint-pos", "0.0", "--noise-joint-vel", "0.0",
        "--noise-base-ang-vel", "0.0", "--noise-imu-tilt-rad", "0.0",
    ]
    measured = plant_noise_argv("measured")
    assert measured[measured.index("--noise-joint-vel") + 1] == "0.025"
    with pytest.raises(ValueError):
        plant_noise_argv("loud")
