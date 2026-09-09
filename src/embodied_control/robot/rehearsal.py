"""The rehearsal gate: hardware runs what the plant already ran.

The plant serves the vendor's RPCs, rejects `rt/lowcmd` in the wrong mode and
hangs the robot from a virtual gantry, so the same lifecycle object drives it
with only the network interface changed. That makes a sim rehearsal cheap
evidence for a hardware run, and the reverse a decision nobody should make by
memory: this module answers "has this exact bundle, on this motion, reached
the end of the ladder against the plant, recently?" from the artifacts the
rehearsal already wrote.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from embodied_control.robot.gates import GateResult

#: A rehearsal counts only if it reached one of these. `VENDOR_RESTORED` is
#: the default end; `STANDING` is the same run taken further.
ACCEPTED_END_STATES = frozenset({"VENDOR_RESTORED", "VENDOR_STAND", "STANDING"})

#: The interface a rehearsal runs on. The plant is reachable on loopback only.
SIM_NETWORKS = frozenset({"lo", "localhost"})


def run_identity(
    *,
    bundle_sha: str,
    bundle_name: str,
    motion: str,
    command_source: str,
    network: str,
    reference_sha: str = "",
    deployment_sha: str = "",
    start_frame: int = 0,
    ticks: int = 0,
) -> dict:
    """What a run has to agree on for one to vouch for the other."""
    return {
        "bundle_sha": str(bundle_sha),
        "bundle_name": str(bundle_name),
        "reference_sha": str(reference_sha),
        "deployment_sha": str(deployment_sha),
        "motion": str(motion),
        "command_source": str(command_source),
        "network": str(network),
        "start_frame": int(start_frame),
        "ticks": int(ticks),
    }


def is_simulated(identity: dict) -> bool:
    return str(identity.get("network", "")) in SIM_NETWORKS


def _matches(candidate: dict, identity: dict) -> bool:
    if not is_simulated(candidate):
        return False
    for key in ("bundle_sha", "motion", "command_source", "reference_sha", "deployment_sha", "start_frame", "ticks"):
        if str(candidate.get(key, "")) != str(identity.get(key, "")):
            return False
    return True


def find_rehearsals(root: str | Path) -> list[dict]:
    """Every finished run under `root`, newest first."""
    base = Path(root)
    if not base.is_dir():
        return []
    runs = []
    for path in base.rglob("lifecycle.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        payload = dict(payload)
        payload["_path"] = str(path)
        runs.append(payload)
    runs.sort(key=lambda run: float(run.get("finished_at", 0.0)), reverse=True)
    return runs


def rehearsal_evidence(
    root: str | Path,
    identity: dict,
    *,
    max_age_days: float = 14.0,
    now: float | None = None,
) -> GateResult:
    """Whether a passing plant rehearsal of this exact run exists."""
    stamp = time.time() if now is None else now
    values = {"rehearsal_root": str(root), **{f"want_{k}": v for k, v in identity.items()}}
    runs = find_rehearsals(root)
    if not runs:
        return GateResult(
            False,
            f"no rehearsal found under {root}; run this job against the plant "
            "with network 'lo' first",
            values,
        )
    same_bundle = [run for run in runs if _matches(run.get("rehearsal", {}), identity)]
    if not same_bundle:
        return GateResult(
            False,
            f"no plant rehearsal of bundle {identity.get('bundle_sha', '')[:12]} "
            f"on motion '{identity.get('motion', '')}' under {root}",
            values,
        )
    newest = same_bundle[0]
    values["rehearsal_path"] = newest.get("_path", "")
    values["rehearsal_state"] = newest.get("state", "")
    values["rehearsal_fault_reason"] = newest.get("fault_reason", "")
    reached = str(newest.get("state", ""))
    failed = int(newest.get("failed_transitions", 0))
    if reached not in ACCEPTED_END_STATES or failed or newest.get("fault_reason"):
        return GateResult(
            False,
            f"the newest rehearsal ended in {reached or 'nothing'} with "
            f"{failed} failed transition(s), fault {newest.get('fault_reason') or 'none'}; "
            "it has to reach the end of the "
            "ladder cleanly",
            values,
        )
    age_days = (stamp - float(newest.get("finished_at", 0.0))) / 86400.0
    values["rehearsal_age_days"] = round(age_days, 2)
    if newest.get("finished_at") and age_days > max_age_days:
        return GateResult(
            False,
            f"the newest rehearsal is {age_days:.1f} days old, past the "
            f"{max_age_days:.0f}-day limit; rehearse again",
            values,
        )
    return GateResult(
        True,
        f"rehearsed on the plant {age_days:.1f} days ago, ending {reached}",
        values,
    )
