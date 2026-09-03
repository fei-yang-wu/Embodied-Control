#!/usr/bin/env python3
"""Write docs/operator_manual.md from the console's own bindings.

The keys, their names in the palette and the ladder all live in code. A
manual typed beside them goes stale the first time a key moves, and the
operator finds out while a robot hangs on a hoist. This regenerates it, and
`pixi run check-manual` fails when the file on disk no longer matches.

    pixi run build-manual
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied_control.console import KEY_DAMP  # noqa: E402
from embodied_control.robot.lifecycle import (  # noqa: E402
    LADDER,
    RECOVERABLE_STATES,
    LifecycleState,
)
from embodied_control.robot.shell import build_session_bindings  # noqa: E402
from embodied_control.robot.tui import NEXT_ACTION, SLASH_KEYS  # noqa: E402

MANUAL = REPO_ROOT / "docs" / "operator_manual.md"

KEY_NAMES = {KEY_DAMP: "Ctrl-D"}

GROUP_TITLES = {
    "safety": "Safety",
    "select": "Choose what to run",
    "ladder": "Climb the ladder",
    "episode": "Run an episode",
    "operator": "Tell the console what you did",
    "end": "End the run",
    "console": "Console",
}

# What each rung proves, in the operator's terms rather than the gate's.
RUNG_MEANING = {
    "PRECHECK": "the link, the vendor and the hoist are all confirmed",
    "VENDOR_DAMP_CONFIRMED": "the vendor has the robot limp, read back from its own FSM",
    "SAFE_EXTERNAL_COMMAND_PRESENT": "our damp frames are on the wire, before anyone lets go",
    "USER_CONTROL_CONFIRMED": "the vendor released and our frames own the joints",
    "START_POSE_RAMP": "the joints reached the start pose without one lagging behind",
    "POSE_SETTLED": "the robot stopped moving there",
    "LOWERED": "you lowered it and it settled again with the feet loaded",
    "POSE_MATCH_VERIFIED": "the pose and the pelvis tilt match the sim start frame",
    "POLICY_COMMAND_FRESH": "the planner answers in time and its first action is sane",
    "PRIMED": "everything is verified and the robot is waiting for you",
}


class _Session:
    """Enough of an ExperimentSession to enumerate the bindings."""

    class _Config:
        frame_step = 25

    config = _Config()

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def _slash_for(key: str) -> str:
    for name, bound in SLASH_KEYS.items():
        if bound == key:
            return f"/{name}"
    return ""


def build() -> str:
    bindings = build_session_bindings(_Session())
    lines = [
        "# Operator manual: one episode, hoist to damp",
        "",
        "Generated from the console's own bindings by",
        "`scripts/build_operator_manual.py`. Do not edit by hand; run",
        "`pixi run build-manual` after changing a key or a state.",
        "",
        "The design and the reasoning behind each gate are in",
        "[docs/design/robot_lifecycle.md](design/robot_lifecycle.md). This page is",
        "what to do, in order, in front of the robot.",
        "",
        "## Before the robot",
        "",
        "**Rehearse first.** A hardware run refuses to start until the same",
        "bundle and motion have reached the end of the ladder against the",
        "MuJoCo plant. The plant serves the vendor's RPCs and rejects",
        "`rt/lowcmd` in the wrong mode, so the rehearsal exercises the code",
        "path the robot will run; only the network interface differs.",
        "",
        "```bash",
        "# 1. the plant, owning the joints until ReleaseMode, robot hoisted",
        "pixi run -e native ec lowlevel plant examples/g1_plant.yaml \\",
        "  --model assets/latent_playkit/model/g1_29dof_rev_1_0.xml \\",
        "  --network lo --vendor --hoist --dds-domain 51",
        "# 2. the console, against the plant",
        "pixi run -e native ec lifecycle console examples/lifecycle_sim_sonic_v1_1.yaml \\",
        "  --enable-writes --confirm ENABLE_G1_LOWLEVEL_NON_REALTIME --allow-non-realtime",
        "```",
        "",
        "The rehearsal has to end cleanly: no failed gate, and a final state of",
        "`VENDOR_RESTORED`, `VENDOR_STAND` or `STANDING`. It expires after two",
        "weeks. Check what is on hand with `ec models list` and pull anything",
        "missing before you start, so a fetch never happens with the robot",
        "hanging.",
        "",
        "## The ladder",
        "",
        "Each rung is a gate the runtime checks, not a key you are trusted",
        "with. `/next` runs the next one and prints its evidence; `/auto`",
        "climbs until a gate fails or a rung needs you.",
        "",
        "| # | State | What it proves |",
        "|---|---|---|",
    ]
    for index, state in enumerate(LADDER):
        if state is LifecycleState.IDLE:
            continue
        meaning = RUNG_MEANING.get(str(state), "")
        lines.append(f"| {index} | `{state}` | {meaning} |")
    lines += [
        "",
        "Then `/go` blends the policy in over half a second and runs the",
        "episode. `/hold` freezes on the last target when you want to stop",
        "early; the budget ends it otherwise.",
        "",
        "## Ending a run",
        "",
        "A run ends with the robot **limp under the vendor's own damp**, so the",
        "next trajectory climbs the whole ladder again. `Ctrl-D` puts our",
        "kd-only frames on the wire immediately and unconditionally, then hands",
        "the joints back once you have acknowledged the hoist. Hook the hoist",
        "and take the load before that hand-back: `SelectMode` restarts the",
        "vendor service in damp, and the robot is limp for about a second while",
        "it comes back.",
        "",
        f"Recovery is available from {', '.join(sorted(str(s) for s in RECOVERABLE_STATES))}.",
        "`/stand` takes it further, to the vendor's balance stand, when you want",
        "the robot on its feet. Our controller never stands the robot itself.",
        "",
        "## Every key",
        "",
    ]
    groups: dict[str, list] = {}
    for binding in bindings:
        groups.setdefault(binding.group, []).append(binding)
    for group, members in groups.items():
        lines += [f"### {GROUP_TITLES.get(group, group.title())}", ""]
        lines += ["| Key | Command | What it does |", "|---|---|---|"]
        for binding in members:
            key = KEY_NAMES.get(binding.key, f"`{binding.key}`")
            slash = _slash_for(binding.key)
            lines.append(
                f"| {key} | {f'`{slash}`' if slash else ''} | {binding.label} |"
            )
        lines.append("")
    lines += [
        "## What the console shows you",
        "",
        "The header carries the episode, the state, whether the vendor still",
        "owns the joints, the writer's mode, the boot heading the fixed anchor",
        "absorbed, and the next command to type. The log colours what it says:",
        "a passed gate reads green, a damp amber, a failure or refusal red.",
        "",
        "`Ctrl-D` always damps, ahead of every gate and every queued key, and",
        "works while the palette is open or the prompt has text. `SPACE` does",
        "nothing on purpose: it is the key a hand rests on.",
        "",
        "## When a gate fails",
        "",
        "The gate prints what it measured and the state does not move. Nothing",
        "is bypassable from the console; fix the cause or change the job. A run",
        "that faults records the runtime's own fault code alongside the writer",
        "counters in `lifecycle.jsonl`, so `writer damped during RUNNING` is",
        "followed by the reason. `/diagnose` sends that snapshot and the recent",
        "log to a read-only coding agent, which can explain but cannot act.",
        "",
        "## Next state at a glance",
        "",
        "| State | Type this |",
        "|---|---|",
    ]
    for state, action in NEXT_ACTION.items():
        lines.append(f"| `{state}` | {action} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    text = build()
    if "--check" in sys.argv:
        if not MANUAL.is_file() or MANUAL.read_text() != text:
            print(
                "FAIL: docs/operator_manual.md is out of date; "
                "run `pixi run build-manual`"
            )
            return 1
        print("ok: docs/operator_manual.md matches the bindings")
        return 0
    MANUAL.parent.mkdir(parents=True, exist_ok=True)
    MANUAL.write_text(text)
    print(f"wrote {MANUAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
