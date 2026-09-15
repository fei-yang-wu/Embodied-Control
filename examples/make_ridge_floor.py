"""Ridge-floor variants of the G1 plant model for the toe-catch test.

A 0.25 m lattice of thin capsule ridges (radius h/2, so a ridge h mm tall)
laid on the plane, clear of the arm pose (x from 0.5 m, |y| from 0.35 m),
so a swing foot's toe sphere (r 5 mm) meets a ridge somewhere along a walk.
Capsules, not boxes: `mj_geomDistance` returns 0 for a far-apart box-mesh
pair (the plant's hoist reads the floor gap as 0, hangs the robot 10 cm too
high and the ankles flop into their limit before ARMED); capsule-mesh pairs
measure correctly.

    python examples/make_ridge_floor.py [--heights-mm 3 5]
    ec lifecycle rehearse <bundle> walking_quip_360_R_002__A428 \
        --model assets/latent_playkit/model/g1_29dof_rev_1_0_ridge3.xml ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

FLOOR = '    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>'


def ridge_xml(source: str, height_mm: float, *, pitch: float = 0.25, x_start: float = 0.5,
              x_end: float = 6.0, y_start: float = 0.35, y_end: float = 2.0) -> str:
    r = height_mm / 2000.0
    geoms = [
        f'    <geom name="ridge_x{x:.3f}" type="capsule" size="{r:.4f} 2.0" pos="{x:.3f} 0 {r:.4f}" '
        f'euler="90 0 0" rgba="0.8 0.3 0.2 1"/>'
        for x in np.arange(x_start, x_end, pitch)
    ]
    ys = list(np.arange(y_start, y_end, pitch)) + list(-np.arange(y_start, y_end, pitch))
    geoms += [
        f'    <geom name="ridge_y{y:+.3f}" type="capsule" size="{r:.4f} 3.5" pos="2.5 {y:.3f} {r:.4f}" '
        f'euler="0 90 0" rgba="0.8 0.3 0.2 1"/>'
        for y in ys
    ]
    if FLOOR not in source:
        raise ValueError("floor plane line not found in the source model")
    return source.replace(FLOOR, FLOOR + "\n" + "\n".join(geoms), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="assets/latent_playkit/model/g1_29dof_rev_1_0.xml")
    parser.add_argument("--heights-mm", type=float, nargs="+", default=[3.0, 5.0])
    args = parser.parse_args()
    source = Path(args.model).read_text()
    for h in args.heights_mm:
        out = Path(args.model).with_name(Path(args.model).stem + f"_ridge{h:g}.xml")
        out.write_text(ridge_xml(source, h))
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
