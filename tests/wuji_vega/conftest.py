"""Shared fixtures for the Vega-Wuji integration tests."""

from pathlib import Path

import pytest


SCENE_PATH = (
    Path(__file__).resolve().parents[3]
    / "simulations/wuji_vega_u_grasp/build/scene_table_cube.xml"
)


@pytest.fixture(scope="session")
def scene_path() -> Path:
    if not SCENE_PATH.is_file():
        pytest.skip(f"portable Vega-Wuji scene has not been built: {SCENE_PATH}")
    return SCENE_PATH
