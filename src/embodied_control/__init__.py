"""Embodied-Control: a thin host orchestrator for isolated VLA policy / simulator evaluation.

This top-level package is intentionally import-light. In particular it does NOT
import pydantic, mujoco, or the CLI at import time, so that the stdlib-only
policy service (``embodied_control.policies.debug_server``) can run inside a
minimal container without the host's dependencies.
"""

__version__ = "0.1.0"
