"""Blank / debug policy services (the "VLA" side of the separation model).

These are deliberately trivial: a ``zero`` policy emits zero commands and a
``random`` policy emits uniform-random commands in the normalized action space
[-1, 1]. They stand in for a real VLA server so the orchestration, transport,
embodiment-adapter, and artifact contracts can be exercised end-to-end before any
heavyweight model is wired in.

Kept stdlib-only so ``debug_server`` can run in a minimal container.
"""

from embodied_control.policies.base import Policy, make_policy

__all__ = ["Policy", "make_policy"]
