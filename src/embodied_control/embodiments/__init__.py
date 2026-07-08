"""Embodiment adapters: the robot/task mapping boundary between policy commands
and simulator actuation (the "lower-level controller").
"""

from embodied_control.embodiments.passthrough import PassthroughController

__all__ = ["PassthroughController"]
