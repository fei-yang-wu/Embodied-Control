"""Runtime adapters: launch a component as a local subprocess or a Docker container.

Both speak the identical policy transport; only the launch/teardown differs. This
is the design's single-``RuntimeSpec``-compiles-to-engine-specific-commands idea,
scoped to what M1 needs.
"""

from embodied_control.runtime.base import RuntimeHandle
from embodied_control.runtime.ports import allocate_free_port

__all__ = ["RuntimeHandle", "allocate_free_port"]
