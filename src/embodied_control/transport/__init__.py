"""Transport between the host orchestrator and the policy service.

The wire protocol is deliberately small and stdlib-only (JSON over HTTP) so the
policy service can run in a minimal container. ``server`` and ``protocol`` import
only the standard library; ``client`` (host side) uses ``urllib``.
"""

PROTOCOL_VERSION = "ec.policy/v1alpha1"
