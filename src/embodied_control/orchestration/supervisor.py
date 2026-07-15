"""Launch + health-check the policy service, or attach to an external one.

Builds the right runtime adapter from the resolved plan, starts it, waits until
the service is healthy over the transport, and tears it down on stop. Returns a
ready ``PolicyClient``.

The launch *command* itself is resolved once, in ``orchestration/planner.py``
(the single source of truth persisted into ``resolved_job.yaml``) -- this
module just hands ``plan.policy_runtime.command`` to the right adapter,
rather than recomputing it (see ``planner.py::_resolve_policy_command`` for
why that used to drift).
"""

from __future__ import annotations

from pathlib import Path

from embodied_control.config.schemas import EvalJob, ExecutionPlan
from embodied_control.logging.logger import EcLogger
from embodied_control.runtime.docker import DockerRuntimeAdapter
from embodied_control.runtime.local import LocalRuntimeAdapter
from embodied_control.transport.factory import make_policy_client


class PolicyServiceSupervisor:
    def __init__(
        self, job: EvalJob, plan: ExecutionPlan, log_path: str, logger: EcLogger | None = None
    ):
        self.job = job
        self.plan = plan
        self.log_path = log_path
        self.logger = logger or EcLogger.null()
        self._adapter = None
        self._handle = None
        self.client = None

    def start(self, health_timeout_s: float = 30.0):
        rt = self.plan.policy_runtime
        ep = self.plan.policy_endpoint
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        runtime_logger = self.logger.child("runtime")

        if rt.type == "external":
            # Attach to an already-running service; nothing to launch.
            self._adapter = None
            self._handle = None
            self.logger.event("runtime.external.attached", phase="policy_launch",
                               endpoint=f"{ep.host}:{ep.port}")
        elif rt.type == "docker":
            # The container binds 0.0.0.0 internally, Docker maps it to
            # host_port; rt.command was resolved by the planner (either the
            # built-in debug-server args, or the job's own override for a
            # real policy server -- see planner.py::_resolve_policy_command).
            self._adapter = DockerRuntimeAdapter(
                name="policy",
                image=rt.image,
                container_name=rt.container_name,
                command=rt.command,
                network="bridge",
                host_port=rt.host_port,
                container_port=rt.container_port,
                log_path=self.log_path,
                shm_size=self.job.policy.runtime.shm_size,
                logger=runtime_logger,
            )
            self._handle = self._adapter.start()
        else:  # local
            self._adapter = LocalRuntimeAdapter(
                name="policy",
                command=rt.command,
                log_path=self.log_path,
                endpoint_host=ep.host,
                endpoint_port=ep.port,
                logger=runtime_logger,
            )
            self._handle = self._adapter.start()

        self.client = make_policy_client(
            ep.scheme, ep.host, ep.port, timeout_s=30.0, logger=self.logger.child("transport"),
            action_dim=ep.action_dim, observation_mapping=ep.observation_mapping,
        )
        try:
            self.client.wait_healthy(timeout_s=health_timeout_s)
        except Exception as exc:
            # Surface the service log to help debugging, then re-raise.
            self.logger.error("runtime.health_check_failed", reason=str(exc))
            self.stop()
            raise
        return self.client

    def logs(self) -> str:
        if self._adapter is not None and self._handle is not None:
            return self._adapter.logs(self._handle)
        return ""

    def stop(self) -> None:
        if self._adapter is not None and self._handle is not None:
            try:
                self._adapter.stop(self._handle)
            finally:
                self._handle = None
