"""Resolve an ``EvalJob`` into an auditable ``ExecutionPlan``.

Binds: run id + run directory, host fingerprint, per-episode seeds, the resolved
policy endpoint, and the resolved policy runtime (local subprocess, Docker
container, or an external already-running service). Persisted as
``resolved_job.yaml`` before any runtime starts.
"""

from __future__ import annotations

import getpass
import platform
import re
import socket
import sys
from datetime import datetime
from pathlib import Path

from embodied_control.config.schemas import (
    EvalJob,
    ExecutionPlan,
    HostFingerprint,
    ResolvedEndpoint,
    ResolvedRuntime,
)
from embodied_control.runtime.ports import allocate_free_port


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower() or "job"


def _host_fingerprint() -> HostFingerprint:
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001
        user = None
    return HostFingerprint(
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        hostname=socket.gethostname(),
        user=user,
    )


def make_run_id(job: EvalJob, now: datetime) -> str:
    template = job.outputs.run_name_template
    return template.format(
        date=now.strftime("%Y%m%d_%H%M%S"),
        name=_slug(job.name),
        seed=job.seed,
    )


def _dedupe_run_id(root_dir: str, run_id: str) -> str:
    """Disambiguate ``run_id`` if its directory already exists.

    The default template has second-level granularity, so two runs with the
    same job name/seed started within the same second (e.g. a fast sweep, or
    back-to-back calls in a test) would otherwise resolve to the identical
    ``run_dir`` and silently overwrite each other's artifacts -- and, since
    the logger's root name is keyed by ``run_id``, would also collide on log
    handlers. Append a counter suffix until the candidate directory is free.
    """
    base_dir = Path(root_dir)
    candidate = run_id
    n = 1
    while (base_dir / candidate).exists():
        candidate = f"{run_id}-{n}"
        n += 1
    return candidate


def build_plan(
    job: EvalJob,
    action_dim: int,
    action_schema_id: str,
    observation_schema_id: str,
    now: datetime | None = None,
) -> ExecutionPlan:
    now = now or datetime.now()
    run_id = _dedupe_run_id(job.outputs.root_dir, make_run_id(job, now))
    run_dir = str((Path(job.outputs.root_dir) / run_id).resolve())
    seeds = [job.seed + i for i in range(job.rollout.num_episodes)]

    pol = job.policy
    if pol.endpoint is not None:
        # Connect to an already-running external service; nothing to launch.
        endpoint = ResolvedEndpoint(
            scheme=pol.endpoint.scheme, host=pol.endpoint.host, port=pol.endpoint.port,
            action_dim=pol.endpoint.action_dim, observation_mapping=pol.endpoint.observation_mapping,
        )
        runtime = ResolvedRuntime(
            name="policy", type="external", host_port=pol.endpoint.port
        )
    elif pol.runtime.type == "docker":
        image = pol.runtime.image
        if not image:
            raise ValueError("policy.runtime.type=docker requires policy.runtime.image")
        host_port = allocate_free_port()
        container_port = pol.runtime.container_port
        container_name = f"ec_{run_id}_policy"
        runtime = ResolvedRuntime(
            name="policy",
            type="docker",
            image=image,
            container_name=container_name,
            host_port=host_port,
            container_port=container_port,
            command=[
                "--type", pol.type,
                "--action-dim", str(action_dim),
                "--action-schema-id", action_schema_id,
                "--host", "0.0.0.0",
                "--port", str(container_port),
                "--seed", str(job.seed),
            ],
        )
        endpoint = ResolvedEndpoint(scheme="http", host="127.0.0.1", port=host_port)
    else:  # local subprocess
        host_port = allocate_free_port()
        runtime = ResolvedRuntime(
            name="policy",
            type="local",
            host_port=host_port,
            command=[
                sys.executable, "-m", "embodied_control.policies.debug_server",
                "--type", pol.type,
                "--action-dim", str(action_dim),
                "--action-schema-id", action_schema_id,
                "--host", "127.0.0.1",
                "--port", str(host_port),
                "--seed", str(job.seed),
            ],
        )
        endpoint = ResolvedEndpoint(scheme="http", host="127.0.0.1", port=host_port)

    return ExecutionPlan(
        job=job,
        run_id=run_id,
        run_dir=run_dir,
        created_at=now.astimezone().isoformat(),
        host=_host_fingerprint(),
        seeds=seeds,
        action_dim=action_dim,
        action_schema_id=action_schema_id,
        observation_schema_id=observation_schema_id,
        policy_endpoint=endpoint,
        policy_runtime=runtime,
    )
