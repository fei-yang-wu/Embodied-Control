"""Policy base + the blank zero/random policies (stdlib only).

A ``Policy`` implements the four wire operations. It works in a *normalized*
action space in [-1, 1]^action_dim; converting normalized commands into concrete
simulator actuation is the job of the embodiment adapter on the host side, not
the policy. This keeps the policy simulator-agnostic — exactly the boundary the
design asks for (the VLA emits commands; a lower-level controller/adapter turns
them into env actions).
"""

from __future__ import annotations

import random
import time

from embodied_control.transport.protocol import PROTOCOL_VERSION, key_str


class Policy:
    """Base policy. Subclasses implement ``_action(env_id, episode_id)``."""

    policy_type = "base"

    def __init__(
        self,
        action_dim: int,
        action_schema_id: str = "ec.action.normalized/v1",
        seed: int = 0,
        max_action_horizon: int = 16,
    ):
        self.action_dim = int(action_dim)
        self.action_schema_id = action_schema_id
        self.base_seed = int(seed)
        self.max_action_horizon = int(max_action_horizon)
        # Per-episode RNG state, keyed by "env:episode". Present-key == was reset.
        self._episodes: dict[str, random.Random] = {}

    # --- wire operations -------------------------------------------------
    def health(self) -> dict:
        return {
            "status": "ok",
            "model_loaded": True,
            "policy_type": self.policy_type,
            "protocol_version": PROTOCOL_VERSION,
        }

    def describe(self) -> dict:
        return {
            "policy_id": f"{self.policy_type}-debug-policy",
            "policy_type": self.policy_type,
            "model_family": "debug",
            "protocol_versions": [PROTOCOL_VERSION],
            "action_dim": self.action_dim,
            "action_schema_ids": [self.action_schema_id],
            "max_action_horizon": self.max_action_horizon,
            "supports_streaming": False,
        }

    def reset(self, req: dict) -> dict:
        seed = int(req.get("seed", self.base_seed))
        keys = req.get("episode_keys", [])
        for key in keys:
            ks = key_str(key)
            # Deterministic per-episode stream so a fixed job seed is reproducible.
            self._episodes[ks] = random.Random((seed * 1_000_003) ^ hash(ks))
        return {"status": "ok", "reset_episodes": [key_str(k) for k in keys]}

    def act(self, req: dict) -> dict:
        t0 = time.perf_counter()
        horizon = max(1, min(int(req.get("requested_horizon", 1)), self.max_action_horizon))
        observations = req.get("observations", [])
        actions = []
        for obs in observations:
            env_id = int(obs.get("env_id", 0))
            episode_id = int(obs.get("episode_id", 0))
            ks = key_str([env_id, episode_id])
            if ks not in self._episodes:
                # Contract: /act on an un-reset episode is a precondition failure.
                return {
                    "protocol_version": PROTOCOL_VERSION,
                    "request_id": req.get("request_id"),
                    "status": "FAILED_PRECONDITION",
                    "error": f"episode {ks} was not reset before act",
                    "actions": [],
                }
            chunk = self._action_chunk(env_id, episode_id, horizon)
            actions.append(
                {
                    "env_id": env_id,
                    "episode_id": episode_id,
                    "action_schema_id": self.action_schema_id,
                    "action_chunk": chunk,
                    "valid_for_steps": horizon,
                }
            )
        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": req.get("request_id"),
            "status": "ok",
            "actions": actions,
            "timing": {"inference_ms": total_ms, "total_ms": total_ms},
        }

    # --- subclass hooks --------------------------------------------------
    def _action(self, env_id: int, episode_id: int) -> list[float]:
        raise NotImplementedError

    def _action_chunk(self, env_id: int, episode_id: int, horizon: int) -> list[list[float]]:
        """Return a chunk of ``horizon`` actions. Default: independent per step."""
        return [self._action(env_id, episode_id) for _ in range(horizon)]


class ZeroPolicy(Policy):
    policy_type = "zero"

    def _action(self, env_id: int, episode_id: int) -> list[float]:
        return [0.0] * self.action_dim


class RandomPolicy(Policy):
    policy_type = "random"

    def _action(self, env_id: int, episode_id: int) -> list[float]:
        rng = self._episodes[key_str([env_id, episode_id])]
        return [rng.uniform(-1.0, 1.0) for _ in range(self.action_dim)]

    def _action_chunk(self, env_id: int, episode_id: int, horizon: int) -> list[list[float]]:
        # One coherent command held across the chunk: a position-controlled arm
        # needs the target held long enough to track it, and this is a truer
        # demonstration of action chunking (one intent per chunk) than per-step
        # white noise. Consecutive chunks re-sample, so the arm explores.
        action = self._action(env_id, episode_id)
        return [list(action) for _ in range(horizon)]


_POLICIES = {"zero": ZeroPolicy, "random": RandomPolicy}


def make_policy(policy_type: str, action_dim: int, **kwargs) -> Policy:
    try:
        cls = _POLICIES[policy_type]
    except KeyError:
        raise ValueError(
            f"unknown policy type {policy_type!r}; choices: {sorted(_POLICIES)}"
        ) from None
    return cls(action_dim=action_dim, **kwargs)
