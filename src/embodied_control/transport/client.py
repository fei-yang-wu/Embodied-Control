"""Host-side transport client for the policy service (stdlib urllib).

This is the concrete ``PolicyAdapter``: it hides HTTP/JSON behind ``health``,
``wait_healthy``, ``describe``, ``reset``, and ``act`` calls. It carries no robot
or task knowledge — that lives in the embodiment adapter.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from embodied_control.logging.logger import EcLogger
from embodied_control.transport.protocol import (
    build_act_request,
    build_reset_request,
)


class PolicyClientError(RuntimeError):
    pass


class PolicyClient:
    def __init__(self, host: str, port: int, timeout_s: float = 30.0, logger: EcLogger | None = None):
        self.host = host
        self.port = int(port)
        self.base = f"http://{host}:{self.port}"
        self.timeout_s = timeout_s
        self.logger = logger or EcLogger.null()

    # --- low-level -------------------------------------------------------
    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        self.logger.debug("policy.request", method=method, path=path)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:  # 4xx/5xx still carry a JSON body
            body = exc.read()
        except OSError as exc:
            # URLError, ConnectionResetError, ConnectionRefusedError, timeouts, etc.
            # (all OSError subclasses). Common while a container is still starting;
            # wait_healthy retries on PolicyClientError, so wrap it as one.
            raise PolicyClientError(f"cannot reach policy service at {self.base}: {exc}") from exc
        return json.loads(body) if body else {}

    # --- operations ------------------------------------------------------
    def health(self) -> dict:
        return self._request("GET", "/health")

    def wait_healthy(self, timeout_s: float = 30.0, interval_s: float = 0.2) -> dict:
        deadline = time.time() + timeout_s
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                h = self.health()
                if h.get("status") == "ok" and h.get("model_loaded"):
                    self.logger.debug("policy.health_check_ok", endpoint=self.base)
                    return h
            except PolicyClientError as exc:
                last_err = exc
                # Expected during service startup (container/process still binding);
                # not logged as a warning unless the deadline is actually exceeded.
                self.logger.debug("policy.health_check_retry", endpoint=self.base, error=str(exc))
            time.sleep(interval_s)
        self.logger.warning("policy.health_check_timeout", endpoint=self.base, timeout_s=timeout_s)
        raise PolicyClientError(
            f"policy service at {self.base} not healthy within {timeout_s}s"
            + (f" (last error: {last_err})" if last_err else "")
        )

    def describe(self) -> dict:
        return self._request("POST", "/describe", {})

    def reset(self, episode_keys, seed: int) -> dict:
        resp = self._request("POST", "/reset", build_reset_request(episode_keys, seed))
        if resp.get("status") != "ok":
            raise PolicyClientError(f"policy reset failed: {resp}")
        return resp

    def act(self, request_id, episode_keys, observations, requested_horizon) -> dict:
        resp = self._request(
            "POST",
            "/act",
            build_act_request(request_id, episode_keys, observations, requested_horizon),
        )
        if resp.get("status") != "ok":
            raise PolicyClientError(f"policy act failed: {resp.get('status')}: {resp.get('error')}")
        return resp
