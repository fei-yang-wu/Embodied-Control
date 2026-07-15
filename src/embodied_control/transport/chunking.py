"""Action-chunk buffer + refill bookkeeping shared by every rollout loop.

Every rollout loop (stepped MuJoCo, ``fake_delegated_eval``, ``libero_eval``)
does the same thing: hold a buffer of actions from the last policy response,
pop one per env step, and call the policy again only once the buffer is
empty -- tracking how many requests that took and what chunk lengths came
back, for ``EpisodePolicyStats``. That bookkeeping was duplicated three times
before this module existed; this is the one place it lives now.

This intentionally does NOT call ``client.act()`` itself -- callers differ on
request-id formatting, exception/fallback handling, and what they do with the
rest of the response (timing, video capture), so that stays call-site-specific.
Stdlib-only: this runs inside every delegated container, including the
stdlib-only ones (see ``docs/gotchas.md``'s ``LogConfig`` story for why that
constraint matters transitively).
"""

from __future__ import annotations


class ChunkScheduler:
    def __init__(self, requested_horizon: int):
        self.requested_horizon = max(1, int(requested_horizon))
        self._buffer: list[list[float]] = []
        self.num_requests = 0
        self._horizons: list[int] = []

    @property
    def empty(self) -> bool:
        return not self._buffer

    def fill(self, chunk: list[list[float]]) -> None:
        self.num_requests += 1
        self._horizons.append(len(chunk))
        self._buffer = list(chunk)

    def pop(self) -> list[float]:
        return self._buffer.pop(0)

    @property
    def mean_action_horizon(self) -> float:
        return round(sum(self._horizons) / len(self._horizons), 4) if self._horizons else 0.0
