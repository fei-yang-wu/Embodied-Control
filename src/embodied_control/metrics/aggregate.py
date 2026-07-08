"""Aggregate per-episode records into run-level metrics.

Preserves raw per-episode outcomes (in episodes.jsonl) and computes only common
aggregates here — success rate, mean length/return, and policy-latency
percentiles — so later statistical analysis stays possible.
"""

from __future__ import annotations

from embodied_control.config.schemas import EpisodeRecord, LatencyStats, RunMetrics


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def latency_stats(latencies_ms: list[float]) -> LatencyStats:
    if not latencies_ms:
        return LatencyStats()
    s = sorted(latencies_ms)
    return LatencyStats(
        count=len(s),
        mean=sum(s) / len(s),
        p50=_percentile(s, 0.50),
        p90=_percentile(s, 0.90),
        p95=_percentile(s, 0.95),
        p99=_percentile(s, 0.99),
    )


def aggregate_run_metrics(
    run_id: str,
    num_requested: int,
    episodes: list[EpisodeRecord],
    latencies_ms: list[float],
) -> RunMetrics:
    completed = [e for e in episodes if e.status == "completed"]
    failed = [e for e in episodes if e.status == "failed"]
    n_completed = len(completed)
    successes = sum(1 for e in completed if e.success)
    success_rate = (successes / n_completed) if n_completed else 0.0
    mean_len = (
        sum(e.episode_length_steps for e in completed) / n_completed if n_completed else 0.0
    )
    mean_return = (sum(e.total_return for e in completed) / n_completed) if n_completed else 0.0
    num_requests = sum(e.policy.num_requests for e in episodes)
    return RunMetrics(
        run_id=run_id,
        num_episodes_requested=num_requested,
        num_episodes_completed=n_completed,
        num_episodes_failed=len(failed),
        success_rate=round(success_rate, 6),
        mean_episode_length_steps=round(mean_len, 4),
        mean_return=round(mean_return, 6),
        num_policy_requests=num_requests,
        policy_latency_ms=latency_stats(latencies_ms),
        extra={"num_successes": successes},
    )
