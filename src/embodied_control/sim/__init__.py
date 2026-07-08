"""Simulator backends. M1 ships an in-process MuJoCo stepped backend.

MuJoCo is flexible enough to drive directly from the host (reset/step), so it does
NOT use the delegated (black-box evaluator) path. Heavier / service-shaped
simulators like LIBERO are the delegated + Docker milestone.
"""
