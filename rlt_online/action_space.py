"""Exactly invertible SO-101 delta/absolute action and quantile conversions."""

import numpy as np

DELTA_JOINTS = np.array([True, True, True, True, True, False])


def _check(value):
    result = np.asarray(value, dtype=np.float32)
    if result.shape[-1:] != (6,) or not np.isfinite(result).all():
        raise ValueError("Action/state must contain six finite joints")
    return result


def denormalize(values, q01, q99):
    values, lo, hi = _check(values), _check(q01), _check(q99)
    if np.any(hi <= lo):
        raise ValueError("Quantile ranges must be strictly positive")
    return (values + 1) * (hi - lo) / 2 + lo


def normalize(values, q01, q99):
    values, lo, hi = _check(values), _check(q01), _check(q99)
    if np.any(hi <= lo):
        raise ValueError("Quantile ranges must be strictly positive")
    return 2 * (values - lo) / (hi - lo) - 1


def delta_to_absolute(model_action, observation_state, q01, q99):
    """The first five joints are relative to the SAME chunk-start state; gripper is absolute."""
    action, state = denormalize(model_action, q01, q99), _check(observation_state)
    result = action.copy()
    result[..., DELTA_JOINTS] += state[..., DELTA_JOINTS]
    return result


def absolute_to_delta(absolute_action, observation_state, q01, q99):
    """Inverse of delta_to_absolute, including the unchanged gripper."""
    result, state = _check(absolute_action).copy(), _check(observation_state)
    result[..., DELTA_JOINTS] -= state[..., DELTA_JOINTS]
    return normalize(result, q01, q99)
