"""The LeRobot checkpoint wrapper accepts the original OpenPI stats asset."""

import json

import pytest

from sft.prepare_rl_checkpoint import _openpi_stats


@pytest.mark.parametrize("layout", ["nested", "flat", "fork"])
def test_stats_wrapper_uses_same_quantiles_as_online_learner(tmp_path, layout):
    state = {"q01": [-1., -2., -3., -4., -5., 0.],
             "q99": [1., 2., 3., 4., 5., 100.]}
    action = {"q01": [-6., -5., -4., -3., -2., 0.],
              "q99": [6., 5., 4., 3., 2., 100.]}
    if layout == "nested":
        data = {"norm_stats": {"state": state, "actions": action}}
    elif layout == "flat":
        data = {"state": state, "actions": action}
    else:
        data = {"observation.state": state, "action": action}
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    result = _openpi_stats(path)
    assert result["observation.state"] == state
    assert result["action"] == action


def test_stats_wrapper_rejects_invalid_quantile_span(tmp_path):
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps({"norm_stats": {
        "state": {"q01": [0] * 6, "q99": [0] * 6},
        "actions": {"q01": [-1] * 6, "q99": [1] * 6},
    }}), encoding="utf-8")
    with pytest.raises(ValueError, match="quantile ranges"):
        _openpi_stats(path)
