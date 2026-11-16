"""SFT JPEG transport maps robot state and returned actions exactly once."""
import importlib
import io
import sys
import types

import numpy as np
from PIL import Image

from rlt_online.joint_mapping import JOINTS, JointMapping


def test_sft_server_joint_mapping_round_trip(monkeypatch):
    openpi = types.ModuleType("openpi")
    openpi.__path__ = []
    for name in ("openpi.policies", "openpi.serving", "openpi.training"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    for module in ("openpi.policies.policy_config", "openpi.serving.websocket_policy_server",
                   "openpi.training.config"):
        monkeypatch.setitem(sys.modules, module, types.ModuleType(module))
    fake_config = types.ModuleType("sft.so101_config")
    fake_config.register = lambda: None
    monkeypatch.setitem(sys.modules, "sft.so101_config", fake_config)
    monkeypatch.setitem(sys.modules, "openpi", openpi)
    sys.modules.pop("sft.serve_policy_fixed_noise", None)
    server = importlib.import_module("sft.serve_policy_fixed_noise")

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(frame).save(stream, format="JPEG")

    class Policy:
        def infer(self, obs, *, noise):
            self.observed = obs["observation/state"].copy()
            assert noise.shape == (50, 32)
            return {"actions": np.repeat(self.observed[None], 50, axis=0)}

    mapping = JointMapping({"joint_names": JOINTS,
                            "scale": [1.0] * 6,
                            "offset": [1., -2., 3., -4., 5., -.5]})
    policy = Policy()
    wrapped = server.FixedNoiseJpegPolicy(policy, action_horizon=50,
                                          action_dim=32, noise_seed=0, mapping=mapping)
    state = np.array([10., 20., 30., 40., 50., 12.], dtype=np.float32)
    reply = wrapped.infer({server.FRONT_JPEG_KEY: stream.getvalue(),
                           server.SIDE_JPEG_KEY: stream.getvalue(),
                           "observation/state": state, "prompt": "stack blocks"})
    np.testing.assert_allclose(policy.observed, mapping.to_model(state))
    np.testing.assert_allclose(reply["actions"][0], state)
    assert wrapped.metadata["noise_shape"] == [50, 32]
