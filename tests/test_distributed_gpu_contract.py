import threading
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from scripts.distributed_gpu import GPUService, absolute_from_delta


class MemoryLearner:
    def __init__(self):
        self.records = []
        self.updates = 0

    def add(self, record):
        self.records.append(record)

    def raise_if_failed(self):
        pass

    def save(self, episode, provenance):
        return f"episode-{episode}.pt"


class DistributedGPUContractTests(unittest.TestCase):
    def make_service(self):
        service = object.__new__(GPUService)
        service.lock = threading.RLock()
        service.device = torch.device("cpu")
        service.learner = MemoryLearner()
        service.pending = None
        service.next_id = 1
        service.current_episode = None
        service.detector = None
        service.critical = service.detected_once = False
        service.args = SimpleNamespace(actor_start_updates=1000)
        service.action_lo = torch.zeros(6)
        service.action_hi = torch.ones(6)
        service.provenance = {}
        def fake_boundary(payload):
            boundary = {"z": np.zeros(2048, dtype=np.float32),
                        "proprio": np.zeros(6, dtype=np.float32),
                        "raw_state": np.arange(10, 16, dtype=np.float32),
                        "ref": np.zeros((10, 6), dtype=np.float32)}
            return boundary, torch.zeros((10, 6)), torch.zeros((1, 10, 6))
        service._boundary = fake_boundary
        return service

    def test_partial_window_and_terminal_reward_enter_replay(self):
        service = self.make_service()
        first = service.plan({"task": "Pick up the red block and place it on top of the green block.",
                              "episode": 1, "step": 0, "critical": True})
        previous = {"id": first["id"], "actual": [[0.1] * 6] * 2,
                    "critical": True, "source": "HUMAN", "episode": 1, "step": 0}
        second = service.plan({"episode": 1, "step": 2, "critical": True, "previous": previous})
        self.assertEqual(len(service.learner.records), 1)
        self.assertEqual(service.learner.records[0].executed_steps, 2)
        self.assertTrue(service.learner.records[0].intervention_flag)
        final = {"id": second["id"], "actual": [[0.2] * 6] * 3,
                 "critical": True, "source": "VLA", "episode": 1, "step": 2}
        result = service.finish({"episode": 1, "previous": final, "reward": 1})
        self.assertTrue(result["ok"])
        self.assertTrue(service.learner.records[-1].done)
        self.assertEqual(service.learner.records[-1].rewards[2], 1)
        self.assertEqual(int(service.learner.records[-1].valid_mask.sum()), 3)

    def test_stale_window_is_rejected(self):
        service = self.make_service()
        service.plan({"episode": 1, "step": 0, "critical": True})
        with self.assertRaisesRegex(ValueError, "Stale"):
            service.plan({"episode": 1, "step": 1, "critical": True,
                          "previous": {"id": -1, "actual": [[0] * 6],
                                       "source": "VLA", "critical": True, "episode": 1, "step": 0}})

    def test_sft_delta_conversion_preserves_absolute_gripper(self):
        normalized = torch.zeros((1, 2, 6))
        low = torch.tensor([-2., -2., -2., -2., -2., 10.])
        high = torch.tensor([2., 2., 2., 2., 2., 30.])
        result = absolute_from_delta(normalized, [1, 2, 3, 4, 5, 99], low, high)
        torch.testing.assert_close(result[0, 0], torch.tensor([1., 2., 3., 4., 5., 20.]))

    def test_classifier_receives_unscaled_joint_angles(self):
        class Detector:
            def reset(self):
                pass
            def update(self, z, proprio):
                self.received = proprio
                return True
        service = self.make_service()
        service.detector = Detector()
        service.plan({"episode": 1, "step": 60})
        np.testing.assert_array_equal(service.detector.received, np.arange(10, 16))


if __name__ == "__main__":
    unittest.main()
