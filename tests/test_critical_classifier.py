"""Classifier and hysteresis tests for the Stage 4 checkpoint."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from lerobot.policies.pi05_rlt.critical import CriticalPhaseClassifier, CriticalPhaseDetector


class CriticalPhaseClassifierTest(unittest.TestCase):
    def test_classifier_output_shape(self):
        model = CriticalPhaseClassifier(z_dim=8, proprio_dim=6, hidden_dim=16)
        z = torch.randn(4, 8)
        p = torch.randn(4, 6)
        logits = model(z, p)
        self.assertEqual(tuple(logits.shape), (4, 2))

    def test_detector_loads_checkpoint_and_applies_hysteresis(self):
        torch.manual_seed(0)  # classifier training below is seed-sensitive
        model = CriticalPhaseClassifier(z_dim=4, proprio_dim=2, hidden_dim=8)
        # Train a trivially separable classifier: z[:, 0] > 0 -> critical.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        z_train = torch.cat([torch.randn(64, 4) + 1.0, torch.randn(64, 4) - 1.0])
        p_train = torch.randn(128, 2)
        labels = torch.cat([torch.ones(64, dtype=torch.long), torch.zeros(64, dtype=torch.long)])
        for _ in range(200):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(z_train, p_train), labels)
            loss.backward()
            optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "cls.pt"
            torch.save({
                "classifier_state_dict": model.state_dict(),
                "proprio_mean": p_train.mean(0).numpy().astype(np.float32),
                "proprio_std": p_train.std(0).numpy().astype(np.float32) + 1e-6,
                "z_dim": 4,
                "proprio_dim": 2,
                "hidden_dim": 8,
                "num_layers": 2,
            }, checkpoint_path)

            detector = CriticalPhaseDetector(
                str(checkpoint_path), "cpu",
                threshold_on=0.7, threshold_off=0.3, smooth_steps=5,
            )
            # Positive z -> on after a few updates; negative z -> off.
            state = None
            for _ in range(10):
                candidate = detector.update(np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32),
                                            np.array([0.0, 0.0], dtype=np.float32))
                if candidate is not None:
                    state = candidate
            self.assertTrue(state)
            detector.reset()
            state = None
            for _ in range(10):
                candidate = detector.update(np.array([-2.0, 0.0, 0.0, 0.0], dtype=np.float32),
                                            np.array([0.0, 0.0], dtype=np.float32))
                if candidate is not None:
                    state = candidate
            self.assertFalse(state)

    def test_detector_rejects_wrong_z_dim(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "cls.pt"
            torch.save({
                "classifier_state_dict": CriticalPhaseClassifier(
                    z_dim=4, proprio_dim=2, hidden_dim=8).state_dict(),
                "proprio_mean": np.zeros(2, dtype=np.float32),
                "proprio_std": np.ones(2, dtype=np.float32),
                "z_dim": 4,
                "proprio_dim": 2,
                "hidden_dim": 8,
                "num_layers": 2,
            }, checkpoint_path)
            detector = CriticalPhaseDetector(str(checkpoint_path), "cpu")
            with self.assertRaises(ValueError):
                detector.update(np.zeros(8, dtype=np.float32), np.zeros(2, dtype=np.float32))
