"""Calibration transfer: first five joints in degrees, gripper in LeRobot range units."""

from __future__ import annotations

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class JointMapping:
    """dataset_joint_value = scale * client_joint_value + offset (per joint)."""

    def __init__(self, config: dict):
        if set(config) != {"joint_names", "scale", "offset"} or tuple(config["joint_names"]) != JOINTS:
            raise ValueError("Mapping must declare the canonical six SO-101 joint_names, scale and offset")
        self.scale = np.asarray(config["scale"], dtype=np.float32)
        self.offset = np.asarray(config["offset"], dtype=np.float32)
        if (self.scale.shape != (6,) or self.offset.shape != (6,) or
            not np.isfinite(self.scale).all() or not np.isfinite(self.offset).all() or np.any(self.scale == 0)):
            raise ValueError("Joint mapping must be finite and invertible")

    def to_model(self, physical):
        physical = self._check(physical)
        return physical * self.scale + self.offset

    def to_physical(self, model):
        model = self._check(model)
        return (model - self.offset) / self.scale

    @staticmethod
    def _check(value):
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape[-1:] != (6,) or not np.isfinite(arr).all():
            raise ValueError("Expected finite SO-101 joint values in canonical order")
        return arr
