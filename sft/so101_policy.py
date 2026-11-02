import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_so101_example() -> dict:
    """Creates a random input example for the SO-101 policy."""
    return {
        "observation/state": np.random.rand(6),
        "observation/front_image": np.random.randint(
            256, size=(480, 640, 3), dtype=np.uint8
        ),
        "observation/side_image": np.random.randint(
            256, size=(480, 640, 3), dtype=np.uint8
        ),
        "prompt": "Pick up the red block and place it on top of the green block.",
    }


def _parse_image(image) -> np.ndarray:
    """
    Convert LeRobot images into uint8 HWC format expected by OpenPI.
    """
    image = np.asarray(image)

    # LeRobot training pipeline may return images as float [0, 1].
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)

    # Convert CHW -> HWC if necessary.
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """
    Converts SO-101 observations/actions into the format expected by pi0/pi0.5.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        front_image = _parse_image(data["observation/front_image"])
        side_image = _parse_image(data["observation/side_image"])

        inputs = {
            "state": data["observation/state"],

            "image": {
                # Main third-person camera.
                "base_0_rgb": front_image,

                # Use the second fixed camera as the second visual input.
                "left_wrist_0_rgb": side_image,

                # SO-101 dataset has no third camera.
                "right_wrist_0_rgb": np.zeros_like(front_image),
            },

            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,

                # Same padding convention as official Libero policy.
                "right_wrist_0_rgb": (
                    np.True_
                    if self.model_type == _model.ModelType.PI0_FAST
                    else np.False_
                ),
            },
        }

        # Available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Language instruction.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """
    Converts model outputs back to SO-101 action format.
    """

    def __call__(self, data: dict) -> dict:
        # SO-101:
        # 5 arm joints + 1 gripper = 6 action dimensions.
        return {
            "actions": np.asarray(data["actions"][..., :6])
        }
