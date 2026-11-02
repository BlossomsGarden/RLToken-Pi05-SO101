"""Register the pick-and-place SFT recipe in an installed OpenPI checkout.

Use these entry points instead of modifying OpenPI's upstream config.py. The
20k weights and their norm stats remain on the GPU server and outside Git.
"""

import dataclasses

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config, optimizer, weight_loaders

from sft.so101_policy import SO101Inputs, SO101Outputs


NAME = "pi05_so101_pickplace_v3"
DATASET = "BlossomsGarden/pick-and-place-v3_20261025_051506"


@dataclasses.dataclass(frozen=True)
class PickPlaceDataConfig(config.DataConfigFactory):
    repo_id: str = DATASET

    def create(self, assets_dirs, model_config):
        # Raw LeRobot action is absolute in degrees. OpenPI normalizes *after*
        # this transform, then reverses it on policy output using request state.
        delta_mask = transforms.make_bool_mask(5, -1)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            action_sequence_keys=("action",),
            prompt_from_task=True,
            repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
                "observation/front_image": "observation.images.front",
                "observation/side_image": "observation.images.side",
                "observation/state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
            })]),
            data_transforms=transforms.Group(
                inputs=[SO101Inputs(model_type=model_config.model_type),
                        transforms.DeltaActions(delta_mask)],
                outputs=[transforms.AbsoluteActions(delta_mask), SO101Outputs()],
            ),
            model_transforms=config.ModelTransformFactory()(model_config),
        )


def register() -> config.TrainConfig:
    """Expose the same config to training, inference, and official conversion."""
    recipe = config.TrainConfig(
        name=NAME,
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=50, discrete_state_input=False,
        ),
        data=PickPlaceDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=2.5e-5,
            decay_steps=20000, decay_lr=2.5e-6,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        batch_size=16,  # Four H200 workers, four examples per worker.
        num_workers=8,
        num_train_steps=20000,
        save_interval=5000,
        keep_period=10000,
    )
    config._CONFIGS_DICT[NAME] = recipe
    return recipe
