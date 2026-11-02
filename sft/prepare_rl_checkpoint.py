"""Add LeRobot metadata around the official converted OpenPI safetensors.

Run on the GPU server. Original converted files remain untouched; the output
contains a symlink to the weights and a LeRobot PI05 config for the RL scripts.
This is metadata preparation, not a second weight conversion.
"""

import argparse
import json
from pathlib import Path


JOINT_NAMES = (
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
)


def _openpi_stats(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for source, target in (("state", "observation.state"), ("actions", "action")):
        if source not in data:
            raise ValueError(f"OpenPI normalization stats missing {source!r}: {path}")
        item = data[source]
        if not isinstance(item, dict):
            raise ValueError(f"Invalid normalization stats for {source}: {path}")
        result[target] = {}
        for name in ("q01", "q99"):
            values = item.get(name)
            if not isinstance(values, list) or len(values) != 6:
                raise ValueError(f"Expected six {name} values in {source}: {path}")
            result[target][name] = values
    return result


def prepare(converted: Path, output: Path, norm_stats: Path) -> Path:
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    converted = converted.resolve(strict=True)
    weights = converted / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    meta = json.loads((converted / "config.json").read_text(encoding="utf-8"))
    if int(meta.get("action_horizon", -1)) != 50 or int(meta.get("action_dim", -1)) != 32:
        raise ValueError(f"Expected OpenPI 50x32 model, got: {meta}")
    if meta.get("precision") not in ("bfloat16", "float32"):
        raise ValueError(f"Unsupported checkpoint precision: {meta.get('precision')}")
    # The OpenPI converter copies its JAX norm_stats.json into assets. These
    # statistics describe DELTA arm actions, not raw absolute LeRobot actions.
    # Resolve explicitly to avoid silently substituting the dataset's abs stats.
    stats = _openpi_stats(norm_stats.resolve(strict=True))
    output = output.resolve()
    if output == converted or output.exists():
        raise FileExistsError("Use a NEW output directory; never overwrite the OpenPI conversion")
    output.mkdir(parents=True)
    features = {
        name: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
        for name in (
            "observation.images.front", "observation.images.side",
            "observation.images.empty_camera_0",
        )
    }
    features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
    cfg = PI05Config(
        input_features=features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
        dtype=meta["precision"],
        chunk_size=50,
        n_action_steps=50,
        paligemma_variant=meta["paligemma_variant"],
        action_expert_variant=meta["action_expert_variant"],
        use_relative_actions=True,
        action_feature_names=list(JOINT_NAMES),
        relative_exclude_joints=["gripper"],
        device="cuda",
        push_to_hub=False,
    )
    cfg.save_pretrained(output)
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    pre.save_pretrained(output)
    post.save_pretrained(output)
    (output / weights.name).symlink_to(weights)
    if (converted / "assets").exists():
        (output / "assets").symlink_to(converted / "assets", target_is_directory=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True,
                        help="OpenPI checkpoint assets/<repo-id>/norm_stats.json (delta action quantiles)")
    args = parser.parse_args()
    print(prepare(args.converted, args.output, args.norm_stats))


if __name__ == "__main__":
    main()
