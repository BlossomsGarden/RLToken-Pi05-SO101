"""Capture front.png and side.png for checking Windows camera assignments."""

import argparse
from pathlib import Path

import cv2

from sft.deploy_openpi import env_value, load_env_file
from sft.camera_env import camera_setting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    env = load_env_file(args.env_file)
    for name, canonical, alias in (("front", "FRONT_CAMERA", "CAM_FRONT"),
                                    ("side", "SIDE_CAMERA", "CAM_SIDE")):
        env_key, index = camera_setting(
            {key: env_value(env, key) for key in (canonical, alias)}, canonical, alias,
        )
        if index is None:
            parser.error(f"Missing {canonical} (or {alias}) in {args.env_file}")
        camera = cv2.VideoCapture(int(index) if index.isdecimal() else index)
        try:
            if not camera.isOpened():
                raise RuntimeError(f"{env_key}={index}: cannot open")
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError(f"{env_key}={index}: no frame")
            output = Path.cwd() / f"{name}.png"
            if not cv2.imwrite(str(output), frame):
                raise RuntimeError(f"Could not save {output}")
            print(f"{env_key}={index} -> {output} ({frame.shape[1]}x{frame.shape[0]})")
        finally:
            camera.release()


if __name__ == "__main__":
    main()
