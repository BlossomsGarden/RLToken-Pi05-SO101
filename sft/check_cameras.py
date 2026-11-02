"""Show camera frames for verifying Windows OpenCV indices before robot motion."""

import argparse
from pathlib import Path

import cv2

from sft.deploy_openpi import load_env_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    env = load_env_file(args.env_file)
    for name in ("FRONT_CAMERA", "SIDE_CAMERA"):
        index = env.get(name)
        if index is None:
            parser.error(f"Missing {name} in {args.env_file}")
        camera = cv2.VideoCapture(int(index) if index.isdecimal() else index)
        try:
            if not camera.isOpened():
                raise RuntimeError(f"{name}={index}: cannot open")
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError(f"{name}={index}: no frame")
            cv2.imshow(f"{name}={index}", frame)
            print(f"{name}={index}: {frame.shape[1]}x{frame.shape[0]}")
            cv2.waitKey(1500)
        finally:
            camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
