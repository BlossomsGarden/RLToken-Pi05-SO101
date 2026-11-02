"""Load the local SO101 config before running an upstream OpenPI entry point."""

import argparse
import runpy
import sys
from pathlib import Path

from sft.so101_config import register


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entry", choices=("train", "compute_norm_stats", "convert"))
    parser.add_argument("--openpi-root", type=Path, required=True)
    args, upstream_args = parser.parse_known_args()
    scripts = {
        "train": "scripts/train.py",
        "compute_norm_stats": "scripts/compute_norm_stats.py",
        "convert": None,
    }
    target = (Path(__file__).with_name("convert_jax_model_to_pytorch.py")
              if args.entry == "convert" else args.openpi_root / scripts[args.entry])
    if not target.is_file():
        parser.error(f"OpenPI entry point missing: {target}")
    register()
    sys.argv = [str(target), *upstream_args]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
