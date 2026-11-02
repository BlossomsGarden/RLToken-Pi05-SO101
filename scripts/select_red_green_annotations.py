"""Filter locally annotated 120-episode intervals to the verified red→green 0..29 subset."""

import argparse
import json
from pathlib import Path


def select(source, manifest):
    included = set(json.loads(Path(manifest).read_text(encoding="utf-8"))["episode_indices"])
    if included != set(range(30)):
        raise ValueError("Expected verified red-on-green episode indices 0 through 29")
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(data.get("annotations"), list):
        raise ValueError("Expected annotation sheet JSON with an annotations list")
    selected = [entry for entry in data["annotations"] if int(entry["episode"]) in included]
    if not selected:
        raise ValueError("No annotations for red-on-green episodes; check the input file")
    if any(int(item["start"]) > int(item["end"]) for item in selected):
        raise ValueError("Invalid annotation interval")
    return {**data, "annotations": selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path("configs/red_on_green_episodes.json"))
    args = parser.parse_args()
    if args.annotations.resolve() == args.out.resolve():
        parser.error("Provide a new output file; leave original 120-episode annotations untouched")
    result = select(args.annotations, args.manifest)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Selected {len(result['annotations'])} intervals from red-on-green episodes")


if __name__ == "__main__":
    main()
