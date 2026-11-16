#!/usr/bin/env python3
"""Label one placement-alignment span in each red-on-green episode (0–29).

Renders numbered front-camera thumbnails and incrementally saves inclusive,
zero-based episode-local frame intervals in an annotations JSON file.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--annotations_out", type=str, required=True,
                        help="JSON output path (incremental: annotated episodes are skipped).")
    parser.add_argument("--camera", type=str, default="observation.images.front")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Dataset frame rate (used to compute episode offsets in shared videos).")
    parser.add_argument("--sheet_dir", type=str, default="annotation_sheets_alignment_30",
                        help="Directory for numbered front-camera contact sheets.")
    return parser.parse_args()


def load_existing(path: Path) -> dict[int, list[tuple[int, int]]]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data.get("annotations"), list):
        raise ValueError("Annotation JSON must contain an annotations list")
    annotations: dict[int, list[tuple[int, int]]] = {}
    for item in data.get("annotations", []):
        episode, start, end = (int(item[key]) for key in ("episode", "start", "end"))
        if episode not in range(30) or episode in annotations or not 0 <= start <= end:
            raise ValueError(f"Invalid or duplicated red-on-green annotation for episode {episode}")
        annotations[episode] = [(start, end)]
    return annotations


def save_annotations(path: Path, annotations: dict[int, list[tuple[int, int]]]) -> None:
    payload = {
        "annotations": [
            {"episode": episode, "start": start, "end": end}
            for episode in sorted(annotations)
            for start, end in annotations[episode]
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.flush()
    tmp.replace(path)
    logger.info("Saved %d intervals across %d episodes to %s",
                len(payload["annotations"]), len(annotations), path)


def episode_video_paths(
    dataset_path: Path,
    fps: float,
    cameras: tuple[str, ...] | None = None,
) -> list[tuple[int, dict[str, tuple[Path, int]], int]]:
    """Return [(episode_index, {camera: (video_path, start_frame)}, length)].

    Newer LeRobot metadata (0.5.x) stores per-episode video references as
    ``videos/<feature>/chunk_index`` + ``file_index``; several episodes share
    one video file, so the episode's position inside it is
    ``from_timestamp * fps`` (a continuous recording). The timestamps are
    RELATIVE TO EACH FILE (every file restarts at 0), and cameras can roll to
    a new file at different times (e.g. top every 688 s but wrist every 344 s),
    so each camera gets its OWN start_frame. Missing cameras (no parquet
    columns / missing file) are omitted from the dict.
    """
    from datasets import Dataset

    if cameras is None:
        with open(dataset_path / "meta" / "info.json", encoding="utf-8") as file:
            features = json.load(file).get("features", {})
        cameras = tuple(
            name for name, feature in features.items() if feature.get("dtype") == "video"
        )
        if not cameras:
            raise ValueError(f"No video cameras found in {dataset_path / 'meta' / 'info.json'}")

    episodes_dir = dataset_path / "meta" / "episodes"
    parquet_files = sorted(episodes_dir.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episodes parquet under {episodes_dir}")
    episodes = Dataset.from_parquet([str(path) for path in parquet_files])

    videos_dir = dataset_path / "videos"
    result: list[tuple[int, dict[str, tuple[Path, int]], int]] = []
    for row in episodes:
        episode_index = int(row["episode_index"])
        if episode_index not in range(30):
            continue
        length = int(row["length"])
        paths: dict[str, tuple[Path, int]] = {}
        for cam in cameras:
            try:
                chunk_index = int(row[f"videos/{cam}/chunk_index"])
                file_index = int(row[f"videos/{cam}/file_index"])
                from_timestamp = float(row[f"videos/{cam}/from_timestamp"])
            except (KeyError, TypeError):
                continue
            video_path = (
                videos_dir / cam / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.mp4"
            )
            if video_path.exists():
                start_frame = int(round(from_timestamp * fps))
                paths[cam] = (video_path, start_frame)
        if not paths:
            raise FileNotFoundError(
                f"Episode {episode_index}: no video files found for cameras {cameras}"
            )
        result.append((episode_index, paths, length))
    return sorted(result, key=lambda item: item[0])



def _seek_to_frame(container, stream, target_frame: int, fps: float) -> None:
    """Seek to the key frame at or before ``target_frame`` (video-global index).

    PyAV's ``Container.seek`` interprets the offset in ``stream.time_base``
    units when a stream is given (NOT microseconds), so convert frame -> ticks.
    """
    offset = int(max(0, target_frame) * (1.0 / fps) / float(stream.time_base))
    container.seek(offset, stream=stream)


def iter_episode_frames(video_path: Path, start_frame: int, length: int, fps: float):
    """Yield (episode_local_frame, bgr24 ndarray) for an episode's frames (PyAV)."""
    import av

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    # Seek a little before the episode start (keyframe-aligned), then decode
    # forward, skipping frames until the episode's first frame.
    _seek_to_frame(container, stream, max(0, start_frame - 10), fps)
    try:
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"Video frame has no timestamp: {video_path}")
            seconds = float(frame.pts * stream.time_base)
            frame_no = int(round(seconds * fps))
            if frame_no >= start_frame + length:
                break
            if frame_no >= start_frame:
                yield frame_no - start_frame, frame.to_ndarray(format="bgr24")
    finally:
        container.close()



def render_contact_sheet(
    video_path: Path,
    start_frame: int,
    length: int,
    fps: float,
    *,
    cols: int,
    thumb_width: int,
) -> np.ndarray:
    """Render one episode as a montage of thumbnails labelled with frame numbers."""
    thumb_height = int(thumb_width * 3 / 4)
    label_height = 22
    cell_w, cell_h = thumb_width, thumb_height + label_height
    rows = (length + cols - 1) // cols
    sheet = np.full((rows * cell_h, cols * cell_w, 3), 30, dtype=np.uint8)
    rendered = set()
    for episode_frame, image in iter_episode_frames(video_path, start_frame, length, fps):
        if episode_frame in rendered:
            continue
        col = episode_frame % cols
        row = episode_frame // cols
        thumb = cv2.resize(image, (thumb_width, thumb_height))
        y0 = row * cell_h + label_height
        x0 = col * cell_w
        sheet[y0:y0 + thumb_height, x0:x0 + thumb_width] = thumb
        cv2.putText(sheet, f"{episode_frame}", (x0 + 2, y0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        rendered.add(episode_frame)
    if len(rendered) != length:
        raise ValueError(f"Expected {length} frames in {video_path}, rendered {len(rendered)}")
    return sheet


def annotate_episode_sheet(
    episode: int,
    video_path: Path,
    start_frame: int,
    length: int,
    fps: float,
    *,
    sheet_dir: Path,
    cols: int,
    thumb_width: int,
) -> tuple[list[tuple[int, int]], bool]:
    """Render a contact sheet, then collect one interval from terminal input.

    Returns (intervals, quit_all).
    """
    sheet_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = sheet_dir / f"episode_{episode:04d}.png"
    sheet = render_contact_sheet(video_path, start_frame, length, fps,
                                 cols=cols, thumb_width=thumb_width)
    if not cv2.imwrite(str(sheet_path), sheet):
        raise OSError(f"Could not write annotation sheet: {sheet_path}")
    print(f"\n=== Episode {episode} | {length} frames | sheet: {sheet_path} ===")
    print("Mark before wrist alignment through release and stable placement.")
    print("  '<start> <end>'  set single interval | 'undo' | 'skip' | 'quit' | blank = next episode")
    intervals: list[tuple[int, int]] = []
    while True:
        try:
            line = input(f"  ep {episode} interval: ").strip().lower()
        except EOFError:
            return intervals, True
        if not line:
            if intervals:
                return intervals, False
            print("  mark an interval or type 'skip'")
            continue
        if line in ("q", "quit"):
            return intervals, True
        if line == "skip":
            return intervals, False
        if line == "undo":
            if intervals:
                removed = intervals.pop()
                print(f"  removed {removed}")
            else:
                print("  nothing to undo")
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError:
                print("  invalid numbers")
                continue
            if not (0 <= start <= end < length):
                print(f"  out of range [0, {length - 1}]")
                continue
            intervals[:] = [(start, end)]
            print(f"  interval [{start}, {end}] ready; Enter to save, or enter a replacement")
        else:
            print("  expected: '<start> <end>'")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path).resolve(strict=True)
    manifest = json.loads((Path(__file__).resolve().parents[1] / "configs/red_on_green_episodes.json").read_text(encoding="utf-8"))
    if manifest["episode_indices"] != list(range(30)) or dataset_path.name != manifest["dataset"].split("/")[-1]:
        raise ValueError("Dataset or episode manifest does not match the red-on-green episodes 0–29")
    info = json.loads((dataset_path / "meta/info.json").read_text(encoding="utf-8"))
    if info["fps"] != args.fps or info["features"].get(args.camera, {}).get("dtype") != "video":
        raise ValueError("Dataset FPS or front-camera name does not match the annotation settings")
    out_path = Path(args.annotations_out)
    annotations = load_existing(out_path)
    if annotations:
        logger.info("Loaded %d previously annotated episodes; they will be skipped.", len(annotations))

    episodes = episode_video_paths(dataset_path, args.fps, cameras=(args.camera,))
    if [item[0] for item in episodes] != list(range(30)):
        raise ValueError("Expected exactly red-on-green episodes 0 through 29")
    logger.info("Found %d red-on-green episodes in %s", len(episodes), dataset_path)
    for episode, _, length in episodes:
        if episode in annotations and annotations[episode][0][1] >= length:
            raise ValueError(f"Saved annotation for episode {episode} exceeds its {length} frames")

    try:
        for episode, video_infos, length in episodes:
            if episode in annotations:
                logger.info("Skipping episode %d (already annotated: %s)", episode, annotations[episode])
                continue
            sheet_video, sheet_start = video_infos[args.camera]
            intervals, should_quit = annotate_episode_sheet(
                episode, sheet_video, sheet_start, length, args.fps,
                sheet_dir=Path(args.sheet_dir), cols=12, thumb_width=160,
            )
            if intervals:
                annotations[episode] = intervals
                save_annotations(out_path, annotations)
            else:
                logger.info("Episode %d: no intervals marked, not saved.", episode)
            if should_quit:
                print("Quit requested.")
                break
    finally:
        if annotations:
            save_annotations(out_path, annotations)
    print(f"\nDone. Annotations: {out_path}")


if __name__ == "__main__":
    main()
