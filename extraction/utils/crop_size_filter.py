#!/usr/bin/env python3

"""
Conservative native crop-size filter for SAM2 segments.

A segment is marked for removal when its tight bounding-box area is below
the native-pixel threshold. No masks or source files are deleted.

Default rule:
    remove = bbox_area < 75

Example:
    python extraction/utils/crop_size_filter.py \
        data/segmentation/vizdoom__defend_center \
        --output data/filtered_segments/vizdoom__defend_center \
        --min-pixels 75
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def tight_bbox_from_mask(mask: np.ndarray):
    """Return (x1, y1, x2, y2) for the nonzero region of a mask."""
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1

    return x1, y1, x2, y2


def load_segments_json(path: Path):
    with path.open("r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data.get("segments", [])

    if isinstance(data, list):
        return data

    raise ValueError(f"Unsupported segments.json structure: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Mark SAM2 crops below a native bounding-box pixel threshold."
    )
    parser.add_argument(
        "segmentation_dir",
        type=Path,
        help="Root segmentation directory containing frame_* folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where filter metadata will be written.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=75,
        help="Remove crops with bbox_area < this value. Default: 75.",
    )

    args = parser.parse_args()

    seg_root = args.segmentation_dir
    output_dir = args.output
    min_pixels = args.min_pixels

    if min_pixels < 1:
        raise ValueError("--min-pixels must be >= 1")

    if not seg_root.exists():
        raise FileNotFoundError(f"Segmentation directory not found: {seg_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    frame_dirs = sorted(p for p in seg_root.iterdir() if p.is_dir())

    for frame_dir in frame_dirs:
        segments_json = frame_dir / "segments.json"

        if not segments_json.exists():
            continue

        segments = load_segments_json(segments_json)

        for segment_idx, segment in enumerate(segments):
            mask_rel = segment.get("mask_path")

            if not mask_rel:
                continue

            mask_path = frame_dir / mask_rel

            if not mask_path.exists():
                continue

            mask = np.load(mask_path).astype(bool)
            bbox = tight_bbox_from_mask(mask)

            if bbox is None:
                width = 0
                height = 0
                bbox_area = 0
                mask_area = 0
                remove = True
                x1 = y1 = x2 = y2 = None
            else:
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                bbox_area = width * height
                mask_area = int(mask.sum())

                # Conservative native-size rule
                remove = bbox_area < min_pixels

            records.append(
                {
                    "frame_name": frame_dir.name,
                    "segment_idx": segment_idx,
                    "mask_path": str(mask_path),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": width,
                    "height": height,
                    "bbox_area": bbox_area,
                    "mask_area": mask_area,
                    "min_pixels": min_pixels,
                    "remove": bool(remove),
                    "keep": bool(not remove),
                }
            )

    df = pd.DataFrame(records)

    csv_path = output_dir / "crop_size_filter_metadata.csv"
    pkl_path = output_dir / "crop_size_filter_metadata.pkl"
    summary_path = output_dir / "crop_size_filter_summary.json"

    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)

    total = len(df)
    removed = int(df["remove"].sum()) if total else 0
    kept = total - removed

    summary = {
        "segmentation_dir": str(seg_root),
        "min_pixels": min_pixels,
        "rule": f"remove if bbox_area < {min_pixels}",
        "total_segments": total,
        "removed": removed,
        "kept": kept,
        "removed_percent": (100.0 * removed / total) if total else 0.0,
        "kept_percent": (100.0 * kept / total) if total else 0.0,
    }

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Total segments: {total}")
    print(
        f"Removed (< {min_pixels} bbox pixels): "
        f"{removed} ({summary['removed_percent']:.2f}%)"
    )
    print(f"Kept: {kept} ({summary['kept_percent']:.2f}%)")
    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {pkl_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
