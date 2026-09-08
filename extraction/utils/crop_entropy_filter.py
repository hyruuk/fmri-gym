#!/usr/bin/env python3
"""
Conservative crop entropy filter using grayscale Shannon entropy.

Removal rule:
    REMOVE if Shannon entropy == 0

The script reconstructs each SAM2 segment crop from:
- the original extracted gameplay frame
- the saved SAM2 .npy mask

Usage:
    python extraction/utils/crop_entropy_filter.py \
        data/extracted_frames/vizdoom__defend_center \
        data/segmentation/vizdoom__defend_center \
        --output data/filtered_segments/vizdoom__defend_center
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray
from skimage.measure import shannon_entropy


ZERO_ATOL = 1e-12


def find_frame(frames_dir: Path, frame_name: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        candidate = frames_dir / f"{frame_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def reconstruct_segment(
    frame: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]] | None:
    """
    Return:
        crop_rgb
        crop_mask
        bbox = (x1, y1, x2, y2)
    """
    mask = np.asarray(mask).astype(bool)

    if mask.shape[:2] != frame.shape[:2]:
        raise ValueError(
            f"Mask/frame shape mismatch: "
            f"mask={mask.shape[:2]}, frame={frame.shape[:2]}"
        )

    ys, xs = np.where(mask)

    if len(xs) == 0:
        return None

    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1

    crop_rgb = frame[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    if crop_rgb.size == 0:
        return None

    return crop_rgb, crop_mask, (x1, y1, x2, y2)


def compute_shannon_entropy_inside_mask(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
) -> float:
    """
    Standard Shannon entropy of grayscale intensities using only
    pixels that belong to the SAM2 segment.
    """
    gray = rgb2gray(crop_rgb)
    pixels = gray[crop_mask]

    if pixels.size == 0:
        return 0.0

    pixels_u8 = np.clip(pixels * 255.0, 0, 255).astype(np.uint8)

    entropy = float(
        shannon_entropy(
            pixels_u8,
            base=2
        )
    )

    if np.isclose(entropy, 0.0, atol=ZERO_ATOL):
        entropy = 0.0

    return entropy



def process_segments(
    frames_dir: Path,
    segments_dir: Path,
) -> pd.DataFrame:

    records = []

    frame_dirs = sorted(
        path
        for path in segments_dir.iterdir()
        if path.is_dir()
    )

    for frame_dir in frame_dirs:

        frame_name = frame_dir.name
        segments_json = frame_dir / "segments.json"

        if not segments_json.exists():
            continue

        frame_path = find_frame(
            frames_dir,
            frame_name
        )

        if frame_path is None:
            print(
                f"[skip] source frame not found: "
                f"{frame_name}"
            )
            continue

        frame = np.asarray(
            Image.open(frame_path).convert("RGB")
        )

        with segments_json.open(
            "r",
            encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        segments = metadata.get("segments", [])

        for segment_idx, segment in enumerate(segments):

            mask_rel = segment.get("mask_path")

            if not mask_rel:
                continue

            mask_path = frame_dir / mask_rel

            if not mask_path.exists():
                print(
                    f"[skip] mask not found: "
                    f"{mask_path}"
                )
                continue

            mask = np.load(mask_path)

            reconstructed = reconstruct_segment(
                frame,
                mask,
            )

            if reconstructed is None:
                continue

            crop_rgb, crop_mask, bbox = reconstructed
            x1, y1, x2, y2 = bbox

            shannon_h = compute_shannon_entropy_inside_mask(
                crop_rgb,
                crop_mask,
            )
            shannon_zero = bool(
                np.isclose(
                    shannon_h,
                    0.0,
                    atol=ZERO_ATOL
                )
            )

            # FINAL FILTER RULE:
            # remove if Shannon entropy is exactly zero
            remove = shannon_zero
            keep = not remove

            records.append(
                {
                    "frame_name": frame_name,
                    "frame_path": str(frame_path),
                    "segment_idx": segment_idx,
                    "mask_path": str(mask_path),

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,

                    "width": x2 - x1,
                    "height": y2 - y1,
                    "mask_area_px": int(
                        np.asarray(mask)
                        .astype(bool)
                        .sum()
                    ),

                    "shannon_entropy": shannon_h,
                    "shannon_zero": shannon_zero,

                    "remove": remove,
                    "keep": keep,
                }
            )

    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame) -> dict:

    total = len(df)
    removed_n = int(df["remove"].sum())
    kept_n = int(df["keep"].sum())

    def pct(n):
        return 100.0 * n / total if total else 0.0

    print("=" * 72)
    print("CROP ENTROPY FILTER")
    print("=" * 72)
    print(f"Total segments:             {total}")
    print()
    print(
        f"Removed (Shannon == 0):     "
        f"{removed_n} "
        f"({pct(removed_n):.2f}%)"
    )
    print(
        f"Kept:                       "
        f"{kept_n} "
        f"({pct(kept_n):.2f}%)"
    )

    return {
        "total_segments": total,
        "removed": removed_n,
        "removed_percentage": pct(removed_n),
        "kept": kept_n,
        "kept_percentage": pct(kept_n),
        "filter_rule": "remove if shannon_entropy == 0",
    }


def save_outputs(
    df: pd.DataFrame,
    output_dir: Path,
    summary: dict,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        output_dir /
        "crop_entropy_filter_metadata.csv"
    )

    pkl_path = (
        output_dir /
        "crop_entropy_filter_metadata.pkl"
    )

    summary_path = (
        output_dir /
        "crop_entropy_filter_summary.json"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    with pkl_path.open("wb") as f:
        pickle.dump(
            df,
            f
        )

    with summary_path.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print()
    print("Saved:")
    print(f"  {csv_path}")
    print(f"  {pkl_path}")
    print(f"  {summary_path}")


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Filter SAM2 crops using zero grayscale Shannon entropy."
        )
    )

    parser.add_argument(
        "frames_dir",
        type=Path,
        help="Path to extracted gameplay frames."
    )

    parser.add_argument(
        "segments_dir",
        type=Path,
        help="Path to SAM2 segmentation output."
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory where filter metadata is saved."
    )

    args = parser.parse_args()

    frames_dir = args.frames_dir.resolve()
    segments_dir = args.segments_dir.resolve()
    output_dir = args.output.resolve()

    if not frames_dir.exists():
        raise FileNotFoundError(
            f"Frames directory does not exist: "
            f"{frames_dir}"
        )

    if not segments_dir.exists():
        raise FileNotFoundError(
            f"Segmentation directory does not exist: "
            f"{segments_dir}"
        )

    print(f"Frames:   {frames_dir}")
    print(f"Segments: {segments_dir}")
    print(f"Output:   {output_dir}")
    print()

    df = process_segments(
        frames_dir=frames_dir,
        segments_dir=segments_dir,
    )

    if df.empty:
        raise RuntimeError(
            "No valid segments were found."
        )

    summary = print_summary(df)

    save_outputs(
        df=df,
        output_dir=output_dir,
        summary=summary,
    )


if __name__ == "__main__":
    main()