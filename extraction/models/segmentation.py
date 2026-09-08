"""
Run SAM2 automatic mask generation on temporally sampled gameplay frames.

The segmentation stage identifies candidate visual regions / objects that can
later be clustered with DINO features, labeled, and incorporated into a
symbolic game-state representation.

All information needed to reconstruct crops and masked regions later is
preserved, while avoiding redundant per-segment image files.
"""

import argparse
import json
import os
import pickle

import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


DEFAULT_CHECKPOINT = "segment-anything-2/checkpoints/sam2.1_hiera_small.pt"
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"


def get_device():
    """Select CUDA, Apple MPS, or CPU."""

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_sam2(
    checkpoint,
    model_cfg,
):
    """Load SAM2 once and return the automatic mask generator."""

    device = get_device()

    print("Device:", device)

    sam2 = build_sam2(
        model_cfg,
        checkpoint,
        device=device,
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=32,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.95,
        crop_n_layers=1,
        min_mask_region_area=20,
    )

    return mask_generator


def segment_frame(
    frame,
    mask_generator,
):
    """Segment one RGB frame."""

    return mask_generator.generate(frame)


def make_json_safe(value):
    """Convert NumPy/SAM2 values into JSON-safe Python types."""

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, dict):
        return {
            key: make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(val)
            for val in value
        ]

    return value


def save_segments(
    masks,
    frame_name,
    output_dir,
):
    """
    Save all information needed to reconstruct segments later.

    The original frame is already stored by the frame extraction stage,
    so only masks and SAM2 metadata are stored here.
    """

    frame_stem = os.path.splitext(frame_name)[0]

    frame_output_dir = os.path.join(
        output_dir,
        frame_stem,
    )

    os.makedirs(
        frame_output_dir,
        exist_ok=True,
    )

    # Preserve SAM2's raw output exactly as returned.
    raw_output_path = os.path.join(
        frame_output_dir,
        "sam2_raw.pkl",
    )

    with open(
        raw_output_path,
        "wb",
    ) as f:
        pickle.dump(
            masks,
            f,
        )

    segments_metadata = []

    for i, mask_data in enumerate(masks):

        mask = mask_data["segmentation"]

        mask_name = f"mask_{i:03d}.npy"

        mask_path = os.path.join(
            frame_output_dir,
            mask_name,
        )

        # Save the full boolean segmentation mask.
        np.save(
            mask_path,
            mask,
        )

        segment_metadata = {
            "segment_id": i,
            "mask_path": mask_name,
        }

        # Preserve every SAM2 field except the segmentation array itself,
        # which is already stored losslessly in mask_XXX.npy.
        for key, value in mask_data.items():

            if key == "segmentation":
                continue

            segment_metadata[key] = make_json_safe(
                value
            )

        segments_metadata.append(
            segment_metadata
        )

    metadata_path = os.path.join(
        frame_output_dir,
        "segments.json",
    )

    frame_metadata = {
        "source_frame": frame_name,
        "num_segments": len(masks),
        "raw_sam2_output": "sam2_raw.pkl",
        "segments": segments_metadata,
    }

    with open(
        metadata_path,
        "w",
    ) as f:
        json.dump(
            frame_metadata,
            f,
            indent=2,
        )


def main():

    parser = argparse.ArgumentParser(
        description="Run SAM2 segmentation on extracted gameplay frames."
    )

    parser.add_argument(
        "game_name",
        help="Game name, e.g. vizdoom__defend_center",
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to the SAM2 checkpoint.",
    )

    parser.add_argument(
        "--model-config",
        default=DEFAULT_MODEL_CFG,
        help="SAM2 model config name.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N frames for testing.",
    )

    args = parser.parse_args()

    frame_dir = os.path.join(
        "data",
        "extracted_frames",
        args.game_name,
    )

    if not os.path.exists(frame_dir):
        raise FileNotFoundError(
            f"Frame directory not found: {frame_dir}"
        )

    frame_names = sorted(
        name
        for name in os.listdir(frame_dir)
        if name.endswith(".png")
    )

    if not frame_names:
        raise FileNotFoundError(
            f"No PNG frames found in: {frame_dir}"
        )

    if args.limit is not None:
        frame_names = frame_names[:args.limit]

    output_dir = os.path.join(
        "data",
        "segmentation",
        args.game_name,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    mask_generator = load_sam2(
        checkpoint=args.checkpoint,
        model_cfg=args.model_config,
    )

    print("Frames:", len(frame_names))
    print("Output:", output_dir)

    for frame_index, frame_name in enumerate(
        frame_names,
        start=1,
    ):

        frame_path = os.path.join(
            frame_dir,
            frame_name,
        )

        frame = np.array(
            Image.open(
                frame_path
            ).convert("RGB")
        )

        masks = segment_frame(
            frame,
            mask_generator,
        )

        save_segments(
            masks=masks,
            frame_name=frame_name,
            output_dir=output_dir,
        )

        print(
            f"[{frame_index}/{len(frame_names)}] "
            f"{frame_name}: {len(masks)} segments"
        )

    print()
    print("Segmentation complete")
    print("---------------------")
    print("Frames processed:", len(frame_names))
    print("Output:", output_dir)


if __name__ == "__main__":
    main()