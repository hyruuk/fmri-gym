"""
DINOv2 EMBEDDING EXTRACTION

Create DINOv2 embeddings for all SAM2-masked regions across sampled gameplay
frames, then compute a full pairwise cosine similarity matrix.

The script reconstructs masked crops from the original sampled frame and the
saved SAM2 mask, so no duplicate segment PNGs are required.

Embeddings are checkpointed periodically so a later failure does not discard
all previously processed regions.
"""

import argparse
import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "facebook/dinov2-base"
CHECKPOINT_EVERY = 100


def get_device():
    """Select CUDA, Apple MPS, or CPU."""

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_dino():
    """Load DINOv2 processor and model."""

    device = get_device()

    print("Device:", device)

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    ).to(device)

    model.eval()

    return processor, model, device


def make_masked_crop(
    frame,
    mask,
    bbox,
    min_size=16,
):
    """
    Reconstruct one SAM2-masked crop.

    Bounding-box coordinates are clipped safely to the frame, and extremely
    small crops are upscaled before being passed to DINOv2.
    """

    x, y, w, h = bbox

    frame_h, frame_w = frame.shape[:2]

    x1 = max(
        0,
        int(np.floor(x)),
    )

    y1 = max(
        0,
        int(np.floor(y)),
    )

    x2 = min(
        frame_w,
        int(np.ceil(x + w)),
    )

    y2 = min(
        frame_h,
        int(np.ceil(y + h)),
    )

    # Prevent zero-width / zero-height crops.
    if x2 <= x1:
        x2 = min(
            frame_w,
            x1 + 1,
        )

    if y2 <= y1:
        y2 = min(
            frame_h,
            y1 + 1,
        )

    crop = frame[
        y1:y2,
        x1:x2,
    ].copy()

    crop_mask = mask[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        raise ValueError(
            f"Empty crop produced from bbox={bbox}"
        )

    masked_crop = np.zeros_like(
        crop
    )

    masked_crop[crop_mask] = crop[
        crop_mask
    ]

    crop_h, crop_w = masked_crop.shape[:2]

    # DINO/HuggingFace image processors can fail on tiny crops.
    # Upscale using nearest-neighbor so the original pixel structure
    # is preserved as much as possible.
    if crop_h < min_size or crop_w < min_size:

        scale = max(
            min_size / crop_h,
            min_size / crop_w,
        )

        new_w = max(
            min_size,
            int(round(crop_w * scale)),
        )

        new_h = max(
            min_size,
            int(round(crop_h * scale)),
        )

        masked_crop = np.array(
            Image.fromarray(
                masked_crop.astype(
                    np.uint8
                )
            ).resize(
                (
                    new_w,
                    new_h,
                ),
                Image.Resampling.NEAREST,
            )
        )

    return masked_crop


def embed_crop(
    crop,
    processor,
    model,
    device,
):
    """Create one normalized DINOv2 CLS embedding."""

    image = Image.fromarray(
        crop.astype(
            np.uint8
        )
    ).convert(
        "RGB"
    )

    inputs = processor(
        images=image,
        return_tensors="pt",
        input_data_format="channels_last",
    )

    inputs = {
        key: value.to(
            device
        )
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model(
            **inputs
        )

        embedding = outputs.last_hidden_state[
            :,
            0,
            :,
        ]

        embedding = F.normalize(
            embedding,
            p=2,
            dim=-1,
        )

    return embedding[
        0
    ].cpu().numpy()


def load_segments_metadata(
    segment_dir,
):
    """Load SAM2 metadata for one sampled frame."""

    metadata_path = os.path.join(
        segment_dir,
        "segments.json",
    )

    with open(
        metadata_path,
        "r",
    ) as f:
        return json.load(f)


def save_checkpoint(
    embeddings,
    metadata,
    output_dir,
):
    """Persist partial DINO results so processing can resume later."""

    if not embeddings:
        return

    embeddings_array = np.stack(
        embeddings
    ).astype(
        np.float32
    )

    np.save(
        os.path.join(
            output_dir,
            "dino_embeddings_partial.npy",
        ),
        embeddings_array,
    )

    with open(
        os.path.join(
            output_dir,
            "dino_metadata_partial.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(
            metadata,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"\nCheckpoint saved: {len(metadata)} segments\n"
    )


def load_checkpoint(
    output_dir,
):
    """Load partial results if a previous run was interrupted."""

    embeddings_path = os.path.join(
        output_dir,
        "dino_embeddings_partial.npy",
    )

    metadata_path = os.path.join(
        output_dir,
        "dino_metadata_partial.pkl",
    )

    if not (
        os.path.exists(
            embeddings_path
        )
        and os.path.exists(
            metadata_path
        )
    ):
        return [], []

    embeddings_array = np.load(
        embeddings_path
    )

    with open(
        metadata_path,
        "rb",
    ) as f:
        metadata = pickle.load(
            f
        )

    embeddings = [
        embedding
        for embedding in embeddings_array
    ]

    print(
        f"Resuming from checkpoint: "
        f"{len(metadata)} segments already processed"
    )

    return embeddings, metadata


def main():

    parser = argparse.ArgumentParser(
        description="Extract DINOv2 embeddings from SAM2 segmented gameplay regions."
    )

    parser.add_argument(
        "game_name",
        help="Game name, e.g. vizdoom__defend_center",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N segmented frames for testing.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing partial checkpoint and start from scratch.",
    )

    args = parser.parse_args()

    frame_dir = os.path.join(
        "data",
        "extracted_frames",
        args.game_name,
    )

    segmentation_dir = os.path.join(
        "data",
        "segmentation",
        args.game_name,
    )

    output_dir = os.path.join(
        "data",
        "dino",
        args.game_name,
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    embeddings_path = os.path.join(
        output_dir,
        "dino_embeddings.npy",
    )

    metadata_path = os.path.join(
        output_dir,
        "dino_metadata.pkl",
    )

    similarity_path = os.path.join(
        output_dir,
        "dino_similarity.npy",
    )

    if not os.path.exists(
        segmentation_dir
    ):
        raise FileNotFoundError(
            f"Segmentation directory not found: {segmentation_dir}"
        )

    frame_names = sorted(
        name
        for name in os.listdir(
            segmentation_dir
        )
        if os.path.isdir(
            os.path.join(
                segmentation_dir,
                name,
            )
        )
    )

    if args.limit is not None:
        frame_names = frame_names[
            :args.limit
        ]

    if not frame_names:
        raise FileNotFoundError(
            f"No segmented frames found in: {segmentation_dir}"
        )

    processor, model, device = load_dino()

    if args.no_resume:
        embeddings = []
        metadata = []
    else:
        embeddings, metadata = load_checkpoint(
            output_dir
        )

    total_segments = 0

    for frame_name in frame_names:

        segment_dir = os.path.join(
            segmentation_dir,
            frame_name,
        )

        frame_metadata = load_segments_metadata(
            segment_dir
        )

        total_segments += frame_metadata[
            "num_segments"
        ]

    print(
        "Frames:",
        len(frame_names),
    )

    print(
        "Total segments:",
        total_segments,
    )

    print(
        "Output:",
        output_dir,
    )

    already_processed = len(
        metadata
    )

    count = 0

    for frame_index, frame_name in enumerate(
        frame_names
    ):

        frame_path = os.path.join(
            frame_dir,
            f"{frame_name}.png",
        )

        frame = np.array(
            Image.open(
                frame_path
            ).convert(
                "RGB"
            )
        )

        segment_dir = os.path.join(
            segmentation_dir,
            frame_name,
        )

        frame_metadata = load_segments_metadata(
            segment_dir
        )

        for segment in frame_metadata[
            "segments"
        ]:

            # Global traversal counter across all regions.
            if count < already_processed:
                count += 1
                continue

            segment_id = segment[
                "segment_id"
            ]

            mask_path = os.path.join(
                segment_dir,
                segment[
                    "mask_path"
                ],
            )

            mask = np.load(
                mask_path
            )

            crop = make_masked_crop(
                frame=frame,
                mask=mask,
                bbox=segment[
                    "bbox"
                ],
            )

            embedding = embed_crop(
                crop=crop,
                processor=processor,
                model=model,
                device=device,
            )

            embeddings.append(
                embedding
            )

            x, y, w, h = segment[
                "bbox"
            ]

            metadata.append(
                {
                    "global_index": count,
                    "frame_index": frame_index,
                    "frame_name": frame_name,
                    "segment_id": segment_id,
                    "mask_path": segment[
                        "mask_path"
                    ],
                    "bbox": segment[
                        "bbox"
                    ],
                    "center_x": x + w / 2,
                    "center_y": y + h / 2,
                    "area": segment.get(
                        "area"
                    ),
                    "predicted_iou": segment.get(
                        "predicted_iou"
                    ),
                    "stability_score": segment.get(
                        "stability_score"
                    ),
                    "point_coords": segment.get(
                        "point_coords"
                    ),
                    "crop_box": segment.get(
                        "crop_box"
                    ),
                }
            )

            count += 1

            print(
                f"{count}/{total_segments} | "
                f"frame={frame_name} | "
                f"segment={segment_id}"
            )

            if (
                len(metadata)
                % CHECKPOINT_EVERY
                == 0
            ):
                save_checkpoint(
                    embeddings=embeddings,
                    metadata=metadata,
                    output_dir=output_dir,
                )

    if not embeddings:
        raise RuntimeError(
            "No embeddings were created."
        )

    embeddings_array = np.stack(
        embeddings
    ).astype(
        np.float32
    )

    print()
    print(
        "Embeddings shape:",
        embeddings_array.shape,
    )

    np.save(
        embeddings_path,
        embeddings_array,
    )

    with open(
        metadata_path,
        "wb",
    ) as f:

        pickle.dump(
            metadata,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    # Embeddings are already L2-normalized, so matrix multiplication
    # gives cosine similarity.
    similarity = (
        embeddings_array
        @ embeddings_array.T
    )

    np.fill_diagonal(
        similarity,
        1.0,
    )

    print(
        "Similarity matrix shape:",
        similarity.shape,
    )

    np.save(
        similarity_path,
        similarity.astype(
            np.float32
        ),
    )

    print()
    print("DINO extraction complete")
    print("------------------------")
    print(
        "Embeddings:",
        embeddings_path,
    )
    print(
        "Metadata:",
        metadata_path,
    )
    print(
        "Similarity:",
        similarity_path,
    )


if __name__ == "__main__":
    main()