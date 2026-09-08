"""
DINOv2 SIMILARITY CLUSTERING

Cluster SAM2 regions using the cosine-similarity matrix produced by
dino_embeddings.py.

Segments with pairwise similarity above a threshold are connected in a graph.
Connected components are treated as clusters.

Cluster assignments are saved for downstream labeling, and contact sheets are
generated for visual inspection.
"""

import argparse
import json
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DEFAULT_SIM_THRESHOLD = 0.80
DEFAULT_MIN_CLUSTER_SIZE = 2
DEFAULT_COLS = 5


def make_masked_crop(
    frame,
    mask,
    bbox,
):
    """Reconstruct a masked crop from the source frame and saved SAM2 mask."""

    x, y, w, h = bbox

    x1 = int(x)
    y1 = int(y)
    x2 = int(x + w)
    y2 = int(y + h)

    crop = frame[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    masked_crop = np.zeros_like(crop)
    masked_crop[crop_mask] = crop[crop_mask]

    return masked_crop


def build_clusters(
    similarity,
    threshold,
    min_cluster_size,
):
    """Build connected-component clusters from a similarity threshold."""

    n = similarity.shape[0]

    neighbors = [
        set(
            np.where(
                similarity[i] >= threshold
            )[0].tolist()
        )
        for i in range(n)
    ]

    for i in range(n):
        neighbors[i].discard(i)

    visited = set()
    clusters = []

    for i in range(n):

        if i in visited:
            continue

        stack = [i]
        component = []

        while stack:

            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            component.append(current)

            stack.extend(
                neighbors[current] - visited
            )

        if len(component) >= min_cluster_size:
            clusters.append(component)

    clusters.sort(
        key=len,
        reverse=True,
    )

    return clusters


def save_cluster_assignments(
    clusters,
    metadata,
    output_dir,
    threshold,
):
    """Save cluster membership for downstream stages."""

    labels = np.full(
        len(metadata),
        -1,
        dtype=np.int32,
    )

    cluster_records = []

    for cluster_id, cluster in enumerate(clusters):

        for global_index in cluster:
            labels[global_index] = cluster_id

        cluster_records.append(
            {
                "cluster_id": cluster_id,
                "size": len(cluster),
                "members": cluster,
            }
        )

    np.save(
        os.path.join(
            output_dir,
            "cluster_labels.npy",
        ),
        labels,
    )

    with open(
        os.path.join(
            output_dir,
            "clusters.json",
        ),
        "w",
    ) as f:
        json.dump(
            {
                "similarity_threshold": threshold,
                "num_clusters": len(clusters),
                "clusters": cluster_records,
            },
            f,
            indent=2,
        )

    return labels


def save_contact_sheets(
    clusters,
    metadata,
    frame_dir,
    segmentation_dir,
    output_dir,
    threshold,
    cols,
):
    """Save one visual contact sheet for each cluster."""

    sheet_dir = os.path.join(
        output_dir,
        "cluster_sheets",
    )

    os.makedirs(
        sheet_dir,
        exist_ok=True,
    )

    for cluster_id, cluster in enumerate(clusters):

        rows = math.ceil(
            len(cluster) / cols
        )

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(
                cols * 3,
                rows * 3,
            ),
        )

        axes = np.array(
            axes,
        ).reshape(-1)

        for ax, global_index in zip(
            axes,
            cluster,
        ):

            meta = metadata[global_index]

            frame_name = meta["frame_name"]
            segment_id = meta["segment_id"]

            frame_path = os.path.join(
                frame_dir,
                f"{frame_name}.png",
            )

            frame = np.array(
                Image.open(
                    frame_path
                ).convert("RGB")
            )

            segment_dir = os.path.join(
                segmentation_dir,
                frame_name,
            )

            mask_path = os.path.join(
                segment_dir,
                meta["mask_path"],
            )

            mask = np.load(
                mask_path
            )

            crop = make_masked_crop(
                frame=frame,
                mask=mask,
                bbox=meta["bbox"],
            )

            ax.imshow(crop)

            ax.set_title(
                f"#{global_index}\n"
                f"{frame_name} S{segment_id}",
                fontsize=8,
            )

            ax.axis("off")

        for ax in axes[len(cluster):]:
            ax.axis("off")

        fig.suptitle(
            f"Cluster {cluster_id:03d} | "
            f"n={len(cluster)} | "
            f"threshold={threshold}",
            fontsize=14,
        )

        plt.tight_layout()

        out_path = os.path.join(
            sheet_dir,
            f"cluster_{cluster_id:03d}_n{len(cluster)}.png",
        )

        plt.savefig(
            out_path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(fig)


def main():

    parser = argparse.ArgumentParser(
        description="Cluster DINOv2 embeddings using cosine-similarity thresholding."
    )

    parser.add_argument(
        "game_name",
        help="Game name, e.g. vizdoom__defend_center",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIM_THRESHOLD,
        help="Cosine similarity threshold. Default: 0.80",
    )

    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=DEFAULT_MIN_CLUSTER_SIZE,
        help="Minimum number of segments in a cluster. Default: 2",
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=DEFAULT_COLS,
        help="Columns in cluster contact sheets. Default: 5",
    )

    args = parser.parse_args()

    dino_dir = os.path.join(
        "data",
        "dino",
        args.game_name,
    )

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

    similarity_path = os.path.join(
        dino_dir,
        "dino_similarity.npy",
    )

    metadata_path = os.path.join(
        dino_dir,
        "dino_metadata.pkl",
    )

    similarity = np.load(
        similarity_path
    )

    with open(
        metadata_path,
        "rb",
    ) as f:
        metadata = pickle.load(f)

    if similarity.shape[0] != len(metadata):
        raise ValueError(
            "Similarity matrix and metadata length do not match."
        )

    clusters = build_clusters(
        similarity=similarity,
        threshold=args.threshold,
        min_cluster_size=args.min_cluster_size,
    )

    print("Segments:", len(metadata))
    print("Clusters:", len(clusters))

    labels = save_cluster_assignments(
        clusters=clusters,
        metadata=metadata,
        output_dir=dino_dir,
        threshold=args.threshold,
    )

    save_contact_sheets(
        clusters=clusters,
        metadata=metadata,
        frame_dir=frame_dir,
        segmentation_dir=segmentation_dir,
        output_dir=dino_dir,
        threshold=args.threshold,
        cols=args.cols,
    )

    print()
    print("Clustering complete")
    print("-------------------")
    print(
        "Assigned segments:",
        int(np.sum(labels >= 0)),
    )
    print(
        "Unclustered segments:",
        int(np.sum(labels == -1)),
    )
    print(
        "Cluster labels:",
        os.path.join(
            dino_dir,
            "cluster_labels.npy",
        ),
    )
    print(
        "Cluster metadata:",
        os.path.join(
            dino_dir,
            "clusters.json",
        ),
    )
    print(
        "Contact sheets:",
        os.path.join(
            dino_dir,
            "cluster_sheets",
        ),
    )


if __name__ == "__main__":
    main()