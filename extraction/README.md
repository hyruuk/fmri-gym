# Extraction Pipeline

This directory contains the pipeline used to convert gameplay trajectories into structured representations for downstream analysis.

The pipeline currently follows:

1. Save temporally sampled gameplay frames and align them with available gameplay metadata
2. Segment relevant visual regions
3. Extract DINO features and cluster regions based on similarity
4. Label objects / regions
5. Format the resulting information into a symbolic game-state representation
6. Extract representations for downstream analysis

Current priority: make the configuration format consistent across games so the same pipeline can be run with minimal game-specific changes.

## Parse game config

```bash
python extraction/utils/config_parser.py \
configs/dbp_games/vizdoom__defend_center.json
```

## Run frame extraction

```bash
python -m extraction.features.frame_extraction vizdoom__defend_center
```

Frames are currently sampled at a default interval of 300 ms based on the FPS specified in the game config.

The extraction run automatically saves sampled frames and aligns them with per-frame metadata logged by `fmri-gym`, including available fields such as:

- actions
- rewards
- session time
- wall time
- episode ID
- termination / truncation state
- backend-specific game variables

Frame extraction and metadata alignment are built on top of the existing `fmri-gym` session pipeline rather than replacing it. The extraction code uses gameplay data already logged by `fmri-gym` and aligns the sampled visual frames with those outputs. This should make it easier to merge the extraction work back into the main fMRI pipeline without maintaining a separate gameplay logging system.

Outputs are saved under:

```text
data/extracted_frames/<game_name>/
```

For example:

```text
data/extracted_frames/vizdoom__defend_center/
├── frame_000006.png
├── frame_000012.png
├── frame_000018.png
└── metadata.npz
```

## SAM2 setup

Clone SAM2 into the repository root:

```bash
git clone https://github.com/facebookresearch/sam2.git segment-anything-2
cd segment-anything-2
pip install -e .
```

Download the SAM2.1 Hiera Small checkpoint:

```bash
cd ..
mkdir -p segment-anything-2/checkpoints

curl -L \
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
-o segment-anything-2/checkpoints/sam2.1_hiera_small.pt
```

## Run SAM2 segmentation

```bash
python -m extraction.models.segmentation vizdoom__defend_center
```

To test a small number of frames:

```bash
python -m extraction.models.segmentation vizdoom__defend_center --limit 5
```

SAM2 runs on the temporally sampled gameplay frames and saves segmentation masks and associated metadata for each frame.

Outputs are saved under:

```text
data/segmentation/<game_name>/
```

For example:

```text
data/segmentation/vizdoom__defend_center/
├── frame_000006/
│   ├── sam2_raw.pkl
│   ├── segments.json
│   ├── mask_000.npy
│   ├── mask_001.npy
│   └── ...
├── frame_000012/
└── ...
```

The original gameplay frame is not duplicated. The saved masks and SAM2 metadata can be used later to reconstruct crops or masked regions as needed.

## Run DINOv2 embedding extraction

```bash
python -m extraction.features.dino_embeddings vizdoom__defend_center
```

DINOv2 embeddings are extracted for all SAM2-masked regions. The script also saves metadata linking each embedding back to its source frame and segment, and computes the full pairwise cosine-similarity matrix.

Outputs are saved under:

```text
data/dino/<game_name>/
```

including:

```text
dino_embeddings.npy
dino_metadata.pkl
dino_similarity.npy
```

## Run DINO similarity clustering

```bash
python -m extraction.features.dino_clustering vizdoom__defend_center
```

Segments are grouped using a cosine-similarity threshold and connected components. Cluster assignments are saved for downstream labeling, and contact sheets are generated for visual inspection.

Additional outputs include:

```text
cluster_labels.npy
clusters.json
cluster_sheets/
```
## TODO

Add a conservative crop-filtering stage between segmentation and DINO embedding extraction.

The filter should only remove unusable regions, such as empty or degenerate crops, while preserving very small but potentially meaningful game elements. Check first what gets removed.

## Next goal

The next goal is to assign reliable semantic labels to the segmented visual regions for game replay.

Low-resolution, stylized game visuals can be difficult to recognize reliably because limited detail makes segmentation and object classification harder. Standard object detectors can also suffer from domain and vocabulary mismatch with game-specific entities (Chen & Jhala, 2025).