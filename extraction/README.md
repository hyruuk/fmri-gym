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

### Parse game config

```bash
python extraction/config_parser.py \
configs/dbp_games/vizdoom__defend_center.json
```

### Run frame extraction

```bash
python -m extraction.frame_extraction vizdoom__defend_center
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
### Run SAM2 segmentation

### SAM2 setup

Clone SAM2 into the repository

```bash
git clone https://github.com/facebookresearch/sam2.git segment-anything-2
cd segment-anything-2
pip install -e .
```

downlaod the checkpoints 

```bash
cd ..
mkdir -p segment-anything-2/checkpoints
curl -L \
https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
-o segment-anything-2/checkpoints/sam2.1_hiera_small.pt
```


```bash
python -m extraction.segmentation vizdoom__defend_center
```

to test a small number of frames 

```bash
python -m extraction.segmentation vizdoom__defend_center --limit 5
```