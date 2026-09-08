"""
METADATA EXTRACTION

Align temporally sampled gameplay frames with the per-frame metadata
already logged by fmri-gym.

For each sampled frame, retain all compatible per-frame variables exposed
by the game block, including actions, rewards, session time, wall time,
episode information, terminal state, and backend-specific variables.

Metadata extraction is called automatically by frame_extraction.py after
the gameplay session ends.
"""

import glob
import os

import numpy as np


def find_game_block(session_dir):
    """Find the game block NPZ created in a session."""

    matches = glob.glob(
        os.path.join(
            session_dir,
            "block-*.npz",
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"No game block found in session: {session_dir}"
        )

    if len(matches) > 1:
        print("Multiple game blocks found:")
        for match in matches:
            print(" ", match)

        print("Using:", matches[0])

    return matches[0]


def extract_metadata(
    session_dir,
    sampled_frames,
    output_dir,
):
    """
    Align sampled rendered frames with fmri-gym per-frame metadata.

    Parameters
    ----------
    session_dir : str
        Session directory created by the current fmri-gym run.

    sampled_frames : list of dict
        Each dict contains:
            render_frame_index
            frame_path

    output_dir : str
        Directory where extracted frame metadata should be saved.
    """

    block_path = find_game_block(session_dir)

    data = np.load(
        block_path,
        allow_pickle=True,
    )

    print()
    print("Game block:", block_path)
    print("Available fields:")

    for key in data.files:
        value = data[key]

        try:
            print(
                f"  {key}: shape={value.shape}, dtype={value.dtype}"
            )
        except AttributeError:
            print(f"  {key}")

    # Use one of the standard per-frame arrays to determine
    # how many gameplay frames were logged.
    if "actions" in data.files:
        n_logged_frames = len(data["actions"])

    elif "session_time" in data.files:
        n_logged_frames = len(data["session_time"])

    else:
        raise ValueError(
            "Could not determine the number of logged gameplay frames."
        )

    render_indices = np.array(
        [
            frame["render_frame_index"]
            for frame in sampled_frames
        ],
        dtype=int,
    )

    # Our render counter starts at 1; NumPy indexing starts at 0.
    log_indices = render_indices - 1

    if len(log_indices) > 0:
        highest_render_index = render_indices.max()

        if log_indices.max() >= n_logged_frames:
            raise ValueError(
                "\nFrame alignment check failed.\n"
                f"Highest sampled render frame: {highest_render_index}\n"
                f"Logged gameplay frames: {n_logged_frames}\n\n"
                "Display.draw_frame() does not appear to map 1:1 onto "
                "the logged game frames. Metadata was NOT aligned by index."
            )

    aligned = {
        "render_frame_index": render_indices,
        "frame_path": np.array(
            [
                frame["frame_path"]
                for frame in sampled_frames
            ]
        ),
    }

    # Collect every per-frame array exposed by fmri-gym.
    #
    # If its first dimension matches the number of logged gameplay
    # frames, treat it as frame-aligned metadata.
    for key in data.files:
        value = data[key]

        if value.ndim == 0:
            continue

        if len(value) == n_logged_frames:
            aligned[key] = value[log_indices]

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    output_path = os.path.join(
        output_dir,
        "metadata.npz",
    )

    np.savez_compressed(
        output_path,
        **aligned,
    )

    print()
    print("Metadata alignment")
    print("------------------")
    print("Logged gameplay frames:", n_logged_frames)
    print("Sampled frames:", len(sampled_frames))

    print("Saved fields:")
    for key in aligned:
        print(" ", key)

    return output_path