"""
Temporal sampling based on game FPS

Start with a 300 ms sampling interval to reduce redundant consecutive frames
while preserving meaningful temporal changes in gameplay.

The sampling interval is converted into a game-specific number of frames
using the FPS defined in each game's configuration.

Initial interval motivated by:
https://pmc.ncbi.nlm.nih.gov/articles/PMC6971407/
"""

import argparse
import glob
import json
import os
import runpy
import sys

from PIL import Image
from fmri_gym.display import Display

from extraction.metadata_extraction import extract_metadata


def get_config_path(game_name):
    """Return the config path for a game."""
    matches = glob.glob(f"configs/dbp_games/{game_name}.json")

    if not matches:
        raise FileNotFoundError(f"No config found for: {game_name}")

    return matches[0]


def get_game_fps(config_path):
    """Read the game's FPS from its config."""
    with open(config_path, "r") as f:
        config = json.load(f)

    curriculum = config["curriculum"] if isinstance(config, dict) else config

    game_block = next(
        block
        for block in curriculum
        if block.get("type") == "game"
    )

    return game_block["fps"]


def main():
    parser = argparse.ArgumentParser(
        description="Run an fmri-gym game and temporally sample rendered frames."
    )

    parser.add_argument(
        "game_name",
        help="Game config name, e.g. vizdoom__defend_center",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.300,
        help="Target frame sampling interval in seconds. Default: 0.300",
    )

    parser.add_argument(
        "--subject",
        default="sub-01",
        help="Subject ID passed to fmri_play.py. Default: sub-01",
    )

    args = parser.parse_args()

    config_path = get_config_path(args.game_name)
    fps = get_game_fps(config_path)

    sample_every = max(1, round(fps * args.interval))
    actual_interval = sample_every / fps

    output_dir = os.path.join(
        "data",
        "extracted_frames",
        args.game_name,
    )

    os.makedirs(output_dir, exist_ok=True)

    frame_counter = 0
    sampled_frames = []

    original_draw_frame = Display.draw_frame

    def extract_temporal_frames(self, frame):
        nonlocal frame_counter

        frame_counter += 1

        if frame_counter % sample_every == 0:
            frame_name = f"frame_{frame_counter:06d}.png"

            frame_path = os.path.join(
                output_dir,
                frame_name,
            )

            Image.fromarray(frame).save(frame_path)

            sampled_frames.append(
                {
                    "render_frame_index": frame_counter,
                    "frame_path": frame_name,
                }
            )

        return original_draw_frame(self, frame)

    Display.draw_frame = extract_temporal_frames

    # Record existing session folders so we can identify
    # the exact session created by this run.
    sessions_before = set(
        glob.glob(
            os.path.join(
                "data",
                f"{args.subject}_*",
            )
        )
    )

    sys.argv = [
        "fmri_play.py",
        "--subject",
        args.subject,
        "--dummy-trigger",
        "--curriculum",
        config_path,
    ]

    try:
        runpy.run_path(
            "fmri_play.py",
            run_name="__main__",
        )

    finally:
        Display.draw_frame = original_draw_frame

    # Find the new session directory created by this run.
    sessions_after = set(
        glob.glob(
            os.path.join(
                "data",
                f"{args.subject}_*",
            )
        )
    )

    new_sessions = sessions_after - sessions_before

    if not new_sessions:
        raise RuntimeError(
            "Could not find the fmri-gym session created by this run."
        )

    session_dir = max(
        new_sessions,
        key=os.path.getmtime,
    )

    metadata_path = extract_metadata(
        session_dir=session_dir,
        sampled_frames=sampled_frames,
        output_dir=output_dir,
    )

    print()
    print("Extraction complete")
    print("-------------------")
    print("Config:", config_path)
    print("FPS:", fps)
    print("Sample every:", sample_every, "frames")
    print(
        "Actual interval:",
        round(actual_interval * 1000, 1),
        "ms",
    )
    print("Rendered frames:", frame_counter)
    print("Saved frames:", len(sampled_frames))
    print("Session:", session_dir)
    print("Frame output:", output_dir)
    print("Metadata:", metadata_path)


if __name__ == "__main__":
    main()