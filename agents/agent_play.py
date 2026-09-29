"""Play a curriculum's game blocks with a policy, not a person.

The comparison the rig is built for needs a model to play the worlds the subject
played and be logged the same way. So this shares everything that defines the
task: the curriculum JSON, the EnvAdapter, the episode seeds, the Logger and its
npz schema. It shares none of the run loop, which exists for a scanner: trigger
wait, fps pacing, a pygame window, a key event queue, a pause menu. A model needs
no window and no wall clock, and faking them would only slow the block to human
speed.

Episode seeds are the block's own (``seed`` + episode index), which is what makes
this the same-worlds condition rather than a fresh sample of the game.

Usage:
    python agents/agent_play.py --curriculum configs/dbp_games/crafter__crafter_L4.json \
        --policy random --outdir data/model-random --max-frames 200
    python agents/agent_play.py --curriculum configs/dbp_games/crafter__crafter_L4.json \
        --policy vlm --model claude-sonnet-5 --max-frames 60
"""

from __future__ import annotations

import argparse
import os
import time
from collections import Counter, defaultdict

from policies import Policy, RandomPolicy, VLMPolicy

from fmri_gym import Clock, Logger, get_adapter
from fmri_gym.config import load_config, validate_config


def no_key_action(adapter) -> tuple:
    """The action for "no key held", if the block's keys define one.

    A real-time phase must map ``""`` (see :mod:`fmri_gym.adapters.keymap`), so
    standing still is an action like any other. A turn-based phase has no such
    entry, on purpose: nothing happens until a key is pressed, and
    ``resolve(frozenset())`` raises rather than invent a step. A policy in that
    mode is allowed to press nothing too, which costs it the frame.

    :param adapter: the block's adapter.
    :return: ``(action, True)``, or ``(None, False)`` in a turn-based phase.
    """
    try:
        return adapter.keymap.resolve(frozenset()), True
    except ValueError:
        return None, False


def build_policy(args, adapter) -> Policy:
    """Construct the policy named on the command line.

    Both policies are built from the adapter's own keymap, so a model is offered
    exactly the keys the subject was taught and nothing more.

    :param args: parsed command-line arguments.
    :param adapter: the block's adapter, already constructed.
    :return: a ready :class:`~policies.Policy`.
    """
    keys = adapter.keymap.turn_actions()
    noop, has_noop = no_key_action(adapter)
    if args.policy == "random":
        # The no-key action is one of the draws where the phase has one: a
        # subject who presses nothing is playing, and the random baseline holds
        # still at the same rate as any other key.
        return RandomPolicy(list(keys.values()) + ([noop] if has_noop else []),
                            seed=args.policy_seed)
    return VLMPolicy(keys, model=args.model, history=args.history, noop=noop)


def play_episode(adapter, policy: Policy, frames: dict, clock: Clock, *,
                 seed: int, episode_id: int, state_stride: int, fps: float,
                 turn_based: bool, max_frames: int) -> tuple[int, int]:
    """Run one episode under ``policy``, appending to ``frames``.

    Mirrors ``Run._episode`` field for field, minus everything that is about a
    person at a screen. Keeping the two in step by hand is the price of not
    bending the run loop around a case it was not written for.

    :param adapter: the block's adapter.
    :param policy: the policy choosing actions.
    :param frames: mutable frame-log dict; lists are appended in place.
    :param clock: the run's clock, for the same time columns humans get.
    :param seed: RNG seed for this episode's ``reset``.
    :param episode_id: index of this episode within the block.
    :param state_stride: save a full state blob every this many frames.
    :param fps: the phase's frames per second, for the HUD's countdown only.
    :param turn_based: the phase's, so the same stretches the run loop steps
        by itself are not put to the policy either
        (:meth:`~fmri_gym.adapters.base.EnvAdapter.autoplay`).
    :param max_frames: stop the episode after this many frames.
    :return: ``(frames stepped, frames the policy pressed nothing)``.
    """
    frames["episode_seeds"].append(seed)
    policy.reset()
    obs, info = adapter.reset(seed)

    terminated = truncated = False
    score = 0.0                         # the episode's cumulative reward
    ep_frame = skipped = 0
    auto = None                         # see EnvAdapter.autoplay
    while not (terminated or truncated) and ep_frame + skipped < max_frames:
        if auto is not None:
            # The env is in a state it takes no action in, so the policy is not
            # asked for one, exactly as the subject is not asked to press. The
            # frames still cost budget: they are engine steps, and the subject
            # pays for them in block time.
            action = auto
        else:
            # The subject's block ends on a wall clock; the model's ends on this
            # frame budget, so the countdown it reads is over its own budget at the
            # block's fps. The alternative, a real clock, would tell the model how
            # long its own inference took, which is not something the game shows.
            remaining = (max_frames - ep_frame - skipped) / fps
            status = list(adapter.hud(score, remaining) or [])
            over = adapter.overlay()
            action = policy.act(adapter.render(), status + list(over[0] if over else []))
            if action is None:
                # A turn-based phase with no key pressed: nothing steps, exactly as
                # in the run loop. It costs the frame, which is what keeps a policy
                # that never names a key from looping here forever.
                skipped += 1
                continue

        obs, reward, terminated, truncated, info = adapter.step(action)
        auto = adapter.autoplay(info) if turn_based else None
        score += float(reward)
        # Anchor a full savestate at episode start and every stride.
        save_blob = (ep_frame % state_stride == 0)
        ep_frame += 1
        fs = adapter.capture(obs, info, want_blob=save_blob)

        frames["action"].append(action)
        frames["reward"].append(reward)
        frames["terminated"].append(bool(terminated))
        frames["truncated"].append(bool(truncated))
        frames["episode_id"].append(episode_id)
        frames["run_time"].append(clock.run_time())
        # A policy has no screen, so no frame of this block was ever flipped.
        # NaN, not the step time: the human column is a measured photon onset,
        # and nothing should be able to average the two together by accident.
        frames["flip_time"].append(float("nan"))
        frames["wall_time"].append(clock.wall_time())
        frames["state_blob"].append(fs.blob)
        for k, v in fs.variables.items():
            frames["variables"][k].append(v)

    # Same vocabulary the human blocks log, from the same method: "playing" is
    # the budget cutting the episode off, where a subject's block clock would.
    outcome, _ = adapter.outcome(terminated, truncated)
    frames["episode_outcome"].append(outcome)
    frames["episode_score"].append(score)
    return ep_frame, skipped


def play_block(phase: dict, index: int, args, logger: Logger,
               clock: Clock) -> str:
    """Play every episode of one game block and write its npz.

    :param phase: the game-phase config, used exactly as a run uses it.
    :param index: phase index in the curriculum (names the output file).
    :param args: parsed command-line arguments.
    :param logger: the run's logger writing the block.
    :param clock: the run's clock.
    :return: path of the written npz.
    """
    backend = phase.get("backend", "gym")
    base_seed = phase.get("seed", 1000 + index)
    state_stride = max(1, int(phase.get("state_stride", 1)))
    # A policy has no ears. Whatever the block's config plays to a subject as a
    # sound, it reads as a line of text, so the two players are told the same
    # things; scanner configs keep that line off the screen to hold the gaze on
    # the frame. Adapters without cues ignore the key.
    adapter = get_adapter(backend, {**phase, "cue_overlay": True})
    policy = build_policy(args, adapter)

    frames = defaultdict(list)
    frames["variables"] = defaultdict(list)
    skipped = 0
    for episode_id in range(args.n_episodes):
        n, n_skipped = play_episode(
            adapter, policy, frames, clock, seed=base_seed + episode_id,
            episode_id=episode_id, state_stride=state_stride,
            fps=phase["fps"], turn_based=bool(phase.get("turn_based", False)),
            max_frames=args.max_frames)
        skipped += n_skipped
        print(f"  episode {episode_id} (seed {base_seed + episode_id}): "
              f"{n} frames, {frames['episode_outcome'][-1]}, "
              f"score {frames['episode_score'][-1]:g}")

    extra = adapter.block_extra() or {}
    # Which policy produced the block belongs in the block, not in a filename:
    # a model npz and a human npz are otherwise indistinguishable by design.
    extra["policy"] = args.policy
    extra["policy_model"] = args.model if args.policy == "vlm" else ""
    adapter.close()
    path = logger.save_game_block(index, backend, phase["game"], frames,
                                  extra=extra)
    logger.log_phase({
        "index": index, "type": "game", "backend": backend,
        "game": phase["game"], "policy": args.policy,
        "n_episodes": args.n_episodes, "n_frames": len(frames["action"]),
        "outcomes": dict(Counter(frames["episode_outcome"])),
        "total_reward": sum(float(r) for r in frames["reward"]),
        # Three ways a model wastes a frame, kept apart: it named no key, it
        # named something that is not a key, or the call never came back.
        "skipped_frames": skipped, "invalid_replies": policy.invalid,
        "dropped_calls": policy.dropped,
        "data_file": path.split("/")[-1]})
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Play a curriculum with a policy.")
    p.add_argument("--curriculum", required=True)
    p.add_argument("--subject", default="model", help="stored in the manifest")
    p.add_argument("--outdir")
    p.add_argument("--policy", default="random", choices=["random", "vlm"])
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--history", type=int, default=4,
                   help="frames (and past keys) a vlm policy sees")
    p.add_argument("--n-episodes", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=1000)
    p.add_argument("--policy-seed", type=int, default=0)
    args = p.parse_args()

    config = load_config(args.curriculum)
    # The same check fmri_play runs, so one config file suits both players.
    problems = validate_config(config)
    if problems:
        raise ValueError(f"{args.curriculum}: " + "; ".join(problems))
    curriculum = config["curriculum"]
    outdir = args.outdir or os.path.join(
        "data", f"{args.subject}_{time.strftime('%Y%m%d-%H%M%S')}")
    clock = Clock()
    # No scanner here, so the run's zero is simply when it started: the time
    # columns stay the same shape as a human run's, measured from that.
    clock.trigger()
    logger = Logger(outdir, args.subject, curriculum, clock)
    logger.set_trigger_time()

    for index, phase in enumerate(curriculum):
        if phase.get("type") != "game":
            continue
        print(f"block {index}: {phase.get('game')} via {args.policy}")
        print(" ->", play_block(phase, index, args, logger, clock))
    print("Manifest:", logger.save_manifest())


if __name__ == "__main__":
    main()
