"""Reconstruction and analysis on top of a logged block (:mod:`fmri_gym.logging`).

An episode is replayed from its seed and its actions through a fresh adapter
built from the block's own ``block_start`` line, so a block's log is enough to
recreate the game the subject saw, frame for frame, on a deterministic engine.
An engine that is exact only from its launch -- a restarted episode is not
quite a fresh one -- is declared by its adapter with ``replay_needs_history =
True``, and the block's earlier episodes are then replayed first, in order,
before the one asked for.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .logging import read_events


def reconstruction_plan(block: str, episode_id: int | None = None) -> dict:
    """One episode's seed and actions.

    :param block: the block's folder.
    :param episode_id: which episode; ``None`` = the last one started (the
        crash-recovery case).
    :return: ``{"episode_id", "seed", "actions"}``.
    :raises ValueError: if the block has no episode.
    """
    events = read_events(block)
    starts = {e["episode_id"]: e for e in events if e["type"] == "episode_start"}
    if not starts:
        raise ValueError(f"no episode_start records in {block}")
    if episode_id is None:
        episode_id = max(starts)
    actions = [e["action"] for e in events
               if e["type"] == "frame" and e["episode_id"] == episode_id]
    return {"episode_id": episode_id, "seed": starts[episode_id]["seed"], "actions": actions}


def reconstruct_episode(
    block: str, episode_id: int | None = None, adapter: Any = None,
    n_actions: int | None = None,
) -> tuple[Any, dict]:
    """Replay one episode through an adapter, leaving it at the episode's last frame.

    :param block: the block's folder.
    :param episode_id: which episode (see :func:`reconstruction_plan`).
    :param adapter: the adapter to drive; built from the block's ``phase`` if omitted.
    :param n_actions: replay only the first this many of the episode's actions
        (``0``: the reset alone), leaving the adapter at the logged frame
        ``ep_frame == n_actions - 1``; ``None`` replays them all.
    :return: ``(adapter, plan)``; the caller owns the adapter's ``close()``.
    """
    plan = reconstruction_plan(block, episode_id)
    if adapter is None:
        from .adapters import get_adapter
        start = read_events(block)[0]
        adapter = get_adapter(start["backend"], start["phase"])
    if getattr(adapter, "replay_needs_history", False):
        _replay_earlier_episodes(block, plan["episode_id"], adapter)
    elif plan["episode_id"] != 0:
        getattr(adapter, "warm_up", lambda: None)()
    adapter.reset(plan["seed"])
    actions = plan["actions"] if n_actions is None else plan["actions"][:n_actions]
    for action in actions:
        obs, _, _, _, info = adapter.step(action)
        adapter.capture(obs, info, want_blob=False)
    return adapter, plan


def _replay_earlier_episodes(block: str, episode_id: int, adapter: Any) -> None:
    """Play the block's episodes before ``episode_id`` through ``adapter``, in order."""
    earlier = sorted({e["episode_id"] for e in read_events(block)
                      if e["type"] == "episode_start" and e["episode_id"] < episode_id})
    for prior in earlier:
        plan = reconstruction_plan(block, prior)
        adapter.reset(plan["seed"])
        for action in plan["actions"]:
            obs, _, _, _, info = adapter.step(action)
            adapter.capture(obs, info, want_blob=False)


def frame_arrays(block: str) -> dict[str, np.ndarray]:
    """A block's ``frame`` lines as parallel arrays, one entry per frame.

    :param block: the block's folder.
    :return: one array per field, the adapter's ``variables`` flattened to
        top-level keys; a field a frame lacks is ``None`` there. Heterogeneous
        fields become object arrays.
    """
    frames = [e for e in read_events(block) if e["type"] == "frame"]
    columns: dict[str, list] = {}
    for i, f in enumerate(frames):
        for key, value in {**f, **f.get("variables", {})}.items():
            if key in ("type", "variables"):
                continue
            columns.setdefault(key, [None] * len(frames))[i] = value
    arrays = {}
    for key, values in columns.items():
        try:
            arrays[key] = np.asarray(values)
        except Exception:
            arrays[key] = np.array(values, dtype=object)
    return arrays
