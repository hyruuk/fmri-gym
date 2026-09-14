"""The config file as a whole: load/save, the ``session`` section, presets.

A config is a JSON dict with ``curriculum`` (the phase list; see README), an
optional ``triggers`` section (:mod:`fmri_gym.triggers`) and an optional
``session`` section that mirrors the CLI flags (subject, outdir, size,
fullscreen, vsync, dummy_trigger). A bare list is still accepted as a
curriculum. Precedence for session settings is CLI flag > ``session`` section
> :data:`SESSION_DEFAULTS`, so a config can carry the rig setup while a flag
still wins for one run.

Everything here is pure (no pygame, no tkinter) so the GUI's data model can be
tested headless and ``fmri_play.py`` can use the same helpers without one.
"""

from __future__ import annotations

import copy
import json
import os
import time
from typing import Any

from .triggers import MarkerSettings, SyncSettings

PHASE_TYPES = ("fixation", "message", "game", "survey")
BACKENDS = ("ale", "retro", "gym", "vgdl", "crafter", "minihack", "nethack",
            "vizdoom", "overcooked", "baba", "rushhour", "supertuxkart",
            "aigamestore")

SESSION_DEFAULTS: dict[str, Any] = {
    "subject": "sub-test",
    "outdir": "",
    "size": "1024x768",
    "fullscreen": False,
    "vsync": True,
    "dummy_trigger": False,
}

#: Trigger presets the GUI offers; ``None`` means "no triggers section".
TRIGGER_PRESETS: dict[str, dict | None] = {
    "fMRI (wait for '=')": None,
    "MEG (start from trigger, serial markers)": {
        "sync": {"mode": "send", "delay": 0.0},
        "markers": {"backend": "serial", "port": "/dev/ttyUSB0", "pulse_ms": 10,
                    "on_frame": True, "frame_every": 1, "on_episode_start": True},
    },
    "EEG (wait, parallel markers)": {
        "sync": {"mode": "wait", "key": "="},
        "markers": {"backend": "parallel", "port": "/dev/parport0", "pulse_ms": 10,
                    "on_frame": True, "frame_every": 1, "on_episode_start": True},
    },
    "Behavioural (no scanner)": {"sync": {"mode": "none"}},
}


def load_config(path: str) -> dict:
    """Load a config file: a bare curriculum list, or a dict with sections.

    :param path: JSON file path.
    :return: a dict with at least ``"curriculum"``.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return {"curriculum": data}


def save_config(config: dict, path: str) -> None:
    """Write ``config`` as indented JSON.

    :param config: config dict (``curriculum`` and optional sections).
    :param path: destination file path.
    """
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def new_config() -> dict:
    """A minimal starting config: one message, one fixation, one game."""
    return {
        "session": dict(SESSION_DEFAULTS),
        "curriculum": [
            {"type": "message", "text": "Get ready\n\n(press SPACE to start)"},
            {"type": "fixation", "duration": 2.0},
            {"type": "game", "backend": "ale", "game": "ALE/Pong-v5", "mode": "duration",
             "duration": 60.0, "fps": 30, "keys": {"UP": 2, "DOWN": 3}},
            {"type": "fixation", "duration": 2.0},
        ],
    }


def resolve_session(config: dict, overrides: dict | None = None) -> dict:
    """Merge session settings: ``overrides`` (CLI) > ``config["session"]`` > defaults.

    Only keys of :data:`SESSION_DEFAULTS` are read; ``None`` in ``overrides``
    means "flag not given". ``outdir`` stays ``""`` when unset (see
    :func:`default_outdir`).

    :param config: full config dict.
    :param overrides: explicit CLI values, ``None`` where the flag was absent.
    :return: a complete session dict.
    """
    out = dict(SESSION_DEFAULTS)
    for key in SESSION_DEFAULTS:
        value = (config.get("session") or {}).get(key)
        if value is not None:
            out[key] = value
    for key, value in (overrides or {}).items():
        if key in SESSION_DEFAULTS and value is not None:
            out[key] = value
    return out


def default_outdir(subject: str) -> str:
    """The output directory used when none is set: ``data/<subject>_<stamp>``."""
    return os.path.join("data", f"{subject}_{time.strftime('%Y%m%d-%H%M%S')}")


def parse_size(size: str) -> tuple[int, int]:
    """``"1024x768"`` -> ``(1024, 768)``.

    :raises ValueError: if the string is not ``<w>x<h>``.
    """
    try:
        w, h = (int(x) for x in size.lower().split("x"))
    except Exception as exc:
        raise ValueError(f"size must look like 1024x768, got {size!r}") from exc
    return w, h


def apply_preset(config: dict, name: str) -> dict:
    """Return a copy of ``config`` with the named trigger preset installed.

    :param config: full config dict.
    :param name: key of :data:`TRIGGER_PRESETS`.
    """
    out = copy.deepcopy(config)
    preset = TRIGGER_PRESETS[name]
    out.pop("triggers", None)
    if preset is not None:
        out["triggers"] = copy.deepcopy(preset)
    return out


def validate_config(config: dict) -> list[str]:
    """Problems that would stop ``fmri_play`` before the first phase.

    Cheap checks only (no env is built, no port opened): phase types, required
    game fields, the session size string and the triggers section, including
    marker code overlaps.

    :param config: full config dict.
    :return: human-readable problems, empty when the config looks runnable.
    """
    problems: list[str] = []
    curriculum = config.get("curriculum")
    if not isinstance(curriculum, list) or not curriculum:
        problems.append("curriculum: needs at least one phase")
        curriculum = []
    for i, phase in enumerate(curriculum):
        problems.extend(f"phase {i}: {p}" for p in _phase_problems(phase))
    session = resolve_session(config)
    try:
        parse_size(str(session["size"]))
    except ValueError as exc:
        problems.append(f"session: {exc}")
    problems.extend(trigger_problems(config.get("triggers")))
    return problems


def _phase_problems(phase: Any) -> list[str]:
    if not isinstance(phase, dict):
        return ["not a JSON object"]
    kind = phase.get("type")
    if kind not in PHASE_TYPES:
        return [f"unknown type {kind!r} (expected one of {PHASE_TYPES})"]
    if kind != "game":
        return []
    out = []
    if not phase.get("game"):
        out.append("game: missing env id")
    if phase.get("mode", "duration") not in ("duration", "episode"):
        out.append(f"mode: expected 'duration' or 'episode', got {phase.get('mode')!r}")
    if not isinstance(phase.get("keys", {}), dict):
        out.append("keys: must be an object of {\"KEY\": action}")
    return out


def trigger_problems(section: Any) -> list[str]:
    if section is None:
        return []
    if not isinstance(section, dict):
        return ["triggers: must be a JSON object"]
    try:
        sync = SyncSettings.from_dict(section.get("sync"))
        markers = MarkerSettings.from_dict(section.get("markers"))
    except (TypeError, ValueError) as exc:
        msg = str(exc)
        return [msg if msg.startswith("triggers:") else f"triggers: {msg}"]
    out = []
    if sync.mode == "send" and markers.backend == "null":
        out.append("triggers: sync.mode 'send' needs a markers backend other than 'null'")
    if markers.backend in ("serial", "parallel") and not markers.port:
        out.append(f"triggers: markers backend {markers.backend!r} needs a port")
    return out
