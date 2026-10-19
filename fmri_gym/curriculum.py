"""The curriculum: what one run plays. Load, save, check, and the trigger presets.

A curriculum is one JSON file: a ``curriculum`` list of phases (see README)
and an optional ``triggers`` section (:mod:`fmri_gym.triggers`);
``_``-prefixed keys are notes. Several runs make a session, which is a shell
script (:func:`fmri_gym.gui.write_session`), not a curriculum.

The file says what is played, not who plays it or where: subject, output
folder, window and the test switches are the run config
(:mod:`fmri_gym.config`), so one file serves every participant at every site.

There is one shape. A bare list, an unknown top-level key or an unknown phase
type is refused at start-up, with the reason.

A phase may start from a shared one: ``"include": "common/freedoom1.json"``
names a file (relative to the curriculum's own) holding one phase object, and
the phase's other keys are laid over it (:func:`overlay`). Curricula of one
game then write the setup once and only what differs per level. Shared
phases live in folders named ``common`` (:data:`PHASE_DIR`), which hold no
curricula. Includes are resolved when the file is loaded, so everything
downstream, the manifest included, sees each phase written out in full.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .keys import KEY_NAMES
from .menu import menu_problems
from .triggers import TriggerError, TriggerSettings

#: The phases of a run; the ``check_*`` ones make a rig check (:mod:`fmri_gym.checks`).
PHASE_TYPES = ("fixation", "message", "game", "survey", "check_display", "check_frames",
               "check_triggers", "check_controls", "check_photodiode")
SECTIONS = ("curriculum", "triggers")
#: A phase's key naming the file of the shared phase it starts from.
INCLUDE = "include"
#: Folders of this name hold shared phases, which are included, not played.
PHASE_DIR = "common"
#: Exit status of ``fmri_play`` when the run was quit (ESC) before its end; a
#: session script (``set -e``) stops on it instead of starting the next run.
EXIT_QUIT = 3

#: Trigger presets the editor offers. Each names ``sync.mode`` and ``backend``
#: outright: a section that leaves them out runs, but is reported as NOT SET.
TRIGGER_PRESETS: dict[str, dict] = {
    "fMRI (wait for '=', no codes)": {
        "sync": {"mode": "wait", "key": "="}, "backend": "null"},
    # MEG and EEG alike: the stimulus PC starts the recording and marks it, over serial.
    "MEG (start from the trigger line, serial codes)": {
        "sync": {"mode": "send", "delay": 0.0}, "backend": "serial", "port": "/dev/ttyUSB0"},
    "EEG (start from the trigger line, serial codes)": {
        "sync": {"mode": "send", "delay": 0.0}, "backend": "serial", "port": "/dev/ttyUSB0"},
    "Behavioural (no scanner)": {
        "sync": {"mode": "none"}, "backend": "null"},
}


#: An episode's possible outcomes: EnvAdapter.outcome's names, and the subject's own endings.
OUTCOMES = frozenset({"won", "lost", "terminated", "truncated", "playing", "forfeit", "reset"})


class BlockMode(str, Enum):
    """What ends a game block: its clock, or its count of episodes."""

    #: the block lasts ``duration`` seconds
    DURATION = "duration"
    #: the block lasts ``n_episodes`` episodes, or ``duration`` seconds at most
    EPISODE = "episode"


@dataclass(frozen=True, slots=True)
class GamePhase:
    """A game phase's settings as the loop plays them, read off its dict once.

    The dict itself stays what the adapter is built from and what the block's
    log keeps, since its backend-specific keys are the adapter's business;
    this is the part the loop reads, every field with its default, so the
    defaults are stated in one place and a wrong value is refused before the
    block starts (:func:`validate_curriculum` builds one per game phase).
    """

    backend: str
    #: the env id, as the backend reads it
    game: str
    mode: BlockMode
    #: seconds the block lasts (``mode`` ``duration``) or lasts at most
    #: (``mode`` ``episode``, where the dict's key is ``max_duration``)
    duration: float
    #: episodes an ``episode`` block lasts
    n_episodes: int
    #: A blocking block: only an episode whose outcome is listed (usually
    #: ``["won"]``) counts toward ``n_episodes``; any other replays the same
    #: instance (the seed follows the count), so the subject stays on a level
    #: until they clear it, ``duration`` runs out, or they forfeit from the
    #: menu. ``None``: every episode counts, whatever its outcome. The dict's
    #: key is ``advancing_outcomes``.
    advance_on: tuple[str, ...] | None
    #: An episode ending in one of these outcomes sends the world back
    #: ``rewind_frames`` frames and play goes on, as a new episode from that
    #: state: session policy, not a rule of the game, so a model played on
    #: the same env ends as the game says. Empty: no ending is rewound.
    rewind_on: tuple[str, ...]
    #: how far back a rewound episode goes; ``None`` when nothing is rewound
    rewind_frames: int | None
    #: seconds the final score and outcome stay on screen after an episode
    #: (the dict's ``outcome_duration``)
    show_outcome_duration: float
    #: the block's base seed: its episode ``e`` plays ``seed + e``
    seed: int
    #: a savestate every this many frames: 1 keeps one per frame, larger
    #: trades savestate density for disk (retro states are ~1 MB/frame)
    state_stride: int
    #: Turn-based games (grid worlds: FrozenLake, CliffWalking, Taxi, ...)
    #: advance ONE step per deliberate key PRESS, not once per frame. In a
    #: real-time loop they would auto-step every frame with the noop action
    #: (which for e.g. FrozenLake is action 0 = LEFT), so the agent "moves on
    #: its own" and a single held key fires many times.
    is_turn_based: bool
    #: Real-time blocks poll HELD keys, so a press that starts and ends
    #: between two frames is never seen -- at a grid world's few frames per
    #: second that loses most taps. ``latched_keys`` lets a fresh keydown win
    #: instead, falling back to the held-key poll (so holding a key still
    #: repeats). Off by default: backends whose actions are key COMBINATIONS
    #: must keep polling.
    has_latched_keys: bool
    #: A turn-based block draws a frame only when the subject presses, so
    #: the seconds its HUD shows stop between presses while the clock the
    #: block actually ends on does not. ``live_hud`` redraws the standing
    #: frame a few times a second through the wait, which changes nothing
    #: but the pixels: no step, no log row, no trigger, no sound. Off by
    #: default, so a block that has run this way keeps running this way.
    has_live_hud: bool
    #: pass the adapter's sound to the speakers; muting never changes what is logged
    has_audio: bool
    #: the block's steps per second. Every game phase states its own: what
    #: the block plays at is the curriculum's business, not a default that
    #: changes with the engine underneath it.
    fps: float

    @property
    def frame_period(self) -> float:
        """Seconds per frame."""
        return 1.0 / self.fps

    @classmethod
    def from_dict(cls, phase: dict, index: int) -> GamePhase:
        """Read a game phase's dict, filling in the defaults.

        :param phase: the phase, of type ``game``, with the fields
            :func:`validate_curriculum` insists on (``game``, ``fps``, ``keys``).
        :param index: its index in the curriculum, which the default seed is from.
        :raises ValueError: ``live_hud`` on a real-time block, or an ``audio``
            that is not true or false.
        """
        is_turn_based = bool(phase.get("turn_based", False))
        has_live_hud = bool(phase.get("live_hud", False))
        if has_live_hud and not is_turn_based:
            raise ValueError('"live_hud" is for turn_based blocks, whose wait is unbounded; '
                             "a real-time block already redraws every frame")
        has_audio = phase.get("audio", True)
        if not isinstance(has_audio, bool):
            raise ValueError(f'"audio" must be true or false, got {has_audio!r}')
        mode = BlockMode(phase.get("mode", "duration"))
        advancing_outcomes = phase.get("advancing_outcomes")
        return cls(
            backend=phase.get("backend", "gym"), game=phase["game"], mode=mode,
            duration=(phase.get("duration", 30.0) if mode is BlockMode.DURATION
                      else phase.get("max_duration", 300.0)),
            n_episodes=phase.get("n_episodes", 1),
            advance_on=None if advancing_outcomes is None else tuple(advancing_outcomes),
            rewind_on=tuple(phase.get("rewind_on", ())),
            rewind_frames=phase.get("rewind_frames"),
            show_outcome_duration=float(phase.get("outcome_duration", 2.0)),
            seed=phase.get("seed", 1000 + index),
            state_stride=max(1, int(phase.get("state_stride", 1))),
            is_turn_based=is_turn_based,
            has_latched_keys=bool(phase.get("latched_keys", False)),
            has_live_hud=has_live_hud, has_audio=has_audio, fps=phase["fps"])


def check_shape(curriculum: Any, where: str) -> dict:
    """Enforce the one curriculum shape.

    :param curriculum: parsed JSON.
    :param where: file path (or another name for the source), for the message.
    :return: ``curriculum``, unchanged.
    :raises ValueError: if it is not an object with a ``curriculum`` list of
        typed phases, or has a top-level key that is neither a section nor a
        ``_`` note.
    """
    if not isinstance(curriculum, dict) or not isinstance(curriculum.get("curriculum"), list):
        raise ValueError(f'{where}: expected a JSON object with a "curriculum" list')
    unknown = [k for k in curriculum if k not in SECTIONS and not k.startswith("_")]
    if unknown:
        raise ValueError(f"{where}: unknown top-level key(s) {unknown}; the sections are "
                         f'{SECTIONS}, and a note must start with "_"')
    for i, phase in enumerate(curriculum["curriculum"]):
        if not isinstance(phase, dict) or phase.get("type") not in PHASE_TYPES:
            raise ValueError(f'{where}: phase {i} must be an object whose "type" is one of '
                             f"{PHASE_TYPES}")
    return curriculum


def load_curriculum(path: str) -> dict:
    """Load a curriculum file, each phase's ``include`` resolved.

    :param path: JSON file path.
    :return: the curriculum dict, every phase in full.
    :raises ValueError: if the file does not have the curriculum shape
        (see :func:`check_shape`).
    """
    with open(path) as f:
        curriculum = json.load(f)
    if isinstance(curriculum, dict) and isinstance(curriculum.get("curriculum"), list):
        curriculum["curriculum"] = [_resolve(phase, path, frozenset())
                                    for phase in curriculum["curriculum"]]
    return check_shape(curriculum, path)


def is_phase_file(path: str) -> bool:
    """Whether ``path`` is a shared phase (in a :data:`PHASE_DIR` folder), not a curriculum."""
    return os.path.basename(os.path.dirname(os.path.abspath(path))) == PHASE_DIR


def overlay(base: dict, top: dict) -> dict:
    """``top`` laid over ``base``: an object in both merges key by key, anything else replaces.

    So a phase can change one line of its instructions, or add a key to its
    ``keys``, and keep the rest; a list (the instructions' ``text``) is
    replaced whole, and a key cannot be taken away.
    """
    merged = dict(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = overlay(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve(phase: Any, path: str, seen: frozenset[str]) -> Any:
    """``phase`` laid over the phase its ``include`` names, which may include another.

    :raises ValueError: if the include is not a path to a JSON object, or comes back round.
    """
    if not isinstance(phase, dict) or INCLUDE not in phase:
        return phase
    name = phase[INCLUDE]
    target = os.path.normpath(os.path.join(os.path.dirname(path), str(name)))
    if not isinstance(name, str) or not os.path.isfile(target):
        raise ValueError(f'{path}: "{INCLUDE}" must name a file next to it, got {name!r}')
    if target in seen:
        raise ValueError(f"{path}: {name} includes itself, through {sorted(seen)}")
    with open(target) as f:
        base = json.load(f)
    if not isinstance(base, dict):
        raise ValueError(f"{target}: a shared phase is one JSON object")
    base = _resolve(base, target, seen | {target})
    return overlay(base, {k: v for k, v in phase.items() if k != INCLUDE})


def save_curriculum(curriculum: dict, path: str) -> None:
    """Write ``curriculum`` as indented JSON.

    :param curriculum: curriculum dict.
    :param path: destination file path.
    """
    with open(path, "w") as f:
        json.dump(curriculum, f, indent=2)
        f.write("\n")


def new_curriculum() -> dict:
    """A minimal starting curriculum: one message, one fixation, one game."""
    return {
        "triggers": copy.deepcopy(next(iter(TRIGGER_PRESETS.values()))),
        "curriculum": [
            {"type": "message", "text": "Get ready\n\n(press SPACE to start)"},
            {"type": "fixation", "duration": 2.0},
            {"type": "game", "backend": "ale", "game": "ALE/Pong-v5", "mode": "duration",
             "duration": 60.0, "fps": 30, "keys": {"UP": 2, "DOWN": 3}},
            {"type": "fixation", "duration": 2.0},
        ],
    }


def fold_cli_options(phases: list[dict], *, has_audio: bool) -> None:
    """Fold the run config's backend options into the game phases they concern.

    Each per-block ``EnvAdapter`` then reads everything it needs from its own
    spec, and the manifest's curriculum shows what was actually played.

    :param phases: the curriculum's phases, changed in place.
    :param has_audio: ``False`` (``--no-audio``) mutes every game phase.
    """
    for phase in phases:
        if phase.get("type") != "game":
            continue
        if not has_audio:
            phase["audio"] = False


def validate_curriculum(curriculum: dict) -> list[str]:
    """Problems that would stop ``fmri_play`` before the first phase.

    Cheap checks only (no env is built, no port opened), for the editor's
    Check: the required game fields and the triggers section.

    :param curriculum: a curriculum dict of the right shape.
    :return: human-readable problems, empty when the curriculum looks runnable.
    """
    problems: list[str] = []
    phases = curriculum["curriculum"]
    if not phases:
        problems.append("curriculum: needs at least one phase")
    for i, phase in enumerate(phases):
        problems.extend(f"phase {i}: {p}" for p in _phase_problems(phase, i))
    problems.extend(_opening_problems(phases))
    problems.extend(trigger_problems(curriculum.get("triggers")))
    problems.extend(trigger_key_clashes(curriculum))
    return problems


def trigger_key_clashes(curriculum: dict) -> list[str]:
    """Phases whose ``keys`` bind the key the scanner types at every volume.

    A button box in a digit mode sends its trigger as a digit too (Current
    Designs: ``5``). With ``sync.mode`` ``wait`` on that key, a game bound to
    it would act at every volume. Only a phase's own ``keys`` are checked: a
    backend's default map is known once its env is built.

    :param curriculum: a curriculum of the right shape.
    :return: one problem per clashing binding.
    """
    try:
        sync = TriggerSettings.from_dict(curriculum.get("triggers")).sync
    except (TypeError, ValueError, TriggerError):
        return []  # trigger_problems says why
    name = sync.key.upper()
    if sync.mode != "wait" or not (len(name) == 1 and name.isalnum()):
        return []  # "=" is no game key: nothing can clash
    return [f"phase {i}: keys: {combo} uses {sync.key!r}, the key the scanner types at every "
            f"volume (triggers.sync.key): bind another key"
            for i, phase in enumerate(curriculum["curriculum"])
            for combo in phase.get("keys", {}) if name in combo.upper().split("+")]


def _opening_problems(phases: list[dict]) -> list[str]:
    """``include_on_resume`` marks the phases a curriculum opens with, and only those.

    A resumed run plays them before carrying on (a fixation to settle in), so
    they are an unbroken run of non-game phases from phase 0.
    """
    out = []
    opening = True
    for i, phase in enumerate(phases):
        flag = phase.get("include_on_resume", False)
        if not isinstance(flag, bool):
            out.append(f"phase {i}: include_on_resume: expected true or false, got {flag!r}")
        elif flag and (phase["type"] == "game" or not opening):
            out.append(f"phase {i}: include_on_resume: only for the non-game phases the "
                       "curriculum opens with, with none without it before them")
        opening = opening and flag is True
    return out


def _phase_problems(phase: dict, index: int) -> list[str]:
    if phase["type"].startswith("check_"):
        from .checks import phase_problems
        return phase_problems(phase)
    if phase["type"] == "message":
        return _dismiss_problems("keys_to_dismiss", phase.get("keys_to_dismiss"))
    if phase["type"] != "game":
        return []
    out = []
    if not phase.get("game"):
        out.append("game: missing env id")
    if phase.get("mode", "duration") not in [mode.value for mode in BlockMode]:
        out.append(f"mode: expected 'duration' or 'episode', got {phase.get('mode')!r}")
    out.extend(_fps_problems(phase))
    out.extend(_keys_problems(phase))
    # advancing_outcomes gates n_episodes, so it means nothing to a block that
    # ends on the clock; the names are EnvAdapter.outcome's plus the menu's.
    if "advancing_outcomes" in phase:
        names = phase["advancing_outcomes"]
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            out.append(f"advancing_outcomes: expected a list of outcome names, got {names!r}")
        elif set(names) - OUTCOMES:
            out.append(f"advancing_outcomes: unknown {sorted(set(names) - OUTCOMES)}; "
                       f"the names are {sorted(OUTCOMES)}")
        elif phase.get("mode", "duration") != "episode":
            out.append("advancing_outcomes: only in mode 'episode' (a duration block ends on the "
                       "clock)")
    if "outcome_duration" in phase and not (
            isinstance(phase["outcome_duration"], (int, float))
            and phase["outcome_duration"] >= 0):
        out.append(f"outcome_duration: expected seconds >= 0, got {phase['outcome_duration']!r}")
    if "menu" in phase:
        out.extend(menu_problems(phase["menu"]))
    if "instructions" in phase:
        out.extend(_instructions_problems(phase["instructions"]))
    out.extend(_rewind_problems(phase))
    if not out:     # the fields above are what from_dict takes as given
        try:
            GamePhase.from_dict(phase, index)
        except ValueError as exc:
            out.append(str(exc))
    return out


#: Outcomes a block may rewind on: the game's own endings. Not `truncated`,
#: the env's step cap, which a rewound episode would hit again for ever.
REWINDABLE = frozenset({"won", "lost", "terminated"})


def _rewind_problems(phase: dict) -> list[str]:
    """``rewind_on`` (outcomes) and ``rewind_frames`` (how far back) come as a pair."""
    if "rewind_on" not in phase and "rewind_frames" not in phase:
        return []
    names, frames = phase.get("rewind_on"), phase.get("rewind_frames")
    out = []
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        out.append(f"rewind_on: expected a list of outcome names, got {names!r}")
    elif set(names) - REWINDABLE:
        out.append(f"rewind_on: {sorted(set(names) - REWINDABLE)} cannot be rewound; the "
                   f"game's own endings are {sorted(REWINDABLE)}")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        out.append(f"rewind_frames: expected the frames to go back, >= 1, got {frames!r}")
    return out


def _instructions_problems(spec: Any) -> list[str]:
    """A game phase's ``instructions``: a title, a text, and what makes the screen go away."""
    if not isinstance(spec, dict):
        return [f"instructions: expected an object, got {spec!r}"]
    out = []
    unknown = set(spec) - {"title", "text", "align", "duration", "keys_to_dismiss"}
    if unknown:
        out.append(f"instructions: unknown field(s) {sorted(unknown)}; the fields are title, "
                   "text, align, duration and keys_to_dismiss")
    if not isinstance(spec.get("title"), str) or not spec["title"]:
        out.append('instructions: title: missing (the screen says "Starting <title>" or '
                   '"Resuming <title>")')
    if spec.get("align", "center") not in ("center", "left"):
        out.append(f"instructions: align: expected 'center' or 'left', got {spec['align']!r}")
    duration = spec.get("duration")
    if duration is not None and (isinstance(duration, bool)
                                 or not isinstance(duration, (int, float)) or duration <= 0):
        out.append(f"instructions: duration: expected seconds > 0, got {duration!r}")
    keys = spec.get("keys_to_dismiss")
    out.extend(_dismiss_problems("instructions: keys_to_dismiss", keys))
    if duration is None and keys is None:
        out.append("instructions: give a duration, keys_to_dismiss, or both")
    return out


def _dismiss_problems(where: str, keys: Any) -> list[str]:
    """``keys_to_dismiss``, if given: a list of key names (:data:`~fmri_gym.keys.KEY_NAMES`)."""
    if keys is None or (isinstance(keys, list) and keys and not set(keys) - KEY_NAMES):
        return []
    return [f"{where}: expected a list of key names (as in fmri_gym/keys.py), got {keys!r}"]


def _keys_problems(phase: dict) -> list[str]:
    """``keys`` is required, and each name in it must be a key that can be pressed.

    There is no default map: which key does what differs between sites (a
    keyboard here, a gamepad that types keys there), so the file states it in
    full. A name outside :data:`~fmri_gym.keys.KEY_NAMES` can never be pressed,
    and the binding would be dead without a word; ``""`` is the no-key action.
    Whether the values fit the env's action space is checked when the env is
    built (:mod:`fmri_gym.adapters.keymap`).
    """
    keys = phase.get("keys")
    if not isinstance(keys, dict) or not keys:
        return ['keys: missing; map each key to the env action it sends, e.g. {"": 0, "LEFT": 3, '
                '"RIGHT": 2, "UP+SPACE": 5} ("" is no key held; names as in fmri_gym/keys.py; '
                "there is no default map)"]
    unknown = sorted({k for combo in keys if combo for k in combo.split("+")} - KEY_NAMES)
    if unknown:
        return [f"keys: {unknown} are not key names; use those in fmri_gym/keys.py (UP, SPACE, "
                "RETURN, LSHIFT, A-Z, 0-9, F1-F12, KP0-KP9, ...)"]
    return []


def _fps_problems(phase: dict) -> list[str]:
    """``fps`` is required: a block's rate is the config's, not the engine's.

    It was the engine's own rate when the phase left it out, so the same file
    played at a different speed depending on the backend underneath it -- and
    silently changed rate when that backend did. The manifest reports the
    engine's own rate against it after a run.
    """
    fps = phase.get("fps")
    if fps is None:
        return ["fps: missing; state the block's steps per second (the engine's own rate is "
                "not assumed: 60 for console cores and Atari, 35 / frame_skip for Doom)"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        return [f"fps: expected a positive number of steps per second, got {fps!r}"]
    return []


def trigger_problems(section: dict | None) -> list[str]:
    """What :class:`~fmri_gym.triggers.TriggerSettings` refuses, as text.

    The port is the one check added here: the settings accept a missing port
    and the backend refuses it on opening, which the editor's Check never does.

    :param section: the ``triggers`` section, or ``None``.
    :return: at most one problem (the first the settings raise).
    """
    try:
        s = TriggerSettings.from_dict(section)
    except (TypeError, ValueError, TriggerError) as exc:
        return [str(exc)]
    if s.backend in ("serial", "parallel") and not s.port:
        return [f"triggers: backend {s.backend!r} needs a port"]
    return []
