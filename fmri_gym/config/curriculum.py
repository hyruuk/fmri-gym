"""The curriculum: what one run plays. Load, save, check.

A curriculum is one JSON file: a ``curriculum`` list of phases (see README);
a key that starts with ``##`` is a comment, anywhere. Several runs make a session, which is a shell
script (:func:`fmri_gym.gui.write_session`), not a curriculum.

The file says what is played, not who plays it or where: subject and test
switches are the run config (:mod:`fmri_gym.config.run`), and the window,
audio, triggers and output folder the rig's (:mod:`fmri_gym.config.rig`), so
one file serves every participant at every site. Nothing here reads a rig.

There is one shape. A bare list, an unknown top-level key, an unknown phase
type or a phase key starting with ``_`` (the one prefix the loader keeps for
itself) is refused at start-up, with the reason.

A phase may start from a shared one: ``"__include": "common/<game>.json"``
names a file (relative to the curriculum's own) holding one phase object, and
the phase's other keys are laid over it (:func:`overlay`). Curricula of one
game then write the setup once and only what differs per level. Shared
phases live in folders named ``common`` (:data:`COMMON_DIRNAME`), which hold no
curricula. Includes are resolved when the file is loaded, so everything
downstream, the manifest included, sees each phase written out in full.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..keys import KEY_NAMES
from ..menu import menu_problems

#: The phases of a run; the ``check_*`` ones make a rig check (:mod:`fmri_gym.checks`).
PHASE_TYPES = ("fixation", "message", "game", "survey", "check_display", "check_frames",
               "check_triggers", "check_controls", "check_photodiode")
SECTIONS = ("curriculum",)
#: A key that starts with this is a comment: nothing reads it.
COMMENT_PREFIX = "##"
#: A phase's key naming the file of the shared phase it starts from.
INCLUDE_KEY = "__include"
#: Folders of this name hold shared phases, which are included, not played.
COMMON_DIRNAME = "common"
#: Game-phase fields a curriculum may change between runs and still carry a
#: block on; any other change makes it a different game. ``seed`` is the
#: run's own (folded in from the run label).
RESUMABLE_CHANGES = frozenset({"duration", "n_episodes", "advance_on", "show_outcome_duration",
                               "instructions", "keys", "menu", "seed", "has_audio",
                               "rewind_on", "rewind_frames"})
#: Exit status of ``fmri_play`` when the run was quit (ESC) before its end; a
#: session script (``set -e``) stops on it instead of starting the next run.
EXIT_QUIT = 3

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

    Each field is the phase's key of the same name. The dict itself stays
    what the adapter is built from and what the block's log keeps, since its
    backend-specific keys are the adapter's business; this is the part the
    loop reads, every field with its default, so the defaults are stated in
    one place and a wrong value is refused before the block starts
    (:func:`validate_curriculum` builds one per game phase).
    """

    backend: str
    #: the env id, as the backend reads it
    game: str
    mode: BlockMode
    #: seconds the block lasts (``mode`` ``duration``, default 30) or lasts
    #: at most (``mode`` ``episode``, default 300)
    duration: float
    #: episodes an ``episode`` block lasts
    n_episodes: int
    #: A blocking block: only an episode whose outcome is listed (usually
    #: ``["won"]``) counts toward ``n_episodes``; any other replays the same
    #: instance (the seed follows the count), so the subject stays on a level
    #: until they clear it, ``duration`` runs out, or they forfeit from the
    #: menu. ``None``: every episode counts, whatever its outcome.
    advance_on: tuple[str, ...] | None
    #: An episode ending in one of these outcomes sends the world back
    #: ``rewind_frames`` frames and play goes on, as a new episode from that
    #: state: session policy, not a rule of the game, so a model played on
    #: the same env ends as the game says. Empty: no ending is rewound.
    rewind_on: tuple[str, ...]
    #: how far back a rewound episode goes; ``None`` when nothing is rewound
    rewind_frames: int | None
    #: seconds the final score and outcome stay on screen after an episode
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
    #: second that loses most taps. ``has_latched_keys`` lets a fresh keydown win
    #: instead, falling back to the held-key poll (so holding a key still
    #: repeats). Off by default: backends whose actions are key COMBINATIONS
    #: must keep polling.
    has_latched_keys: bool
    #: A turn-based block draws a frame only when the subject presses, so
    #: the seconds its HUD shows stop between presses while the clock the
    #: block actually ends on does not. ``has_live_hud`` redraws the standing
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
        :raises ValueError: ``has_live_hud`` on a real-time block, or a
            ``has_audio`` that is not true or false.
        """
        is_turn_based = bool(phase.get("is_turn_based", False))
        has_live_hud = bool(phase.get("has_live_hud", False))
        if has_live_hud and not is_turn_based:
            raise ValueError('"has_live_hud" is for is_turn_based blocks, whose wait is '
                             "unbounded; a real-time block already redraws every frame")
        has_audio = phase.get("has_audio", True)
        if not isinstance(has_audio, bool):
            raise ValueError(f'"has_audio" must be true or false, got {has_audio!r}')
        mode = BlockMode(phase.get("mode", "duration"))
        advance_on = phase.get("advance_on")
        return cls(
            backend=phase.get("backend", "gym"), game=phase["game"], mode=mode,
            duration=phase.get("duration", 30.0 if mode is BlockMode.DURATION else 300.0),
            n_episodes=phase.get("n_episodes", 1),
            advance_on=None if advance_on is None else tuple(advance_on),
            rewind_on=tuple(phase.get("rewind_on", ())),
            rewind_frames=phase.get("rewind_frames"),
            show_outcome_duration=float(phase.get("show_outcome_duration", 2.0)),
            seed=phase.get("seed", 1000 + index),
            state_stride=max(1, int(phase.get("state_stride", 1))),
            is_turn_based=is_turn_based,
            has_latched_keys=bool(phase.get("has_latched_keys", False)),
            has_live_hud=has_live_hud, has_audio=has_audio, fps=phase["fps"])


def check_shape(curriculum: Any, where: str) -> dict:
    """Enforce the one curriculum shape.

    :param curriculum: parsed JSON.
    :param where: file path (or another name for the source), for the message.
    :return: ``curriculum``, unchanged.
    :raises ValueError: if it is not an object with a ``curriculum`` list of
        typed phases, has a top-level key that is neither a section nor a
        comment, or a phase key that starts with ``_``.
    """
    if not isinstance(curriculum, dict) or not isinstance(curriculum.get("curriculum"), list):
        raise ValueError(f'{where}: expected a JSON object with a "curriculum" list')
    unknown = [k for k in curriculum if k not in SECTIONS and not is_comment(k)]
    if unknown:
        raise ValueError(f"{where}: unknown top-level key(s) {unknown}; the sections are "
                         f'{SECTIONS}, and a comment starts with "{COMMENT_PREFIX}"')
    for i, phase in enumerate(curriculum["curriculum"]):
        if not isinstance(phase, dict) or phase.get("type") not in PHASE_TYPES:
            raise ValueError(f'{where}: phase {i} must be an object whose "type" is one of '
                             f"{PHASE_TYPES}")
        underscored = [k for k in phase if k.startswith("_") and k != INCLUDE_KEY]
        if underscored:
            raise ValueError(f"{where}: phase {i} has {underscored}; a comment starts with "
                             f'"{COMMENT_PREFIX}", and "{INCLUDE_KEY}" names a file of '
                             f"{COMMON_DIRNAME}/")
    return curriculum


def is_comment(key: str) -> bool:
    """Whether ``key`` is a comment (it starts with :data:`COMMENT_PREFIX`)."""
    return key.startswith(COMMENT_PREFIX)


def game_changes(then: dict, now: dict) -> list[str]:
    """The keys that make ``now`` another game than ``then``, sorted.

    Comments and :data:`RESUMABLE_CHANGES` do not count: a block played under
    ``then`` carries on under ``now`` when nothing else differs.

    :param then: a game phase as a run played it (its manifest's copy).
    :param now: the phase at the same place in the curriculum today.
    """
    return sorted(k for k in then.keys() | now.keys() if then.get(k) != now.get(k)
                  and k not in RESUMABLE_CHANGES and not is_comment(k))


def load_curriculum(path: str, keep_includes: bool = False) -> dict:
    """Load a curriculum file, each phase's ``__include`` resolved.

    :param path: JSON file path.
    :param keep_includes: a phase that includes a shared one also keeps its
        ``__include``, as a path from the working directory, so that
        :func:`save_curriculum` writes it as an include again.
    :return: the curriculum dict, every phase in full.
    :raises ValueError: if the file does not have the curriculum shape
        (see :func:`check_shape`).
    """
    with open(path) as f:
        curriculum = json.load(f)
    if isinstance(curriculum, dict) and isinstance(curriculum.get("curriculum"), list):
        curriculum["curriculum"] = [_resolve(phase, path, frozenset(), keep_includes)
                                    for phase in curriculum["curriculum"]]
    return check_shape(curriculum, path)


def is_common_file(path: str) -> bool:
    """Whether ``path`` is a shared phase (in a :data:`COMMON_DIRNAME` folder), not a curriculum."""
    return os.path.basename(os.path.dirname(os.path.abspath(path))) == COMMON_DIRNAME


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


def _resolve(phase: Any, path: str, seen: frozenset[str], keep: bool = False) -> Any:
    """``phase`` laid over the phase its ``__include`` names, which may include another.

    :param keep: the result keeps ``__include``, as a path from the working directory.
    :raises ValueError: if the include is not a path to a JSON object, or comes back round.
    """
    if not isinstance(phase, dict) or INCLUDE_KEY not in phase:
        return phase
    name = phase[INCLUDE_KEY]
    target = os.path.normpath(os.path.join(os.path.dirname(path), str(name)))
    if not isinstance(name, str) or not os.path.isfile(target):
        raise ValueError(f'{path}: "{INCLUDE_KEY}" must name a file next to it, got {name!r}')
    if target in seen:
        raise ValueError(f"{path}: {name} includes itself, through {sorted(seen)}")
    own = overlay(_shared_phase(target, seen | {target}),
                  {k: v for k, v in phase.items() if k != INCLUDE_KEY})
    return {INCLUDE_KEY: os.path.relpath(target), **own} if keep else own


def _shared_phase(path: str, seen: frozenset[str] = frozenset()) -> dict:
    """The shared phase in ``path``, its own includes resolved.

    :raises ValueError: if the file is not one JSON object.
    """
    with open(path) as f:
        base = json.load(f)
    if not isinstance(base, dict):
        raise ValueError(f"{path}: a shared phase is one JSON object")
    return _resolve(base, path, seen | {path})


def shared_overrides(curriculum: dict) -> list[str]:
    """What the phases that keep their ``__include`` change of the shared phase, one line each.

    Laying a phase over its shared one only adds and replaces, so a changed
    value is saved as this phase's own; a removed one cannot be, and the
    phase is saved in full, without its include (:func:`save_curriculum`).
    Either way the other curricula that include the shared phase keep it as
    it is.

    :param curriculum: as :func:`load_curriculum` with ``keep_includes`` gives it, edited.
    """
    lines = []
    for i, phase in enumerate(curriculum["curriculum"]):
        if INCLUDE_KEY not in phase:
            continue
        shared = phase[INCLUDE_KEY]
        own = {k: v for k, v in phase.items() if k != INCLUDE_KEY}
        changed, removed = _departures(_shared_phase(shared), own)
        if changed:
            lines.append(f"phase {i}: {', '.join(changed)} differ from {shared}; saved as "
                         "this phase's own")
        if removed:
            lines.append(f"phase {i}: {', '.join(removed)} removed from what {shared} gives; "
                         f'saved in full, without "{INCLUDE_KEY}"')
    return lines


def _departures(base: dict, phase: dict, at: str = "") -> tuple[list[str], list[str]]:
    """The keys ``base`` sets that ``phase`` gives another value, and those it lacks."""
    changed, removed = [], []
    for key, value in base.items():
        name = at + key
        if key not in phase:
            removed.append(name)
        elif isinstance(value, dict) and isinstance(phase[key], dict):
            more = _departures(value, phase[key], name + ".")
            changed += more[0]
            removed += more[1]
        elif phase[key] != value:
            changed.append(name)
    return changed, removed


def _difference(phase: dict, base: dict) -> dict:
    """What ``phase`` lays over ``base`` to become itself (:func:`overlay`'s inverse)."""
    out = {}
    for key, value in phase.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            inner = _difference(value, base[key])
            if inner:
                out[key] = inner
        elif key not in base or base[key] != value:
            out[key] = value
    return out


def _compact(phase: dict, path: str) -> dict:
    """``phase`` as the file at ``path`` writes it: its include and what it adds, if it can."""
    if INCLUDE_KEY not in phase:
        return phase
    shared = _shared_phase(phase[INCLUDE_KEY])
    own = {k: v for k, v in phase.items() if k != INCLUDE_KEY}
    if _departures(shared, own)[1]:
        return own
    name = os.path.relpath(phase[INCLUDE_KEY], os.path.dirname(os.path.abspath(path)))
    return {INCLUDE_KEY: name, **_difference(own, shared)}


def save_curriculum(curriculum: dict, path: str) -> None:
    """Write ``curriculum`` as indented JSON.

    A phase that keeps its ``__include`` (:func:`load_curriculum`) is written
    as the include and what the phase adds to the shared one, or in full when
    it lacks something the shared phase sets (:func:`shared_overrides`).

    :param curriculum: curriculum dict.
    :param path: destination file path.
    """
    phases = [_compact(phase, path) for phase in curriculum["curriculum"]]
    with open(path, "w") as f:
        json.dump({**curriculum, "curriculum": phases}, f, indent=2)
        f.write("\n")


def new_curriculum() -> dict:
    """A minimal starting curriculum: one message, one fixation, one game."""
    return {
        "curriculum": [
            {"type": "message", "text": "Get ready\n\n(press SPACE to start)"},
            {"type": "fixation", "duration": 2.0},
            {"type": "game", "backend": "ale", "game": "ALE/Pong-v5", "mode": "duration",
             "duration": 60.0, "fps": 30, "keys": {"UP": 2, "DOWN": 3}},
            {"type": "fixation", "duration": 2.0},
        ],
    }


def validate_curriculum(curriculum: dict) -> list[str]:
    """Problems that would stop ``fmri_play`` before the first phase.

    Cheap checks only (no env is built), for the editor's Check: the phases'
    required fields and their shapes.

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
    return problems


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
        from ..checks import phase_problems
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
    # advance_on gates n_episodes, so it means nothing to a block that ends
    # on the clock; the names are EnvAdapter.outcome's plus the menu's.
    if "advance_on" in phase:
        names = phase["advance_on"]
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            out.append(f"advance_on: expected a list of outcome names, got {names!r}")
        elif set(names) - OUTCOMES:
            out.append(f"advance_on: unknown {sorted(set(names) - OUTCOMES)}; "
                       f"the names are {sorted(OUTCOMES)}")
        elif phase.get("mode", "duration") != "episode":
            out.append("advance_on: only in mode 'episode' (a duration block ends on the clock)")
    seconds = phase.get("show_outcome_duration", 0)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds < 0:
        out.append(f"show_outcome_duration: expected seconds >= 0, got {seconds!r}")
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
