"""The task chain: one Crafter goal named at a time, on the levels that ask for it.

Crafter's tech tree is 22 achievements and no order. The paradigm's task levels
(L1-L3) name one entry of a fixed chain at a time instead -- the rig's own
(crafter-for-brain-scan v0.33, ``core.py``: ``TASK_CHAIN``, ``TASK_GOALS``,
``TASK_LABELS``, ``SKIP_ENTRY``). It is a wrapper, not a frontend feature, so a
model evaluated on a level is asked for the same things in the same order out
of the same ``info``. The chain does not change the game: crafter's own
achievements and reward are untouched, and a finished chain is free play.

An entry is closed when crafter's own counters for its goals are non-zero, so
the pointer is derived from the world each step rather than kept in step with
it. The one entry crafter has no achievement for, ``build_stone_shelter``, is
tested on the live world (:func:`_shelter_around`), read-only: a test that drew
from the world's RNG would break replay.

Two kinds of entry wait until they mean something. On a level whose homeostat
runs (``stat_tasks``), drink, eat and sleep are asked for the first time their
stat is seen at or below :data:`STAT_LOW`; on a level with hostiles
(``hostile_tasks``), defeating a zombie or a skeleton is asked for the first
time one is on screen. An armed entry goes just ahead of the first base entry
still open, never over the entry being asked for, and several armed at once
wait in :data:`_INTERRUPT_ORDER`. All of them are in the chain from the first
frame, so its length -- the score's denominator -- never changes mid-episode.

The menu's ``skip_task`` entry (:mod:`crafter_gym.menu`) puts the current task
at the back of this episode's order. The chain pickles with the env, so a
restored world carries on with the task, the order and the arming it had.

``info`` gains ``task`` (the entry asked for during this step, ``None`` once the
chain is done), ``task_done`` (it closed), ``task_next`` (the entry asked for
now), ``task_passed`` (``build_stone_shelter`` closed this step, the one
completion no achievement counter records), ``task_skip`` (the step put the task
off) and ``tasks_closed`` (how many entries of the chain are closed).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .env import import_crafter
from .levels import level_rules

TASK_CHAIN = (
    "collect_wood", "place_table", "collect_place_plant", "make_wood_pickaxe",
    "make_stone_pickaxe", "build_stone_shelter", "make_iron_pickaxe", "collect_diamond",
    "make_wood_sword", "make_stone_sword", "make_iron_sword", "eat_plant",
)
#: Entries that close on more than one achievement.
TASK_GOALS = {"collect_place_plant": ("collect_sapling", "place_plant")}
TASK_LABELS = {
    "collect_place_plant": "collect and place plant",
    "eat_plant": "eat a ripe plant",
    "build_stone_shelter": "build a stone shelter",
    "collect_drink": "drink water",
    "eat_cow": "eat a cow",
    "wake_up": "sleep",
    "defeat_zombie": "defeat a zombie",
    "defeat_skeleton": "defeat a skeleton",
}
#: The menu entry that puts the current task off.
SKIP_ENTRY = "skip_task"
#: (crafter object class, entry): asked for once one is on screen.
HOSTILE_TASKS = (("Zombie", "defeat_zombie"), ("Skeleton", "defeat_skeleton"))
#: (inventory stat, entry): asked for once the stat is at or below STAT_LOW.
STAT_TASKS = (("drink", "collect_drink"), ("food", "eat_cow"), ("energy", "wake_up"))
#: 8 of 9 rather than the rig's 6: the first point a stat moves, so an errand
#: arrives while it is still a plan, at presses 21, 26 and 31 of a world played
#: from full rather than 63, 78 and 93 (Fan, 2026-10-07).
STAT_LOW = 8
#: Interrupts waiting together are asked in this order: hostiles, then stats.
_INTERRUPT_ORDER = {entry: i for i, entry in
                    enumerate([e for _, e in HOSTILE_TASKS] + [e for _, e in STAT_TASKS])}


def chain_for(level: str) -> tuple[str, ...]:
    """The chain a level asks for, empty on a level without tasks.

    ``eat_plant`` only means something while the homeostat runs, so a level
    that freezes the stats (L1) is not asked for it.
    """
    rules = level_rules(level)
    if not rules.get("tasks"):
        return ()
    if rules.get("frozen_stats"):
        return tuple(e for e in TASK_CHAIN if e != "eat_plant")
    interrupts = ([e for _, e in HOSTILE_TASKS] if rules.get("hostile_tasks") else []) + \
                 ([e for _, e in STAT_TASKS] if rules.get("stat_tasks") else [])
    return TASK_CHAIN + tuple(interrupts)


#: What the subject reads once the chain is done.
ALL_DONE_TEXT = "ALL TASKS DONE! Free play."


def task_label(task: str) -> str:
    """A task as the subject reads it ("make wood pickaxe", "drink water")."""
    return TASK_LABELS.get(task, task.replace("_", " "))


def task_text(task: str | None) -> str:
    """The strip's line for the task asked for now."""
    return ALL_DONE_TEXT if task is None else "TASK: " + task_label(task)


class TaskWrapper(gym.Wrapper):
    """Names one entry of a chain at a time (see the module docstring).

    :param env: a level-wrapped crafter env, behind the menu if there is one.
    :param chain: the entries, in the order they are asked for.
    """

    def __init__(self, env: gym.Env, chain: tuple[str, ...]) -> None:
        super().__init__(env)
        self.base_chain = tuple(chain)
        self.chain = list(self.base_chain)
        self._unlocked: set[str] = set()
        self._passed: set[str] = set()
        self._armed: set[str] = set()

    @property
    def action_names(self) -> list[str]:
        """The wrapped env's action names (Gymnasium forwards no attribute)."""
        return list(self.env.action_names)

    @property
    def task(self) -> str | None:
        """The entry asked for now: the first open one, interrupts only once armed."""
        for entry in self.chain:
            if (entry not in _INTERRUPT_ORDER or entry in self._armed) and self._open(entry):
                return entry
        return None

    @property
    def closed(self) -> int:
        """How many entries of the chain are closed."""
        return sum(not self._open(e) for e in self.base_chain)

    def _open(self, entry: str) -> bool:
        if entry == "build_stone_shelter":
            return entry not in self._passed
        return any(g not in self._unlocked for g in TASK_GOALS.get(entry, (entry,)))

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        """Start the chain over with the episode.

        :param kwargs: passed through (``seed``, ``options``).
        :return: ``(obs, info)`` with the task fields.
        """
        obs, info = self.env.reset(**kwargs)
        self.chain = list(self.base_chain)
        self._unlocked, self._passed, self._armed = set(), set(), set()
        self._arm_hostiles()   # a zombie can be in frame from the start
        return obs, self._with_tasks(info, self.task)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Step, then close, arm and skip entries as this step's world says.

        :param action: passed through.
        :return: ``(obs, reward, terminated, truncated, info)`` with the task fields.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        task = self.task
        self._unlocked = {name for name, n in info["achievements"].items() if n > 0}
        if terminated:
            return obs, reward, terminated, truncated, self._with_tasks(info, task)
        self._arm_hostiles()
        self._arm_stats(info["inventory"])
        passed = task == "build_stone_shelter" and self._open(task) and _shelter_around(
            self.env.unwrapped.game)
        if passed:
            self._passed.add(task)
        skip = (info.get("menu_extra") == SKIP_ENTRY and task is not None
                and self.task == task)
        if skip:
            self.chain.remove(task)
            self.chain.append(task)
        return obs, reward, terminated, truncated, self._with_tasks(
            info, task, done=task is not None and not self._open(task), passed=passed,
            skip=skip and self.task != task)

    def _arm_stats(self, inventory: dict) -> None:
        for stat, entry in STAT_TASKS:
            if entry in self.chain and entry not in self._armed and inventory[stat] <= STAT_LOW:
                self._arm(entry)

    def _arm_hostiles(self) -> None:
        objects = import_crafter().objects
        game = self.env.unwrapped.game
        for name, entry in HOSTILE_TASKS:
            if (entry in self.chain and entry not in self._armed
                    and _on_screen(game, getattr(objects, name))):
                self._arm(entry)

    def _arm(self, entry: str) -> None:
        """Let ``entry`` be asked for, ahead of the base entries but not over the current task."""
        current = self.task
        self._armed.add(entry)
        if not self._open(entry):
            return
        first_base = self._first_open_base()
        waiting = [e for e in self.chain[:first_base]
                   if e in self._armed and e != current and self._open(e)]
        waiting = sorted(set(waiting) | {entry}, key=_INTERRUPT_ORDER.__getitem__)
        for e in waiting:
            self.chain.remove(e)
        at = self._first_open_base()
        self.chain[at:at] = waiting

    def _first_open_base(self) -> int:
        return next((i for i, e in enumerate(self.chain)
                     if e not in _INTERRUPT_ORDER and self._open(e)), len(self.chain))

    def _with_tasks(self, info: dict, task: str | None, done: bool = False,
                    passed: bool = False, skip: bool = False) -> dict:
        return {**info, "task": task, "task_done": done, "task_next": self.task,
                "task_passed": passed, "task_skip": skip, "tasks_closed": self.closed}


# --------------------------------------------------------------------------- #
# Reading the world, read-only
# --------------------------------------------------------------------------- #

#: Cells a zombie or skeleton can walk on, and how close to the player each
#: kind spawns on them (crafter's balance rules: zombies on grass, skeletons on
#: path); a room is open if any such cell is reachable and far enough out.
_WALKABLE = ("grass", "path", "sand")
_SPAWN_RANGE = {"grass": 6, "path": 7}
_DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))
#: A room bigger than this is not a shelter.
_MAX_ROOM = 200
#: Cells an arrow flies over, and how far: a room must be shut against a shot
#: from a walkable cell outside it, at least _ARROW_MIN cells out.
_ARROW_PATH = ("grass", "path", "sand", "water", "lava")
_ARROW_RANGE, _ARROW_MIN = 5, 2


def _shelter_around(game: Any) -> bool:
    """Whether the player stands in a sealed room nothing can spawn in or shoot into."""
    fence = import_crafter().objects.Fence
    world = game._world
    start = tuple(int(p) for p in game._player.pos)
    room, stack = {start}, [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in _DIRS:
            cell = (x + dx, y + dy)
            if cell in room:
                continue
            material, obj = world[cell]
            if material is None:
                return False
            if material not in _WALKABLE or isinstance(obj, fence):
                continue
            reach = _SPAWN_RANGE.get(material)
            if reach is not None and abs(cell[0] - start[0]) + abs(cell[1] - start[1]) >= reach:
                return False
            room.add(cell)
            if len(room) > _MAX_ROOM:
                return False
            stack.append(cell)
    return not any(_shot_from_outside(world, start, d, room, fence) for d in _DIRS)


def _shot_from_outside(world: Any, start: tuple, step: tuple, room: set, fence: type) -> bool:
    """Whether an arrow could reach ``start`` along ``step`` from outside the room."""
    for k in range(1, _ARROW_RANGE + 1):
        cell = (start[0] + step[0] * k, start[1] + step[1] * k)
        material, obj = world[cell]
        if material is None or material not in _ARROW_PATH or isinstance(obj, fence):
            return False
        if k >= _ARROW_MIN and cell not in room and material in _WALKABLE:
            return True
    return False


def _on_screen(game: Any, kind: type) -> bool:
    """Whether a creature of ``kind`` is inside the window the player is shown.

    The window is the renderer's (crafter's ``LocalView``): ``view`` tiles less
    the rows the inventory takes, centred on the player.
    """
    view = np.asarray(game._view)
    rows = int(np.ceil(len(import_crafter().constants.items) / view[0]))
    grid = np.array([view[0], view[1] - rows])
    centre = np.asarray(game._player.pos)
    for obj in game._world.objects:
        if not isinstance(obj, kind):
            continue
        pos = np.asarray(obj.pos) - centre + grid // 2
        if (pos >= 0).all() and (pos < grid).all():
            return True
    return False
