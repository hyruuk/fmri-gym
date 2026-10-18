"""The four Crafter levels of the scanner paradigm: rule variants of one game.

The levels differ by rules, not by worlds: L1 freezes and hides the life stats
and keeps hostiles out, L2 turns the stats back on, L3 lets hostiles in, L4 is
stock crafter. :data:`LEVELS` is the rig's own table (crafter-for-brain-scan
v0.33, ``core.py:LEVELS``), field for field.

The rules are a wrapper because they are the game: a model evaluated on a
level has to meet them too. They also have to survive a savestate, which for
crafter is a pickle of the whole env chain, so each rule is an instance of a
module-level class rather than a closure (``pickle`` cannot serialise a local
function). A rule reaches the engine's own method through ``type(player)``,
never through the instance attribute it replaces, so unpickling a player and
the rules installed on it does not resolve a rule back to itself.

Two rules hold in every level, the stock-like L4 included: health cannot
regenerate from 0 (stock crafter can add a point on the tick a death set it to
0, which makes a death an unreliable episode end), and the view is tinted red
while the player is dead (the engine's own commented-out tint), the one death
signal that survives L1's hidden health icon.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar

import gymnasium as gym
import numpy as np

from .env import import_crafter

LEVELS: dict[str, dict[str, Any]] = {
    # No hostiles, no homeostatic death, the four life stats frozen at 9 and hidden.
    "L1_affordance": dict(hostiles=False, homeostatic_death=False,
                          hidden_items=("health", "food", "drink", "energy"),
                          tasks=True, frozen_stats=True),
    # The homeostat back on and visible, still nothing hostile.
    "L2_homeostasis": dict(hostiles=False, homeostatic_death=True, hidden_items=(),
                           tasks=True, stat_tasks=True),
    # Zombies and skeletons too.
    "L3_predation": dict(hostiles=True, homeostatic_death=True, hidden_items=(),
                         tasks=True, stat_tasks=True, hostile_tasks=True),
    # Stock crafter.
    "L4_survival": dict(hostiles=True, homeostatic_death=True, hidden_items=()),
}

#: The achievements a level without hostiles can never unlock. Every other one
#: stays reachable with the stats frozen, because crafter awards an unlock for
#: the act, not its effect: drinking at a full stat still pays ``collect_drink``,
#: and :class:`_InstantSleep` keeps ``wake_up`` reachable with energy at 9.
_HOSTILE_ACHIEVEMENTS = ("defeat_skeleton", "defeat_zombie")


def level_rules(level: str) -> dict[str, Any]:
    """A level's row of :data:`LEVELS`.

    :raises ValueError: if ``level`` is not one of them.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown crafter level {level!r}; the levels are {', '.join(LEVELS)}")
    return LEVELS[level]


def reachable_achievements(level: str | None) -> tuple[str, ...]:
    """The achievements a level can unlock, in crafter's own order.

    :param level: a key of :data:`LEVELS`, or ``None`` for stock crafter.
    :return: a subsequence of ``crafter.constants.achievements``.
    """
    names = tuple(import_crafter().constants.achievements)
    if level is None or level_rules(level)["hostiles"]:
        return names
    return tuple(n for n in names if n not in _HOSTILE_ACHIEVEMENTS)


class _PlayerRule:
    """A rule installed over one method of one player, picklable.

    :param player: the player it is installed on.
    :param inner: the rule it wraps, or ``None`` for the engine's own method,
        reached through the class (see the module docstring).
    """

    slot: ClassVar[str] = ""

    def __init__(self, player: Any, inner: Callable[[], None] | None = None) -> None:
        self.player = player
        self.inner = inner

    def _call_inner(self) -> None:
        if self.inner is None:
            getattr(type(self.player), self.slot)(self.player)
        else:
            self.inner()


class _NoRevive(_PlayerRule):
    """Every level: health stays at 0 once it gets there."""

    slot = "_degen_or_regen_health"

    def __call__(self) -> None:
        dead = self.player.health <= 0
        self._call_inner()
        if dead:
            self.player.health = 0


class _NoHomeostaticDeath(_PlayerRule):
    """L1: hunger, thirst and fatigue cannot cost health; lava still can (it acts earlier)."""

    slot = "_degen_or_regen_health"

    def __call__(self) -> None:
        before = self.player.health
        self._call_inner()
        self.player.health = max(self.player.health, before)


class _FrozenLifeStats(_PlayerRule):
    """L1: food, drink and energy stay at 9 (this is the only method that moves them)."""

    slot = "_update_life_stats"

    def __call__(self) -> None:
        pass


class _InstantSleep(_PlayerRule):
    """L1: the sleep button still works with energy frozen at full.

    Stock crafter refuses to fall asleep at full energy, which would make the
    button a silent no-op here; honouring it gives a one-tick nap (the stock
    wake-up fires on the next tick) and keeps ``wake_up`` reachable.
    """

    slot = "update"

    def __call__(self) -> None:
        want = not self.player.sleeping and self.player.action == "sleep"
        self._call_inner()
        if want:
            self.player.sleeping = True


class _HiddenItems:
    """Blank the named cells of the item panel; the other icons keep their places.

    :param view: the engine's own ``ItemView``.
    :param hidden: inventory names to blank.
    """

    def __init__(self, view: Any, hidden: tuple[str, ...]) -> None:
        self.view = view
        self.hidden = tuple(hidden)

    def __call__(self, inventory: dict, unit: Any) -> np.ndarray:
        return self.view({**inventory, **dict.fromkeys(self.hidden, 0)}, unit)


class _DeathTint:
    """Every level: tint the world view red while the player is dead.

    :param view: the engine's own ``LocalView``.
    """

    def __init__(self, view: Any) -> None:
        self.view = view

    def __call__(self, player: Any, unit: Any) -> np.ndarray:
        canvas = self.view(player, unit)
        if player.health < 1:
            canvas = 0.4 * np.asarray(canvas, np.float64) + 0.6 * np.array((128.0, 0.0, 0.0))
        return canvas


class _NoHostiles:
    """Keep the engine's periodic creature rebalance from placing anything hostile.

    :param game: the ``crafter.Env`` whose ``_balance_object`` this replaces.
    """

    def __init__(self, game: Any) -> None:
        self.game = game

    def __call__(self, chunk: Any, objs: Any, cls: type, *args: Any, **kwargs: Any) -> None:
        if issubclass(cls, _hostile_classes()):
            return None
        return type(self.game)._balance_object(self.game, chunk, objs, cls, *args, **kwargs)


def _hostile_classes() -> tuple[type, ...]:
    objects = import_crafter().objects
    return (objects.Zombie, objects.Skeleton, objects.Arrow)


class LevelWrapper(gym.Wrapper):
    """One of :data:`LEVELS` imposed on a :class:`~crafter_gym.env.CrafterEnv`.

    Changes no space and no API: same actions, same frame, same ``info``.
    Worldgen runs before the rules, so a level without hostiles is a world
    generated like any other with them removed, and one seed gives the same
    terrain in all four levels.

    :param env: a ``CrafterEnv``.
    :param level: a key of :data:`LEVELS`.
    :raises ValueError: if ``level`` is not one of them.
    """

    def __init__(self, env: gym.Env, level: str) -> None:
        level_rules(level)
        super().__init__(env)
        self.level = level

    @property
    def action_names(self) -> list[str]:
        """crafter's action names (Gymnasium forwards no attribute through a wrapper)."""
        return list(self.env.action_names)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        """Generate a new world, impose the level on it, and draw it again.

        The rules go on after every reset, since crafter builds a new player
        and a new world per episode; the frame the reset returned predates
        them.

        :param kwargs: ``seed`` and ``options``.
        :return: ``(frame, info)``.
        """
        _, info = self.env.reset(**kwargs)
        self._impose()
        return self.env.unwrapped.redraw(), info

    def _impose(self) -> None:
        """Install this level's rules on the current game and player."""
        rules = LEVELS[self.level]
        game = self.env.unwrapped.game
        player = game._player
        item_view = game._item_view
        item_view = item_view.view if isinstance(item_view, _HiddenItems) else item_view
        game._item_view = (_HiddenItems(item_view, rules["hidden_items"])
                           if rules["hidden_items"] else item_view)
        local_view = game._local_view
        game._local_view = _DeathTint(local_view.view if isinstance(local_view, _DeathTint)
                                      else local_view)
        if not rules["hostiles"]:
            for obj in [o for o in game._world.objects if isinstance(o, _hostile_classes())]:
                game._world.remove(obj)
            game._balance_object = _NoHostiles(game)
        health: _PlayerRule = _NoRevive(player)
        if not rules["homeostatic_death"]:
            health = _NoHomeostaticDeath(player, health)
        player._degen_or_regen_health = health
        if rules.get("frozen_stats"):
            player._update_life_stats = _FrozenLifeStats(player)
            player.update = _InstantSleep(player)
