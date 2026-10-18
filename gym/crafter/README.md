# crafter-gym

[Crafter](https://github.com/danijar/crafter) as a Gymnasium env. `crafter.Env` speaks the old `gym` API -- `reset` returns the observation alone, `step` a 4-tuple -- and is seeded at construction only, so this holds one `crafter.Env`, presents the Gymnasium contract in front of it, and rebuilds the game on `reset(seed=)`. The game is untouched.

```bash
pip install -e .            # from this directory
```

```python
import crafter_gym
env = crafter_gym.CrafterEnv(size=(512, 512))    # or gym.make("Crafter-v0", size=(512, 512))
obs, info = env.reset(seed=0)                    # obs is the (512, 512, 3) frame
obs, reward, terminated, truncated, info = env.step(5)   # do
```

Keyword arguments are `crafter.Env`'s: `area`, `view`, `size`, `reward`, `length`, `seed`. Actions are crafter's Discrete(17): 0 = noop, 1..4 = move left/right/up/down, 5 = do, 6 = sleep, 7..10 = place stone/table/furnace/plant, 11..16 = make wood/stone/iron pickaxe, wood/stone/iron sword. `info` carries `inventory`, `achievements`, `player_pos`, `semantic`. There is no savestate API, but the env pickles whole, so a caller that wants one takes it that way; an episode also replays from its seed and its actions, and replays exactly only on the crafter fork this package pins (see `pyproject.toml`).

## The eight-button menu (`CrafterMenu-v0`)

A button box has fewer buttons than crafter has actions. `MenuWrapper` gives six of them a button of their own (the four moves, `do`, `sleep`) and puts the other ten in a list the player steps through, so the whole tech tree stays reachable from eight buttons. The action space is `Discrete(19)`: crafter's own 17, then `cycle` (17) and `confirm` (18). Cycling spends a turn, stepping the engine with `noop`; otherwise looking through the menu would be free and any of the ten would cost one press. `info` gains `env_action` (what the engine was given), `menu_idx` and `menu_sel` (where the cursor ended up, and the name it is on), so a log keeps both the button and its effect.

```python
env = crafter_gym.make_menu(size=(384, 384))     # or gym.make("CrafterMenu-v0", size=(384, 384))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(17)   # cycle: the cursor moves, the world steps
```

`extra=` adds entries after the actions with no crafter action behind them: confirming one steps the engine with `noop` and names it in `info["menu_extra"]`, for a wrapper above the menu to act on (the task chain's `skip_task`).

The cursor is game state rather than a display detail: which action `confirm` fires depends on how many times `cycle` was pressed, so an agent evaluated on this interface meets the same cursor a subject does and a replay reproduces it. `direct=` chooses which action ids keep a button. Reach the wrapper's own attributes (`menu`, `menu_idx`, `action_names`) through `make_menu`: `gym.make` hands back an `OrderEnforcing` around the wrapper, and Gymnasium 1.3 forwards no attribute through it, so `env.menu_idx` raises `AttributeError` and `env.unwrapped` skips the wrapper entirely.

## The paradigm's four levels (`level=`)

`make_plain(level=...)`, `make_menu(level=...)` and both registered ids take `level`, one of `crafter_gym.LEVELS`: the scanner paradigm's rule variants of one game, the rig's own table (crafter-for-brain-scan v0.33, `core.py:LEVELS`).

| level | hostiles | life stats | task chain |
|-------|----------|------------|------------|
| `L1_affordance` | none | frozen at 9, hidden; only lava kills | 11 entries |
| `L2_homeostasis` | none | running, shown | 15: the 12 base entries, plus drink / eat / sleep |
| `L3_predation` | zombies, skeletons | running, shown | 17: as L2, plus defeating a zombie / a skeleton |
| `L4_survival` | stock | stock | none |

`LevelWrapper` imposes the rules (`levels.py`), and in every level health cannot regenerate from 0 and the view is tinted red while the player is dead. The rules are module-level picklable objects, so a pickled env comes back with its level. One seed gives the same terrain in all four levels: a peaceful world is generated like any other, with its hostiles removed afterwards. `reachable_achievements(level)` is what a level can unlock (20 of 22 without hostiles).

On L1-L3 a `TaskWrapper` (`tasks.py`, outermost) names one entry of the rig's task chain at a time and adds `task`, `task_done`, `task_next`, `task_passed`, `task_skip` and `tasks_closed` to `info`. An entry closes when crafter's own achievement counters say so, except `build_stone_shelter`, which is tested on the live world. Drink, eat and sleep are asked for the first time their stat drops to 8 of 9, and defeating a zombie or a skeleton the first time one is on screen; each then goes ahead of the base entries still open, but never replaces the one being asked for. The menu's `skip_task` entry puts the current task at the back. A finished chain is free play: the game itself is unchanged.

```python
env = crafter_gym.make_menu(level="L2_homeostasis", size=(384, 384), length=0)
obs, info = env.reset(seed=0)
print(info["task"])                      # collect_wood
tasks = crafter_gym.find_wrapper(env, crafter_gym.TaskWrapper)   # Gymnasium forwards no attribute
```
