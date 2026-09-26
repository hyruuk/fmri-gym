# AI GameStore — Game Environments

Source code for the 10 game environments featured on the AI GameStore platform,
plus `aigamestore_gym`, a Gymnasium env that plays them in lock step.

Each directory (`game1` … `game10`) is a self-contained, browser-based game
implemented in plain HTML and JavaScript (no build step). Open a game's
`index.html` to run it.

```
game1/ … game10/
  index.html      # entry point
  *.js            # game logic (ES modules)
aigamestore_gym/
  env.py          # AIGameStoreEnv: MultiBinary keys in, canvas + getGameState() out
  lockstep.js     # init script: the page's clock and Math.random, under the env's control
```

Rendering is done on an HTML canvas in code; no external art assets are
required. The games have no sound.

## The gym env

```bash
pip install -e .            # from this directory; needs a Chromium browser (system Chrome by default)
```

```python
import gymnasium as gym, aigamestore_gym
env = gym.make("AIGameStore/game4-v0")       # or aigamestore_gym.AIGameStoreEnv("game4", frame_skip=6)
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step([1, 0])   # hold env.keys[0] == "SPACE"
```

One `step()` holds the keys the action marks and advances the game by exactly
`frame_skip` of its 60 Hz frames -- no time passes between steps, so the game is
the same however long the agent takes to decide, and `reset(seed=...)` plus the
action sequence replays an episode. The observation is the canvas (RGB), the
reward is the change in the game's `score`, and `info["state"]` is the scalar
part of `getGameState()` (`gamePhase`, `score`, `currentLevel`, ...). An episode
ends when the game returns to its START screen; game overs and level clears move
on by themselves after 3 s of game time. See the docstring of `env.py`.

## Changes to the games

The games differ from the ones on the platform in two ways, both so they can be
played without a keyboard in front of them: level clears and game overs advance
by themselves after 3 s (no "press ENTER" / "press R" prompts), and the on-canvas
control hints take their key names from `?label_<KEY>=` query parameters, so a
host that remaps keys (a button box, say) can show the key the player really
presses.
