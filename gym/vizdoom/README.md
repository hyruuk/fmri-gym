# vizdoom-gym

Two wrappers over [ViZDoom](https://github.com/Farama-Foundation/ViZDoom)'s own Gymnasium envs (`vizdoom.gymnasium_wrapper`, the `Vizdoom*-v1` ids).

`KeyboardOnlyAction(env)`: a scenario with mouse axes (Deathmatch, the full-game maps) has a `Dict{"binary", "continuous"}` action space; this holds the axes at 0 and exposes the binary part, for play with keys only.

`StartedEpisodes(env)`: ViZDoom starts its first episode with `game.init()` and every later one with `game.new_episode()`, and the two mix an episode's sound slightly differently (pixels and game variables agree). This resets once on construction, so every episode you reset begins the same way and a replay of any of them matches its audio too.

```python
import gymnasium as gym
from vizdoom import gymnasium_wrapper  # noqa: F401  (registers Vizdoom*-v1)
from vizdoom_gym import KeyboardOnlyAction, StartedEpisodes
env = StartedEpisodes(KeyboardOnlyAction(gym.make("VizdoomDeathmatch-v1", render_mode="rgb_array")))
```
