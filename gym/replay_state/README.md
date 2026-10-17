# replay-state-gym

Two Gymnasium wrappers for engines that are deterministic but keep their state out of Python's reach (a separate process, a library loaded per env, a browser page, a C++ object without a serializer).

`ReplayState(env)` records the episode's `reset` seed and every `step` action, and offers them as a savestate: `get_state()` returns bytes, `set_state(blob)` resets with the seed and replays the actions. The names are the ones envs with a real snapshot use (procgen, retro's emulator), so code over either reads the same. A restore costs one reset plus one step per recorded action, and the blob grows by one action per step.

`RelaunchOnReset(make)` builds a fresh inner env on every reset but the first, for an engine whose reset leaves something of the previous episode behind and is exact only from a fresh launch. Under `ReplayState` it makes `set_state` exact for such an engine too.

```python
from replay_state_gym import RelaunchOnReset, ReplayState
env = ReplayState(gym.make("MiniHack-Room-5x5-v0"))
env.reset(seed=3); env.step(1); env.step(2)
blob = env.get_state()          # seed 3, actions [1, 2]
env.step(0)
env.set_state(blob)             # back to after the second step
```

Both are exact only if the engine is: `reset(seed=)` must determine the episode. Test it, in a second process, rather than assume it.
