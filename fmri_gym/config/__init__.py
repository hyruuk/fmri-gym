"""The three kinds of settings a run is played from, kept apart because they are
shared differently:

- the **curriculum** (:mod:`.curriculum`, its phases :mod:`.phases`): what is
  played, the sequence of phases of one run -- a game, or several. It is
  shared by every subject at every site, and knows nothing of any rig;
- the **rig config** (:mod:`.rig`): how one rig plays every run -- what the rig
  is, its window, controller, audio, triggers and data folder. It is tied to a
  site's setup, shared by every subject and every game played on it, and knows
  nothing of any curriculum;
- the **run config** (:mod:`.run`): one curriculum on one rig, and the few
  things only the run says -- who plays, which run of which session, how long,
  the test switches.

A setting goes in the kind it is shared like. ``fmri_play`` reads the three
into a :class:`~.run.RunConfig` and hands it to :class:`fmri_gym.run.Run`
(``Run.from_config``), which plays the curriculum on the rig.
"""
