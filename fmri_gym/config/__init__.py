"""The three kinds of settings a run is played from, kept apart because they are
shared differently:

- the **curriculum** (:mod:`.curriculum`): what is played, the sequence of
  phases of one run. It is tied to a game and shared by every subject at every
  site;
- the **rig config** (:mod:`.rig`): how one rig plays every run -- what the rig
  is, its window, controller, audio and data folder. It is tied to a site's
  setup and shared by every subject and every game played on it;
- the **run config** (:mod:`.run`): one run -- who plays, which run of which
  session, the test switches -- and the rig it is played on.

A setting goes in the kind it is shared like. ``fmri_play`` reads the three
and hands them to :class:`fmri_gym.run.Run`, which plays the curriculum on the
rig.
"""
