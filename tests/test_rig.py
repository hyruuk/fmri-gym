"""The rig config: picking a rig file, reading it, and refusing a wrong one; and every
rig file in the repo is one a run would start on."""

from __future__ import annotations

import json

import pytest

from fmri_gym.config import rig
from fmri_gym.config.rig import RigConfig

FIELDS = {**{k: "x" for k in rig.FIELDS}, "site": "site1", "pi": "MT", "modality": "fmri",
          "notes": ""}


REPO_RIGS = rig.names()  # read before any test points the folder elsewhere


@pytest.mark.parametrize("name", REPO_RIGS)
def test_repo_rig_files_are_valid(name):
    assert RigConfig.open(name).name == name


@pytest.fixture(autouse=True)
def rigs_folder(tmp_path, monkeypatch, request):
    if request.node.originalname != "test_repo_rig_files_are_valid":
        monkeypatch.setenv("FMRI_GYM_RIGS", str(tmp_path / "rigs"))
    monkeypatch.chdir(tmp_path)


def test_defaults_fill_what_the_file_leaves_out():
    rig.save({**FIELDS, "rig": "desk"})
    desk = RigConfig.open(None)
    screen = desk.screen
    assert (screen.size, screen.is_fullscreen, screen.monitor, screen.has_vsync) == (
        (1024, 768), False, 0, True)
    assert (desk.has_pad, desk.has_audio, desk.data_root) == (True, True, "data")
    assert desk.record()["screen"] == rig.DEFAULTS["screen"]


def test_settings_are_read():
    rig.save({**FIELDS, "rig": "scan", "screen": {"size": "800x600", "is_fullscreen": True,
                                                  "monitor": 1, "has_vsync": False},
              "has_pad": False, "has_audio": False, "data_root": "/data"})
    scan = RigConfig.open("site1/scan")
    screen = scan.screen
    assert (screen.size, screen.is_fullscreen, screen.monitor, screen.has_vsync) == (
        (800, 600), True, 1, False)
    assert (scan.has_pad, scan.has_audio, scan.data_root) == (False, False, "/data")


def test_none_or_several_rigs_need_a_name():
    with pytest.raises(ValueError, match="rig-check-long"):
        RigConfig.open(None)
    rig.save({**FIELDS, "rig": "a"})
    rig.save({**FIELDS, "rig": "b"})
    with pytest.raises(ValueError, match="pick one with --rig"):
        RigConfig.open(None)
    assert RigConfig.open("site1/b").name == "site1/b"
    with pytest.raises(ValueError, match="there are site1/a, site1/b"):
        RigConfig.open("site1/c")
    with pytest.raises(ValueError, match="a rig is named <site>/<rig>"):
        RigConfig.open("b")


@pytest.mark.parametrize("change, problem", [
    ({"pi": "mt"}, "initials"),
    ({"screen": {"size": "big"}}, '"size"'),
    ({"screen": {"monitor": -1}}, '"monitor"'),
    ({"has_audio": "no"}, '"has_audio" must be true or false'),
    ({"audio": False}, "unknown key 'audio'"),
    ({"data_root": ""}, '"data_root"'),
    ({"bogus": 1}, "unknown key 'bogus'"),
    ({"rig": "other"}, "sits at <site>/<rig>.json"),
])
def test_a_wrong_file_is_refused(change, problem):
    rig.save({**FIELDS, "rig": "r"})
    rig.path("site1/r").write_text(json.dumps({**FIELDS, "rig": "r", **change}))
    with pytest.raises(ValueError, match=problem):
        RigConfig.open("site1/r")


def test_a_loose_rig_json_is_pointed_at(tmp_path):
    (tmp_path / "rig.json").write_text("{}")
    with pytest.raises(ValueError, match="move rig.json there"):
        RigConfig.open(None)


def test_triggers_are_the_rigs():
    rig.save({**FIELDS, "rig": "meg", "triggers": {"sync": {"mode": "send"}, "backend": "lsl"}})
    meg = RigConfig.open("site1/meg")
    assert meg.has_trigger_setup and meg.triggers.backend == "lsl"
    assert meg.without_trigger_line().triggers.sync.mode == "none"
    rig.save({**FIELDS, "rig": "plain"})
    assert not RigConfig.open("site1/plain").has_trigger_setup
    rig.path("site1/plain").write_text(json.dumps({**FIELDS, "rig": "plain",
                                                   "triggers": {"backend": "serial"}}))
    with pytest.raises(ValueError, match="needs a port"):
        RigConfig.open("site1/plain")


def test_a_game_key_the_scanner_types_is_refused_on_that_rig():
    from fmri_gym.config.curriculum import Curriculum
    from fmri_gym.config.run import RunConfig
    rig.save({**FIELDS, "rig": "box", "triggers": {"sync": {"mode": "wait", "key": "5"}}})
    box = RigConfig.open("site1/box")

    def run_of(keys: dict) -> RunConfig:
        game = {"type": "game", "game": "CartPole-v1", "fps": 30, "keys": keys}
        curriculum = Curriculum.from_dict({"curriculum": [game]}, "cartpole.json")
        return RunConfig(curriculum=curriculum, rig=box, subject="sub-01", ses=1, run=1,
                         duration=None, end_fixation=10.0, resume=False, dummy_trigger=True)

    assert run_of({"": 0, "5": 1}).problems() and not run_of({"": 0, "LEFT": 1}).problems()
