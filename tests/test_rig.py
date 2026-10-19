"""The rig config: picking a rig file, reading it, and refusing a wrong one."""

from __future__ import annotations

import json

import pytest

from fmri_gym import rig
from fmri_gym.rig import RigConfig

FIELDS = {**{k: "x" for k in rig.FIELDS}, "site": "site1", "pi": "MT", "modality": "fmri",
          "notes": ""}


@pytest.fixture(autouse=True)
def config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)


def test_defaults_fill_what_the_file_leaves_out():
    rig.save({**FIELDS, "rig": "desk"})
    desk = RigConfig.open(None)
    assert (desk.size, desk.is_fullscreen, desk.monitor, desk.has_vsync) == ((1024, 768), False,
                                                                              0, True)
    assert (desk.has_pad, desk.has_audio, desk.data_root) == (True, True, "data")
    assert desk.record()["screen"] == rig.DEFAULTS["screen"]


def test_settings_are_read():
    rig.save({**FIELDS, "rig": "scan", "screen": {"size": "800x600", "fullscreen": True,
                                                  "monitor": 1, "vsync": False},
              "pad": False, "audio": False, "data_root": "/data"})
    scan = RigConfig.open("scan")
    assert (scan.size, scan.is_fullscreen, scan.monitor, scan.has_vsync) == ((800, 600), True,
                                                                              1, False)
    assert (scan.has_pad, scan.has_audio, scan.data_root) == (False, False, "/data")


def test_none_or_several_rigs_need_a_name():
    with pytest.raises(ValueError, match="rig-check-long"):
        RigConfig.open(None)
    rig.save({**FIELDS, "rig": "a"})
    rig.save({**FIELDS, "rig": "b"})
    with pytest.raises(ValueError, match="pick one with --rig"):
        RigConfig.open(None)
    assert RigConfig.open("b").name == "b"
    with pytest.raises(ValueError, match="this machine has a, b"):
        RigConfig.open("c")


@pytest.mark.parametrize("change, problem", [
    ({"pi": "mt"}, "initials"),
    ({"screen": {"size": "big"}}, '"size"'),
    ({"screen": {"monitor": -1}}, '"monitor"'),
    ({"audio": "no"}, '"audio" must be true or false'),
    ({"data_root": ""}, '"data_root"'),
    ({"bogus": 1}, "unknown key 'bogus'"),
    ({"rig": "other"}, "named after its rig"),
])
def test_a_wrong_file_is_refused(change, problem):
    rig.save({**FIELDS, "rig": "r"})
    rig.path("r").write_text(json.dumps({**FIELDS, "rig": "r", **change}))
    with pytest.raises(ValueError, match=problem):
        RigConfig.open("r")


def test_a_loose_rig_json_is_pointed_at(tmp_path):
    (tmp_path / "rig.json").write_text("{}")
    with pytest.raises(ValueError, match="move .*rig.json there"):
        RigConfig.open(None)
