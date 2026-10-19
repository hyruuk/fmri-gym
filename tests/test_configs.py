"""Every config in the repo is one ``fmri-play`` would start, and the suite's table covers every backend.

Engine-free: nothing is built. ``validate_curriculum`` is the check ``fmri_play``
runs before the window opens, so a config that fails here fails at the desk.
"""

from __future__ import annotations

import re

import pytest

from fmri_gym.config.curriculum import load_curriculum, validate_curriculum
from tests.backends import BACKENDS, REPO, all_configs, game_phases, phase_backend

CONFIGS = all_configs()


@pytest.mark.parametrize("path", CONFIGS, ids=[str(p.relative_to(REPO)) for p in CONFIGS])
def test_config_validates(path):
    problems = validate_curriculum(load_curriculum(str(path)))
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("path", CONFIGS, ids=[str(p.relative_to(REPO)) for p in CONFIGS])
def test_config_names_known_backends(path):
    unknown = {phase_backend(g) for g in game_phases(path)} - set(BACKENDS) - {"p5"}
    assert not unknown, f"{unknown}: not a backend get_adapter knows"


def test_table_matches_get_adapter():
    """A backend added to ``get_adapter`` needs a row in ``tests/backends.py`` to be tested."""
    source = (REPO / "fmri_gym" / "adapters" / "__init__.py").read_text()
    registered = set(re.findall(r'backend == "(\w+)"', source))
    for group in re.findall(r"backend in \(([^)]*)\)", source):
        registered |= set(re.findall(r'"(\w+)"', group))
    registered -= {"p5"}      # an alias of aigamestore
    missing = registered - set(BACKENDS)
    extra = set(BACKENDS) - registered
    assert not missing, f"get_adapter knows {sorted(missing)}: add a Backend row in tests/backends.py"
    assert not extra, f"tests/backends.py lists {sorted(extra)}, which get_adapter does not know"


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_smoke_configs_stand_for_their_backend(backend):
    for rel in BACKENDS[backend].smoke:
        path = REPO / rel
        assert path.exists(), f"{rel}: smoke config missing"
        games = game_phases(path)
        assert games, f"{rel}: no game phase"
        assert phase_backend(games[0]) == backend, f"{rel}: first game phase is not {backend}'s"
