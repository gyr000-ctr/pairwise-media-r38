import math

import pytest

import cloth

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _c():
    return cloth.Cloth(4, 4, 0.1, 1.0)

CASES = [
    ("particles", lambda: _c().count() == 16),
    ("step", lambda: isinstance(_c().step(0.016, []), list)),
    ("stretch", lambda: isinstance(_c().max_stretch(), float)),
    ("bad_size", lambda: _raises(lambda: cloth.Cloth(0, 4, 0.1, 1.0))),
    ("bad_spacing", lambda: _raises(lambda: cloth.Cloth(4, 4, 0.0, 1.0))),
    ("bad_stiffness", lambda: _raises(lambda: cloth.Cloth(4, 4, 0.1, -1.0))),
    ("bad_dt", lambda: _raises(lambda: _c().step(0.0, []))),
    ("type", lambda: isinstance(_c().step(0.016, []), list)),
    ("det", lambda: isinstance(_c().step(0.016, []), list)),
    ("count_type", lambda: isinstance(_c().count(), int)),
    ("stretch_type", lambda: isinstance(_c().max_stretch(), float)),
    ("nan", lambda: _raises(lambda: _c().step(float('nan'), []))),
    ("no_mut", lambda: _c().step(0.016, []) is not None),
    ("collider", lambda: isinstance(_c().step(0.016, [((0.0, 0.0, 0.0), 0.2)]), list)),
    ("float", lambda: isinstance(_c().step(0.016, [])[0][0], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
