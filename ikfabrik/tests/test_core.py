import math

import pytest

import ikfabrik

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _f():
    return ikfabrik.Fabrik([1.0, 1.0], 16, 1e-6)

CASES = [
    ("joints_len", lambda: len(_f().solve((0.0, 0.0), (1.0, 1.0))) == 3),
    ("reach", lambda: abs(math.dist(_f().solve((0.0, 0.0), (1.0, 1.0))[-1], (1.0, 1.0))) < 1e-6),
    ("bad_lengths", lambda: _raises(lambda: ikfabrik.Fabrik([], 16, 1e-6))),
    ("bad_iters", lambda: _raises(lambda: ikfabrik.Fabrik([1.0], 0, 1e-6))),
    ("bad_tol", lambda: _raises(lambda: ikfabrik.Fabrik([1.0], 16, 0.0))),
    ("joint_type", lambda: isinstance(_f().solve((0.0, 0.0), (1.0, 1.0)), list)),
    ("det", lambda: _f().solve((0.0, 0.0), (1.0, 1.0)) == _f().solve((0.0, 0.0), (1.0, 1.0))),
    ("length_keep", lambda: abs(math.dist(_f().solve((0.0, 0.0), (1.0, 1.0))[0], _f().solve((0.0, 0.0), (1.0, 1.0))[1]) - 1.0) < 1e-9),
    ("unreachable", lambda: isinstance(_f().solve((0.0, 0.0), (10.0, 0.0)), list)),
    ("nan", lambda: _raises(lambda: _f().solve((0.0, 0.0), (float('nan'), 0.0)))),
    ("no_mut", lambda: _f().solve((0.0, 0.0), (0.5, 0.5)) is not None),
    ("pair", lambda: all(len(p) == 2 for p in _f().solve((0.0, 0.0), (1.0, 1.0)))),
    ("float", lambda: isinstance(_f().solve((0.0, 0.0), (1.0, 1.0))[0][0], float)),
    ("root", lambda: abs(_f().solve((2.0, 3.0), (1.0, 1.0))[0][0] - 2.0) < 1e-9),
    ("three_segments", lambda: isinstance(ikfabrik.Fabrik([1.0, 1.0, 1.0]).solve((0.0, 0.0), (2.0, 0.0)), list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
