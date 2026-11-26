import math

import pytest

import probevol

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _p():
    return probevol.ProbeVolume((2, 2, 2), [[1.0, 0.0, 0.0, 0.0]] * 8)

CASES = [
    ("weights_sum", lambda: abs(sum(_p().weights((0.5, 0.5, 0.5))) - 1.0) < 1e-12),
    ("weights_count", lambda: len(_p().weights((0.5, 0.5, 0.5))) == 8),
    ("corner", lambda: abs(_p().weights((0.0, 0.0, 0.0))[0] - 1.0) < 1e-9),
    ("sample", lambda: isinstance(_p().sample((0.5, 0.5, 0.5), (0.0, 1.0, 0.0)), float)),
    ("bad_res", lambda: _raises(lambda: probevol.ProbeVolume((0, 2, 2), [[1.0]] * 8))),
    ("bad_probes", lambda: _raises(lambda: probevol.ProbeVolume((2, 2, 2), [[1.0]]))),
    ("clamp", lambda: isinstance(_p().weights((5.0, 5.0, 5.0)), list)),
    ("type", lambda: isinstance(_p().weights((0.5, 0.5, 0.5)), list)),
    ("det", lambda: _p().weights((0.5, 0.5, 0.5)) == _p().weights((0.5, 0.5, 0.5))),
    ("nan", lambda: _raises(lambda: _p().weights((float('nan'), 0.0, 0.0)))),
    ("no_mut", lambda: _p().sample((0.5, 0.5, 0.5), (1.0, 0.0, 0.0)) is not None),
    ("dir_zero", lambda: _raises(lambda: _p().sample((0.5, 0.5, 0.5), (0.0, 0.0, 0.0)))),
    ("sample_type", lambda: isinstance(_p().sample((0.5, 0.5, 0.5), (1.0, 0.0, 0.0)), float)),
    ("weights_range", lambda: all(w >= 0.0 for w in _p().weights((0.3, 0.3, 0.3)))),
    ("float", lambda: isinstance(_p().weights((0.5, 0.5, 0.5))[0], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
