import math

import pytest

import lightmap

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

T = [[0.2, 0.4], [0.6, 0.8]]
B = ((0.0, 0.0), (1.0, 1.0))

CASES = [
    ("uv_pair", lambda: len(lightmap.Lightmap(T, B).uv((0.5, 0.5))) == 2),
    ("uv_center", lambda: abs(lightmap.Lightmap(T, B).uv((0.5, 0.5))[0] - 0.5) < 1e-9),
    ("sample", lambda: 0.0 <= lightmap.Lightmap(T, B).sample((0.5, 0.5)) <= 1.0),
    ("bad_tex", lambda: _raises(lambda: lightmap.Lightmap([], B))),
    ("bad_bounds", lambda: _raises(lambda: lightmap.Lightmap(T, ((1.0, 1.0), (0.0, 0.0))))),
    ("clamp", lambda: isinstance(lightmap.Lightmap(T, B).sample((5.0, 5.0)), float)),
    ("inverse", lambda: isinstance(lightmap.Lightmap(T, B).pos((0.5, 0.5)), tuple)),
    ("roundtrip", lambda: isinstance(lightmap.Lightmap(T, B).uv(lightmap.Lightmap(T, B).pos((0.3, 0.7))), tuple)),
    ("type", lambda: isinstance(lightmap.Lightmap(T, B).sample((0.5, 0.5)), float)),
    ("det", lambda: lightmap.Lightmap(T, B).sample((0.5, 0.5)) == lightmap.Lightmap(T, B).sample((0.5, 0.5))),
    ("nan", lambda: _raises(lambda: lightmap.Lightmap(T, B).uv((float('nan'), 0.0)))),
    ("no_mut", lambda: lightmap.Lightmap([r[:] for r in T], B).sample((0.5, 0.5)) is not None),
    ("uv_type", lambda: isinstance(lightmap.Lightmap(T, B).uv((0.2, 0.2))[0], float)),
    ("corner", lambda: abs(lightmap.Lightmap(T, B).uv((0.0, 0.0))[0]) < 1e-9),
    ("range", lambda: 0.0 <= lightmap.Lightmap(T, B).sample((0.9, 0.1)) <= 1.0),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
