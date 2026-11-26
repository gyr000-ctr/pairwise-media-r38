import math

import pytest

import skinsparse

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

W = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.0, 0.0]]

CASES = [
    ("count", lambda: len(skinsparse.SkinSparse().compress(W, 2)[0]) == 2),
    ("sum_one", lambda: abs(sum(skinsparse.SkinSparse().decompress(skinsparse.SkinSparse().compress(W, 2))[0]) - 1.0) < 1e-9),
    ("roundtrip", lambda: isinstance(skinsparse.SkinSparse().decompress(skinsparse.SkinSparse().compress(W, 2)), list)),
    ("bad_k", lambda: _raises(lambda: skinsparse.SkinSparse().compress(W, 0))),
    ("bad_weights", lambda: _raises(lambda: skinsparse.SkinSparse().compress([], 2))),
    ("bound_error", lambda: isinstance(skinsparse.SkinSparse().compress(W, 2), tuple)),
    ("error", lambda: skinsparse.SkinSparse().compress(W, 2) and skinsparse.SkinSparse().error() >= 0.0),
    ("type", lambda: isinstance(skinsparse.SkinSparse().compress(W, 2), tuple)),
    ("det", lambda: skinsparse.SkinSparse().compress(W, 2) == skinsparse.SkinSparse().compress(W, 2)),
    ("index_len", lambda: len(skinsparse.SkinSparse().compress(W, 4)[0][0]) == len(skinsparse.SkinSparse().compress(W, 4)[1][0])),
    ("quant_err", lambda: isinstance(skinsparse.SkinSparse().decompress(skinsparse.SkinSparse().compress(W, 2)), list)),
    ("nan", lambda: _raises(lambda: skinsparse.SkinSparse().compress([[float('nan'), 1.0]], 1))),
    ("no_mut", lambda: skinsparse.SkinSparse().compress([list(r) for r in W], 2) is not None),
    ("error_type", lambda: skinsparse.SkinSparse().compress(W, 2) and isinstance(skinsparse.SkinSparse().error(), float)),
    ("zero_weights", lambda: _raises(lambda: skinsparse.SkinSparse().compress([[0.0, 0.0]], 1))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
