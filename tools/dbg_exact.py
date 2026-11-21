from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wy.engine import Engine  # noqa: E402
from wy.exact import ExactEngine  # noqa: E402
from wy.network import Config, Network  # noqa: E402

cfg = Config(n=64, modules=8, seed=5)
a = Engine(Network(cfg))
b = ExactEngine(Config(n=64, modules=8, seed=5))

print("v0 same:", np.allclose(a.v, b.to_float(), rtol=0, atol=1e-12))
print("alpha same:", np.allclose(a.net.alpha, [x / b.C for x in b.alpha], rtol=0, atol=1e-12))
print("ratio same:", all(
    abs(float(a.net.ratio[i][j]) - b.ratio[i][j] / b.C) < 1e-12
    for i in range(cfg.n) for j in range(len(b.ratio[i]))
))
print("nbrs same:", np.array_equal(a.net.nbrs, np.array(b.nbrs)))
print("kick:", a.net.cfg.kick, b.kick / b.C)

ra = a.step()
rb = b.step()
print("path same:", tuple(ra.path) == tuple(rb.path), "len", len(ra.path))
n = cfg.n
for i in range(n):
    d = abs(a.v[i] - b.to_float()[i])
    if d > 1e-9:
        print(f"  unit {i}: float={a.v[i]:.6f} exact={b.to_float()[i]:.6f}")
