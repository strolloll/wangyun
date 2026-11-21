"""看时间花在哪：演算 / 群表 / 布局 / 画 / 拼版。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy import render  # noqa: E402
from wy.cli import run_once  # noqa: E402
from wy.layout import broadcast, make_shape  # noqa: E402
from wy.network import Config  # noqa: E402
from wy.shape import group_table, k_curve, scale_of  # noqa: E402

FRACS = np.array([1e-1, 1e-2, 1e-3, 1e-4, 3e-4, 3e-3, 3e-2, 3e-1])


def t(label, fn):
    t0 = time.perf_counter()
    out = fn()
    print(f"{label:34s} {time.perf_counter() - t0:8.2f}s")
    return out


n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 400
cfg = Config(n=n)

res = t(f"run_once n={n} ticks={ticks}", lambda: run_once(cfg, ticks))
eng, net, mon = res["_engine"], res["_net"], res["_mon"]
scale = scale_of(eng.v)

t("k_curve (8 档)", lambda: k_curve(eng.v, net, FRACS))

fracs = [1e-1, 1e-2, 1e-3]
tables = {}
for f in fracs:
    t(f"group_table {f:g}", lambda f=f: group_table(eng.v, net, f * scale, mon, 4))
    print(f"    k={len(tables[f]) if f in tables else '-'}")
    tables[f] = group_table(eng.v, net, f * scale, mon, 4)
    print(f"    k={len(tables[f])}")

shapes = {}
shapes[1e-2] = t("make_shape mid (spring)", lambda: make_shape(tables[1e-2], net, seed=7))
shapes[1e-3] = t(
    "make_shape fine (broadcast)",
    lambda: make_shape(
        tables[1e-3], net, seed=7, iters=120,
        init=broadcast(shapes[1e-2], tables[1e-3], net, seed=9),
    ),
)
shapes[1e-1] = t(
    "make_shape coarse (one point)",
    lambda: make_shape(tables[1e-1], net, seed=7, pos=np.zeros((len(tables[1e-1]), 2))),
)

panels = []
for f in fracs:
    k = len(tables[f])
    img, _ = t(f"draw_shape k={k}", lambda f=f: render.draw_shape(shapes[f], 380, "x", 4, 7))
    panels.append(img)
t("sheet", lambda: render.sheet(panels, 3))
