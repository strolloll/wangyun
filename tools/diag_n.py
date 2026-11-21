"""诊断：模块内/模块间的离散度，跟 n 和 tick 数的关系。

用来回答一个问题：模块到底是「还没收」还是「收不动」。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.engine import Engine  # noqa: E402
from wy.network import Config, Network  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
for ticks in [int(x) for x in (sys.argv[2:] or ["1000", "4000", "16000"])]:
    cfg = Config(n=n)
    net = Network(cfg)
    eng = Engine(net)
    for _ in range(ticks):
        eng.step()
    v = eng.v
    groups = [np.arange(m, n, cfg.modules) for m in range(cfg.modules)]
    within = float(np.mean([v[m].std() for m in groups]))
    means = np.array([v[m].mean() for m in groups])
    between = float(means.std())
    scale = float(np.abs(v).mean())
    print(
        f"n={n} ticks={ticks:6d} scale={scale:9.1f} within={within:9.2f} "
        f"between={between:9.2f} ratio={within / max(between, 1e-9):7.3f} "
        f"rel={within / scale:9.2e}"
    )
