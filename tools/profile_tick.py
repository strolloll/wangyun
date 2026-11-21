"""一个 tick 里到底在干什么：引擎 / 划分 / 记账。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.engine import Engine  # noqa: E402
from wy.network import Config, Network  # noqa: E402
from wy.shape import partition, scale_of  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 400
net = Network(Config(n=n))
eng = Engine(net)

t0 = time.perf_counter()
for _ in range(ticks):
    eng.step()
t1 = time.perf_counter()
print(f"engine.step x{ticks:<5} {t1 - t0:7.2f}s   {1000 * (t1 - t0) / ticks:6.2f} ms/tick")

v = eng.v.copy()
s = scale_of(v)
t0 = time.perf_counter()
for _ in range(ticks):
    partition(v, net, 2e-2 * s)
t1 = time.perf_counter()
print(f"partition   x{ticks:<5} {t1 - t0:7.2f}s   {1000 * (t1 - t0) / ticks:6.2f} ms/tick")

t0 = time.perf_counter()
for _ in range(ticks):
    eng.defect()
t1 = time.perf_counter()
print(f"defect      x{ticks:<5} {t1 - t0:7.2f}s   {1000 * (t1 - t0) / ticks:6.2f} ms/tick")

path, _ = eng.walk(eng.tick % net.n)
print(f"一条路的长度: {len(path)}（上界 n={net.n}）")
