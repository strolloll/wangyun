"""诊断：挑一个卡住的单位，盯着它，看它到底被没被弛豫过。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.engine import Engine  # noqa: E402
from wy.network import Config, Network  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
target = int(sys.argv[2]) if len(sys.argv) > 2 else 599

net = Network(Config(n=n))
eng = Engine(net)
for _ in range(8000):
    eng.step()

res = eng.residual()
target = int(np.argmax(res))
nb = net.nbrs[target]
print(f"卡住的单位 u={target}  v={eng.v[target]:.1f}  residual={res[target]:.1f}")
print("  邻居：", nb.tolist())
print("  邻居值：", np.round(eng.v[nb], 1).tolist())
print("  关系比例：", np.round(net.ratio[target], 3).tolist())
print("  它自己被谁指向：", np.flatnonzero((net.nbrs == target).any(axis=0)).shape)
in_edges = [ (u, i) for u in range(n) for i in range(net.nbrs.shape[1]) if net.nbrs[u, i] == target ]
print("  入边：", in_edges[:8], "共", len(in_edges))

before = eng.v[target]
hits = 0
for t in range(2000):
    src = eng.tick % n
    path, edges = eng.walk(src)
    if target in path[1:]:
        hits += 1
    eng.step()
print(f"  2000 tick 里它作为「被弛豫的 w」出现 {hits} 次")
print(f"  值 {before:.1f} -> {eng.v[target]:.1f}   新 residual={eng.residual()[target]:.1f}")
print("  defect:", f"{eng.defect():.3e}")
