"""诊断：收敛停住的时候，是谁在拖后腿。

看三件事：入度分布（有没有永远不被弛豫的单位）、残差最大的那批单位是谁、
以及只走「最大比例边」和走「全部边」的差别。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.engine import Engine  # noqa: E402
from wy.network import Config, Network  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

net = Network(Config(n=n))
eng = Engine(net)
for _ in range(ticks):
    eng.step()

indeg = np.bincount(net.nbrs.ravel(), minlength=n)
print("入度分布：", dict(zip(*np.unique(indeg, return_counts=True))))
print("defect =", f"{eng.defect():.3e}")

res = eng.residual()
order = np.argsort(-res)
print("残差最大的 10 个单位：", order[:10].tolist())
print("它们的残差：", np.round(res[order[:10]], 1).tolist())
print("它们的值：", np.round(eng.v[order[:10]], 1).tolist())
mod = order[:10] % 8
print("分属模块：", mod.tolist())
print("残差中位：", f"{np.median(res):.1f}", " 均值：", f"{res.mean():.1f}")

# 每个单位用到的最大比例边是哪些
w = net.ratio
best = np.argmax(w, axis=1)
print("最大比例边覆盖了", len(set(zip(range(n), net.nbrs[np.arange(n), best].tolist()))), "条有向边")

lens = []
for _ in range(50):
    rec = eng.step()
    lens.append(len(rec.path))
print("路径长度 均值", f"{np.mean(lens):.1f}", " 最大", max(lens))
print("defect 之后：", f"{eng.defect():.3e}")
