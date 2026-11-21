"""诊断：走 2048 个 tick，看有多少单位从来没被走到过。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.engine import Engine  # noqa: E402
from wy.network import Config, Network  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
net = Network(Config(n=n))
eng = Engine(net)
visits = np.zeros(n, dtype=np.int64)
for _ in range(n):
    path, _e = eng.walk(eng.tick % n)
    visits[np.asarray(path, dtype=np.int64)] += 1
    eng.step()

print("跑了", n, "个 tick（源刚好轮一遍）")
print("  被走到过的单位：", int((visits > 0).sum()), "/", n)
print("  从没被走过的：", int((visits == 0).sum()))
print("  访问次数 min/中位/最大：", visits.min(), int(np.median(visits)), visits.max())
print("  访问次数为 0 的样例：", np.flatnonzero(visits == 0)[:10].tolist())
