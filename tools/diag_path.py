"""诊断：路径多长、收敛多快。n 变大时先看这两个数。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.engine import Engine  # noqa: E402
from wy.network import Config, Network  # noqa: E402

for n in [int(x) for x in (sys.argv[1:] or ["128", "2048"])]:
    net = Network(Config(n=n))
    eng = Engine(net)
    lens = []
    for t in range(400):
        rec = eng.step()
        lens.append(len(rec.path))
    print(f"n={n:5d} path mean={sum(lens) / len(lens):6.2f} max={max(lens):4d} min={min(lens):3d}")
    for t in range(400, 40000, 8000):
        for _ in range(8000):
            eng.step()
        print(f"   ticks={t + 8000:6d} defect={eng.defect():10.3e}")
