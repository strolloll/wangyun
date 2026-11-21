"""标定：这张图到底能不能读——先量瓶颈，再看读数。

第 23 节说「同一个仪器、种下去 ARI=1.0、真图 0.34」。这一节要回答的是更实用的问题：

    给定一张图，什么时候可以信它的读数？

做法：把合成图的瓶颈强度按倍率调弱（`inter_ratio` 整体乘 f），看读数在哪一档开始崩；
同时量两个**不依赖标准答案**的量：

    contrast   = 跨群关系的平均比例 / 群内关系的平均比例   （瓶颈有多陡）
    dispersion = 群 span 的 p90 / p10                      （各群的松紧差多少倍）

    python tools\\readable.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.digits import StabilityMonitor  # noqa: E402
from wy.engine import Engine  # noqa: E402
from wy.gauge import contrast_and_dispersion  # noqa: E402
from wy.network import Config, Network  # noqa: E402
from wy.shape import group_table, partition_rel  # noqa: E402


def ari(pred: dict[int, int], ref: dict[int, int]) -> float:
    """调整兰德指数（这里不引用 tools，保持工具之间不互相依赖）。"""
    units = sorted(ref)
    n = len(units)
    pairs = {}
    for i, u in enumerate(units):
        for w in units[i + 1:]:
            key = (pred[u] == pred[w], ref[u] == ref[w])
            pairs[key] = pairs.get(key, 0) + 1
    a = pairs.get((True, True), 0)
    b = pairs.get((True, False), 0)
    c = pairs.get((False, True), 0)
    d = pairs.get((False, False), 0)
    total = a + b + c + d
    if total == 0:
        return 1.0
    exp = (a + b) * (a + c) + (c + d) * (b + d)
    exp = exp / total
    mx = ((a + b) + (a + c)) / 2.0 + ((c + d) + (b + d)) / 2.0
    if mx == exp:
        return 1.0
    return (a + d - exp) / (mx - exp)


def main() -> int:
    print(f"{'f':>5} {'设计对比':>9} {'实测contrast':>12} {'dispersion':>10} {'k':>3} {'ARI':>6}")
    for f in (1, 2, 4, 8, 16, 32, 64, 128):
        lo, hi = 0.002 * f, 0.006 * f
        cfg = Config(n=128, seed=7, walk_power=2.0, inter_ratio=(lo, hi))
        if hi >= 0.7:
            continue
        net = Network(cfg)
        eng = Engine(net)
        mon = StabilityMonitor(net.n, cfg.window, cfg.digits)
        mon.push(eng.v)
        for _ in range(3000):
            eng.step()
            mon.push(eng.v)
        v = eng.values()
        lab = partition_rel(v, net, 2.0)
        gs = group_table(v, net, 0.0, mon, 4, touched=eng.recent(cfg.window), labels=lab)
        pred = {int(u): g.gid for g in gs for u in g.members}
        ref = {u: int(u) % cfg.modules for u in range(cfg.n)}
        c, disp = contrast_and_dispersion(lab, v, net)
        design = ((lo + hi) / 2.0) / 0.85
        print(f"{f:>5} {design:>9.3f} {c:>12.3f} {disp:>10.2f} {len(gs):>3} {ari(pred, ref):>6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
