"""松紧的第三种读法：把 span 除以"邻居典型差"，看能不能与规模无关地分开紧/松。"""

from __future__ import annotations

import sys

import numpy as np

from wy.digits import StabilityMonitor
from wy.engine import Engine
from wy.network import Config, Network
from wy.shape import group_table, local_scale, scale_of


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    cfg = Config(n=n, seed=7, walk_power=2.0, weak_modules=(4, 5, 6, 7),
                 weak_intra=(0.10, 0.20))
    net = Network(cfg)
    eng = Engine(net)
    mon = StabilityMonitor(net.n, cfg.window, cfg.digits)
    for _ in range(3000):
        eng.step()
        mon.push(eng.v)
    gs = group_table(eng.v, net, 0.01 * scale_of(eng.v), mon, 3, touched=eng.recent(cfg.window))
    ls = local_scale(eng.v, net)
    mod = np.arange(net.n) % cfg.modules
    print(f"{'gid':>4} {'人':>3} {'span':>9} {'median(s)':>10} {'span/s':>8}  模块")
    tight, loose = [], []
    for g in gs:
        mem = np.asarray(sorted(g.members), dtype=int)
        s = float(np.median(ls[mem]))
        rel = g.span / max(1e-12, s)
        weak = float((mod[mem] >= 4).mean())
        print(f"{g.gid:>4} {len(mem):>3} {g.span:9.3f} {s:10.3f} {rel:8.2f}  {'松' if weak > 0.6 else '紧'}")
        (loose if weak > 0.6 else tight).append(rel)
    if tight and loose:
        print(f"紧组 span/s 中位 {np.median(tight):.2f}；松组 {np.median(loose):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
