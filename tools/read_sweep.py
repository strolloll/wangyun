"""在一张真图上扫时间：望云什么时候读到「像样」的划分。

真图的终态一定是 1 团（连通分量），结构只活在过程里——所以读真图必须带上 tick 数。
这个工具把「跑到第几个 tick」当横轴，逐点报 k(r) 台阶、群数、以及跟参照分组的 ARI。

    python tools/read_sweep.py data/miserables.txt data/miserables_labels.txt --ticks 200 400 800 1600 3200

**参照分组只用来事后对账，不进入演算。**
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.cli import FRACS  # noqa: E402
from wy.load import load_graph  # noqa: E402
from wy.network import Config  # noqa: E402
from wy.shape import group_table, k_curve, plateau, scale_of  # noqa: E402
from wy.engine import Engine  # noqa: E402
from wy.digits import StabilityMonitor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_groups import adjusted_rand, purity, read_labels  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("edges")
    ap.add_argument("labels")
    ap.add_argument("--ticks", type=int, nargs="+", default=[200, 400, 800, 1600, 3200, 6400])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--walk-power", type=float, default=2.0, dest="walk_power")
    ap.add_argument("--need", type=int, default=4)
    args = ap.parse_args()

    cfg = Config(n=0, seed=args.seed, walk_power=args.walk_power)
    net = load_graph(args.edges, cfg)
    cfg = net.cfg
    ref = read_labels(args.labels)

    eng = Engine(net)
    mon = StabilityMonitor(net.n, cfg.window, cfg.digits)
    mon.push(eng.v)
    marks = sorted(set(args.ticks))
    print(f"{'tick':>6} {'k(r) 八档':>36} {'最佳 r':>8} {'群数':>4} {'纯度':>5} {'ARI':>6}")
    t = 0
    for goal in marks:
        while t < goal:
            eng.step()
            mon.push(eng.v)
            t += 1
        fv = eng.values()
        sc = scale_of(fv)
        kk = k_curve(fv, net, FRACS)
        best = (-2.0, None, 0, 0.0)
        for frac in FRACS:
            groups = group_table(fv, net, frac * sc, mon, args.need,
                                 touched=eng.recent(cfg.window))
            if len(groups) < 2:
                continue
            pred = {int(u): g.gid for g in groups for u in g.members}
            ar = adjusted_rand(pred, ref)
            if ar > best[0]:
                best = (ar, frac, len(groups), purity(pred, ref))
        ar, frac, ng, pu = best
        print(f"{goal:>6} {str(kk):>36} {('—' if frac is None else f'{frac:g}'):>8} "
              f"{ng:>4} {pu:>5.2f} {ar:>6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
