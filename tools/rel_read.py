"""相对半径能不能把真图读出来：τ 扫描 + 跟参照分组对账。

全局半径在第 22 节失败了（最好 ARI 0.28）。这里换成相对半径：
每个单位用自己的「邻居差的中位数」当尺子，判据变成 |Δ| ≤ τ × 局部尺度。

事先说好的判据：如果 §22 的诊断对（失败的原因是「一个全局半径伺候不了松紧差 20 倍的块」），
那相对半径下应该出现一段 ARI 明显更高的 τ；如果还是平在地上，那诊断就错了。

    python tools/rel_read.py data/miserables.txt data/miserables_labels.txt --ticks 100 200 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wy.digits import StabilityMonitor  # noqa: E402
from wy.engine import Engine  # noqa: E402
from wy.load import load_graph  # noqa: E402
from wy.network import Config  # noqa: E402
from wy.shape import group_table, k_rel, partition_rel  # noqa: E402

from score_groups import adjusted_rand, purity, read_labels  # noqa: E402

TAUS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("edges")
    ap.add_argument("labels")
    ap.add_argument("--ticks", type=int, nargs="+", default=[50, 100, 200, 400])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--walk-power", type=float, default=1.0, dest="walk_power")
    ap.add_argument("--need", type=int, default=4)
    ap.add_argument("--taus", type=float, nargs="+", default=TAUS)
    ap.add_argument("--mode", choices=["min", "max"], default="min")
    args = ap.parse_args()

    net = load_graph(args.edges, Config(n=0, seed=args.seed, walk_power=args.walk_power))
    cfg = net.cfg
    ref = read_labels(args.labels)
    eng = Engine(net)
    mon = StabilityMonitor(net.n, cfg.window, cfg.digits)
    mon.push(eng.v)

    print(f"{'tick':>5} {'τ':>5} {'k':>4} {'群数':>4} {'纯度':>5} {'ARI':>6}")
    best = (-2.0, None, None, 0, 0.0)
    t = 0
    for goal in sorted(set(args.ticks)):
        while t < goal:
            eng.step()
            mon.push(eng.v)
            t += 1
        fv = eng.values()
        for tau in args.taus:
            labels = partition_rel(fv, net, tau, mode=args.mode)
            k = int(np.unique(labels).size)
            gs = group_table(fv, net, 0.0, mon, args.need,
                             touched=eng.recent(cfg.window), labels=labels)
            pred = {int(u): g.gid for g in gs for u in g.members}
            ar = adjusted_rand(pred, ref)
            print(f"{goal:>5} {tau:>5.1f} {k:>4} {len(gs):>4} {purity(pred, ref):>5.2f} {ar:>6.3f}")
            if ar > best[0]:
                best = (ar, goal, tau, len(gs), purity(pred, ref))
    print(f"\n最好: ARI {best[0]:.3f}  在 t={best[1]} τ={best[2]}  群数 {best[3]} 纯度 {best[4]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
