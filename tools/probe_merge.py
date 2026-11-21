"""读真图的第二种读法：固定半径，看合并的先后顺序。

第一种读法（某一 tick 的 k(r) 阶梯）在真图上不好用——因为值一直在全局趋同，
阶梯的每一档都在移动。换一个问法：

    把半径 r 钉死在初值尺度的一个比例上，然后看 k 随时间往下掉。
    k 掉到 11 的那一刻，划分像不像「已知的 11 个组」？

这条读法的道理：扩散先抹平组内，再抹平组间，所以**合并的先后**就是结构的层级。
半径钉住不动，就是为了让「什么时候并」变成一个可比较的量。

    python tools/probe_merge.py data/miserables.txt data/miserables_labels.txt --frac 2e-2 --targets 11 5 3 2
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
from wy.shape import group_table, k_of, scale_of  # noqa: E402

from score_groups import adjusted_rand, purity, read_labels  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("edges")
    ap.add_argument("labels")
    ap.add_argument("--frac", type=float, default=2e-2, help="半径 = 这个比例 × 初值尺度")
    ap.add_argument("--targets", type=int, nargs="+", default=[11, 5, 3, 2])
    ap.add_argument("--ticks", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--walk-power", type=float, default=2.0, dest="walk_power")
    ap.add_argument("--need", type=int, default=4)
    ap.add_argument("--sharpness", type=float, default=1.0,
                    help="把关系比例再取一次幂（r → r**p）：锐化强弱对比，看结构会不会浮出来")
    args = ap.parse_args()

    net = load_graph(args.edges, Config(n=0, seed=args.seed, walk_power=args.walk_power))
    cfg = net.cfg
    ref = read_labels(args.labels)
    if args.sharpness != 1.0:
        p = float(args.sharpness)
        net.ratio = np.where(net.mask, net.ratio ** p, net.ratio)
        net.r_edge = net.r_edge ** p
    eng = Engine(net)
    mon = StabilityMonitor(net.n, cfg.window, cfg.digits)
    mon.push(eng.v)

    r_abs = args.frac * float(np.abs(eng.v).mean())  # 钉死在初值尺度上
    targets = sorted(set(args.targets), reverse=True)
    hits: dict[int, int] = {}
    print(f"半径 r = {r_abs:.3f}（初值尺度的 {args.frac:g}），钉住不动")
    print(f"{'tick':>6} {'k':>4} {'群数':>4} {'纯度':>5} {'ARI':>6}  说明")
    for t in range(1, args.ticks + 1):
        eng.step()
        mon.push(eng.v)
        fv = eng.values()
        k = k_of(fv, net, r_abs)
        for tg in list(targets):
            if k <= tg and tg not in hits:
                hits[tg] = t
                groups = group_table(fv, net, r_abs, mon, args.need,
                                     touched=eng.recent(cfg.window))
                pred = {int(u): g.gid for g in groups for u in g.members}
                print(f"{t:>6} {k:>4} {len(groups):>4} {purity(pred, ref):>5.2f} "
                      f"{adjusted_rand(pred, ref):>6.3f}  k 第一次 ≤ {tg}")
                targets.remove(tg)
        if not targets:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
