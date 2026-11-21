"""同一个连通块里住着两种形：k 看不出来，span 看得出来。

做法：8 个模块、外连一圈（连通块数 = 1）全都不变，只把 4,5,6,7 这四个模块的
内部比例压低一档。密度分成两档，但**模块数、连通块数、拓扑全都一样**。

于是问题变成：望云读得出「这里有两种形」吗？

    python tools\\two_shapes.py --weak-lo 0.30 --weak-hi 0.50

输出每个群的成员数、群内极差（span），以及它的成员主要落在哪些模块上、
那个模块是被压过的还是原样的。判据很简单：如果 span 分成两档且跟强弱对得上，
那就是读出来了；如果糊成一锅，就是没读出来。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.digits import StabilityMonitor  # noqa: E402
from wy.engine import Engine  # noqa: E402
from wy.network import Config, Network  # noqa: E402
from wy.shape import group_table, scale_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ticks", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--frac", type=float, default=1e-2, help="认群用的相对半径")
    ap.add_argument("--weak-set", type=str, default="", help="逗号隔开；空=不弱化")
    ap.add_argument("--weak-lo", type=float, default=0.30)
    ap.add_argument("--weak-hi", type=float, default=0.50)
    ap.add_argument("--cycles", type=int, default=1, help="外连连成几个独立的环（=几个连通块）")
    args = ap.parse_args()

    weak = tuple(int(x) for x in args.weak_set.split(",") if x.strip())
    cfg = Config(
        n=args.n,
        seed=args.seed,
        weak_modules=weak,
        weak_intra=(args.weak_lo, args.weak_hi),
        inter_cycles=args.cycles,
    )
    net = Network(cfg)
    eng = Engine(net)
    mon = StabilityMonitor(net.n, cfg.window, cfg.digits)
    mon.push(eng.v)
    for _ in range(args.ticks):
        eng.step()
        mon.push(eng.v)

    r = args.frac * scale_of(eng.v)
    rows = group_table(eng.v, net, r, mon, 4, touched=eng.recent(cfg.window))
    mod_of = np.arange(cfg.n) % cfg.modules
    print(f"weak 模块 = {weak}，比例 {args.weak_lo}~{args.weak_hi}；其余 0.70~1.00")
    print(f"外连连成 {args.cycles} 个环（= 连通块数）")
    print(f"n={cfg.n} ticks={args.ticks} 认群半径 r={r:.3g}\n")
    print(f"{'gid':>4} {'人数':>5} {'span':>9}  成员落在的模块（弱=*）")
    for g in rows:
        mods = sorted({int(mod_of[u]) for u in g.members})
        tag = " ".join(f"{m}*" if m in weak else str(m) for m in mods)
        print(f"{g.gid:>4} {len(g.members):>5} {g.span:>9.3f}  {tag}")

    tight = [g.span for g in rows if not ({int(mod_of[u]) for u in g.members} & set(weak))]
    loose = [g.span for g in rows if {int(mod_of[u]) for u in g.members} & set(weak)]
    if tight and loose:
        print(
            f"\n没被压过的群 span 均值 {np.mean(tight):.3f}"
            f"（{min(tight):.2f}~{max(tight):.2f}，{len(tight)} 个）"
        )
        print(
            f"被压过的群 span 均值 {np.mean(loose):.3f}"
            f"（{min(loose):.2f}~{max(loose):.2f}，{len(loose)} 个）"
        )
        overlap = min(loose) <= max(tight)
        print("两档" + ("有重叠——没分开。" if overlap else "没有重叠——分开了。"))


if __name__ == "__main__":
    main()
