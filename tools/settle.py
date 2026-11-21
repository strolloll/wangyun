"""形长齐了吗：稳住的群数会不会爬到「模块数」然后停住。

到上一轮为止，settled 一直是个别扭的读数——4000 tick 时只有 1~3 个群稳得住，
再多跑就塌成一团（§19：寿命 t ≈ modules/(alpha·r)）。换句话说：
**结构还没长齐就先死了**，所以「长齐」这件事从来没被观测到。

γ=2（§20）把寿命拉长 2.6 倍，于是第一次有机会问这个问题：

    稳住的群数会不会爬到 modules（8）然后停住？
    k(2%) 会不会同时稳定在 8？

如果会，那「成形」就有一个可检的定义：**每个模块都成为一组稳住的群，且 k 停在 modules 上**。
如果不会（比如爬到 5 就停、或者一路涨到十几），那说明 settled 数的上限另有原因。

跑法：

    & $py -u tools\\settle.py                       # γ=2
    & $py -u tools\\settle.py --walk-power 1        # 对照：原规则
"""

from __future__ import annotations

import argparse

import numpy as np

from wy.engine import Engine
from wy.network import Config, Network
from wy.shape import StabilityMonitor, group_table, k_curve, plateau, scale_of

FRACS = (0.3, 0.1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)


def main() -> int:
    ap = argparse.ArgumentParser(description="稳住的群数会不会爬到 modules")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ticks", type=int, nargs="+", default=[4000, 8000, 16000, 32000])
    ap.add_argument("--walk-power", type=float, default=2.0, dest="walk_power")
    ap.add_argument("--need", type=int, default=4)
    ap.add_argument("--ratio", type=float, default=0.004)
    args = ap.parse_args()

    cfg = Config(n=args.n, seed=args.seed, inter_ratio=(args.ratio, args.ratio),
                 walk_power=args.walk_power)
    net = Network(cfg)
    eng = Engine(net)
    mon = StabilityMonitor(net.n, cfg.window, cfg.digits)

    print(f"n={net.n} modules={cfg.modules} seed={args.seed} γ={args.walk_power} need={args.need} 位")
    print(f"{'tick':>7}  {'k(3%)':>5}  {'稳住':>6}  {'最大群':>6}  {'k(r)':>28}  {'平台':>10}  {'宽度':>5}  {'defect':>10}")

    target = 0
    for t in range(1, max(args.ticks) + 1):
        eng.step()
        mon.push(eng.v)
        if t not in args.ticks:
            continue
        fv = eng.values()
        sc = scale_of(fv)
        live = eng.recent(cfg.window)
        kk = k_curve(fv, net, FRACS)
        pl = plateau(FRACS, kk)
        # 在哪个尺度上问「稳没稳」——必须问 k == modules 的那一档：
        # 尺度太粗时群是几十人的大团，span 大、位数短，永远不会 settled；
        # 那是尺度选的错，不是结构没长齐。
        idx = next((i for i, v in enumerate(kk) if v == cfg.modules), None)
        if idx is None:
            print(f"{t:>7}  {'-':>5}  {'-':>6}  {'-':>6}  {'/'.join(map(str, kk)):>28}  "
                  f"{str(pl['k']) if pl else '-':>10}  {eng.defect():>10.3g}   （这一 tick 没有 k==modules 的档）")
            continue
        frac = FRACS[idx]
        groups = group_table(fv, net, frac * sc, mon, need=args.need, touched=live)
        settled = sum(1 for g in groups if g.settled)
        biggest = max((len(g.members) for g in groups), default=0)
        target = max(target, settled)
        # 8 这一档跨了几个数量级：平台宽度才是「结构对比度」的量，settled 数不是。
        idx8 = [i for i, v in enumerate(kk) if v == cfg.modules]
        width = float(np.log10(FRACS[idx8[0]] / FRACS[idx8[-1]])) if len(idx8) >= 2 else 0.0
        print(f"{t:>7}  {kk[2]:>5}  {settled:>6}  {biggest:>6}  {'/'.join(map(str, kk)):>28}  "
              f"{str(pl['k']) if pl else '-':>10}  {width:>5.2f}  {eng.defect():>10.3g}   frac={frac:g}")

    print()
    print(f"稳住的最多见过 {target} 个（modules = {cfg.modules}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
