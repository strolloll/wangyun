"""关系的寿命：哪条关系早干活、哪条晚干活、哪条一直在干。

§16 说明白了一件事：一个窗口里「谁在干活」在换班（群内早、群外晚）。但那是整批的份额。
这里把**每条关系**的时间序列摊开，问三件事：

1. 一条关系的「重心时刻」是什么——它的负荷集中在哪一段？
2. 群内关系和群外关系的重心，谁早谁晚？
3. 焊接（形反作用于关系）之后，负荷挪到哪儿去了？

跑法：

    & $py -u tools\\load_life.py --ticks 4000 --window 100
    & $py -u tools\\load_life.py --ticks 4000 --window 100 --feedback 0.05
    & $py -u tools\\load_life.py --ticks 4000 --window 100 --feedback 0.05 --until 1500

读法：`重心` 是被负荷加权的平均时刻，`散布` 是加权标准差（这条关系是「一次性干完」还是「一直零敲碎打」）。
两个都是窗口序号换算成 tick 之后的值。
"""

from __future__ import annotations

import argparse

import numpy as np

from wy.engine import Engine
from wy.network import Config, Network
from wy.shape import StabilityMonitor, group_table, scale_of


def main() -> int:
    ap = argparse.ArgumentParser(description="关系的寿命：谁早干活、谁晚干活")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ticks", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--feedback", type=float, default=0.0)
    ap.add_argument("--until", type=int, default=0, help="只焊到第 N tick（0 = 一直焊）")
    args = ap.parse_args()

    cfg = Config(n=args.n, seed=args.seed, shape_feedback=args.feedback)
    net = Network(cfg)
    eng = Engine(net)
    mon = StabilityMonitor(net.n, cfg.window, cfg.digits)

    mod_of = np.arange(net.n) % cfg.modules
    is_inter = (mod_of[:, None] != mod_of[net.nbrs]).ravel()
    ratio = np.asarray(net.ratio, dtype=np.float64)

    frames: list[np.ndarray] = []
    prev = np.zeros(net.nbrs.shape, dtype=np.float64)
    welds = 0
    for t in range(args.ticks):
        eng.step()
        mon.push(eng.v)

        # 和 cli.run_once 一样：每 feedback_every tick 认一次群，够格的群焊一下。
        if (args.feedback > 0.0 and t > 0 and t % cfg.feedback_every == 0
                and (args.until <= 0 or t < args.until)):
            fv = eng.v
            groups = group_table(fv, net, cfg.feedback_frac * scale_of(fv), mon,
                                 need=4, touched=eng.recent(cfg.window))
            if groups:
                welds += eng.feedback(groups, args.feedback)

        if (t + 1) % args.window:
            continue
        cum = np.array(eng.load, dtype=np.float64)
        frames.append((cum - prev).ravel())
        prev = cum

    if not frames:
        print("窗口比 tick 还长，什么都没收到。")
        return 2

    M = np.stack(frames)                      # (窗口, 关系)
    W = M.shape[0]
    center_t = (np.arange(W) + 1) * args.window
    tot = M.sum(axis=0)
    live = tot > 0

    center = np.where(live, (M * center_t[:, None]).sum(axis=0) / np.maximum(tot, 1e-300), np.nan)
    var = np.where(live, (M * (center_t[:, None] - center[None, :]) ** 2).sum(axis=0)
                   / np.maximum(tot, 1e-300), np.nan)
    spread = np.sqrt(np.maximum(var, 0.0))

    def stat(mask: np.ndarray, name: str) -> None:
        c = center[mask & live]
        s = spread[mask & live]
        if c.size == 0:
            print(f"{name:<6} 没有活着的")
            return
        print(f"{name:<6} {c.size:>3} 条  重心 {c.mean():7.0f}±{c.std():5.0f}"
              f"  中位 {np.median(c):7.0f}  散布 {s.mean():7.0f}"
              f"  末窗仍活 {int((M[-1][mask] > 0).sum()):>3}")

    print(f"n={net.n}  ticks={args.ticks}  window={args.window}  "
          f"feedback={args.feedback}  until={args.until or '∞'}  改动关系 {welds} 条次")
    print(f"有向关系 {M.shape[1]} 条（群外 {int(is_inter.sum())}），其中活过的 {int(live.sum())} 条")
    print()
    stat(~is_inter, "群内")
    stat(is_inter, "群外")
    print()

    # 早干的排前面：重心时刻分三档，看群内/群外各占多少
    q = np.nanquantile(center[live], [1 / 3, 2 / 3])
    early = live & (center <= q[0])
    late = live & (center > q[1])
    print(f"重心前三分之一（早干）：群内 {int((early & ~is_inter).sum())} 条 / 群外 {int((early & is_inter).sum())} 条")
    print(f"重心后三分之一（晚干）：群内 {int((late & ~is_inter).sum())} 条 / 群外 {int((late & is_inter).sum())} 条")
    print()

    share = M[:, is_inter].sum(axis=1) / np.maximum(M.sum(axis=1), 1e-300)
    print("群外份额随时间：" + "  ".join(f"{v:.0%}" for v in share[:: max(1, W // 10)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
