"""关系负荷：哪条关系在干活。

k 数团、span 量团内松紧，但「哪条关系在干活」以前完全看不见。这个工具按有向关系记账：
每 tick 沿路弛豫时，这条关系实际搬了多少值，就累加到它头上。

跑法：

    & $py -u tools\\load.py --ticks 4000 --window 200
    & $py -u tools\\load.py --ticks 12000 --window 500 --exact

三列读数的意思：

- `搬运`：这个窗口里所有关系一共搬了多少值（绝对值之和）。它是「这个窗口还在动吗」。
- `集中`：搬得最多的那 10% 关系，承担了总搬运量的百分之多少。
  1.0 = 全靠少数几条；0.1 = 均匀摊到所有关系上。
- `换人`：和上一个窗口相比，前 10% 的名单里有多少是新的（Jaccard 距离）。
  0 = 名单没变，1 = 全换了。

负荷只在「还在动」的时候有定义：一个窗口的搬运量掉到 0 附近，集中和换人都没有意义。
"""

from __future__ import annotations

import argparse

import numpy as np

from wy.engine import Engine
from wy.network import Config, Network
from wy.shape import k_of, scale_of


def _decile(flat: np.ndarray) -> np.ndarray:
    """搬得最多的那 10% 关系的下标。"""
    live = np.flatnonzero(flat > 0)
    if live.size == 0:
        return np.empty(0, dtype=np.int64)
    order = live[np.argsort(-flat[live])]
    take = max(1, int(round(0.10 * order.size)))
    return order[:take]


def _share(flat: np.ndarray, idx: np.ndarray) -> float:
    total = float(flat.sum())
    if total <= 0.0:
        return float("nan")
    return float(flat[idx].sum() / total)


def _turnover(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    sa, sb = set(a.tolist()), set(b.tolist())
    union = len(sa | sb)
    return 1.0 - len(sa & sb) / union if union else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="关系负荷：哪条关系在干活")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ticks", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--exact", action="store_true")
    args = ap.parse_args()

    cfg = Config(n=args.n, seed=args.seed)
    net = Network(cfg)
    if args.exact:
        print("负荷记账目前只在 float64 引擎里（整数演算没接这条读数）。")
        return 2
    eng = Engine(net)
    total_edges = net.nbrs.size

    print(f"n={net.n}  有向关系 {total_edges} 条（每单位 {net.nbrs.shape[1]}）  window={args.window}")
    print()
    print(f"{'tick':>7}  {'搬运':>11}  {'活跃边':>7}  {'集中':>6}  {'换人':>6}  {'k@2%':>5}  {'defect':>10}  {'群外':>7}")

    prev_cum = np.zeros(net.nbrs.shape, dtype=np.float64)
    prev_top: np.ndarray = np.empty(0, dtype=np.int64)
    # 每条有向关系是群内还是群外：模块按 id 取模（见 Network 的成员表构造）。
    mod_of = np.arange(net.n) % cfg.modules
    is_inter = (mod_of[:, None] != mod_of[net.nbrs]).ravel()
    ratio_flat = np.asarray(net.ratio, dtype=np.float64).ravel()
    ratio_share = float(ratio_flat.sum())  # 群外关系的比例总和 / 全体比例总和
    rows = []
    for t in range(args.ticks):
        eng.step()
        if (t + 1) % args.window:
            continue
        cum = _cum(eng)
        inc = np.asarray(cum, dtype=np.float64).ravel() - prev_cum.ravel()
        prev_cum = np.array(cum, dtype=np.float64)  # 必须拷贝：cum 是引擎里的活数组
        top = _decile(inc)
        fv = eng.values()
        kk = k_of(fv, net, 0.02 * scale_of(fv))
        tot = float(inc.sum())
        inter_share = float(inc[is_inter].sum() / tot) if tot > 0 else float("nan")
        rows.append((t + 1, tot, int((inc > 0).sum()), _share(inc, top),
                     _turnover(prev_top, top), kk, eng.defect(), inter_share))
        prev_top = top
        r = rows[-1]
        print(f"{r[0]:>7}  {r[1]:>11.4g}  {r[2]:>7}  {r[3]:>6.3f}  {r[4]:>6.3f}  {r[5]:>5}  {r[6]:>10.3g}  {r[7]:>7.3f}")

    if len(rows) >= 3:
        first, last = rows[0], rows[-1]
        print()
        print(f"搬运: {first[1]:.4g} -> {last[1]:.4g}（{last[1] / max(first[1], 1e-30):.3f}×）")
        print(f"活跃边: {first[2]} -> {last[2]} / {total_edges}")
        print(f"集中: {first[3]:.3f} -> {last[3]:.3f}")
        print(f"换人: 首窗 {rows[0][4] if len(rows) > 1 else float('nan'):.3f} -> 末窗 {last[4]:.3f}")
        print()
        print(f"群外关系: {int(is_inter.sum())}/{total_edges} 条"
              f"（{is_inter.mean():.1%} 的关系），比例总和只占 {ratio_share and float(ratio_flat[is_inter].sum() / ratio_flat.sum()):.4%}，"
              f"搬的量占 {last[7]:.1%}")
    return 0


def _cum(eng) -> np.ndarray:
    return eng.load


if __name__ == "__main__":
    raise SystemExit(main())
