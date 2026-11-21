"""给弱关系打折：比例的指数 γ，一头是走廊，一头是纯差值。

现在的走动规则是 `score = 比例 × |值差|`（§9.6）。它的副作用是：群内收敛之后，
群外那些弱关系因为 |值差| 大，score 反超群内（0.004 × 450 vs 0.85 × 1.2），
于是路掉头去干群外，结构被自己拆掉（§16）。这条解释有一个可以直接反证的改动：

    score = 比例**γ × |值差|

- γ = 1 是现在这条；
- γ 调大，更看重「强关系」本身，弱关系的差值补不回来——结构该活得久；
- γ 调小到 0，就变成「只看差值」，比例完全不算数。

两个失效方向要对着一块看：**走廊**（少数关系干所有活，touches 的 min 掉到个位数）
和 **寿命**（t2 太短）。如果 γ 真的只是在两者之间移动，那就说明 §16 的解释成立。

跑法：

    & $py -u tools\\power.py
    & $py -u tools\\power.py --gammas 0 1 2 4 --ratio 0.004
"""

from __future__ import annotations

import argparse

import numpy as np

from wy.engine import Engine
from wy.network import Config, Network
from wy.strip import lifetimes


def _coverage(cfg: Config, warmup: int, window: int) -> tuple[int, float, float, float]:
    """跑满一轮源，看每个单位轮到几次：min / 中位 / 末窗负荷集中度 / 群外负荷份额。"""
    net = Network(cfg)
    eng = Engine(net)
    for _ in range(warmup):
        eng.step()
    base = eng.touches.copy()
    load0 = np.array(eng.load, dtype=np.float64)
    for _ in range(window):
        eng.step()
    got = eng.touches - base
    load = np.array(eng.load, dtype=np.float64) - load0
    flat = load.ravel()
    live = flat[flat > 0]
    if live.size:
        top = np.sort(live)[-max(1, live.size // 10):]
        conc = float(top.sum() / flat.sum())
    else:
        conc = float("nan")
    mod_of = np.arange(net.n) % cfg.modules
    is_inter = (mod_of[:, None] != mod_of[net.nbrs]).ravel()
    inter_share = float(flat[is_inter].sum() / flat.sum()) if flat.sum() > 0 else float("nan")
    return int(got.min()), float(np.median(got)), conc, inter_share


def main() -> int:
    ap = argparse.ArgumentParser(description="比例的指数 γ：走廊 vs 寿命")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ratio", type=float, default=0.004)
    ap.add_argument("--ticks", type=int, default=12000)
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 1.0, 1.5, 2.0, 3.0])
    args = ap.parse_args()

    print(f"n={args.n} seed={args.seed} inter_ratio={args.ratio}  (γ = score 里比例的指数)")
    print(f"{'γ':>5}  {'t(k<=4)':>8}  {'t(k<=2)':>8}  {'峰k':>5}  {'终态k':>6}"
          f"  {'覆盖min':>7}  {'覆盖中位':>8}  {'集中':>6}  {'群外负荷':>8}")
    for g in args.gammas:
        cfg = Config(n=args.n, seed=args.seed, inter_ratio=(args.ratio, args.ratio),
                     walk_power=g)
        d = lifetimes(cfg, args.ticks, sample=args.sample)
        mn, med, conc, inter_share = _coverage(cfg, warmup=100, window=args.n)
        f = lambda v: "-" if v is None else str(v)  # noqa: E731
        print(f"{g:>5.1f}  {f(d['t4']):>8}  {f(d['t2']):>8}  {d['peak']:>5}  {d['k_end']:>6}"
              f"  {mn:>7}  {med:>8.1f}  {conc:>6.3f}  {inter_share:>8.1%}")
    print()
    print("读法：γ 调大 t2 变长 = 结构活得更久；但覆盖 min 掉下去、集中升上去 = 走廊回来了。")
    print("      两头都坏（γ=0 寿命最短、γ 太大走廊回来）才说明这条解释成立。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
