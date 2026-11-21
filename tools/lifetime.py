"""结构活多久：把 §9.3 那条「寿命 ≈ 1/inter_ratio」重新量一遍。

旧结论是在「走比例最大的邻居」那个版本上测的。现在走动规则换成「走欠得最多的那条」，
而且负荷读数告诉我们路会掉头去干群外，所以寿命到底还跟不跟 inter_ratio 成反比，得重新量。

三个时刻（都用 2% 那一档判 k，且都从「最后一次还看得见整套模块」之后算起）：

- `t(k<=4)`：k 第一次掉到 4 或以下（八个模块并掉了四个）
- `t(k<=2)`：第一次掉到 2 或以下（结构基本没了）
- `最后分`：最后一次「分」发生在哪——还在重排，说明结构还有得选

跑法：

    & $py -u tools\\lifetime.py
    & $py -u tools\\lifetime.py --n 256 --ratios 0.002 0.004 0.008
"""

from __future__ import annotations

import argparse

import numpy as np

from wy.network import Config
from wy.strip import lifetimes


def main() -> int:
    ap = argparse.ArgumentParser(description="结构寿命 vs inter_ratio")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ticks", type=int, default=12000)
    ap.add_argument("--sample", type=int, default=10)
    ap.add_argument("--ratios", type=float, nargs="+",
                    default=[0.001, 0.002, 0.004, 0.008, 0.016, 0.032])
    args = ap.parse_args()

    print(f"n={args.n} seed={args.seed} ticks<={args.ticks} 每 {args.sample} tick 采一次，判 k 用 2% 档")
    print(f"{'inter_ratio':>12}  {'1/r':>8}  {'峰k':>5}  {'峰后':>7}  {'t(k<=4)':>9}  {'t(k<=2)':>9}  {'最后分':>8}  {'k_end':>6}")
    rows = []
    for r in args.ratios:
        cfg = Config(n=args.n, seed=args.seed, inter_ratio=(r, r))
        d = lifetimes(cfg, args.ticks, sample=args.sample)
        rows.append((r, d))
        f = lambda v: "-" if v is None else str(v)  # noqa: E731
        print(f"{r:>12.4f}  {1 / r:>8.0f}  {d['peak']:>5}  {d['last_full']:>7}"
              f"  {f(d['t4']):>9}  {f(d['t2']):>9}  {f(d['last_split']):>8}  {d['k_end']:>6}")

    xs, ys = [], []
    for r, d in rows:
        if d["t2"]:
            xs.append(np.log10(1.0 / r))
            ys.append(np.log10(float(d["t2"])))
    if len(xs) >= 2:
        slope, icept = np.polyfit(xs, ys, 1)
        print()
        print(f"log t(k<=2) vs log(1/r) 斜率 = {slope:.2f}（1.0 = 正好反比；旧说法就是这么来的）")
        print(f"拟合: t(k<=2) ≈ {10 ** icept:.0f} · (1/r)^{slope:.2f}")
        print(f"粗估: modules/(alpha·r) = {8 / 0.65 / 0.004:.0f}（r=0.004，modules=8, alpha≈0.65）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
