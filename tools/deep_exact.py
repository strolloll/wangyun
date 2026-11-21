"""深跑：用整数精确演算测「16 位之后」的稳定性。

    python -m tools.deep_exact --ticks 50000 --kick 1e-16

跟 tools/deep.py 的区别：那边是 float64，位数天花板大约在 15~16 位（DESIGN §13.1）；
这边值的表示是 10**-30 的整数，位数只受「系数定标」和「时间」限制。

输出表每十倍 tick 打一行：tick / defect / 稳定位数（均值）/ 最近在算的比例 / vmin / vmax。
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from wy.exact import ExactEngine, StableDigits  # noqa: E402
from wy.network import Config  # noqa: E402


def defect_of(eng: ExactEngine) -> float:
    """违约读数：|v_i - 加权邻居均值| 的平均，除以 mean|v|。

    只在打点的时候算——它要读整张网，不适合每 tick 做。
    """
    v = eng.to_float()
    net = eng.net
    num = np.zeros(eng.n)
    den = np.zeros(eng.n)
    for i in range(eng.n):
        row, rats = net.nbrs[i], net.ratio[i]
        w = np.asarray(rats, dtype=np.float64)
        num[i] = float(np.dot(w, v[row]))
        den[i] = float(w.sum())
    pred = num / den
    resid = np.abs(v - pred)
    scale = float(np.mean(np.abs(v))) or 1.0
    return float(resid.mean() / scale)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ticks", type=int, default=50000)
    ap.add_argument("--kick", type=float, default=1e-16)
    ap.add_argument("--digits", type=int, default=30, help="值的定标位数")
    ap.add_argument("--window", type=int, default=8)
    args = ap.parse_args()

    cfg = Config(n=args.n, seed=args.seed, kick=args.kick)
    eng = ExactEngine(cfg, digits=args.digits)
    mon = StableDigits(eng.n, window=args.window, keep=28)

    print(f"n={cfg.n} seed={cfg.seed} kick={args.kick:g} 定标 10**-{args.digits} "
          f"系数定标 10**-{eng.C and 24}")
    print(f"{'tick':>8} {'defect':>11} {'digits':>8} {'active':>7} {'vmin':>12} {'vmax':>12}")
    marks = []
    t = 2
    while t <= args.ticks:
        marks.append(t)
        t *= 10
    t0 = time.time()
    next_mark = 0
    for step in range(args.ticks):
        eng.step()
        mon.push([eng.digit_text(i) for i in range(eng.n)])
        if next_mark < len(marks) and step + 1 == marks[next_mark]:
            v = eng.to_float()
            touched = np.array([x >= 0 for x in eng.last_touch])
            recent = np.array([(x >= 0) and (eng.tick - x <= args.window)
                               for x in eng.last_touch])
            d = mon.mean_length(recent if recent.any() else None)
            print(f"{eng.tick:>8} {defect_of(eng):>11.3g} {d:>8.2f} {touched.mean():>7.2f} "
                  f"{v.min():>12.6g} {v.max():>12.6g}")
            next_mark += 1
    dt = time.time() - t0
    print(f"\n{args.ticks} tick / {dt:.1f}s = {1e3 * dt / args.ticks:.2f} ms/tick")
    print(f"最终稳定位数（均值）: {mon.mean_length():.2f}")
    print("注意：位数读数要连 kick、tick 数、定标位数一起报（DESIGN §13.1）。")


if __name__ == "__main__":
    main()
