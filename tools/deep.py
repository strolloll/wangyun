"""跑到很深的 tick 数，看「位数天花板」到底是被 kick 定的，还是被 float64 定的。

`run` 的代价大头是每 tick 算一次 k(r)（为了画曲线）。这里不要曲线，只要
defect 和稳定位数，所以一步只做 step + push —— 1e6 tick 也跑得动。

    python tools\\deep.py --ticks 1000000 --kick 1e-12

每 10 倍 tick 打印一行：tick / defect / 稳定位数 / 值域 / 「一 tick 之后值真的变了」的单位数。
最后一列是关键：如果某个单位整 tick 一点没动（changed=0），说明它的增量已经掉到
浮点能表示的下面去了 —— 那时候位数数的是舍入，不是物理。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.digits import StabilityMonitor  # noqa: E402
from wy.engine import Engine  # noqa: E402
from wy.network import Config, Network  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ticks", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--kick", type=float, default=1e-12)
    ap.add_argument("--digits", type=int, default=0, help="稳定前缀最多数到几位（0=用 Config 默认）")
    ap.add_argument("--every-log", type=float, default=1.0, help="每隔多少个数量级报一行")
    args = ap.parse_args()

    cfg = Config(n=args.n, seed=args.seed, kick=args.kick)
    if args.digits:
        # 量尺要够长：默认 digits=12 只能数到 13 位，撞上去会以为是物理天花板，
        # 其实只是尺子到头了。
        cfg = Config(n=args.n, seed=args.seed, kick=args.kick, digits=args.digits)
    net = Network(cfg)
    eng = Engine(net)
    mon = StabilityMonitor(net.n, cfg.window, cfg.digits)
    mon.push(eng.v)

    # 报点：10^k，外加最后一 tick
    marks = sorted({int(round(10 ** (k * args.every_log))) for k in range(0, 9)})
    marks = [m for m in marks if 0 < m <= args.ticks]

    print(f"n={args.n} seed={args.seed} kick={args.kick:g} ticks={args.ticks}")
    print(f"{'tick':>9}  {'defect':>10}  {'stable':>7}  {'vmin':>10}  {'vmax':>10}  changed")
    t0 = time.time()
    done = 0
    for mark in marks + [args.ticks]:
        while done < mark:
            prev = eng.v.copy()
            eng.step()
            mon.push(eng.v)
            done += 1
        changed = int(np.count_nonzero(eng.v != prev))
        live = eng.recent(cfg.window)
        touched = int(live.sum())
        stable = mon.mean_length(live)
        print(
            f"{done:>9}  {eng.defect():>10.3e}  {stable:>7.3f}  "
            f"{eng.v.min():>10.4f}  {eng.v.max():>10.4f}  "
            f"{changed}/{touched}  ({time.time() - t0:.1f}s)"
        )


if __name__ == "__main__":
    main()
