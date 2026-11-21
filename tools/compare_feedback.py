"""形反作用的对照实验：同一份关系，开/关反馈，看结构是活得久了还是被锁死了。

跑法（在 D:\\望云 下）：

    & $py tools\\compare_feedback.py --ticks 4000 8000 12000

每一行是一次完整的确定性演算：k 的终值、k(r) 的台阶、违约读数、稳定位数。
别只看 k 变小就说「散了」——要看 k(r) 在各档是不是一起塌。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.cli import FRACS, run_once  # noqa: E402
from wy.network import Config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--modules", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ticks", type=int, nargs="+", default=[4000, 8000, 12000])
    ap.add_argument("--feedback", type=float, nargs="+", default=[0.0, 0.01, 0.05])
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--age", type=int, default=64)
    args = ap.parse_args()

    print(
        f"{'ticks':>6} {'fb':>6} {'k_end':>6} {'stable':>7} {'defect':>10} "
        f"{'k(r)':>26} {'plateau':>10} {'fb×':>4}"
    )
    for ticks in args.ticks:
        for fb in args.feedback:
            cfg = Config(
                n=args.n, modules=args.modules, seed=args.seed,
                shape_feedback=fb, feedback_every=args.every, feedback_age=args.age,
            )
            res = run_once(cfg, ticks)
            ks = res["k_curve"]["k"]
            plat = res["plateau"]
            print(
                f"{ticks:>6} {fb:>6.3f} {res['final']['k_at_trace']:>6} "
                f"{res['final']['stable_mean']:>7.2f} {res['final']['defect']:>10.3e} "
                f"{str(ks):>26} "
                f"{(plateau_text(plat)):>10} {res['shape_feedback']['times']:>4}"
            )
    return 0


def plateau_text(plat: dict | None) -> str:
    if not plat:
        return "-"
    return f"k={plat['k']}"


if __name__ == "__main__":
    raise SystemExit(main())
