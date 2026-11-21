"""对账：整数演算 vs float64 演算，看两边在第几步开始分开。

结论用来说明「两条路一致」到底能一致到哪一步——它是数值地基的实测边界，
不是「实现有 bug」的意思（真到了分歧步，两边各自都是自洽的）。
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from wy.engine import Engine  # noqa: E402
from wy.exact import ExactEngine  # noqa: E402
from wy.network import Config, Network  # noqa: E402


def main() -> None:
    cfg = Config(n=64, modules=8, seed=5)
    a = Engine(Network(cfg))
    b = ExactEngine(Config(n=64, modules=8, seed=5))
    scale = float(np.max(np.abs(a.v)))
    print(f"{'tick':>6} {'rel_err':>12} {'same_path':>10}")
    for k in range(1, 121):
        ra = a.step()
        rb = b.step()
        err = float(np.max(np.abs(a.v - b.to_float())) / scale)
        same = tuple(ra.path) == tuple(rb.path)
        if k <= 10 or k % 10 == 0 or not same:
            print(f"{k:>6} {err:>12.3e} {str(same):>10}")
        if not same:
            print("  第一次分歧：路径不同了（上面那一步两边还一致）")
            break


if __name__ == "__main__":
    main()
