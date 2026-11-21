"""色相能不能验：关系最强的两个群，色相必须最接近。

这是「色相由关系定」这条说法的红线检查——不是看着像，是能量出来。
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

from wy.color import color_space


def hue_gap(a: float, b: float) -> float:
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def check(strength: np.ndarray, gids: list[int] | None = None) -> tuple[bool, str]:
    k = len(strength)

    class G:
        def __init__(self, gid: int) -> None:
            self.gid = gid
            self.span = 1.0
            self.stable = 3.0

    groups = [G(i if gids is None else gids[i]) for i in range(k)]
    cs = color_space(groups, None, strength=strength)
    hue = [cs[g.gid][0] for g in groups]

    best_pair, best_w = None, -1.0
    for i, j in itertools.combinations(range(k), 2):
        if strength[i, j] > best_w:
            best_w, best_pair = float(strength[i, j]), (i, j)
    i, j = best_pair
    gap_best = hue_gap(hue[i], hue[j])
    gaps = [hue_gap(hue[a], hue[b]) for a, b in itertools.combinations(range(k), 2)]
    med = float(np.median(gaps))
    ok = gap_best <= med
    return ok, f"最强关系 ({i},{j}) W={best_w:.3f} 色相差 {gap_best:.3f}；全体色相差中位 {med:.3f}"


def main() -> int:
    rng = np.random.default_rng(3)
    fails = 0
    for t in range(20):
        k = int(rng.integers(4, 9))
        W = rng.uniform(0.0, 0.2, size=(k, k))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        ok, msg = check(W)
        if not ok:
            fails += 1
            print(f"  FAIL#{t}: {msg}")
    print(f"{20 - fails}/20 通过（最强关系的色相必须不宽于中位）")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
