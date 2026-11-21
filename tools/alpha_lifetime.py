"""常数项 12 是不是 modules/alpha：扫 alpha，看 t(k<=2)·alpha 是不是常数。"""

from __future__ import annotations

import sys

from wy.network import Config
from wy.strip import lifetimes


def run(alpha: float, r: float, ticks: int = 16000) -> tuple[float, int]:
    lo, hi = max(0.05, alpha - 0.05), min(0.95, alpha + 0.05)
    cfg = Config(n=128, seed=7, alpha_range=(lo, hi), inter_ratio=(r, r), walk_power=2.0)
    out = lifetimes(cfg, ticks, frac=1e-2)
    return float(out.get("t2") or float("nan")), int(out.get("last_full") or 0)


def main() -> int:
    print("alpha    r      t(k<=2)   t*alpha*r   t*r    last_full")
    for r in (0.04, 0.02, 0.01):
        for a in (0.5, 0.9):
            t2, last = run(a, r)
            print(f"{a:.1f}   {r:.3f}   {t2:8.0f}   {t2 * a * r:9.1f}   {t2 * r:7.1f}   {last}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
