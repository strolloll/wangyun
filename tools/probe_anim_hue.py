"""探一探：动画里色相逐帧记路以后，移动总量在什么时候起来。

找一条**短**而且一定并群的配置，好把「并群那一 tick 移动总量是峰值」写成契约。
用法：python tools\probe_anim_hue.py
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from wy.anim import frames
from wy.network import Config


def run(name: str, cfg: Config, ticks: int, count: int = 8, size: int = 160) -> None:
    trace: list = []
    t0 = time.time()
    imgs = frames(cfg, ticks, count=count, frac=3e-2, need=4, size=size, trace=trace)
    dt = time.time() - t0
    print(f"\n== {name}  n={cfg.n} modules={cfg.modules} ticks={ticks} "
          f"frames={len(imgs)}  {dt:.1f}s")
    prev_k = None
    peak, peak_t, merge_peak = -1.0, None, False
    for r in trace:
        flag = ""
        merged = False
        if prev_k is not None:
            if r["k"] < prev_k:
                flag, merged = "  <-- 并", True
            elif r["k"] > prev_k:
                flag = "  <-- 分"
        print(f"  t={r['t']:5d}  k={r['k']:2d}  moved={r['moved']:.3f}{flag}")
        if r["moved"] > peak:
            peak, peak_t, merge_peak = r["moved"], r["t"], merged
        prev_k = r["k"]
    print(f"  peak t={peak_t} moved={peak:.3f} 是并群帧={merge_peak}")


def main() -> int:
    fast = dict(alpha_range=(0.85, 0.95), inter_ratio=(0.04, 0.04), walk_power=2.0)
    run("默认 n=128", Config(n=128, seed=7), 1200)
    run("塌得快 n=128", Config(n=128, seed=7, **fast), 2000)
    run("塌得快 n=64", Config(n=64, modules=4, seed=3, **fast), 1200)
    run("塌得快 n=64 / 800", Config(n=64, modules=4, seed=3, **fast), 800)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
