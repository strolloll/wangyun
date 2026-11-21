"""重跑一遍 γ 扫描，多个种子取均值和范围——单种子的非单调可能只是噪声。"""

import numpy as np

from wy.network import Config
from wy.strip import lifetimes

GAMMAS = (1.0, 1.5, 2.0, 4.0)
SEEDS = (7, 8, 9)

print(f"{'γ':>5}  {'t2 均值':>9}  {'范围':>13}  {'各种子':>20}")
for g in GAMMAS:
    vals = []
    for s in SEEDS:
        cfg = Config(n=128, seed=s, inter_ratio=(0.004, 0.004), walk_power=g)
        d = lifetimes(cfg, 20000, sample=20)
        if d["t2"]:
            vals.append(d["t2"])
    if vals:
        print(f"{g:>5.1f}  {np.mean(vals):>9.0f}  {min(vals):>5}~{max(vals):<7}  {str(vals):>20}")
