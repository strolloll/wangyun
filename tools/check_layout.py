"""spring 在不同种子下稳不稳：8 环应该每次都闭合成圈。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.layout import spring  # noqa: E402


def cycle_w(k: int) -> np.ndarray:
    W = np.zeros((k, k))
    for i in range(k):
        j = (i + 1) % k
        W[i, j] = W[j, i] = 1.0
    return W


def score(pos: np.ndarray) -> float:
    p = pos[np.argsort(np.arctan2(pos[:, 1], pos[:, 0]))]
    d = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    return float(d.std() / max(d.mean(), 1e-12))


W = cycle_w(8)
bad = []
for seed in range(20):
    s = score(spring(W, seed=seed))
    if s > 0.02:
        bad.append((seed, round(s, 3)))
print(f"默认参数：坏 {len(bad)}/20  {bad}")

for iters, tag in ((600, "600 步"), (1200, "1200 步")):
    bad = []
    for seed in range(20):
        s = score(spring(W, seed=seed, iters=iters))
        if s > 0.02:
            bad.append((seed, round(s, 3)))
    print(f"{tag}：坏 {len(bad)}/20  {bad}")
