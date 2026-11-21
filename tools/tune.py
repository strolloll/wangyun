"""参数探针：只跑数值，不出图。用来找「结构能站住、数能定住」的区域。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.digits import StabilityMonitor  # noqa: E402
from wy.engine import Engine  # noqa: E402
from wy.network import Config, Network  # noqa: E402
from wy.shape import k_of, k_curve, scale_of  # noqa: E402


def probe(cfg: Config, ticks: int, frac: float = 2e-2) -> dict:
    net = Network(cfg)
    eng = Engine(net)
    mon = StabilityMonitor(cfg.n, cfg.window, cfg.digits)
    mon.push(eng.v)
    marks = {}
    for t in range(ticks):
        eng.step()
        mon.push(eng.v)
        if (t + 1) in (ticks // 4, ticks // 2, ticks):
            marks[str(t + 1)] = (round(mon.mean_length(), 2), eng.defect())
    ks = k_curve(eng.v, net, np.array([1e-3, 1e-2, 1e-1]))
    return {
        "k": k_of(eng.v, net, frac * scale_of(eng.v)),
        "ks": ks.tolist(),
        "stable": round(mon.mean_length(), 2),
        "stmax": int(mon.length.max()),
        "defect": round(eng.defect(), 4),
        "marks": marks,
    }


def main() -> None:
    for inter in (0.30, 0.10, 0.03, 0.01, 0.003):
        for ticks in (500, 2000, 8000):
            cfg = Config(inter_ratio=(inter * 0.4, inter), seed=7)
            r = probe(cfg, ticks)
            print(
                f"inter<= {inter:<6} ticks={ticks:<5} k={r['k']:<4} k(r)={r['ks']} "
                f"stable={r['stable']:<5} max={r['stmax']:<3} defect={r['defect']:<7} {r['marks']}"
            )


if __name__ == "__main__":
    main()
