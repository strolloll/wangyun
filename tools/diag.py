"""诊断：模块结构到底形成了没有，残差为什么下不去。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.digits import StabilityMonitor  # noqa: E402
from wy.engine import Engine  # noqa: E402
from wy.network import Config, Network  # noqa: E402


def report(ticks: int, inter: tuple[float, float], seed: int = 7,
           alpha: tuple[float, float] = (0.05, 0.60), kick: float = 1e-6) -> None:
    cfg = Config(inter_ratio=inter, seed=seed, alpha_range=alpha, kick=kick)
    net = Network(cfg)
    eng = Engine(net)
    mon = StabilityMonitor(cfg.n, cfg.window, cfg.digits)
    mon.push(eng.v)
    for _ in range(ticks):
        eng.step()
        mon.push(eng.v)

    v = eng.v
    mod = np.arange(cfg.n) % cfg.modules
    scale = np.abs(v).mean()
    res = eng.residual()

    within, between = [], []
    means = [v[mod == m].mean() for m in range(cfg.modules)]
    for m in range(cfg.modules):
        vv = v[mod == m]
        within.append(vv.max() - vv.min())
        for m2 in range(m + 1, cfg.modules):
            between.append(abs(means[m] - means[m2]))

    print(f"--- ticks={ticks} inter={inter} seed={seed}")
    print(f"  scale(mean|v|) = {scale:.4g}")
    print(f"  residual  mean {res.mean():.4g}  median {np.median(res):.4g}  p90 {np.percentile(res,90):.4g}")
    print(f"  residual/scale  mean {res.mean()/scale:.4g}  median {np.median(res)/scale:.4g}")
    print(f"  within-module spread  mean {np.mean(within):.4g}  max {np.max(within):.4g}")
    print(f"  between-module spread mean {np.mean(between):.4g}  min {np.min(between):.4g}")
    print(f"  module means {np.round(means, 3).tolist()}")
    print(f"  stable len  mean {mon.mean_length():.2f}")
    per = [round(float(mon.length[mod == m].mean()), 2) for m in range(cfg.modules)]
    print(f"  stable len per module {per}")
    print(f"  step size |v_t - v_t-1| mean {np.abs(mon._buf[-1] - mon._buf[-2]).mean()/scale:.3g} (relative)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, nargs="+", default=[500, 2000, 8000])
    ap.add_argument("--inter-lo", type=float, default=0.004)
    ap.add_argument("--inter-hi", type=float, default=0.01)
    ap.add_argument("--kick", type=float, default=1e-6)
    ap.add_argument("--alpha-lo", type=float, default=0.05)
    ap.add_argument("--alpha-hi", type=float, default=0.60)
    a = ap.parse_args()
    for t in a.ticks:
        report(t, (a.inter_lo, a.inter_hi), alpha=(a.alpha_lo, a.alpha_hi), kick=a.kick)
