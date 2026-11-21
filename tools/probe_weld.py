import numpy as np

from wy.engine import Engine
from wy.network import Config, Network
from wy.shape import StabilityMonitor, group_table, scale_of


def run(seed: int, ticks: int, fb: float, until: int = 0):
    cfg = Config(n=128, seed=seed, shape_feedback=fb)
    net = Network(cfg)
    eng = Engine(net)
    mon = StabilityMonitor(net.n, cfg.window, cfg.digits)
    mod = np.arange(net.n) % cfg.modules
    inter = (mod[:, None] != mod[net.nbrs])
    welds = 0
    for t in range(ticks):
        eng.step()
        mon.push(eng.v)
        if fb > 0 and t > 0 and t % cfg.feedback_every == 0 and (until <= 0 or t < until):
            g = group_table(eng.v, net, cfg.feedback_frac * scale_of(eng.v), mon, need=4,
                            touched=eng.recent(cfg.window))
            if g:
                welds += 1
                eng.feedback(g, fb)
    r = np.asarray(net.ratio, dtype=np.float64)
    at_floor = int((r <= 1e-3 + 1e-12).sum())
    print(f"seed {seed} fb={fb} until={until or 'inf'} 焊 {welds} 轮")
    print(f"  群内 比例 {r[~inter].mean():.4f}±{r[~inter].std():.4f}  群外 {r[inter].mean():.5f}±{r[inter].std():.5f}")
    print(f"  在 1e-3 地板上的关系: {at_floor}/{r.size}  群外里有 {int((r[inter] <= 1e-3 + 1e-12).sum())}/{int(inter.sum())}")
    dv = np.abs(eng.v[:, None] - eng.v[net.nbrs])
    print(f"  score 群内 {(r * dv)[~inter].mean():.4f}  群外 {(r * dv)[inter].mean():.4f}")
    print(f"  defect {eng.defect():.3e}  k@2% {len(np.unique(np.round(eng.v / (0.02 * scale_of(eng.v))))) }")


for fb, until in ((0.0, 0), (0.05, 0), (0.05, 1500)):
    run(7, 4000, fb, until)
