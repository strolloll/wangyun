import numpy as np

from wy.engine import Engine
from wy.network import Config, Network


def probe(seed: int, ticks: int = 800):
    cfg = Config(n=128, seed=seed)
    net = Network(cfg)
    eng = Engine(net)
    for _ in range(ticks):
        eng.step()
    mod_of = np.arange(net.n) % cfg.modules
    is_inter = (mod_of[:, None] != mod_of[net.nbrs])
    ratio = np.asarray(net.ratio, dtype=np.float64)
    dv = np.abs(eng.v[:, None] - eng.v[net.nbrs])
    load = eng.load
    tot_load = load.sum()
    tot_ratio = ratio.sum()
    print(
        f"seed {seed:>2}  t={ticks}"
        f"  比例占比 {ratio[is_inter].sum() / tot_ratio:7.4%}"
        f"  负荷占比 {load[is_inter].sum() / tot_load:7.1%}"
        f"  平均|dv| 内 {dv[~is_inter].mean():9.2f}  外 {dv[is_inter].mean():9.2f}"
        f"  比例 内 {ratio[~is_inter].mean():.3f}  外 {ratio[is_inter].mean():.4f}"
        f"  集中 {np.sort(load.ravel())[-51:].sum() / tot_load:.3f}"
    )


for s in (7, 8, 9, 11):
    probe(s)
