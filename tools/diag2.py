"""诊断之二：路径到底走到了谁身上。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.engine import Engine  # noqa: E402
from wy.network import Config, Network  # noqa: E402


def main(ticks: int = 2000, inter=(0.004, 0.01)) -> None:
    cfg = Config(inter_ratio=inter, seed=7)
    net = Network(cfg)
    eng = Engine(net)
    mod = np.arange(cfg.n) % cfg.modules

    visits = np.zeros(cfg.n, dtype=int)
    lengths = []
    intra_edges = 0
    total_edges = 0
    source_hist = np.zeros(cfg.n, dtype=int)

    for _ in range(ticks):
        rec = eng.step()
        path = rec.path
        lengths.append(len(path))
        visits[list(path)] += 1
        source_hist[rec.source] += 1
        for u, w in zip(path, path[1:]):
            total_edges += 1
            if mod[u] == mod[w]:
                intra_edges += 1

    print(f"ticks={ticks} inter={inter}")
    print(f"  path length: mean {np.mean(lengths):.2f} median {np.median(lengths)} max {max(lengths)}")
    print(f"  intra-edge share {intra_edges/max(total_edges,1):.3f}")
    print(f"  visits per unit: min {visits.min()} median {np.median(visits)} max {visits.max()}")
    print(f"  units with 0 visits: {int((visits==0).sum())}, <10 visits: {int((visits<10).sum())}")
    print(f"  visits per module {[int(visits[mod==m].min()) for m in range(cfg.modules)]}")
    # 出度利用率：每个单位被外面选中的次数（入边被走过）
    indeg = np.zeros(cfg.n, dtype=int)
    for u in range(cfg.n):
        for w in net.nbrs[u]:
            indeg[w] += 1
    print(f"  structural in-degree: min {indeg.min()} median {np.median(indeg)} max {indeg.max()}")
    print(f"  units with in-degree 0: {int((indeg==0).sum())}")


if __name__ == "__main__":
    main()
