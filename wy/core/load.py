"""读一张真实的关系组：把外来图变成望云能吃的东西。

到这一步为止，望云只看过自己造的图（8 个模块、每个 4 条有向关系）。真图不长那样：
度数不均、有叶子、有枢纽、有孤立点。这个模块把「一行一条关系」的边表读进来，
补齐成 `Network` 那副样子，剩下的引擎和读数一行都不用改。

三件必须说清楚的事：

1. **比例从哪来**。带权图：r_uw = w_uw / max(w)，落在 (0,1]。不带权：每条关系
   都是 1.0（一样强）。比例是关系的强弱，不是它的绝对值——跟合成图同一条口径。
2. **无向输入按双向读**。边表里写了 u-v，就当 u→v 和 v→u 各有一次，
   两条比例相同。真实关系常常是单方面的，有心的话边表就该是有向的。
3. **度数用自环补齐**。engine 和形状读数都假定邻接表是 (n, d) 的方阵，
   d 取最大度数，空位填「自己」、比例 0。走动时自环必定已被走过，所以永远
   不会被选中；`mask` 记着哪些格子是真的关系，反馈那一步靠它跳过空位。

初值仍然是**纯随机**的：图的模块结构（如果有）不能从初值里漏进去。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from .network import Config


class RealNetwork:
    """一张外来图，接口与 `Network` 一致（render/shape/engine 都只认这几个字段）。"""

    def __init__(self, cfg: Config, edges, n: int, labels=None, weighted: bool = True) -> None:
        self.cfg = cfg
        self.n = n
        adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        wmax = max((abs(w) for _, _, w in edges), default=1.0) or 1.0
        # 每个单位自己最强的那条关系（归一化的分母）。
        node_max: dict[int, float] = {}
        for u, v, w in edges:
            cur = node_max.get(u, 0.0)
            if abs(w) > cur:
                node_max[u] = abs(w)
        seen = set()
        for u, v, w in edges:
            if u == v:
                continue
            key = (u, v)
            if key in seen:  # 重边只算一次
                continue
            seen.add(key)
            # 比例按**以这个单位为基准**归一：自己最强的那条关系记 1.0，其余按比值。
            # 用全图最大值归一的话，权重量纲一大，所有比例就都趴到 0.01 以下，
            # 弛豫一步只挪 alpha·0.01·Δ——几万 tick 都收敛不动（实测 defects 停在 0.019）。
            r = (abs(w) / node_max.get(u, wmax)) if weighted else 1.0
            r = float(min(max(r, 2e-2), 1.0))
            adj[u].append((v, r))
        # 无向输入：反向补上（只在原表里没有反向边时补，给有向输入留出余地）
        for u in range(n):
            for v, r in list(adj[u]):
                if (v, u) not in seen:
                    adj[v].append((u, r))
                    seen.add((v, u))
        for row in adj:
            row.sort()  # 行内 id 升序，平手取小 id 的确定性来源之一

        deg = max((len(row) for row in adj), default=1)
        deg = max(deg, 1)
        nbrs = np.zeros((n, deg), dtype=np.int64)
        ratio = np.zeros((n, deg), dtype=np.float64)
        mask = np.zeros((n, deg), dtype=bool)
        for u, row in enumerate(adj):
            for j, (v, r) in enumerate(row):
                nbrs[u, j] = v
                ratio[u, j] = r
                mask[u, j] = True
            for j in range(len(row), deg):
                nbrs[u, j] = u  # 自环占位：走动时必定已被走过
        self.nbrs = nbrs
        self.ratio = ratio
        self.mask = mask
        self.nbrs_list = [tuple(int(x) for x in row) for row in nbrs]
        self.degree = deg

        # 划分（成团）读的是**无向**关系表：一条关系两个方向取较大的那个比例，
        # 和合成图同一条口径（见 Network.__init__ 里的 edge_r）。
        edge_r: dict[tuple[int, int], float] = {}
        for u in range(n):
            for j in range(deg):
                if not mask[u, j]:
                    continue
                v = int(nbrs[u, j])
                lo, hi = (u, v) if u <= v else (v, u)
                edge_r[(lo, hi)] = max(edge_r.get((lo, hi), 0.0), float(ratio[u, j]))
        self.i_edge = np.asarray([k[0] for k in edge_r], dtype=np.int64)
        self.j_edge = np.asarray([k[1] for k in edge_r], dtype=np.int64)
        self.r_edge = np.asarray(list(edge_r.values()), dtype=np.float64)

        rng_net = np.random.default_rng([cfg.seed, 0])
        rng_val = np.random.default_rng([cfg.seed, 1])
        self.alpha = rng_net.uniform(*cfg.alpha_range, size=n)
        self.values = rng_val.uniform(cfg.v_lo, cfg.v_hi, size=n)
        self.labels = list(labels) if labels is not None else list(range(n))
        self.isolated = np.array([len(row) == 0 for row in adj])
        self.degree_vec = np.array([len(row) for row in adj])

    # ---------------------------------------------------------------- 查询

    def ratio_between(self, u: int, w: int) -> float:
        hit = np.flatnonzero(self.nbrs[u] == w)
        if hit.size == 0:
            return 0.0
        return float(self.ratio[u, hit[0]])

    def kick_delta(self, tick: int) -> float:
        rng = np.random.default_rng([self.cfg.seed, 2, tick])
        mag = self.cfg.kick * (1.0 + 0.5 * float(rng.random()))
        sign = 1.0 if float(rng.random()) < 0.5 else -1.0
        return sign * mag

    def signature(self) -> dict:
        return {
            "n": self.n,
            "degree": int(self.degree),
            "mean_degree": float(self.degree_vec.mean()),
            "max_degree": int(self.degree_vec.max()),
            "isolated": int(self.isolated.sum()),
            "seed": self.cfg.seed,
            "kick": self.cfg.kick,
            "window": self.cfg.window,
            "strong": self.cfg.strong,
            "source_rule": self.cfg.source_rule,
            "walk_rule": self.cfg.walk_rule,
            "walk_power": self.cfg.walk_power,
        }


def read_edgelist(path: str | Path, base: int = 0, weighted: bool | None = None):
    """读一行一条关系的边表。`#` 开头是注释。

    返回 (边表 [(u, v, w)], 节点数, 原始 id 表)。id 会被重排成 0..n-1。
    """
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("%"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        u, v = int(float(parts[0])), int(float(parts[1]))
        w = float(parts[2]) if len(parts) > 2 else 1.0
        rows.append((u, v, w))
    if not rows:
        raise ValueError(f"{path} 里没读到关系")
    raw = sorted({x for u, v, _ in rows for x in (u, v)})
    if weighted is None:
        weighted = any(u != v and w != 1.0 for u, v, w in rows)
    idx = {x: i for i, x in enumerate(raw)}
    edges = [(idx[u], idx[v], w) for u, v, w in rows]
    return edges, len(raw), raw


def load_graph(path: str | Path, cfg: Config | None = None, base: int = 0,
               weighted: bool | None = None) -> RealNetwork:
    """把边表读成一张能直接喂给 engine 的图。"""
    edges, n, labels = read_edgelist(path, base=base, weighted=weighted)
    base_cfg = cfg or Config()
    cfg2 = replace(base_cfg, n=n)
    use_w = bool(weighted) if weighted is not None else any(w != 1.0 for _, _, w in edges)
    return RealNetwork(cfg2, edges, n, labels=labels, weighted=use_w)
