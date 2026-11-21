"""拿望云读出来的群，跟「人给的分组」（npm 的 scope）对一对。

问题：望云在依赖图上认出的群，是不是就是 `@deepseek-ai` / `@aws-sdk` 这些 scope？
答完这个才知道「按依赖关系长出来的群」和「按包管理器分组」差多少。
"""

from __future__ import annotations

import io
import sys

sys.path[:0] = ["D:\\望云", "D:\\望云\\tools"]

import numpy as np

from score_groups import adjusted_rand
from wy.engine import Engine
from wy.load import load_graph
from wy.network import Config
from wy.shape import group_table, scale_of


def scope_of(name: str) -> str:
    """人给的分组：有 scope 的按 scope，没有的一律算 core。

    注意不能用「包名第一段」——那几乎是唯一 id，ARI 会恒等于 0，等于没测。
    """
    if name.startswith("@"):
        return name.split("/")[0]
    return "core"


def main() -> None:
    edges = "data/deps.txt"
    labels_path = "data/deps_labels.txt"
    net = load_graph(edges, Config(seed=7, kick=1e-6, walk_power=2.0))
    eng = Engine(net)
    for _ in range(400):
        eng.step()

    names = {}
    for line in io.open(labels_path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        i, name = line.split(None, 1)
        names[int(i)] = name

    scopes = np.array([scope_of(names.get(u, f"#{u}")) for u in range(net.n)])
    fv = eng.values()
    scale = scale_of(fv)

    best = None
    for f in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3):
        groups = group_table(fv, net, f * scale, None, 4, touched=eng.recent(net.cfg.window))
        k = len(groups)
        if k < 2:
            continue
        lab = np.zeros(net.n, dtype=np.int64)
        for gi, g in enumerate(groups):
            lab[list(g.members)] = gi
        # 纯度：每个群里最常见的 scope 占比，按人数加权
        tot = 0
        for g in groups:
            s = scopes[list(g.members)]
            _, cnt = np.unique(s, return_counts=True)
            tot += cnt.max()
        purity = tot / net.n
        if best is None or abs(k - 12) < abs(best[1] - 12):
            best = (f, k, purity, groups, lab)

    f, k, purity, groups, lab = best
    ari = adjusted_rand({i: int(lab[i]) for i in range(net.n)},
                        {i: scopes[i] for i in range(net.n)})
    print(f"frac={f:g}  k={k}  纯度={purity:.3f}  ARI={ari:.3f}")
    print("\n最大的几个群：")
    for g in sorted(groups, key=lambda g: -len(g.members))[:8]:
        s = scopes[list(g.members)]
        vals, cnt = np.unique(s, return_counts=True)
        top = sorted(zip(cnt, vals), reverse=True)[:3]
        desc = "  ".join(f"{v}×{c}" for c, v in top)
        print(f"  {len(g.members):4d} 人  span {float(g.span):8.3g}  {desc}")

    # 按 scope 聚合看：最大的几个 scope 落在几个群里
    print("\n几个大 scope 的分布：")
    for sc in ("@deepseek-ai", "@aws-sdk", "@types", "@babel", "@esbuild"):
        idx = np.where(scopes == sc)[0]
        if idx.size == 0:
            continue
        labs, cnt = np.unique(lab[idx], return_counts=True)
        main_share = cnt.max() / idx.size
        print(f"  {sc:14s} {idx.size:4d} 个包，其中 {main_share:.1%} 在同一个群里（跨 {len(labs)} 个群）")


if __name__ == "__main__":
    main()
