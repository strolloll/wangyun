"""真图上最要紧的一件事：**该在哪个 tick 看**。

结构有寿命（§19）：早看还没分开，晚看已经并成一团。所以「读一张真图」不是挑参数，
是挑时刻。`scan` 把若干 tick 上的读数摆成一张表，让你自己看那一格什么时候最值得信。
"""

from __future__ import annotations

import numpy as np

from .engine import Engine
from .gauge import report
from .shape import group_table, scale_of


def scan(net, ticks: list[int], frac: float = 1e-2, need: int = 4,
         target_k: int | None = None) -> list[dict]:
    """跑一遍，按给定的 tick 打点。返回每行：tick / k / defect / contrast / q / 判语。

    `target_k` 给了就在几档半径里挑 k 最接近它的那一档（画面/读数都好读）。
    """
    ticks = sorted({int(t) for t in ticks if t > 0})
    if not ticks:
        return []
    rows: list[dict] = []
    eng = Engine(net)
    mon = None
    from .digits import StabilityMonitor

    mon = StabilityMonitor(net.n, net.cfg.window, net.cfg.digits)
    for t in range(1, ticks[-1] + 1):
        eng.step()
        mon.push(eng.v)
        if t not in ticks:
            continue
        fv = eng.values()
        scale = scale_of(fv)
        cand = [frac]
        if target_k is not None:
            cand = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
        groups, used = None, frac
        for f in cand:
            gs = group_table(fv, net, f * scale, mon, need, touched=eng.recent(net.cfg.window))
            if len(gs) < 2:
                continue
            if groups is None or (target_k is not None
                                  and abs(len(gs) - target_k) < abs(len(groups) - target_k)):
                groups, used = gs, f
        if not groups:
            rows.append({"tick": t, "k": 0, "defect": float("nan"),
                         "contrast": float("nan"), "q": float("nan"),
                         "verdict": "没有群", "frac": used})
            continue
        lab = np.zeros(net.n, dtype=np.int64)
        for gi, g in enumerate(groups):
            lab[list(g.members)] = gi
        g = report(lab, fv, net)
        rows.append({"tick": t, "k": len(groups), "defect": float(eng.defect()),
                     "contrast": float(g["contrast"]), "q": float(g["q_excess"]),
                     "verdict": str(g["verdict"]), "frac": used})
    return rows


def best(rows: list[dict]) -> dict | None:
    """判语优先，其次 q_excess 高、k 不太碎。"""
    rank = {"能读": 3, "勉强": 2, "别信": 1}
    ok = [r for r in rows if r["k"] >= 2]
    if not ok:
        return None
    return max(ok, key=lambda r: (rank.get(r["verdict"], 0), r["q"] - 0.002 * r["k"]))
