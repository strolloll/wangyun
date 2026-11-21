"""色彩空间：让颜色也由关系定，而不是随便挑。

形空间里，群是关系网络收出来的团；团与团之间的关系有强度 W。
色彩空间里，同理：色相不是每个群各自抽一个，而是**排在一条环上，
由 W 决定谁挨着谁**——关系强的两个群，色相挨得近。

于是"形"和"色"是同一个东西的两次投影：
    关系网络 --(收敛)--> 形空间（团、大小、松紧）
    关系网络 --(排色)--> 色彩空间（色相相邻、明度/纯度由读数定）
两个空间用同一张 W 串起来。
"""

from __future__ import annotations

import math

import numpy as np

from .layout import group_strength


def _ring_order(W: np.ndarray) -> list[int]:
    """按最强关系排成一个环：从全图最强的那一对开始，两头长。"""
    k = len(W)
    if k <= 2:
        return list(range(k))
    i, j = np.unravel_index(int(np.argmax(W)), W.shape)
    order = [int(i), int(j)]
    left = set(range(k)) - set(order)
    while left:
        head, tail = order[0], order[-1]
        best_n, best_side, best_w = None, 0, -1.0
        for n in left:
            w_head, w_tail = float(W[n, head]), float(W[n, tail])
            if w_head > best_w:
                best_n, best_side, best_w = n, 0, w_head
            if w_tail > best_w:
                best_n, best_side, best_w = n, 1, w_tail
        if best_side == 0:
            order.insert(0, best_n)
        else:
            order.append(best_n)
        left.discard(best_n)
    return order


def color_space(groups, net, strength=None, base: float = 0.52) -> dict[int, tuple[float, float, float]]:
    """给每个群一个 (hue, sat, val)。色相由关系定，明度/纯度由松紧与稳定度定。"""
    W = group_strength(groups, net) if strength is None else np.asarray(strength)
    order = _ring_order(W)
    k = max(1, len(order))
    out: dict[int, tuple[float, float, float]] = {}
    spans = np.array([max(1e-9, g.span) for g in groups], dtype=float)
    med = float(np.median(spans))
    for pos, gi in enumerate(order):
        g = groups[gi]
        # 色相：均匀分布在环上，但被关系强度轻轻拉一拉——
        # 连得紧的一对，色相差半个格；连得松的，拉开。
        # 色相：按环上位置均匀分布。相邻 = 关系最强的邻居。
        # （曾经试过按强度再多拉一点，但那会让最强的一对反而被推远——
        #  均匀分布 + 正确的排环顺序，才是"强关系色相相邻"的严格保证。）
        hue = (base + pos / k) % 1.0
        # 纯度：团内部越松，颜色越灰（它本来就不像一个东西）
        sat = float(np.clip(0.18 + 0.52 * (1.0 - (g.span / (med + 1e-12)) / 2.4), 0.10, 0.72))
        # 明度：稳住的群亮一点（测试里的假群可能没有 stable 字段）
        st = float(getattr(g, "stable", 0.0))
        val = float(np.clip(0.62 + 0.30 * math.tanh(0.6 * (st - 2.0)), 0.45, 0.95))
        out[int(getattr(g, "gid", pos))] = (hue, sat, val)
    return out


def _arc(a: float, b: float) -> float:
    """色相环上两个色相的短弧距离（0 = 同色，0.5 = 正对面）。"""
    d = abs(float(a) - float(b)) % 1.0
    return min(d, 1.0 - d)


def hue_track(prev: dict[frozenset, float], groups, strength,
              base: float = 0.52) -> tuple[dict[frozenset, float], float]:
    """跨帧颜色：老群沿用上一帧的色相，新群继承「成员重合最多的旧群」的色相。

    返回 (成员集合 -> 色相, 本帧色相移动总量)。移动总量按**人**算：每个人这一帧
    拿到的色相和上一帧差多远（环上短弧），加起来除以总人数。于是它读出三件事：

        老群还在    → 0（一个人都没动，颜色接得上）
        分开        → 小（老大原色不动，后来的孩子各挪一格）
        并群        → 大（输掉的那半整体跳到赢家的色相上）

    并群是唯一「有人不得不换色」的时刻（分开时只有后面的兄弟挪一格）。所以移动总量
    的峰值大多落在并群那一 tick；但 k 小时，分开走掉一大团也要每人挪一格，可能更疼
    （§30.5 有实测，别再写死「峰值一定在并群」）。
    """
    keys = [frozenset(int(m) for m in g.members) for g in groups]
    fresh = color_space(groups, None, strength=strength, base=base)
    k = max(1, len(keys))
    total = float(sum(len(key) for key in keys)) or 1.0
    hues: dict[frozenset, float] = {}
    claims: dict[frozenset, int] = {}
    moved = 0.0
    for i, key in enumerate(keys):
        want = float(fresh[int(getattr(groups[i], "gid", i))][0])
        # 1) 成员完全对得上：沿用旧色相，一个人都不动
        if key in prev:
            hues[key] = float(prev[key])
            continue
        # 2) 新群：找重合最多的旧群，继承它的色相
        best_key, best_j = None, -1.0
        for old in prev:
            inter = len(key & old)
            if not inter:
                continue
            j = inter / len(key | old)
            if j > best_j:
                best_key, best_j = old, j
        if best_key is None:
            hues[key] = want  # 没有旧血统：照关系排出来的颜色从零开始
            continue
        # 同一个旧群里分出来的几个孩子按出生顺序各挪一格：老大原色不动，
        # 后面的偏出去。不然分开的两半会同色，画回场面像还没分开。
        step = claims.get(best_key, 0)
        claims[best_key] = step + 1
        hues[key] = (float(prev[best_key]) + step / k) % 1.0
        # 这个新群的人上一帧散在两个（或更多）旧群里：按重合人数加权，算他们真挪了多远
        jumped = 0.0
        for old, old_h in prev.items():
            inter = len(key & old)
            if inter:
                jumped += inter * _arc(old_h, hues[key])
        moved += jumped / total
    return hues, moved


def hue_vec(hues: dict[frozenset, float], groups) -> np.ndarray:
    """把每个群的色相按群表排成向量，交给 `draw_scene(hue=)`。

    只认成员集合：群号、顺序都不参与——两处画布（动画、诊断带缩略图）都靠这一句
    保证「同一批人同一个色」。
    """
    return np.array([hues[frozenset(int(m) for m in g.members)] for g in groups],
                    dtype=float)
