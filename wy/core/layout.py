"""布局：把群表变成一个形状。

形是输出端。可是要**看见**形，得先决定谁靠着谁。在环上排开只是占位——
真正决定位置的是**群之间的关系强度**：关系强的群挨得近，关系弱的被推开。

这里有两条规矩：

1. 布局只读群表和一张 k×k 的群间强度矩阵。它看不见 n² 条关系，也看不见 n。
   所以视觉系统的成本是 k²，而 k ≪ n——这就是「视觉系统省支出」的实际机制。
2. 粗视图**不重新布局**。粗群的坐标 = 它成员（细群）的加权重心。
   粗化于是就是真并起来，而不是换一张图。LOD 不矛盾律在画面上也成立，
   不只是划分上成立。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import Network
from .shape import Group


@dataclass(frozen=True)
class Shape:
    """视图层的东西：群表 + 群间强度 + 坐标。视觉系统只吃这个。"""

    groups: list[Group]
    strength: np.ndarray  # (k, k) 对称，群 i 与 j 之间平均一条关系的比例
    pos: np.ndarray  # (k, 2)，落在单位圆内
    degraded: bool = False  # k 太大、摆不动了：位置退成占位，含义是「视角收太细了」

    @property
    def k(self) -> int:
        return len(self.groups)


# 布局的成本是 k²。超过这个数就别摆了——不是省时间，是这个视角本来就没有
# 可读的结构：屏幕上放不下两千个团，摆出来的也只是一堆点。该做的是收视角。
SPRING_MAX_K = 400


# --------------------------------------------------------------------- 群间强度


def group_strength(groups: list[Group], net: Network) -> np.ndarray:
    """把单位级的关系汇总成群级强度。

    W[i, j] = 群 i 的成员指向群 j 的那些关系的平均比例。
    对称化后作为布局的吸引力：比例越大，两群越近。
    """
    n = net.nbrs.shape[0]
    k = len(groups)
    if k == 1:
        return np.zeros((1, 1), dtype=np.float64)

    idx = np.full(n, -1, dtype=np.int64)
    for gi, g in enumerate(groups):
        idx[list(g.members)] = gi

    W = np.zeros((k, k), dtype=np.float64)
    a = idx[:, None]
    b = idx[net.nbrs]
    cross = (a >= 0) & (b >= 0) & (a != b)
    np.add.at(W, (np.broadcast_to(a, b.shape)[cross], b[cross]), net.ratio[cross])

    counts = np.bincount(idx[idx >= 0], minlength=k).astype(np.float64)
    W = W / np.maximum(counts, 1.0)[:, None]
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


# --------------------------------------------------------------------- 弹簧


def _energy(Wm: np.ndarray, pos: np.ndarray, kk: float) -> float:
    """布局的能量：关系强的要近（弹簧能），点与点要开（斥力能）。用来比几次重启。"""
    delta = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((delta * delta).sum(-1))
    off = ~np.eye(pos.shape[0], dtype=bool)
    d = np.maximum(dist[off], 1e-9)
    att = 0.5 * float((Wm[off] * d * d).sum())
    rep = -0.5 * float((kk * kk * np.log(d)).sum())
    return att + rep


def _crossings(W: np.ndarray, pos: np.ndarray) -> int:
    """有多少对关系线交叉。

    只比能量是不够的：8 个点等距排成一圈时，「环」和「八角星」能量一样低，
    但星的连线是交错的，读出来就是错的拓扑。所以交叉数优先于能量。
    """
    k = W.shape[0]
    iu = np.triu_indices(k, 1)
    keep = W[iu] > 0.0
    a, b = iu[0][keep], iu[1][keep]
    m = a.size
    if m < 2 or m > 400:
        return 0

    ii, jj = np.triu_indices(m, 1)
    a1, b1, a2, b2 = a[ii], b[ii], a[jj], b[jj]
    share = (a1 == a2) | (a1 == b2) | (b1 == a2) | (b1 == b2)
    p, q, r, s = pos[a1], pos[b1], pos[a2], pos[b2]

    def orient(o, u, v):
        return (u[:, 0] - o[:, 0]) * (v[:, 1] - o[:, 1]) - (u[:, 1] - o[:, 1]) * (v[:, 0] - o[:, 0])

    d1 = orient(p, q, r)
    d2 = orient(p, q, s)
    d3 = orient(r, s, p)
    d4 = orient(r, s, q)
    hit = (d1 * d2 < 0) & (d3 * d4 < 0) & ~share
    return int(hit.sum())


def spring(
    W: np.ndarray,
    seed: int = 0,
    iters: int = 420,
    restarts: int = 4,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """Fruchterman–Reingold，多跑几次取最好的那个。

    确定性：初始位置全由 seed 派生，重启次数固定，比较规则固定。

    斥力 k²/d² 把群推开，引力 W·d/k 把关系强的群拉近。平衡距离 d ~ k/W^(1/3)：
    关系强 → 近，关系弱 → 远。

    为什么要重启：力和位置纠缠在一起，同样的 8 环，有的种子会折成麻花，
    那是另一个局部极小。多撒几次网，取最好的那个，环才稳定是环。

    init 给了就从那儿开始（细视图从粗视图的坐标开始——粗视图定形，细视图调细节）。
    """
    k = W.shape[0]
    if k == 1:
        return np.zeros((1, 2), dtype=np.float64)
    if k > SPRING_MAX_K:
        # 降级：按 id 均匀撒在半径 0.5 的圆上，画出来就是个占位圈。
        ang = np.arange(k) / k * 2.0 * np.pi
        return 0.5 * np.stack([np.cos(ang), np.sin(ang)], axis=1)
    if k == 2:
        w = float(W[0, 1])
        d = np.zeros((2, 2))
        d[1, 0] = 1.0 / (1.0 + 4.0 * w)
        d[1, 0] -= d[1, 0] / 2
        d[0, 0] -= d[1, 0]
        return d

    Wm = np.array(W, dtype=np.float64, copy=True)
    np.fill_diagonal(Wm, 0.0)
    peak = float(Wm.max())
    if peak <= 0.0:
        Wm = np.ones((k, k)) - np.eye(k)
        peak = 1.0
    # 只跟相对强弱有关：把最大强度归一化到 1。
    # 不归一化的话，关系整体乘以 100 和除以 100 会给出不同形状——
    # 比例是绝对的，「强度」本身没有绝对量纲。
    Wm = Wm / peak

    kk = float(np.sqrt(1.0 / k))
    if init is not None:
        starts: list[np.ndarray | None] = [init]
        iters_eff = iters
    else:
        # 撒网的次数和跑多少步，按群的个数缩：
        # 一个有一千个团的视角，谁也没打算看清每一对的位置关系——
        # 那是「这个尺度上本来没有可读结构」，跟 k>SPRING_MAX_K 是同一件事，
        # 只是程度轻一点。小图上一次不能少（8 环要靠重启才稳定是环）。
        if k <= 32:
            starts = [None] * restarts
        elif k <= 128:
            starts = [None] * max(1, restarts // 2)
        else:
            starts = [None]
        # 每步是 O(k²)，总预算按 k 反过来定：k 越大，步数越少。
        iters_eff = int(min(iters, max(60, 60000 // k)))
    best_pos = None
    best_key = (np.inf, np.inf)
    for r, start in enumerate(starts):
        pos = _relax(Wm, kk, k, seed * 7919 + r, iters_eff, start)
        key = (_crossings(W, pos), _energy(Wm, pos, kk))
        if key < best_key:
            best_key, best_pos = key, pos

    assert best_pos is not None
    pos = best_pos - best_pos.mean(axis=0)
    span = float(np.abs(pos).max())
    if span > 0:
        pos = pos / span * 0.92
    return pos


def _relax(
    Wm: np.ndarray, kk: float, k: int, seed: int, iters: int, start: np.ndarray | None = None
) -> np.ndarray:
    # 初始位置撒在一个圆上（角序随机）。这不是装饰：从方块里随机撒点的话，
    # 环上会有一个点正好落在正中间——它到两边拉力相等，是个鞍点，会被困在那儿。
    if start is not None:
        pos = np.array(start, dtype=np.float32, copy=True)
    else:
        rng = np.random.default_rng(seed)
        ang = (rng.permutation(k) / k) * 2.0 * np.pi + rng.normal(0.0, 0.05, size=k)
        rad = 0.5 + rng.normal(0.0, 0.02, size=k)
        pos = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32) * rad[:, None].astype(
            np.float32
        )

    # 内部一律 float32：这一步完全是几何，用不着双精度的有效位；
    # 而它每一步都在搬 k² 个浮点数，精度翻倍就是带宽翻倍（实测能把大 k 的布局砍掉一半时间）。
    Wf = np.asarray(Wm, dtype=np.float32)
    temp = np.float32(0.25)
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]  # (k, k, 2)
        dist = np.sqrt((delta * delta).sum(-1)) + np.float32(1e-9)
        np.fill_diagonal(dist, np.inf)

        rep = (kk * kk / (dist * dist))[..., None] * delta / dist[..., None]
        att = Wf[..., None] * delta / np.float32(kk)
        disp = (rep - att).sum(axis=1)

        mag = np.sqrt((disp * disp).sum(-1))
        step = np.minimum(mag, temp)
        pos += disp * (step / np.maximum(mag, np.float32(1e-12)))[:, None]

        # 定容弛豫：每步把整体缩放回固定大小。
        # 不然力的大小随 W 变化，弱关系时整体塌成一点，形状就丢了。
        # 只保住「谁离谁近」，不保住绝对距离——量级本来就该由画面来定。
        pos -= pos.mean(axis=0)
        rms = float(np.sqrt((pos.astype(np.float64) ** 2).sum(axis=1).mean()))
        if rms > 1e-12:
            pos *= np.float32(0.5 / rms)

        temp = np.float32(temp * 0.996)
    return pos.astype(np.float64)


# --------------------------------------------------------------------- 视图


def make_shape(
    groups: list[Group],
    net: Network,
    seed: int = 0,
    pos: np.ndarray | None = None,
    init: np.ndarray | None = None,
    iters: int = 420,
) -> Shape:
    """一张视图：群表 + 群间强度 + 坐标。

    pos 给了就直接用；否则跑弹簧（init 给了就从上一次视图的坐标接着调）。
    """
    W = group_strength(groups, net)
    if pos is None:
        pos = spring(W, seed=seed, iters=iters, init=init)
    return Shape(
        groups=groups,
        strength=W,
        pos=np.asarray(pos, dtype=np.float64),
        degraded=len(groups) > SPRING_MAX_K,
    )


def broadcast(parent: Shape, children: list[Group], net: Network, seed: int = 0) -> np.ndarray:
    """粗视图的坐标 → 细视图的坐标：细节长在粗形状**里面**。

    一个细群落在哪个粗群里是确定的（细划分是粗划分的细化），所以它的位置只能
    由它所属的粗群决定。一个粗群拆成 c 个细群时，它们围在粗群中心的一个小圈上。

    这就是 LOD 在画面上的样子：粗视图定形，细视图只填内部细节。
    缩放的时候画面不许跳——跳的话，读的人会以为结构变了。
    """
    unit_parent = np.full(net.nbrs.shape[0], -1, dtype=np.int64)
    for gi, g in enumerate(parent.groups):
        unit_parent[list(g.members)] = gi

    k = len(children)
    pos = np.zeros((k, 2), dtype=np.float64)
    siblings: dict[int, list[int]] = {}
    for ci, g in enumerate(children):
        owner = int(unit_parent[int(g.members[0])])
        siblings.setdefault(owner, []).append(ci)

    rng = np.random.default_rng(seed)
    for owner, kids in siblings.items():
        base = parent.pos[owner]
        if len(kids) == 1:
            pos[kids[0]] = base
            continue
        c = len(kids)
        ring = 0.09 * c
        phase = float(rng.random()) * 2.0 * np.pi
        for n, ci in enumerate(kids):
            a = phase + 2.0 * np.pi * n / c
            pos[ci] = base + ring * np.array([np.cos(a), np.sin(a)])
    return pos


def carry_pos(prev_groups: list[Group], prev_pos: np.ndarray,
              groups: list[Group]) -> np.ndarray:
    """把上一帧的坐标搬到这一帧：按成员重合认人，不按第几号群。

    群数变了也能搬——搬的是「谁跟谁在一起」，这才是画面跳不跳的关键。
    这里是纯演算版（渲染层的 `scene.carry` 只是转手调它），缩略图和网页都用它。
    """
    k = len(groups)
    out = np.zeros((k, 2), dtype=np.float64)
    if len(prev_groups) == 0 or np.asarray(prev_pos).size == 0:
        return out
    prev_pos = np.asarray(prev_pos, dtype=np.float64)
    prev_sets = [set(map(int, g.members)) for g in prev_groups]
    for i, g in enumerate(groups):
        cur = set(map(int, g.members))
        best, best_j = -1, 0.0
        for j, ps in enumerate(prev_sets):
            inter = len(cur & ps)
            if inter == 0:
                continue
            jac = inter / max(1, len(cur | ps))
            if best >= 0:
                jac += 0.001 * float(np.linalg.norm(prev_pos[j]))  # 平手时用位置挑一个稳的
            if jac > best_j:
                best, best_j = j, jac
        out[i] = prev_pos[best] if best >= 0 else prev_pos.mean(axis=0)
    return out
