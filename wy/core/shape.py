"""形（输出端）：按类型成群的参数表。

形不是渲染对象，是输出端：它是一组数值，交给建模软件当参数用。
形住在关系网络里：网络变量一变，形的数值跟着变。

群的定义（DESIGN.md §5.4）：

    分辨率 r 下，两个单位连成一群 ⟺ 它们的关系比例 ≥ strong，且数值差 ≤ r
    群 = 连通分量

这个定义有个好处：r 变大只会加边，所以

    r2 > r1  ⇒  划分(r2) 是 划分(r1) 的粗化（每个粗群恰好是若干细群之并）

这就是 LOD 不矛盾律的可证形式，可以写成测试。粗视图不许推翻细视图，只许把细视图并起来。

涌现的读数：
    k(t)  —— 固定 r，类型数随时间下沉（n → 少数几群）
    k(r)  —— 固定 t，类型数随分辨率呈阶梯，出现平台就是真有结构（不是噪音）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .digits import StabilityMonitor
from .network import Network


@dataclass(frozen=True)
class Group:
    """群表里的一行。视觉系统只认这些东西，不认网络。"""

    gid: int
    members: tuple[int, ...]
    value: float  # 群内均值
    span: float  # 群内极差，离散度
    stable: float  # 群内稳定前缀长度的均值
    age: int  # 这一群有多久没变过（群身份稳定度）
    settled: bool  # 群内数值是否已稳住到 need 位（且这个群有资格说「稳住」）

    @property
    def count(self) -> int:
        return len(self.members)


# ---------------------------------------------------------------------- 划分


class _UF:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)  # 小 id 当根，划分是规范的


def partition(v: np.ndarray, net: Network, r_abs: float, strong: float | None = None) -> np.ndarray:
    """分辨率 r_abs 下的类型划分。返回每个单位的群标签。"""
    strong = net.cfg.strong if strong is None else strong
    n = v.shape[0]
    uf = _UF(n)

    i, j, rr = net.i_edge, net.j_edge, net.r_edge
    close = np.abs(v[i] - v[j]) <= r_abs
    keep = (rr >= strong) & close
    for a, b in zip(i[keep].tolist(), j[keep].tolist()):
        uf.union(a, b)

    roots = np.fromiter((uf.find(x) for x in range(n)), dtype=np.int64, count=n)
    return roots


def scale_of(v: np.ndarray) -> float:
    """相对分辨率的参照尺度。视角是用「当前的量级」来量的，不是绝对数。"""
    return max(float(np.abs(v).mean()), 1e-30)


# ------------------------------------------------------------------ 相对半径
def local_scale(v: np.ndarray, net: Network) -> np.ndarray:
    """每个单位自己的「远近尺度」：它跟邻居们差多少的中位数。

    为什么需要这个：全局半径要求整张图的块「差不多紧」，可真图的块
    组内离散能差二十倍（§22）。用自己邻居差的中位数当尺子，紧的组用紧的尺子、
    松的组用松的尺子，半径就变成一个**相对量**——判据跟着群走，不跟着全图走。

    孤立点（没有邻居）没有尺子，返回 0：它跟谁都不连。
    """
    n = v.shape[0]
    out = np.zeros(n, dtype=np.float64)
    mask = getattr(net, "mask", None)
    if mask is None:
        mask = np.ones_like(net.nbrs, dtype=bool)
    for u in range(n):
        row = net.nbrs[u][mask[u]]
        if row.size == 0:
            continue
        out[u] = float(np.median(np.abs(v[row] - v[u])))
    return out


def partition_rel(v: np.ndarray, net: Network, tau: float,
                  strong: float | None = None, mode: str = "min") -> np.ndarray:
    """相对半径下的划分：|Δ| ≤ τ × 两端的局部尺度。

    两端怎么合取是个真问题：
      - `min`：两边都觉得自己这一差「算小」才连——边界上那条跨组的关系会被拒，
        因为它对紧的那一端来说是大差。**默认用这个**。
      - `max`：宽松的那侧说了算。实测在真图上太松：枢纽节点邻居差的中位数很大，
        一条边只要挨着枢纽就被判「算小」，整张图 τ=1 就并成一团（ARI 0.05）。
    """
    strong = net.cfg.strong if strong is None else strong
    n = v.shape[0]
    uf = _UF(n)
    s = local_scale(v, net)
    i, j, rr = net.i_edge, net.j_edge, net.r_edge
    scale_e = np.minimum(s[i], s[j]) if mode == "min" else np.maximum(s[i], s[j])
    close = np.abs(v[i] - v[j]) <= tau * scale_e
    # 尺度为 0（两端都是孤立点，或差恒为 0）：只有差真的为 0 才算一群
    close &= ~((scale_e <= 0.0) & (np.abs(v[i] - v[j]) > 0.0))
    keep = (rr >= strong) & close
    for a, b in zip(i[keep].tolist(), j[keep].tolist()):
        uf.union(a, b)
    roots = np.fromiter((uf.find(x) for x in range(n)), dtype=np.int64, count=n)
    return roots


def k_rel(v: np.ndarray, net: Network, tau: float, mode: str = "min") -> int:
    return int(np.unique(partition_rel(v, net, tau, mode=mode)).size)


def k_of(v: np.ndarray, net: Network, r_abs: float) -> int:
    return int(np.unique(partition(v, net, r_abs)).size)


def k_curve(v: np.ndarray, net: Network, fracs: np.ndarray) -> np.ndarray:
    """相对分辨率 frac 下的 k。rs 升序 ⇒ 返回的 k 必须非增。"""
    s = scale_of(v)
    return np.array([k_of(v, net, float(f) * s) for f in fracs], dtype=np.int64)


def plateau(
    fracs: np.ndarray, ks: np.ndarray, min_k: int = 2, tol: float = 0.10
) -> dict | None:
    """k(r) 上最宽的一段平台。

    平台意味着：存在一个尺度，在这个尺度上结构是稳的——那才是真结构，不是噪音。
    没有平台就没有涌现，这条让「涌现」变成可否证的。

    「平」是**变化很小**，不是**完全相等**。要求相邻两档 k 一字不差，等于拿一个
    整数去卡浮点结果：13 → 12 明明是一个台阶，却不算平台。所以这里用相对变化
    `|Δk| / k ≤ tol` 判平——缩一个数量级，类型数几乎不动，才叫看得见的结构。
    """
    best = None
    start = 0
    for idx in range(1, len(ks) + 1):
        flat = idx < len(ks) and abs(float(ks[idx]) - float(ks[start])) <= tol * max(
            float(ks[start]), 1.0
        )
        if idx == len(ks) or not flat:
            k = int(round(float(ks[start:idx].mean())) if idx > start else ks[start])
            if k >= min_k and idx - start >= 2:
                width = float(np.log10(fracs[idx - 1] / fracs[start]))
                if best is None or width > best["width"]:
                    best = {
                        "k": k,
                        "frac_lo": float(fracs[start]),
                        "frac_hi": float(fracs[idx - 1]),
                        "width": width,
                    }
            start = idx
    return best


# ---------------------------------------------------------------------- 群表


class PartitionTracker:
    """群身份也有稳定度：这个划分多久没变过。

    数的位会颤，群的划分也会颤。不给群一个 W，类型数每 tick 都变，画面会一直跳。
    """

    def __init__(self, window: int = 8) -> None:
        self.window = window
        self._last: np.ndarray | None = None
        self._frames: list[np.ndarray] = []
        self.age = 0
        self.events: list[tuple[int, int]] = []  # (tick, 事件码)：1 生 2 分 3 合

    def push(self, labels: np.ndarray, tick: int) -> int:
        canon = _canonical(labels)
        if self._last is not None and not np.array_equal(canon, self._last):
            before, after = _count(self._last), _count(canon)
            self.events.append((tick, 3 if after < before else 2))
            self.age = 0
        else:
            self.age += 1
        self._last = canon
        self._frames.append(canon)
        if len(self._frames) > self.window:
            self._frames.pop(0)
        return self.age


def _canonical(labels: np.ndarray) -> np.ndarray:
    """同一划分的不同编号要看成同一个。按最小成员 id 重新编号。"""
    order = {}
    out = np.zeros_like(labels)
    for i, lab in enumerate(labels.tolist()):
        if lab not in order:
            order[lab] = len(order)
        out[i] = order[lab]
    return out


def _count(labels: np.ndarray) -> int:
    return int(np.unique(labels).size)


def describe_change(prev_members: list, cur_members: list) -> dict:
    """把一次划分的变化说成人话：谁跟谁并了、谁裂成了几块。

    `prev_members` / `cur_members` 是逐群的成员集合。判据是「过半」：
    一个旧群有超过一半的成员落进同一个新群，就算它整体过去了；落进两个以上，就是分。

    这个东西不是装饰。带子上如果只写 k 从 7 变 6，看的人知道变了却不知道为什么；
    写「A+B → C」，位置又接着上一帧，才看得出那两团是被谁拉在一起的。
    """
    prev_sets = [set(m) for m in prev_members]
    cur_sets = [set(m) for m in cur_members]

    merges = []  # (新群下标, [旧群下标...])
    for ci, cs in enumerate(cur_sets):
        src = []
        for pi, ps in enumerate(prev_sets):
            ov = len(cs & ps)
            if ov and ov * 2 >= len(ps):  # 旧群过半都进了这个新群
                src.append(pi)
        if len(src) >= 2:
            merges.append((ci, src))

    splits = []  # (旧群下标, [新群下标...])
    for pi, ps in enumerate(prev_sets):
        dst = []
        for ci, cs in enumerate(cur_sets):
            ov = len(cs & ps)
            if ov and ov * 2 >= len(cs):
                dst.append(ci)
        if len(dst) >= 2:
            splits.append((pi, dst))

    if merges:
        ci, src = max(merges, key=lambda m: len(m[1]))
        kind = "merge"
        text = f"并：{len(src)} 群 → 1 群"
        short = f"并 {len(src)}→1"
        touched = [ci]
    elif splits:
        pi, dst = max(splits, key=lambda s: len(s[1]))
        kind = "split"
        text = f"分：1 群 → {len(dst)} 群"
        short = f"分 1→{len(dst)}"
        touched = list(dst)
    else:
        kind, text, short, touched = "shift", "挪动（划分变了，群数没变）", "挪动", []

    return {
        "kind": kind,
        "text": text,
        "short": short,
        "k_before": len(prev_sets),
        "k_after": len(cur_sets),
        # 圈谁：给**成员**而不是下标。群表是按 gid 重排过的，下标对不上，
        # 拿下标去圈会圈到别的群——这种错在图上看着像真的，最难发现。
        "cur_members": [tuple(sorted(cur_members[i])) for i in touched],
        "merges": merges,
        "splits": splits,
    }


def group_table(
    v: np.ndarray,
    net: Network,
    r_abs: float,
    monitor: StabilityMonitor | None = None,
    need: int = 8,
    touched: np.ndarray | None = None,
    labels: np.ndarray | None = None,
) -> list[Group]:
    """按当前的 r 把单位收成群表。视觉系统只吃这张表。

    `settled` 是三件事同时成立，缺一不算：

    1. **群里有至少两个成员**。一个孤零零的单位，极差恒为 0——那不是「成员彼此吻合」，
       是「没有人跟它比」。拿它当收敛证据，等于说一个人自己待着就算意见一致。
    2. **所有成员都被算过**（`touched`）。没算过的位数一位没变，那是没算，不是稳。
    3. **位数到 need**。前两条是资格，这一条才是读数。

    这三条一加，细视角（大多是单元素群）就不会再整片标成「稳住了」——
    它本来就不是「稳」，是「这里还什么都没成立」。
    """
    if labels is None:
        labels = partition(v, net, r_abs)
    groups: list[Group] = []
    for gid, root in enumerate(sorted(set(labels.tolist()))):
        members = tuple(int(i) for i in np.flatnonzero(labels == root).tolist())
        vals = v[list(members)]
        if monitor is not None:
            stable = float(monitor.length[list(members)].mean())
            qualifies = len(members) >= 2
            if touched is not None:
                qualifies = qualifies and bool(np.all(touched[list(members)]))
            settled = bool(qualifies and np.all(monitor.settled_mask(need)[list(members)]))
            age = int(np.min(monitor.age[list(members)]))
        else:
            stable, settled, age = 0.0, False, 0
        groups.append(
            Group(
                gid=gid,
                members=members,
                value=float(vals.mean()),
                span=float(vals.max() - vals.min()),
                stable=stable,
                age=age,
                settled=settled,
            )
        )
    return groups


def table_signature(groups: list[Group]) -> dict:
    """群表的摘要，方便落盘与对比。"""
    return {
        "k": len(groups),
        "sizes": [g.count for g in groups],
        "values": [round(g.value, 6) for g in groups],
        "stable": [round(g.stable, 3) for g in groups],
        "settled": [g.settled for g in groups],
    }
