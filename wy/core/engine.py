"""迭代核（编辑器）：一个 tick = 一次不重复的传播。

规则（DESIGN.md §5.3 的实现）：

1. 源 A = 轮转（tick % n）。不挑「残差最大的那个」——那是反扩散的，DESIGN.md §9.1 有账。
2. 从 A 出发走一条**极大简单路径**：每步走还没走过、且**欠得最多**的邻居
   （`比例 × |值差|`，不是「比例最大」——见 walk() 的注释）；
   走到「下一步会踩到走过的单位」就停——环在这里被切断。
3. 在 A 处注入一个小扰动（它是输入，不是解）。
4. 沿路径依次弛豫：v_w += alpha_w · r_uw · (v_u − v_w)。

成本 = 路径长度，与 n² 无关。永不试图解完。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import Network


@dataclass(frozen=True)
class TickRecord:
    """一次 tick 的可复现账本。"""

    tick: int
    source: int
    path: tuple[int, ...]
    edges: tuple[float, ...]
    kick: float
    defect: float  # tick 之前的关系偏离（违约读数）


class Engine:
    def __init__(self, net: Network) -> None:
        self.net = net
        self.v = net.values.copy()
        self.tick = 0
        # 哪些单位真正被算过。没被碰过的单位位数当然一位没变，但那是「没算」不是「稳」，
        # 读数不能把这两种情况混在一起（DESIGN.md §4 的 A 型模糊）。
        self.touched = np.zeros(net.n, dtype=bool)
        # 最近一次被算是在哪一 tick。累积的 touched 只说明「历史上算过」，
        # 而位数要问「最近还在算吗」——一个几百 tick 前算过一次、之后没人再碰的单位，
        # 它的位数会一路「稳」到量尺顶，那不是稳，是没人理它（DESIGN §13.1）。
        self.last_touch = np.full(net.n, -1, dtype=np.int64)
        # 形反作用：只对已经稳住的群动手，账记在这里，读数里要写出来。
        self.feedback_times = 0
        self.feedback_edges = 0
        self._group_born: dict[frozenset[int], int] = {}
        # 关系负荷：每条有向关系至今搬过多少值（按 (单位, 邻居列) 记账）。
        # 这是第一条「按关系」而不是「按单位」的读数——k 数团、span 量团内松紧，
        # 但「哪条关系在干活」以前完全看不见（DESIGN §16）。
        self.load = np.zeros(net.nbrs.shape, dtype=np.float64)
        # 每个单位被弛豫过几次。和 load 不同：load 问「搬了多少」，touches 问「轮到过没有」。
        # 走廊那个坑（§9.6）就是 touches 的 min 掉到 1，而 load 总量看着还挺大。
        self.touches = np.zeros(net.n, dtype=np.int64)
    # ---------------------------------------------------------------- 观测量

    def recent(self, window: int) -> np.ndarray:
        """最近 `window` 个 tick 内被算过的单位。

        稳定位数只在这里有定义：别处的位数没变，是因为没人动它，不是因为算准了。
        """
        return (self.last_touch >= 0) & ((self.tick - self.last_touch) <= int(window))

    def weighted_mean(self) -> np.ndarray:
        w = self.net.ratio
        vn = self.v[self.net.nbrs]
        den = w.sum(axis=1)
        num = (w * vn).sum(axis=1)
        out = num / np.where(den > 0.0, den, 1.0)
        # 孤立点（真图里真的有）：没有人跟它比，它的「邻居均值」就是它自己，
        # 残差 0。不这么做的话 0 会被当成真实均值，残差变成它的值本身。
        lonely = den <= 0.0
        if lonely.any():
            out[lonely] = self.v[lonely]
        return out

    def residual(self) -> np.ndarray:
        """每个单位与它邻居的加权均值差多少。源就从这里挑。"""
        return np.abs(self.v - self.weighted_mean())

    def defect(self, res: np.ndarray | None = None) -> float:
        """违约读数：当前有形状态偏离关系组多少（相对量）。这个数难看没关系，得在。"""
        if res is None:
            res = self.residual()
        scale = max(float(np.abs(self.v).mean()), 1e-30)
        return float(res.mean() / scale)

    # ---------------------------------------------------------------- 一步

    def walk(self, source: int) -> tuple[tuple[int, ...], list[float], list[int]]:
        """从 source 出发的一条极大简单路径。走到会重访就停。

        下一个走哪：**欠得最多的那条关系**，即 `比例 × |值差|` 最大的邻居。

        为什么不是「比例最大的邻居」：比例是静态的，按它走等于走一台确定的自动机——
        大部分源都汇进同一条高比例走廊，少数单位一辈子等不到自己那条边被弛豫
        （实测 n=2048 时访问次数 min 1 / 中位 52 / 最大 2017，defect 卡在 3e-2 不动）。
        按欠得多少走，路自己会去找还没抚平的地方，扫完一条换一条。

        内层用纯 Python 的元组和 bytearray，不用 numpy：一步只看 4 个邻居，
        numpy 的调用开销比算这 4 个数还贵（n=2048 一条路近两千步，差一个数量级）。
        """
        nbr_rows = self.net.nbrs_list  # 邻接表，静态
        ratio = self.net.ratio
        v = self.v
        use_ratio = self.net.cfg.walk_rule == "ratio"
        gamma = float(self.net.cfg.walk_power)
        seen = bytearray(self.net.n)
        seen[source] = 1
        path = [source]
        edges: list[float] = []
        cols: list[int] = []
        cur = source
        while True:
            row = nbr_rows[cur]
            rat_row = ratio[cur]
            best, best_score, best_r, best_j = -1, -1.0, 0.0, -1
            vc = v[cur]
            for j in range(len(row)):
                nxt = row[j]
                if seen[nxt]:
                    continue
                r = rat_row[j]
                if use_ratio:
                    score = r
                elif gamma == 1.0:
                    score = r * abs(vc - v[nxt])
                else:  # 比例的指数：调大更信强关系，调小更信差值
                    score = (r ** gamma) * abs(vc - v[nxt])
                if score > best_score:  # 平手保留先出现的（行内 id 升序 → 取最小 id）
                    best, best_score, best_r, best_j = nxt, score, r, j
            if best < 0:
                break
            seen[best] = 1
            path.append(best)
            edges.append(float(best_r))
            cols.append(best_j)
            cur = best
        return tuple(path), edges, cols

    def values(self) -> np.ndarray:
        """给几何/读数用的浮点视图。float64 引擎就是 v 本身（零成本）。"""
        return self.v

    def source_of(self, res: np.ndarray) -> int:
        """谁当这个 tick 的「变量」。

        默认轮转：不偏、不看谁的残差大。这不是省事——挑残差最大的单位当源，
        等于每 tick 都用系统里最离群的那一个当输入，邻居被拉向它，偏差越放越大。
        """
        if self.net.cfg.source_rule == "cycle":
            return self.tick % self.net.n
        return int(np.argmax(res))

    def feedback(self, groups, strength: float, floor: float = 1e-3) -> int:
        """形反作用于关系比例：已经稳住的那几团，群内收紧、群外放松。

        什么算「已经稳住」：**这个成员集合从哪一 tick 起就在一起**，连续 `feedback_age`
        个 tick 没变过才算（群自己记生日，不靠数值的年龄——数值每 tick 都在动低几位，
        拿数值年龄当门槛等于永远不开门）。还要求至少两个成员、成员都被算过。

        为什么只对稳住的群动手：早期乱划的群让它反过来加固关系，等于把噪声锁死成「真相」。
        **只有已经站住的形才有资格改关系。**

        为什么是这个方向：群内收紧、群外放松，等于成形的东西自我加固。上一步的账
        （DESIGN §9.3）是结构寿命太短、收敛太快；正反馈正好把它拉长——但正反馈也能
        把偶然锁死，所以要配一个关得掉的开关，并且在读数里写明关系已不是静态数据。

        就地改 net.ratio（它同时是 residual、walk、群强度矩阵的输入），返回改动的有向关系条数。
        """
        if strength <= 0.0:
            return 0
        net = self.net
        min_age = int(getattr(net.cfg, "feedback_age", 64))
        now = self.tick
        # 群自己记生日：同一个成员集合从哪一 tick 起就在一起。
        born = self._group_born
        fresh: dict[frozenset[int], int] = {}
        gid = np.full(net.n, -1, dtype=np.int64)
        for gi, g in enumerate(groups):
            key = frozenset(int(u) for u in g.members)
            fresh[key] = born.get(key, now)
            if len(key) < 2 or len(key) >= net.n or now - fresh[key] < min_age:
                continue
            mem = list(key)
            if not self.touched[mem].all():
                continue
            gid[mem] = gi
        self._group_born = fresh
        active = gid >= 0
        if not active.any():
            return 0

        ratio = net.ratio
        mj = gid[net.nbrs]
        inside = active[:, None] & (mj == gid[:, None])
        outside = active[:, None] & (mj != gid[:, None])
        # 外来图用自环补齐过，空位不是关系，不许被收紧（真图的孤立点尤其明显）。
        real = getattr(net, "mask", None)
        if real is not None:
            inside &= real
            outside &= real
        if inside.any():
            ratio[inside] += strength * (1.0 - ratio[inside])
        if outside.any():
            ratio[outside] *= 1.0 - strength
        np.clip(ratio, floor, 1.0, out=ratio)

        changed = int(inside.sum()) + int(outside.sum())
        self.feedback_times += 1
        self.feedback_edges += changed
        return changed

    def step(self) -> TickRecord:
        before = self.defect()
        res = self.residual()
        source = self.source_of(res)

        path, edges, cols = self.walk(source)

        kick = self.net.kick_delta(self.tick)
        self.v[source] *= 1.0 + kick

        alpha = self.net.alpha
        load = self.load
        for u, w, r, j in zip(path, path[1:], edges, cols):
            delta = alpha[w] * r * (self.v[u] - self.v[w])
            self.v[w] += delta
            # 关系负荷：这一 tick 这条有向关系实际搬了多少值。
            # 值不动的时候负荷是 0，所以负荷只在「还在动」的时候有定义（同位数那条）。
            load[u, j] += abs(delta)

        self.touched[list(path)] = True
        self.last_touch[list(path)] = self.tick
        for w in path[1:]:
            self.touches[w] += 1

        rec = TickRecord(self.tick, source, path, tuple(edges), kick, before)
        self.tick += 1
        return rec
