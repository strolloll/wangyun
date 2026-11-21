"""精确整数演算：把「位数」这条读数从 float64 的天花板里解放出来。

为什么要有这个：`DESIGN.md` §13.1 记着两个假天花板——量尺（`digits`）和「冻住」。
放长量尺之后实测 float64 大约在 15~16 位有效数字上到顶（kick=1e-15 的演算读出 15.70，
再往下增量掉到可表示精度以下，值冻住，位数开始朝量尺顶爬）。
也就是说：**用 float64 就永远测不出「16 位之后」的稳定性**，因为那里已经没有数字了。

这里换一套表示：值的单位是 10**(-D) 的整数（Python 大整数，精确），
关系的比例和 alpha 各存成 10**(-C) 的整数。每一步的更新是

    v_w += (a_num * (v_u - v_w)) // C

全是整数除法的确定性运算——没有舍入误差累积，只有「这一步截断了多少」。
截断的量级是 1 个最小单位（10**-D），比 D=30 时我们关心的任何一位都小得多。

注意这不是「无限精度」：
  * 系数（alpha、比例）是量化到 10**-C 的**模型定义**，不是误差；
  * 每步的整数除法向下取整，累计的偏移最多 ~tick 个最小单位，
    那是「低位的抖动」，和 float64 的抖动同性质，只是起点低了 15 个数量级。

跟 float64 引擎的关系：两条路独立实现（`wy/engine.py` 用浮点、这里用整数），
测试里用短跑逐位对齐到 1e-12 交叉验证——两套代码同时错成一样是不容易的。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .network import Config, Network


@dataclass
class ExactTrace:
    """一步的记录，够用来对账就行。"""

    tick: int
    source: int
    path: list[int]
    changed: bool


class ExactEngine:
    """整数精确演算的核心：跟 `Engine` 同一个模型，换一套数值表示。

    参数
    ----
    digits:
        值的定标位数，v 存的是 round(真实值 * 10**digits)。
    cdigits:
        系数（alpha、关系比例、kick）的定标位数。
    """

    def __init__(self, cfg: Config | Network, digits: int = 30, cdigits: int = 24) -> None:
        # 既可以传 Config（自己建网），也可以传一张现成的网——
        # 后者是为了和 `run_once` 共用同一份关系（形反作用要就地改它）。
        if isinstance(cfg, Network):
            self.net = cfg
            cfg = self.net.cfg
        else:
            self.net = Network(cfg)
        self.cfg = cfg
        self.n = self.net.n
        self.digits = digits
        self.S = 10**digits
        self.C = 10**cdigits

        # 值：整数定标。初值是 float64 抽出来的，乘 S 之后取整——
        # 初值只有 16 位有效数字，但那是「数据的精度」，不是演算的精度。
        self.v: list[int] = [int(round(float(x) * self.S)) for x in self.net.values]

        # 系数：量化到 10**-C。这是模型定义的一部分（和 float64 版本差在 1e-24 量级）
        self.alpha = [int(round(float(a) * self.C)) for a in self.net.alpha]
        self.ratio = [[int(round(float(r) * self.C)) for r in row]
                      for row in np.asarray(self.net.ratio).tolist()]
        self.nbrs = [list(row) for row in np.asarray(self.net.nbrs).tolist()]

        self.tick = 0
        self.last_touch = [-1] * self.n
        self.feedback_times = 0
        self.feedback_edges = 0
        self._group_born: dict[frozenset[int], int] = {}

    # ---------------------------------------------------------------- 走动

    def _score(self, cur: int, nxt: int, idx: int) -> float:
        """一条关系的「还欠多少」：比例 × 值差。比较只用 float，不参与数值累积。"""
        gap = abs(self.v[nxt] - self.v[cur]) / self.S
        return (self.ratio[cur][idx] / self.C) * gap if self.cfg.walk_rule != "ratio" \
            else (self.ratio[cur][idx] / self.C)

    def walk(self, source: int) -> tuple[list[int], list[int]]:
        """极大简单路径：从 source 出发，每次走「还欠得最多」且没走过的邻居。"""
        path = [source]
        edges: list[int] = []
        seen = bytearray(self.n)
        seen[source] = 1
        cur = source
        while True:
            row, rats = self.nbrs[cur], self.ratio[cur]
            best_i = -1
            best_s = -1.0
            for idx, nxt in enumerate(row):
                if seen[nxt]:
                    continue
                if self.cfg.walk_rule == "ratio":
                    s = rats[idx] / self.C
                else:
                    s = (rats[idx] / self.C) * (abs(self.v[nxt] - self.v[cur]) / self.S)
                if s > best_s:
                    best_s, best_i = s, idx
            if best_i < 0:
                break
            nxt = row[best_i]
            seen[nxt] = 1
            edges.append(best_i)
            path.append(nxt)
            cur = nxt
        return path, edges

    # ---------------------------------------------------------------- 一步

    def step(self) -> ExactTrace:
        src = self.tick % self.n
        path, edges = self.walk(src)

        # 源是输入：只被 kick，不被弛豫（同一 tick 里它是「已知量」）
        # kick 不是常数：幅度带 1~1.5 倍抖动、符号随机，只由 (seed, tick) 决定。
        # 这里必须调同一个函数，否则两套演算从第一步起就分道扬镳（差 1e-7 量级）。
        kick = int(round(float(self.net.kick_delta(self.tick)) * self.C))
        self.v[src] = self.v[src] * (self.C + kick) // self.C

        changed = False
        for u, w, idx in zip(path, path[1:], edges):
            # idx 是 **u 那一行**里的位置（这条关系从 u 指向 w），别再拿它去索引 w 的行
            a_num = self.alpha[w] * self.ratio[u][idx] // self.C    # alpha_w * r(u→w)
            delta = a_num * (self.v[u] - self.v[w]) // self.C
            if delta:
                self.v[w] += delta
                changed = True

        self.last_touch_tick(path)
        self.tick += 1
        return ExactTrace(self.tick - 1, src, path, changed)

    def last_touch_tick(self, path: list[int]) -> None:
        t = self.tick
        for i in path:
            self.last_touch[i] = t

    def recent(self, window: int) -> np.ndarray:
        """最近 window 个 tick 里被算到过的单位（位数只在它们身上有定义）。"""
        t = self.tick
        lt = np.asarray(self.last_touch, dtype=np.int64)
        return (lt >= 0) & ((t - lt) <= int(window))

    @property
    def touched(self) -> np.ndarray:
        return np.asarray(self.last_touch, dtype=np.int64) >= 0

    # ---------------------------------------------------------------- 读数

    def to_float(self) -> np.ndarray:
        return np.array([x / self.S for x in self.v], dtype=np.float64)

    def values(self) -> np.ndarray:
        """给几何/读数用的浮点视图，跟 `Engine.values()` 同一个接口。"""
        return self.to_float()

    def defect(self, fv: np.ndarray | None = None) -> float:
        """违约读数：|v_i − 加权邻居均值| 的平均，除以 mean|v|。

        和 `Engine.defect()` 同一个定义，只是从整数表示换算成浮点再算——
        几何和读数用 15 位浮点足够，位数读数才需要精确（那部分走 `digit_text`）。
        """
        v = self.to_float() if fv is None else fv
        num = np.zeros(self.n)
        den = np.zeros(self.n)
        for i in range(self.n):
            row, rats = self.net.nbrs[i], self.net.ratio[i]
            w = np.asarray(rats, dtype=np.float64)
            num[i] = float(np.dot(w, v[row]))
            den[i] = float(w.sum())
        resid = np.abs(v - num / den)
        scale = float(np.mean(np.abs(v))) or 1.0
        return float(resid.mean() / scale)

    def texts(self) -> list[str]:
        """每个单位的有效数字串，给 `StableDigits` 用。"""
        return [self.digit_text(i) for i in range(self.n)]

    # ---------------------------------------------------------------- 双向

    def feedback(self, groups, strength: float, floor: float = 1e-3) -> int:
        """形反作用于关系：跟 `Engine.feedback` 同一套门槛和方向，改的是同一张网的 ratio。

        改完必须重建系数缓存——整数演算把比例量化成了整数，不重建就等于没改。
        """
        if strength <= 0.0:
            return 0
        net = self.net
        min_age = int(getattr(net.cfg, "feedback_age", 64))
        now = self.tick
        born = self._group_born
        fresh: dict[frozenset[int], int] = {}
        gid = np.full(net.n, -1, dtype=np.int64)
        touched = self.touched
        for gi, g in enumerate(groups):
            key = frozenset(int(u) for u in g.members)
            fresh[key] = born.get(key, now)
            if len(key) < 2 or len(key) >= net.n or now - fresh[key] < min_age:
                continue
            mem = list(key)
            if not touched[mem].all():
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
        if inside.any():
            ratio[inside] += strength * (1.0 - ratio[inside])
        if outside.any():
            ratio[outside] *= 1.0 - strength
        np.clip(ratio, floor, 1.0, out=ratio)

        # 系数缓存重建：整数演算读的是量化后的比例，不重建 = 白改
        self.ratio = [[int(round(float(r) * self.C)) for r in row]
                      for row in np.asarray(ratio).tolist()]

        changed = int(inside.sum()) + int(outside.sum())
        self.feedback_times += 1
        self.feedback_edges += changed
        return changed

    def digit_text(self, i: int, keep: int = 28) -> str:
        """把一个值写成「有效数字串」，用来数稳定前缀。

        值大约是 1e0~1e4，取整数部分 + 小数部分拼起来，去掉前导零，
        不足 keep 位就在右边补零（补零代表「后面确实是 0」）。
        """
        v = abs(self.v[i])
        s = str(v).rjust(self.digits + 1, "0")
        int_part, frac = s[:-self.digits], s[-self.digits:]
        text = (int_part + frac).lstrip("0")
        if not text:
            text = "0"
        return text[:keep].ljust(keep, "0")


class StableDigits:
    """按「十进制有效数字串」数稳定前缀——不经过 log10，也不受量尺限制。

    判据和 `wy/digits.py` 一样是**时间轴**上的：最近 window 个 tick 里
    第 d 位没变过，就认为第 d 位是稳的。区别只在读数的材料从 float64 换成整数串。

    实现上不做「窗口内两两比较」：前缀相等是可传递的（a、b 前十位相同、b、c 前十位相同
    → a、c 也相同），所以**相邻两两的前缀长度的最小值**就是整个窗口的公共前缀长度。
    这样每次 push 只比一条串，O(keep)，窗口长短不影响成本。
    """

    def __init__(self, n: int, window: int = 8, keep: int = 28) -> None:
        self.n = n
        self.window = max(2, int(window))
        self.keep = int(keep)
        self._last: list[str | None] = [None] * n
        self._pairs: list[deque[int]] = [deque(maxlen=self.window - 1) for _ in range(n)]
        self._count: list[int] = [0] * n
        self._age: list[int] = [0] * n

    @staticmethod
    def _prefix(a: str, b: str, cap: int) -> int:
        d = 0
        while d < cap and d < len(a) and d < len(b) and a[d] == b[d]:
            d += 1
        return d

    def push(self, texts: list[str]) -> None:
        for i, t in enumerate(texts):
            last = self._last[i]
            if last is not None:
                self._pairs[i].append(self._prefix(last, t, self.keep))
                self._age[i] = 0 if t != last else self._age[i] + 1
            self._last[i] = t
            self._count[i] += 1

    def prefix(self, i: int) -> int:
        q = self._pairs[i]
        return min(q) if q else 0

    @property
    def length(self) -> np.ndarray:
        """和 `StabilityMonitor.length` 同一个形状：每个单位当前的稳定前缀长度。"""
        return np.fromiter((self.prefix(i) for i in range(self.n)), np.int64, self.n)

    @property
    def age(self) -> np.ndarray:
        """这个单位的有效数字串已经多久没变过（tick 数）。"""
        return np.asarray(self._age, dtype=np.int64)

    def settled_mask(self, need: int) -> np.ndarray:
        return self.length >= int(need)

    def mean_length(self, mask: np.ndarray | None = None) -> float:
        vals = self.length
        if mask is not None:
            vals = vals[np.asarray(mask, dtype=bool)]
        return float(vals.mean()) if vals.size else 0.0


