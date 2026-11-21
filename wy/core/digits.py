"""稳定前缀：按时间轴判断「这几位多久没动过」。

判据（DESIGN.md §4）：

    第 d 位锁定 ⟺ 最近 W 个 tick 内第 d 位没变，且尾幅 < 10^(−d)

问法变了才是关键：原来问「这几位算得准吗」（得先知道正确答案，没法问），
现在问「这几位多久没动过了」（查历史就行）。研究工具能自己判断，靠的就是这种能自己查的问题。

输出的不只是前缀长度，还有尾部摘要：
    前缀告诉你它到了哪，尾部告诉你它还在往哪走。
尾部不许删，只能压——删掉它等于删掉正在发生的事。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tail:
    """变动尾部的摘要。压成两三个数，但那个「还在动」的记号必须在。"""

    amp: float  # 幅度 |v_t − v_{t−1}|
    direction: int  # +1 / -1 / 0
    narrowing: bool  # 幅度是否在收窄


class StabilityMonitor:
    """每个参数随身的一份稳定度档案。"""

    def __init__(self, n: int, window: int = 8, digits: int = 12) -> None:
        self.n = n
        self.window = window
        self.digits = digits
        self._buf = np.zeros((window, n), dtype=np.float64)
        self._filled = 0
        self.length = np.zeros(n, dtype=np.int64)  # 稳定的十进制位数
        self.age = np.zeros(n, dtype=np.int64)  # 这个长度保持多久了
        self.tail = np.zeros(n, dtype=np.float64)  # 幅度
        self.direction = np.zeros(n, dtype=np.int64)  # 方向
        self._prev_tail = np.zeros(n, dtype=np.float64)
        self._narrowing = np.zeros(n, dtype=bool)

    # ------------------------------------------------------------------

    def push(self, v: np.ndarray) -> None:
        v = np.asarray(v, dtype=np.float64)
        if self._filled < self.window:
            self._buf[self._filled] = v
            self._filled += 1
        else:
            self._buf = np.roll(self._buf, -1, axis=0)
            self._buf[-1] = v
        samples = self._buf[: self._filled]

        if self._filled >= 2:
            prev = samples[-2]
            self.tail = np.abs(v - prev)
            self.direction = np.sign(v - prev).astype(np.int64)
            # 收窄 = 这一步比上一步小。上一步的幅度就是上一轮的 tail。
            self._narrowing = self.tail <= self._prev_tail
            self._prev_tail = self.tail
        else:
            self.tail = np.zeros_like(v)
            self.direction = np.zeros_like(v, dtype=np.int64)
            self._narrowing = np.zeros_like(v, dtype=bool)

        length = self._stable_length(samples)
        changed = length != self.length
        self.age = np.where(changed, 0, self.age + 1)
        self.length = length

    # ------------------------------------------------------------------

    def _stable_length(self, samples: np.ndarray) -> np.ndarray:
        """从最高位往下数，连续的、在窗口内没变过的十进制位数。"""
        if samples.shape[0] < 2:
            # 只见过一个值，什么都不能证明。
            return np.zeros(self.n, dtype=np.int64)

        v_now = samples[-1]
        mag = np.maximum(np.abs(v_now), 1e-12)
        pmax = np.floor(np.log10(mag)).astype(np.int64)

        length = np.zeros(self.n, dtype=np.int64)
        active = np.ones(self.n, dtype=bool)
        for j in range(self.digits + 1):
            p = (pmax - j).astype(np.float64)
            scale = np.power(10.0, p)
            trunc = np.floor(samples / scale)
            ok = np.all(trunc == trunc[-1], axis=0)
            hit = active & ok
            length += hit.astype(np.int64)
            active = hit
            if not np.any(active):
                break
        return length

    # ------------------------------------------------------------------

    def tails(self) -> list[Tail]:
        return [
            Tail(float(self.tail[i]), int(self.direction[i]), bool(self._narrowing[i]))
            for i in range(self.n)
        ]

    def mean_length(self, mask: np.ndarray | None = None) -> float:
        """平均稳定位数。

        mask 用来把「还没算到」的单位排除在外：一个没被碰过的参数，它的位数当然一位没变，
        但那是**没算**，不是**稳**。混进平均里会让读数凭空变好看——这正是 A 型模糊要标出来的事。
        """
        if mask is None:
            return float(self.length.mean())
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return 0.0
        return float(self.length[mask].mean())

    def settled_mask(self, need: int) -> np.ndarray:
        """稳住到 need 位以上的参数。视觉系统用它区分 A/B 两型模糊。"""
        return self.length >= need
