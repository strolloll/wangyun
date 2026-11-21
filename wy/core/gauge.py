"""这张图能不能读：先量瓶颈有多陡、这次的切是不是真的，再看读数。

第 22、23 节学到的：望云读的是**瓶颈**（切断它要花多少代价），不是**语义**（人按故事分的组）。
所以「读数可不可信」不该是一句态度，得是能算的数。三个，都不需要标准答案：

    contrast   = 跨群关系的平均比例 / 群内关系的平均比例
                 小 = 权重上真有瓶颈。**单位权重或权重全一样的图上它必然等于 1**
                 （每个单位最强的那条被归一成 1），那时它说明「这图不带权重信息」，
                 而不是「这图不能读」。

    q_excess   = 这个划分的模块度 − 同尺寸随机划分的模块度
                 衡量「这一刀是不是真的切在结构上」。噪声切出来的一刀，q_excess ≈ 0。

    dispersion = 群 span 的 p90 / p10
                 大 = 各群松紧差得多，一个判据伺候不了全部。

三者合起来才说得清：contrast 说权重有没有信息，q_excess 说这一刀真不真，
dispersion 说这一刀是不是靠某个参数勉强凑出来的。
"""

from __future__ import annotations

import numpy as np

from .network import Network

# 阈值来自 §24 的标定（把合成图的瓶颈一点点调弱，看读数什么时候崩）
CONTRAST_OK = 0.10
CONTRAST_MAYBE = 0.20
Q_OK = 0.40
Q_MAYBE = 0.20
DISPERSION_OK = 8.0
DISPERSION_MAYBE = 12.0


def modularity(labels: np.ndarray, net: Network) -> float:
    """按关系比例加权的模块度。labels 是每单位的群标签。"""
    i, j, r = net.i_edge, net.j_edge, net.r_edge
    total = float(r.sum())
    if total <= 0.0:
        return 0.0
    q = 0.0
    for gid in set(labels.tolist()):
        members = np.flatnonzero(labels == gid)
        if members.size == 0:
            continue
        inside = (labels[i] == gid) & (labels[j] == gid)
        win = float(r[inside].sum())
        in_a = np.isin(i, members)
        in_b = np.isin(j, members)
        cut = float(r[in_a ^ in_b].sum())
        degree_sum = 2.0 * win + cut
        q += win / total - (degree_sum / (2.0 * total)) ** 2
    return float(q)


def q_excess(labels: np.ndarray, net: Network, trials: int = 8, seed: int = 0) -> float:
    """模块度减掉「群大小一样、成员乱换」的对照。

    不减这一下的话，群越多 Q 越高——一个切成 180 份的读数和 8 份的读数没法比。
    """
    rng = np.random.default_rng(seed)
    q = modularity(labels, net)
    base = [modularity(rng.permutation(labels), net) for _ in range(trials)]
    return float(q - np.mean(base))


def contrast_and_dispersion(labels: np.ndarray, v: np.ndarray,
                            net: Network) -> tuple[float, float]:
    """返回 (contrast, dispersion)。"""
    i, j, rr = net.i_edge, net.j_edge, net.r_edge
    same = labels[i] == labels[j]
    inside = rr[same]
    cross = rr[~same]
    if inside.size == 0 or cross.size == 0:
        return (1.0, 1.0)
    contrast = float(cross.mean() / max(inside.mean(), 1e-12))

    spans = []
    for gid in set(labels.tolist()):
        members = np.flatnonzero(labels == gid)
        if members.size < 2:
            continue
        vals = v[members]
        spans.append(float(vals.max() - vals.min()))
    if len(spans) < 2:
        return (contrast, 1.0)
    lo = float(np.percentile(spans, 10))
    hi = float(np.percentile(spans, 90))
    return (contrast, hi / max(lo, 1e-12))


def verdict(contrast: float, qx: float, dispersion: float) -> str:
    """一句话结论。"""
    if qx >= Q_OK and contrast <= CONTRAST_OK and dispersion <= DISPERSION_OK:
        return "能读：权重上有瓶颈，这一刀切在结构上"
    if qx >= Q_OK and contrast > CONTRAST_MAYBE:
        return "能读拓扑：这一刀是真的，但权重不带信息（图上没有强弱之分）"
    if qx >= Q_MAYBE and dispersion <= DISPERSION_MAYBE:
        return "勉强：切得动，但读数挑 tick 和 τ"
    return "别信：这一刀不是瓶颈切出来的（多半是噪声，或者图里没有块）"


def report(labels: np.ndarray, v: np.ndarray, net: Network) -> dict[str, object]:
    c, d = contrast_and_dispersion(labels, v, net)
    qx = q_excess(labels, net)
    return {
        "contrast": round(c, 4),
        "q_excess": round(qx, 4),
        "dispersion": round(d, 2),
        "verdict": verdict(c, qx, d),
    }
