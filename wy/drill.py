"""点开一个群，看它里面的人长什么样。

`scene` 把群画成物体；`drill` 再往里一层：把某个群的成员当成最小单位，
按它们之间真实的关系摆开——于是「一个群内部」也成了可以看的东西。

同一个群为什么能看：群内部的关系比例高（0.7~1.0），弹簧摆出来就是一团；
如果它其实该拆，摆出来的就是一串两团——**画面上看得出来，不用等 k 变**。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .core.layout import spring
from .render import _font
from .scene import SceneParams, draw_scene
from .core.shape import Group


@dataclass
class Cell:
    """把一个单位伪装成「只有一个成员的群」，好复用场面那套画法。"""

    members: list
    value: float
    span: float = 0.0
    stable: float = 0.0
    settled: bool = True


def member_adjacency(net, members: list[int]) -> np.ndarray:
    """群内部的邻接矩阵（无向，取两端比例的较大者）。"""
    idx = {u: i for i, u in enumerate(members)}
    k = len(members)
    adj = np.zeros((k, k), dtype=float)
    for i in range(k):
        row = net.nbrs[i if False else members[i]]
        for j_col, w in enumerate(row):
            w_id = int(w)
            if w_id in idx:
                r = float(net.ratio[members[i], j_col])
                a, b = idx[members[i]], idx[w_id]
                if a != b and r > adj[a, b]:
                    adj[a, b] = adj[b, a] = r
    return adj


def face(groups: list[Group], net, vals: np.ndarray, gid: int,
         size: int = 300, seed: int = 7, label: bool = True) -> Image.Image:
    """某一个群的内部：成员当物体，内部关系当连线。"""
    g = next((x for x in groups if x.gid == gid), None) or groups[gid]
    members = [int(u) for u in g.members]
    v = np.array([float(vals[u]) for u in members], dtype=float)
    span = float(v.max() - v.min()) or 1.0
    dev01 = (v - v.mean()) / span + 0.5          # 群内偏离：0=最低 1=最高
    cells = [Cell([u], float(vals[u]), 0.0, float(getattr(g, "stable", 0.0)), True)
             for u in members]
    adj = member_adjacency(net, members)
    deg = adj.sum(axis=1)
    w = 0.55 + 0.9 * (deg / max(1e-9, deg.max()))   # 内部度数 → 大小（绑得紧的更醒目）
    hue = 0.58 - 0.52 * np.clip(dev01, 0, 1)        # 群内偏离 → 色相：暖=高于本群，冷=低于
    pos = (np.asarray(spring(adj, seed=seed, iters=300, restarts=2), dtype=float)
           if len(members) > 1 else np.zeros((1, 2)))
    title = f"群 {g.gid} · {len(members)} 人 · span {float(g.span):.3g}"
    note = (f"内部平均关系 {adj[adj > 0].mean():.3f} · 色相=群内高低，大小=内部度数"
            if (adj > 0).any() else "内部没有关系")
    return draw_scene(cells, adj, SceneParams(size=size, facets=0.0),
                      title=title if label else "",
                      note=note if label else "",
                      seed=seed, pos=pos, hue=hue, weights=w)


def zoom(net, vals: np.ndarray, member: int, size: int = 420, seed: int = 7,
         hops: int = 1) -> Image.Image:
    """第三层：盯住一个人，看他直接连着谁。

    hops=1 只到邻居；颜色分三类：自己（暖）、同群的人（青）、别群的人（紫）。
    这一层回答的是「这个单位靠谁活着」——群内度数高说明他是骨架，跨界邻居多说明他是桥。
    """
    nodes = [int(member)]
    for _ in range(max(1, hops)):
        for u in list(nodes):
            for j, w in enumerate(net.nbrs[u]):
                v = int(w)
                if v == u or v in nodes:
                    continue
                if getattr(net, "mask", None) is not None and not bool(net.mask[u, j]):
                    continue
                if float(net.ratio[u, j]) > 0:
                    nodes.append(v)
    idx = {u: i for i, u in enumerate(nodes)}
    k = len(nodes)
    adj = np.zeros((k, k), dtype=float)
    for u in nodes:
        for j, w in enumerate(net.nbrs[u]):
            v = int(w)
            if v in idx and v != u:
                r = float(net.ratio[u, j])
                a, b = idx[u], idx[v]
                adj[a, b] = adj[b, a] = max(adj[a, b], r)

    # 谁和中心在同一个群里
    same = np.ones(k, dtype=bool)
    same[0] = False
    cells = [Cell([u], float(vals[u])) for u in nodes]
    deg = adj.sum(axis=1)
    w = 0.55 + 1.0 * (deg / max(1e-9, deg.max()))
    w[0] = 1.0
    hues = np.where(same, 0.50, 0.74)
    hues = np.asarray(hues, dtype=float)
    hues[0] = 0.07
    pos = (np.asarray(spring(adj, seed=seed, iters=320, restarts=2), dtype=float)
           if k > 1 else np.zeros((1, 2)))
    title = f"单位 {member} · 邻居 {k - 1} 个"
    note = f"暖=它自己 · 青=同群 · 紫=跨群 · 大小=度数 · 图上只有一跳（hops={hops}）"
    return draw_scene(cells, adj, SceneParams(size=size, facets=0.0),
                      title=title, note=note, seed=seed, pos=pos, hue=hues, weights=w)


def inset(groups, net, vals: np.ndarray, gid: int, strength: np.ndarray,
          size: int = 900, seed: int = 7) -> Image.Image:
    """连续放大：父图照画，把那个群的内部**贴在它自己在画面上的位置**。

    和 `face` 的区别：`face` 是另起一张图；`inset` 用的是同一套坐标
    （由 `scene.frame_of` 给出），所以放大的位置对得上——看图的人不会「迷路」。
    """
    from .scene import SceneParams, draw_scene, frame_of

    p = SceneParams(size=size)
    base = draw_scene(groups, strength, p,
                      title=f"望云 · {len(groups)} 个群 · 放大群 {gid}",
                      note="白圈里的那张图，就是这个群内部", seed=seed)
    xy, rf = frame_of(groups, strength, p, seed=seed)
    idx = next((i for i, g in enumerate(groups) if g.gid == gid), None)
    if idx is None:
        return base

    r_px = max(70.0, float(rf[idx]) * size)
    inner_size = int(min(size * 0.92, max(220, r_px * 2.4)))
    inner = face(groups, net, vals, gid, size=inner_size, seed=seed, label=False)

    mask = Image.new("L", (inner_size, inner_size), 0)
    ImageDraw.Draw(mask).ellipse([2, 2, inner_size - 3, inner_size - 3], fill=255)
    cx, cy = xy[idx][0] * size, xy[idx][1] * size
    base.paste(inner, (int(cx - inner_size / 2), int(cy - inner_size / 2)), mask)

    d = ImageDraw.Draw(base)
    d.ellipse([cx - inner_size / 2, cy - inner_size / 2,
               cx + inner_size / 2, cy + inner_size / 2],
              outline=(250, 244, 220), width=max(2, size // 300))
    g = groups[idx]
    d.text((cx - inner_size / 2, cy + inner_size / 2 + 6),
           f"群 {g.gid} · {len(g.members)} 人 · span {float(g.span):.3g}",
           fill=(232, 240, 244), font=_font(12))
    return base


def sheet(groups: list[Group], net, vals: np.ndarray, top: int = 6, cols: int = 3,
          size: int = 300, seed: int = 7) -> Image.Image:
    """按人数从大到小，给前 top 个群各画一张内部图，拼成一张。"""
    order = sorted(groups, key=lambda g: len(g.members), reverse=True)[:top]
    imgs = [face(groups, net, vals, g.gid, size=size, seed=seed) for g in order]
    if not imgs:
        return Image.new("RGB", (10, 10))
    rows = (len(imgs) + cols - 1) // cols
    gap = 8
    out = Image.new("RGB", (cols * size + (cols - 1) * gap,
                            rows * size + (rows - 1) * gap), (10, 16, 20))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        out.paste(im, (c * (size + gap), r * (size + gap)))
    return out
