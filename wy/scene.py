"""把一张图的读数画成**场面**：每个群是一个物体，关系强度决定远近和大小。

和 `paint.py` 的区别：`paint` 画的是一张脸（参数化肖像），`scene` 画的是**这张图长什么样**。
这里没有「构图」这一说——位置来自关系的弹簧布局，远近来自每个群的平均关系强度，
大小来自它的人数。颜色、明暗、表面、连线也全都从读数来：

    人数      → 大小
    关系强度  → 远近（强 = 近 = 大）+ 连线的亮度和粗细
    内部极差  → 表面碎不碎（紧的群接近球，松的群是多面体带毛刺）
    稳定位数  → 透明度（信得多的实，信得少的半透明）
    稳没稳住  → 实心 / 点阵（和 sheet、board 同一套语言）

所以同一张图、同一个种子，画出来的场面逐像素相同；换一张图，场面就换一个样子。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .paint import _hsv, _radial
from .render import _font

BG_TOP = (6, 12, 17)
BG_BOT = (10, 24, 26)
INK = (232, 240, 244)
DIM = (128, 152, 162)


@dataclass(frozen=True)
class SceneParams:
    size: int = 900
    fov: float = 2.2         # 透视强度：越大越像长焦（远近差别小）
    min_scale: float = 0.34  # 最远的物体缩到多小
    near_boost: float = 0.85  # 平均关系强 → 往前拉多少
    glow: float = 0.5
    stars: float = 0.35
    edges: float = 0.75      # 关系线的存在感
    facets: float = 1.0      # 表面碎不碎这个映射的强度（0 = 全画成球）
    span_ref: float = 0.0    # 动画用：沿用第一帧的尺度，别每帧自己归一化


def _depth_of(groups, strength: np.ndarray, p: SceneParams) -> np.ndarray:
    """每个群的平均关系强度 → 远近。关系越强越近（越靠前、越大）。"""
    counts = np.array([max(1, len(g.members)) for g in groups], dtype=float)
    w = strength @ (counts / counts.sum()) if strength.size else np.zeros(len(groups))
    if w.size == 0:
        return np.zeros(0)
    lo, hi = float(w.min()), float(w.max())
    norm = np.zeros_like(w) if hi - lo < 1e-12 else (w - lo) / (hi - lo)
    return 1.0 - p.near_boost * norm


def _looseness(groups) -> np.ndarray:
    """内部极差 → 0（紧）到 1（松）。按本场最紧/最松的两团归一。

    松紧是相对量：同一张图里「谁比谁松」有意义，「0.3 算松吗」没有意义。
    """
    span = np.array([float(getattr(g, "span", 0.0)) for g in groups], dtype=float)
    if span.size == 0:
        return span
    lo, hi = float(span.min()), float(span.max())
    if hi - lo < 1e-12:
        return np.zeros_like(span)
    return (span - lo) / (hi - lo)


def _facet(loose: float, seed: int) -> tuple[int, float, float]:
    """松紧 → (边数, 毛刺幅度, 相位)。紧的接近球（18 边、几乎无毛刺），松的像碎石头（5 边）。"""
    rng = np.random.default_rng(seed)
    sides = int(round(18 - 13 * float(np.clip(loose, 0, 1))))
    jag = 0.012 + 0.085 * float(np.clip(loose, 0, 1))
    phase = float(rng.uniform(0, 2 * math.pi))
    return max(4, sides), jag, phase


def carry(prev_groups, prev_pos: np.ndarray, groups) -> np.ndarray:
    """把上一帧的坐标搬到这一帧：按成员重合认人，不按第几号群。

    真身在 `wy.core.layout.carry_pos()`——色相、缩略图、网页共用同一套「谁还是谁」；
    这里只是转手，方便老的调用点接着用。
    """
    from .core.layout import carry_pos

    return carry_pos(prev_groups, np.asarray(prev_pos, dtype=np.float64), groups)


def frame_of(groups, strength: np.ndarray, p: SceneParams = SceneParams(),
             pos: np.ndarray | None = None, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """每个物体在画面上的**位置和半径**（都化成 0~1 的比例）。

    `drill --focus` 用它把「群内部的图」贴在父图里那一团的位置上——
    这样三层共享同一套坐标，放大是连续的，不是另起一张图。
    """
    k = len(groups)
    if k == 0:
        return np.zeros((0, 2)), np.zeros(0)
    if pos is None or len(pos) != k:
        from .core.layout import spring

        pos = np.asarray(spring(strength, seed=seed, iters=420, restarts=2), dtype=float)
    pos = np.asarray(pos, dtype=float)
    pos = pos - pos.mean(axis=0)
    rad = float(np.abs(pos).max()) or 1.0
    pos = pos / rad * 0.86

    depth = _depth_of(groups, strength, p)
    persp = 1.0 / (1.0 + p.fov * depth)
    span_p = float(persp.max() - persp.min())
    persp = p.min_scale + (1.0 - p.min_scale) * (
        (persp - persp.min()) / span_p if span_p > 1e-9 else np.ones_like(persp))
    xy = np.stack([0.5 + pos[:, 0] * 0.42 * persp,
                   0.54 + pos[:, 1] * 0.42 * persp * 0.92], axis=1)
    counts = np.array([max(1, len(g.members)) for g in groups], dtype=float)
    rf = np.sqrt(counts / counts.max()) * 0.085 * persp
    return xy, rf


def draw_scene(groups, strength: np.ndarray, p: SceneParams = SceneParams(),
               title: str = "", note: str = "", seed: int = 7,
               pos: np.ndarray | None = None,
               hue: np.ndarray | None = None,
               weights: np.ndarray | None = None) -> Image.Image:
    """群表 + 强度矩阵 → 一张有远近的场面。

    `pos` 给了就用它（动画要接着上一帧）；`hue`/`weights` 给了就覆盖默认映射
    （drill 用它们把「群内偏离」和「内部度数」画出来）。
    """
    k = len(groups)
    S = int(p.size)
    ss = 2
    W = S * ss
    rng = np.random.default_rng(seed)

    # ---- 背景：上暗下亮的雾
    yy = np.linspace(0, 1, W, dtype=np.float32)[:, None]
    bg = np.zeros((W, W, 3), dtype=np.float32)
    for i in range(3):
        bg[..., i] = BG_TOP[i] + (BG_BOT[i] - BG_TOP[i]) * (yy ** 1.3)
    fog = np.clip(1 - _radial(W, W * 0.78, W * 0.5, W * 0.7, W * 0.42), 0, 1) ** 2
    bg[..., 1] += fog * 12
    bg[..., 2] += fog * 8
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

    # ---- 星尘
    dust = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dust)
    for _ in range(int(120 + 300 * p.stars)):
        x, y = rng.uniform(0, W), rng.uniform(0, W)
        r = float(rng.uniform(0.4, 1.2)) * ss
        a = int(rng.integers(40, 120))
        dd.ellipse([x - r, y - r, x + r, y + r], fill=(190, 226, 230, a))
    img.alpha_composite(dust.filter(ImageFilter.GaussianBlur(0.3 * ss)))
    d = ImageDraw.Draw(img)

    if k == 0:
        return img.convert("RGB").resize((S, S), Image.LANCZOS)

    # ---- 位置
    if pos is None or len(pos) != k:
        from .core.layout import spring

        pos = np.asarray(spring(strength, seed=seed, iters=420, restarts=2), dtype=float)
    pos = np.asarray(pos, dtype=float)
    pos = pos - pos.mean(axis=0)
    rad = float(np.abs(pos).max()) or 1.0
    # 动画要接着上一帧的尺度，否则每帧自己归一化、画面会一跳一跳
    ref = float(p.span_ref) if getattr(p, "span_ref", 0.0) > 0 else rad
    pos = pos / ref * 0.86

    depth = _depth_of(groups, strength, p)
    persp = 1.0 / (1.0 + p.fov * depth)
    span_p = float(persp.max() - persp.min())
    persp = p.min_scale + (1.0 - p.min_scale) * (
        (persp - persp.min()) / span_p if span_p > 1e-9 else np.ones_like(persp))

    cx, cy = W * 0.5, W * 0.54
    span = W * 0.42
    xy = np.stack([cx + pos[:, 0] * span * persp,
                   cy + pos[:, 1] * span * persp * 0.92], axis=1)

    counts = np.array([max(1, len(g.members)) for g in groups], dtype=float)
    if weights is not None and len(weights) == len(groups):
        counts = counts * np.asarray(weights, dtype=float)
    base = np.sqrt(counts / max(1e-9, counts.max())) * W * 0.085
    radii = base * persp
    loose = _looseness(groups) * p.facets

    order = np.argsort(-depth)  # 远的先画
    hues = (np.asarray(hue, dtype=float) if hue is not None and len(hue) == len(groups) else None)
    if hues is None:
        # 色相也由关系定：关系强的群在色轮上挨着（与形空间共用同一张 W）
        from .core.color import color_space

        cs = color_space(groups, None, strength=strength)
        if cs:
            hues = np.array([cs[int(getattr(g, "gid", i))][0] for i, g in enumerate(groups)], dtype=float)

    # ---- 关系线（先画，压在所有物体下面）
    wmax = max(1e-9, float(strength.max())) if strength.size else 1.0
    for i in order:
        for j in order:
            if j <= i:
                continue
            w = float(strength[i, j]) if strength.size else 0.0
            if w <= 0:
                continue
            t = min(1.0, w / wmax)
            a = int((10 + 46 * t) * p.edges)
            lw = max(1, int((0.5 + 1.6 * t) * ss))
            d.line([tuple(xy[i]), tuple(xy[j])], fill=(126, 172, 178, a), width=lw)

    # ---- 物体：由远到近
    for i in order:
        g = groups[i]
        r = float(radii[i])
        x, y = float(xy[i][0]), float(xy[i][1])

        # 投影阴影（给纵深一点锚）
        sh = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        ImageDraw.Draw(sh).ellipse(
            [x - r * 0.9, y + r * 0.72, x + r * 0.9, y + r * 0.98],
            fill=(0, 0, 0, 120))
        img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(r * 0.22)))

        hue_i = (float(hues[i]) if hues is not None
                 else 0.455 + 0.085 * (float(g.value) % 1.0))
        ys, xs = np.mgrid[max(0, int(y - r * 1.2 - 2)):min(W, int(y + r * 1.2 + 3)),
                          max(0, int(x - r * 1.2 - 2)):min(W, int(x + r * 1.2 + 3))]
        if xs.size == 0:
            continue
        nx = (xs - x) / r
        ny = (ys - y) / r
        rho = np.sqrt(nx ** 2 + ny ** 2)
        theta = np.arctan2(ny, nx)

        # 表面：紧的接近球，松的是多面体 + 毛刺（边数、毛刺幅度都由极差定）
        sides, jag, phase = _facet(float(loose[i]), seed + 17 * i)
        if sides >= 16:
            edge_r = np.ones_like(rho)
        else:
            seg = 2 * math.pi / sides
            edge_r = math.cos(seg * 0.5) / np.cos(((theta + phase) % seg) - seg * 0.5)
        edge_r = edge_r * (1.0 + jag * np.sin(sides * theta + phase * 3))
        inside = np.clip((edge_r - rho) / 0.05, 0, 1)

        nz = np.sqrt(np.clip(1 - (rho / np.maximum(edge_r, 1e-6)) ** 2, 0, 1))
        lam = np.clip(nx * -0.45 + ny * -0.62 + nz * 0.75, 0, 1) ** 1.1
        lit = _hsv(hue_i, 0.34, 1.0)
        dark = _hsv(hue_i + 0.02, 0.55, 0.52)
        rgb = np.zeros((*nx.shape, 3), dtype=np.float32)
        for ch in range(3):
            rgb[..., ch] = dark[ch] + (lit[ch] - dark[ch]) * (0.30 + 0.70 * lam)
        spec = np.clip(1 - (((nx + 0.42) / 0.42) ** 2 + ((ny + 0.40) / 0.36) ** 2), 0, 1) ** 2
        rgb += spec[..., None] * 255 * (0.18 + 0.30 * p.glow)

        # 棱边：多面体要有棱，不然看起来只是圆
        if sides < 16:
            rim = np.clip(1 - np.abs(edge_r - rho) / 0.020, 0, 1)
            rgb *= (1.0 - 0.22 * rim)[..., None]

        # 透明度：稳定位数少的群更"虚"（信得少就别画得像铁板一块）
        stable = float(getattr(g, "stable", 0.0))
        alpha_mul = float(np.clip(0.45 + 0.14 * stable, 0.45, 1.0))
        patch = Image.fromarray(
            np.dstack([np.clip(rgb, 0, 255), inside * 255 * alpha_mul]).astype(np.uint8), "RGBA")
        img.alpha_composite(patch, (int(xs.min()), int(ys.min())))

        # 纹理：稳住的实心，还在动的点阵
        if not bool(getattr(g, "settled", False)) and r > 10 * ss:
            tex = Image.new("RGBA", (W, W), (0, 0, 0, 0))
            td = ImageDraw.Draw(tex)
            n_dots = int(np.clip(r * r * 0.012, 40, 1200))
            for _ in range(n_dots):
                ang = rng.uniform(0, 2 * math.pi)
                rr = r * math.sqrt(rng.uniform(0, 0.94))
                px, py = x + math.cos(ang) * rr, y + math.sin(ang) * rr
                t = 0.55 * ss
                td.ellipse([px - t, py - t, px + t, py + t], fill=(28, 52, 56, 110))
            img.alpha_composite(tex)

    out = img.convert("RGB").resize((S, S), Image.LANCZOS)
    d2 = ImageDraw.Draw(out)
    if title:
        d2.text((14, 10), title, fill=INK, font=_font(15))
    if note:
        d2.text((14, S - 22), note, fill=DIM, font=_font(11))
    return out
