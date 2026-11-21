"""自带插件：一个零依赖的 SVG 渲染通道、一条读数、一个边表数据源。

它们同时是写法示例——真正接自己的东西时，照着这三个函数的形状来就行。
"""

from __future__ import annotations

import math

from . import plugins

BG = "#0a1a1a"


def _hsl(hue: float, sat: int = 58, light: int = 64) -> str:
    return f"hsl({int((hue % 1.0) * 360)}, {sat}%, {light}%)"


@plugins.renderer("svg", "一帧画成 SVG 文本（零依赖，网页和别的软件都能直接用）")
def svg(bundle: dict, index: int = -1, size: int = 520) -> str:
    """bundle 的一帧 → SVG。位置/半径/色相都来自 bundle，不另算一遍。"""
    frames = bundle.get("frames") or []
    if not frames:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"></svg>'
    f = frames[index]
    k = f["k"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 100 100">',
        f'<rect width="100" height="100" fill="{BG}"/>',
    ]
    pos = f["pos"]
    hue = [g["hue"] for g in f["groups"]]
    count = [g["count"] for g in f["groups"]]
    settled = [g["settled"] for g in f["groups"]]

    def xy(i: int) -> tuple[float, float]:
        return 50.0 + 44.0 * pos[i][0], 50.0 - 44.0 * pos[i][1]

    # 群间强度：线越亮/越粗，关系越紧
    for i in range(k):
        for j in range(i + 1, k):
            w = float(f["strength"][i][j])
            if w <= 0.02:
                continue
            x1, y1 = xy(i)
            x2, y2 = xy(j)
            parts.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
                f'stroke="#7fd6d0" stroke-opacity="{min(0.55, 0.25 + 0.9 * w):.2f}" '
                f'stroke-width="{0.25 + 0.7 * w:.2f}"/>'
            )

    for i in range(k):
        x, y = xy(i)
        col = _hsl(hue[i])
        r = 1.5 + 1.7 * math.sqrt(max(1, count[i]))
        if settled[i]:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{col}" '
                         f'fill-opacity="0.88"/>')
        else:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{col}" '
                         f'fill-opacity="0.28" stroke="{_hsl(hue[i], 58, 72)}" '
                         f'stroke-width="0.35" stroke-dasharray="1.3 1.1"/>')

    parts.append(f'<text x="2" y="4.5" fill="#d8e8e4" font-size="3.2" '
                 f'font-family="monospace">t={f["t"]}  k={k}</text>')
    parts.append(f'<text x="2" y="97.5" fill="#7f9a96" font-size="2.6" '
                 f'font-family="monospace">n={bundle["meta"]["n"]} '
                 f'seed={bundle["meta"]["seed"]}</text>')
    parts.append("</svg>")
    return "".join(parts)


@plugins.reading("hue-moved", "采样帧上的色相移动序列：[(t, moved), ...]")
def hue_moved(bundle: dict) -> list[tuple[int, float]]:
    return [(int(f["t"]), float(f["hue_moved"] or 0.0)) for f in bundle.get("frames", [])]


@plugins.datasource("edgelist", "读一个边表文件，得到真图（交给 Engine 跑）")
def edgelist(path: str, cfg=None):
    from .core.load import load_graph

    return load_graph(path, cfg)
