"""让场面动起来：按时间采样若干帧，群的位置一路搬过去。

规矩和 `strip` 一样：**采样，不插值**。每一帧都是某一次 tick 上真实的读数，
中间那些没画出来的 tick 不是被平滑掉了，是没被采集——图上不说「中间怎么样了」。
"""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image

from .core.color import hue_track, hue_vec
from .core.digits import StabilityMonitor
from .core.engine import Engine
from .core.layout import group_strength
from .core.network import Config, Network
from .scene import SceneParams, carry, draw_scene
from .core.shape import group_table, scale_of


def frames(cfg: Config, ticks: int, count: int = 8, frac: float = 3e-2,
           need: int = 4, size: int = 360, window: int | None = None,
           trace: list | None = None) -> list[Image.Image]:
    """一次演算，采样 count 帧。位置按成员重合从上帧搬过来，画面不跳。

    颜色和位置一样是「记路」的：老群沿用上一帧的色相，新群继承成员重合最多的
    那个旧群（`color.hue_track`）。传一个空列表给 `trace`，就按帧把颜色读数记
    下来——每帧一项，含 `t`、`k`、本帧色相移动总量 `moved`、以及
    「成员集合 → 色相」。结构在不在重排，读这条序列，不靠看图猜。
    """
    net = Network(cfg)
    eng = Engine(net)
    win = cfg.window if window is None else window
    hist: deque = deque(maxlen=win)
    hist.append(eng.v.copy())

    marks = sorted({int(round(x)) for x in np.linspace(ticks / count, ticks, count)})
    shots: list[tuple[int, np.ndarray, list]] = []
    for t in range(1, ticks + 1):
        eng.step()
        mon = StabilityMonitor(net.n, win, cfg.digits)
        for v in hist:
            mon.push(v)
        hist.append(eng.v.copy())
        if marks and t == marks[0]:
            marks.pop(0)
            shots.append((t, eng.v.copy(), list(hist)))

    out: list[Image.Image] = []
    prev_groups, prev_pos = [], np.zeros((0, 2))
    prev_hues: dict[frozenset, float] = {}
    ref = 0.0
    for t, v, h in shots:
        mon = StabilityMonitor(net.n, win, cfg.digits)
        for vv in h:
            mon.push(vv)
        scale = scale_of(v)
        groups = group_table(v, net, frac * scale, mon, need, touched=eng.recent(win))
        if len(groups) == 0:
            continue
        strength = group_strength(groups, net)
        # 色相逐帧接着走：老群不动，新群跟着它的旧血统挪一格
        hues, moved = hue_track(prev_hues, groups, strength)
        colours = hue_vec(hues, groups)
        prev_hues = hues
        pos = carry(prev_groups, prev_pos, groups) if prev_pos.size else np.zeros((0, 2))
        if pos.size == 0:
            from .layout import spring

            pos = np.asarray(spring(strength, seed=cfg.seed, iters=320, restarts=2), dtype=float)
        pos = np.asarray(pos, dtype=float)
        centred = pos - pos.mean(axis=0)
        # 搬过来的坐标再松弛几步：既接着上一帧，又不至于被硬拉住变形
        from .layout import spring

        if ref > 0:
            centred = np.asarray(
                spring(strength, seed=cfg.seed, iters=60, init=centred), dtype=float)
            centred = centred - centred.mean(axis=0)
        rad = float(np.abs(centred).max()) or 1.0
        if ref <= 0:
            ref = rad  # 第一帧定尺度，后面都沿用它
        img = draw_scene(groups, strength, SceneParams(size=size, span_ref=ref),
                         title=f"t={t}  k={len(groups)}", note="", seed=cfg.seed,
                         pos=centred, hue=colours)
        out.append(img)
        if trace is not None:
            trace.append({"t": t, "k": len(groups), "moved": float(moved),
                          "hues": dict(hues)})
        prev_groups, prev_pos = groups, centred
    return out


def filmstrip(imgs: list[Image.Image], cols: int = 4, gap: int = 8) -> Image.Image:
    """把若干帧拼成一条带子。"""
    if not imgs:
        return Image.new("RGB", (10, 10))
    w, h = imgs[0].size
    rows = (len(imgs) + cols - 1) // cols
    out = Image.new("RGB", (cols * w + (cols - 1) * gap,
                            rows * h + (rows - 1) * gap), (10, 16, 20))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        out.paste(im, (c * (w + gap), r * (h + gap)))
    return out


def save_gif(imgs: list[Image.Image], path, duration: int = 400) -> None:
    """存成 GIF。帧序就是采样序，不补中间帧。"""
    if not imgs:
        raise ValueError("没有帧可存")
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=duration, loop=0, optimize=True)
