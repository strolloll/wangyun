"""把「形」画成一张图：参数驱动，不画具体故事。

DESIGN §5.4 说形是**输出端**：形是一组数值，交给建模/渲染去当真数。
这个模块就是那条线的第二半——它不认识「史莱姆」也不认识「角色」，
它只认识一张参数表（`Params`），并把参数表变成像素。

参数从哪来：`params_from_engine()` 把一次演算的读数映射成参数（映射规则写在函数里，
一行一条，可复查）。所以图上的每个数都能追回到某次 tick 的某个读数——
这正是望云要的样子：**画面是读数的函数，不是画师的心情**。

用法：

    python -m wy paint --n 128 --ticks 4000 --seed 7
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------- 参数


@dataclass(frozen=True)
class Params:
    """一张图的全部旋钮。"""

    size: int = 900              # 画布边长（会 2 倍超采样，最后缩回来）
    hue_body: float = 0.470       # 身体主色（HSV 色相 0..1）
    body_sat: float = 0.50
    body_val: float = 0.93
    eye_h: float = 0.36          # 眼睛高 / 身体半径
    eye_gap: float = 0.19        # 两眼中心距 / 身体半径
    eye_tilt: float = 0.0        # 眼睛微微外倾（度）
    eye_hue: float = 0.075       # 眼睛色相（橙）
    swirl: float = 0.45          # 侧面花纹的存在感 0..1
    fins: int = 2                # 小翅的数量
    fin_len: float = 0.22        # 翅长 / 半径
    fin_tilt: float = 42.0       # 翅的仰角（度）
    stars: float = 0.5           # 背景星密度 0..1
    glow: float = 0.55           # 边缘辉光 0..1
    light: tuple = (-0.45, -0.62)  # 光来自左上


def params_from_engine(k, spans, stable, defect, need=4, base: Params | None = None) -> Params:
    """读数 → 参数。每一行都是一条可复查的映射，不是魔法。"""
    base = base or Params()
    spans = np.asarray(spans, dtype=float) if len(spans) else np.array([1.0])
    k = int(max(1, k))
    # 群越多、花纹越多；但花纹最多到 0.8，否则糊成一片
    swirl = float(np.clip(0.35 + 0.06 * (k - 1), 0.30, 0.80))
    # 群内极差的均值 → 眼睛大小（结构越松，眼越大，越"呆"）。
    # 对数尺度：span 从 1 到 40 跨一个半数量级，线性映射会一直顶在上限。
    sp = float(np.mean(np.abs(spans))) if len(spans) else 1.0
    eye_h = float(np.clip(0.22 + 0.18 * (math.log10(max(sp, 1.0)) / 1.6), 0.22, 0.40))
    # 稳定位数 → 高光多硬（越稳越清亮）
    glow = float(np.clip(0.25 + 0.09 * float(stable), 0.25, 0.85))
    # 违约读数 → 星星密度（没和解的差别，就是背景里的碎点）。
    # 用对数：defect 从 1e-6 到 1e-1 跨五个数量级，线性映射会把整张图压成「没星」。
    d = max(float(defect), 1e-9)
    stars = float(np.clip(0.22 + 0.16 * (math.log10(d) + 6.0), 0.20, 0.95))
    # 身体色相随 defcet 微移到青绿
    hue = float(np.clip(0.47 + 0.10 * float(defect), 0.45, 0.55))
    return Params(
        size=base.size,
        hue_body=hue,
        eye_h=eye_h,
        swirl=swirl,
        glow=glow,
        stars=stars,
    )


# ---------------------------------------------------------------- 小工具


def _hsv(c, s, v, a=1.0):
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(c % 1.0, s, v)
    return (int(r * 255), int(g * 255), int(b * 255), int(a * 255))


def _radial(S, cy, cx, rx, ry, power=2.0):
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    return d  # 0 中心，1 边缘


def _blur_mask(mask: np.ndarray, r: float) -> np.ndarray:
    img = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), "L")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32) / 255.0


def _paste_rgba(canvas: Image.Image, layer: Image.Image, mask: Image.Image | None = None) -> Image.Image:
    if mask is None:
        canvas.alpha_composite(layer)
    else:
        canvas.paste(layer, (0, 0), mask)
    return canvas


# ---------------------------------------------------------------- 画


def draw_character(p: Params, ss: int = 2) -> Image.Image:
    """画一张：一个圆润的小东西，正面，左上打光，背景是暗的星空。"""
    S = int(p.size * ss)
    c = S / 2.0
    R = S * 0.335                      # 身体半径

    # ---- 背景：暗星空 + 底部一点绿雾
    bg = np.zeros((S, S, 3), dtype=np.float32)
    yy = np.linspace(0, 1, S, dtype=np.float32)[:, None]
    bg[..., 0] = 8 + 6 * (1 - yy)      # R
    bg[..., 1] = 20 + 26 * (1 - yy)    # G
    bg[..., 2] = 30 + 28 * (1 - yy)    # B
    # 底部绿雾
    fog = _radial(S, S * 1.02, c, S * 0.85, S * 0.45)
    fog = np.clip(1 - fog, 0, 1) ** 2 * 0.55
    bg[..., 1] += fog * 46
    bg[..., 2] += fog * 26
    # 暗角
    vig = _radial(S, c, c, S * 0.95, S * 0.95)
    bg *= np.clip(1.15 - 0.55 * vig, 0.35, 1.15)[..., None]
    canvas = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

    # ---- 星星
    rng = np.random.default_rng(7)
    n_star = int(50 + 430 * p.stars)
    stars = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stars)
    for _ in range(n_star):
        x, y = rng.uniform(0, S), rng.uniform(0, S)
        if (x - c) ** 2 + (y - c) ** 2 < (R * 1.05) ** 2:
            continue
        a = int(rng.integers(90, 200))
        r = float(rng.uniform(0.4, 1.3)) * ss
        col = (210, 250, 242) if rng.random() < 0.7 else (178, 214, 255)
        sd.ellipse([x - r, y - r, x + r, y + r], fill=col + (a,))
        # 少数亮星带一点极淡的光晕
        if rng.random() < 0.06:
            hr = r * 2.4
            sd.ellipse([x - hr, y - hr, x + hr, y + hr], fill=col + (16,))
    stars = stars.filter(ImageFilter.GaussianBlur(0.3 * ss))
    canvas.alpha_composite(stars)

    # ---- 小翅：从背后伸出来，先画（会被身体压住内侧）
    for i in range(max(0, p.fins)):
        sign = -1 if i % 2 == 0 else 1
        ang = math.radians(p.fin_tilt)
        bx = c + sign * R * 0.74
        by = c - R * 0.63
        L = R * p.fin_len * 1.55
        tipx = bx + sign * math.sin(ang) * L
        tipy = by - math.cos(ang) * L
        fin = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        fd = ImageDraw.Draw(fin)
        # 一片叶子：沿轴线取两侧宽度（根部宽、尖端收）
        ax, ay = tipx - bx, tipy - by
        nx, ny = -ay, ax
        nl = math.hypot(nx, ny) or 1.0
        nx, ny = nx / nl, ny / nl
        left, right = [], []
        for t in np.linspace(0, 1, 24):
            w = (R * 0.082) * math.sin(math.pi * min(1.0, t * 1.15)) ** 0.7
            px, py = bx + ax * t, by + ay * t
            left.append((px + nx * w, py + ny * w))
            right.append((px - nx * w, py - ny * w))
        fd.polygon(left + right[::-1], fill=(248, 253, 252, 240))
        fd.line(left, fill=(210, 236, 231, 150), width=max(1, int(1.1 * ss)))
        fd.line(right, fill=(210, 236, 231, 110), width=max(1, int(1.1 * ss)))
        canvas.alpha_composite(fin.filter(ImageFilter.GaussianBlur(0.45 * ss)))

    # ---- 身体：球面光照
    d = _radial(S, c, c, R, R)
    inside = np.clip((1.0 - d) * R / (1.6 * ss), 0, 1)  # 边缘软化
    lx, ly = p.light
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    nx = (xx - c) / R
    ny = (yy - c) / R
    nz = np.sqrt(np.clip(1 - nx**2 - ny**2, 0, 1))
    l = np.array([lx, ly, 0.82], dtype=np.float32)
    l /= np.linalg.norm(l)
    lam = np.clip(nx * l[0] + ny * l[1] + nz * l[2], 0, 1) ** 1.05
    lit = _hsv(p.hue_body, p.body_sat * 0.80, 0.99)
    dark = _hsv(p.hue_body + 0.020, p.body_sat * 1.35, 0.70)
    rgb = np.zeros((S, S, 3), dtype=np.float32)
    for i in range(3):
        rgb[..., i] = dark[i] + (lit[i] - dark[i]) * (0.30 + 0.70 * lam)
    # 底部再压暗一点（体积感）
    rgb *= (1.0 - 0.16 * np.clip(ny, 0, 1) ** 1.5)[..., None]
    # 柔光：左上大面积 + 顶部一道细亮
    soft = np.clip(1 - _radial(S, c - R * 0.36, c - R * 0.34, R * 0.86, R * 0.80), 0, 1) ** 1.5
    rgb += soft[..., None] * 255 * 0.24 * p.glow
    sheen = np.clip(1 - _radial(S, c - R * 0.50, c - R * 0.02, R * 0.16, R * 0.34), 0, 1) ** 2.0
    rgb += sheen[..., None] * 255 * 0.22 * p.glow
    inner_sh = np.clip(1 - _radial(S, c + R * 0.34, c + R * 0.40, R * 1.02, R * 1.02), 0, 1) ** 1.6
    rgb *= (1.0 - 0.20 * inner_sh)[..., None]
    body = np.dstack([np.clip(rgb, 0, 255), inside * 255]).astype(np.uint8)

    # ---- 身体外的一圈柔光（先铺，身体盖在上面）
    halo = np.clip(1 - _radial(S, c, c, R * 1.22, R * 1.22), 0, 1) ** 2.4
    glow_rgb = _hsv(p.hue_body, 0.55, 1.0)
    halo_img = np.dstack([
        np.full((S, S), glow_rgb[0], np.float32),
        np.full((S, S), glow_rgb[1], np.float32),
        np.full((S, S), glow_rgb[2], np.float32),
        (halo * 96 * p.glow).astype(np.float32),
    ]).astype(np.uint8)
    canvas.alpha_composite(Image.fromarray(halo_img, "RGBA"))
    canvas.alpha_composite(Image.fromarray(body, "RGBA"))

    # ---- 轮廓（柔和的一圈深色，不是黑边）
    ring = np.clip((d - 0.955) * 26, 0, 1) * np.clip((1.03 - d) * 26, 0, 1)
    ring_rgb = _hsv(p.hue_body + 0.02, 0.55, 0.28)
    edge = np.dstack([
        np.full((S, S), ring_rgb[0], np.float32),
        np.full((S, S), ring_rgb[1], np.float32),
        np.full((S, S), ring_rgb[2], np.float32),
        (ring * 150 * inside * 255 / 255).astype(np.float32),
    ]).astype(np.uint8)
    canvas.alpha_composite(Image.fromarray(edge, "RGBA"))

    # ---- 花纹（两侧的螺旋）
    if p.swirl > 0.05:
        deco = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        dd = ImageDraw.Draw(deco)
        col = _hsv(p.hue_body - 0.015, 0.18, 1.0, int(20 + 140 * p.swirl))
        w = max(2, int((2.5 + 7.0 * p.swirl) * ss))
        for sign in (-1, 1):
            bx = c + sign * R * 0.63
            by = c - R * 0.01
            pts = []
            for t in np.linspace(0.15, 3.9 * math.pi, 120):
                rr = R * 0.265 * (t / (3.9 * math.pi)) ** 0.78
                ang = t * sign + (0 if sign > 0 else math.pi * 0.0)
                pts.append((bx + sign * rr * math.cos(ang) * 0.92,
                            by + rr * math.sin(ang) * 1.25))
            dd.line(pts, fill=col, width=w, joint="curve")
            # 起笔处一个小圆点，让螺旋看起来是"画上去的"
            r0 = max(2, int(2.6 * ss))
            dd.ellipse([pts[0][0] - r0, pts[0][1] - r0, pts[0][0] + r0, pts[0][1] + r0], fill=col)
        deco = deco.filter(ImageFilter.GaussianBlur(0.8 * ss))
        # 只在球内
        clip = Image.fromarray((inside * 255).astype(np.uint8), "L")
        canvas.paste(deco, (0, 0), Image.fromarray(
            (np.asarray(clip, dtype=np.float32) * (np.asarray(deco.split()[-1], dtype=np.float32) / 255.0)).astype(np.uint8), "L"))

    # ---- 眼睛（两个立着的橙色椭圆）
    for sign in (-1, 1):
        ex = c + sign * R * p.eye_gap
        ey = c - R * 0.035
        eh = R * p.eye_h
        ew = eh * 0.62
        eye = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        ed = ImageDraw.Draw(eye)
        # 外圈（橙）
        ed.ellipse([ex - ew / 2, ey - eh / 2, ex + ew / 2, ey + eh / 2],
                   fill=_hsv(p.eye_hue, 0.62, 0.95))
        eye = eye.filter(ImageFilter.GaussianBlur(0.5 * ss))
        canvas.alpha_composite(eye)
        # 内芯（淡黄）
        inner = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        idr = ImageDraw.Draw(inner)
        ih, iw = eh * 0.74, ew * 0.55
        idr.ellipse([ex - iw / 2, ey - ih / 2 + eh * 0.06, ex + iw / 2, ey + ih / 2 + eh * 0.06],
                    fill=_hsv(p.eye_hue - 0.02, 0.22, 1.0, 235))
        canvas.alpha_composite(inner.filter(ImageFilter.GaussianBlur(0.8 * ss)))
        # 眼上沿一点更亮的反光
        hl = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        hd = ImageDraw.Draw(hl)
        hd.ellipse([ex - iw * 0.34, ey - ih * 0.30, ex + iw * 0.10, ey - ih * 0.02],
                   fill=(255, 255, 255, 90))
        canvas.alpha_composite(hl.filter(ImageFilter.GaussianBlur(1.4 * ss)))

    out = canvas.convert("RGB")
    if ss > 1:
        out = out.resize((p.size, p.size), Image.LANCZOS)
    return out


def caption(params: Params, note: str) -> Image.Image:
    """在图下面垫一条读数脚注（每个数都能追回到某次演算）。"""
    img = Image.new("RGB", (params.size, 44), (8, 14, 18))
    d = ImageDraw.Draw(img)
    d.text((10, 12), note, fill=(150, 176, 186))
    return img
