"""画像 v1：带遮罩和色阶的肖像。

v0 的问题：平涂叠多边形，像贴纸。这一版做三件事——
① 每个部件先画成遮罩，再按遮罩上两三个色阶；② 头发分层（后发→侧发→刘海）并互相投影；
③ 眼睛按「眼白 / 虹膜三环 / 上睑压低 / 两点高光」重画。

映射不变：k→发束与亮度，极差→刘海遮多少，稳定位数→眼神，defect→颗粒与背景。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .render import _font

INK = (240, 233, 232)


@dataclass(frozen=True)
class Portrait:
    size: int = 600
    hair_hue: float = 0.075
    hair_lit: float = 0.36
    skin: tuple = (247, 217, 202)
    skin_shadow: tuple = (209, 160, 148)
    eye_hue: float = 0.44
    iris_lit: float = 0.44
    cloth: tuple = (116, 42, 48)
    cloth_dark: tuple = (80, 28, 34)
    inner: tuple = (30, 34, 52)
    bg_top: tuple = (64, 34, 38)
    bg_bot: tuple = (22, 15, 19)
    bang: float = 0.55
    strands: int = 5
    gloss: float = 0.55
    grain: float = 0.12


def _hsv(h, s, v):
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h % 1.0, min(1, max(0, s)), min(1, max(0, v)))
    return int(r * 255), int(g * 255), int(b * 255)


def _mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _bez(p0, p1, p2, n=22):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
            for t in (i / n for i in range(n + 1))]


def params_from_readings(k: int, span: float, stable: float, defect: float) -> Portrait:
    bang = float(np.clip(0.55 + 0.10 * math.log10(max(span, 1e-6)), 0.25, 0.85))
    return Portrait(
        bang=bang,
        strands=int(np.clip(3 + (k - 2) * 0.6, 3, 12)),
        gloss=float(np.clip(0.25 + 0.12 * stable, 0.25, 0.95)),
        grain=float(np.clip(0.05 + 6.0 * defect, 0.05, 0.4)),
        hair_lit=float(np.clip(0.28 + 0.015 * min(k, 8), 0.28, 0.40)),
    )


def draw_portrait(p: Portrait) -> Image.Image:
    S = int(p.size)
    ss = 2
    W = S * ss
    cx, cy = W * 0.5, W * 0.46

    # ---------- 背景
    yy = np.linspace(0, 1, W, dtype=np.float32)[:, None]
    bg = np.zeros((W, W, 3), dtype=np.float32)
    for i in range(3):
        bg[..., i] = p.bg_top[i] + (p.bg_bot[i] - p.bg_top[i]) * yy ** 1.2
    vig = np.clip(1 - ((np.arange(W)[None, :] - cx) ** 2 + (np.arange(W)[:, None] - cy) ** 2)
                  / (W * W * 0.60), 0, 1)
    bg *= (0.50 + 0.50 * vig)[..., None]
    img = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), "RGB")

    hair_dark = _hsv(p.hair_hue, 0.62, p.hair_lit * 0.55)
    hair_mid = _hsv(p.hair_hue, 0.55, p.hair_lit)
    hair_lite = _hsv(p.hair_hue - 0.008, 0.30, p.hair_lit * 1.55)

    d = ImageDraw.Draw(img, "RGBA")

    # ---------- 衣服
    d.polygon([(-W * 0.05, W * 1.05), (W * 0.08, cy + W * 0.32), (W * 0.30, cy + W * 0.255),
               (W * 0.70, cy + W * 0.255), (W * 0.92, cy + W * 0.32), (W * 1.05, W * 1.05)],
              fill=p.cloth)
    for s in (-1, 1):
        d.polygon([(cx + s * W * 0.085, cy + W * 0.245), (cx + s * W * 0.30, cy + W * 0.27),
                   (cx + s * W * 0.255, cy + W * 0.42), (cx + s * W * 0.095, cy + W * 0.37)],
                  fill=p.cloth_dark)
    # 高领
    d.ellipse([cx - W * 0.082, cy + W * 0.20, cx + W * 0.082, cy + W * 0.30], fill=p.inner)
    d.rectangle([cx - W * 0.082, cy + W * 0.25, cx + W * 0.082, cy + W * 0.40], fill=p.inner)
    # 脖子
    d.rounded_rectangle([cx - W * 0.055, cy + W * 0.10, cx + W * 0.055, cy + W * 0.265],
                        radius=W * 0.025, fill=_mix(p.skin, p.skin_shadow, 0.5))

    # ---------- 后发：一条闭合曲线围出来的波波头（不是圆角矩形）
    hair_top = cy - W * 0.258

    def _half(sign: float):
        pts = [(cx + sign * W * 0.015, hair_top)]
        pts += _bez((cx + sign * W * 0.015, hair_top),
                    (cx + sign * W * 0.275, cy - W * 0.245),
                    (cx + sign * W * 0.248, cy + W * 0.035), 18)
        pts += _bez((cx + sign * W * 0.248, cy + W * 0.035),
                    (cx + sign * W * 0.242, cy + W * 0.245),
                    (cx + sign * W * 0.112, cy + W * 0.312), 18)
        return pts

    d.polygon(_half(-1) + _half(1)[::-1], fill=hair_dark)

    # ---------- 脸（遮罩 + 两阶色）
    fw, top_y, jaw_y, chin_y = W * 0.148, cy - W * 0.168, cy + W * 0.035, cy + W * 0.198
    face_mask = Image.new("L", (W, W), 0)
    fm = ImageDraw.Draw(face_mask)
    fm.ellipse([cx - fw, top_y, cx + fw, cy + W * 0.140], fill=255)
    fm.polygon([(cx - fw * 0.97, jaw_y), (cx + fw * 0.97, jaw_y),
                (cx + W * 0.070, chin_y), (cx - W * 0.070, chin_y)], fill=255)
    fm.ellipse([cx - W * 0.070, chin_y - W * 0.045, cx + W * 0.070, chin_y + W * 0.012], fill=255)

    skin = np.zeros((W, W, 3), dtype=np.float32)
    skin[...] = p.skin
    # 体积：右上暗、左下亮
    gx = (np.arange(W)[None, :] - cx) / (fw * 1.6)
    gy = (np.arange(W)[:, None] - cy) / (W * 0.22)
    sh = np.clip(0.55 + 0.45 * (-gx * 0.55 - gy * 0.35), 0, 1)
    # 赛璐璐：把连续渐变量化成三级硬色阶，交界才有「画」的感觉
    q = np.where(sh > 0.74, 1.00, np.where(sh > 0.46, 0.88, 0.74))[..., None]
    warm = np.array(p.skin_shadow, dtype=np.float32)
    skin = np.array(p.skin, dtype=np.float32) * q
    skin = np.where(q < 0.80, skin * 0.55 + warm * 0.45, skin)
    face_img = Image.fromarray(np.clip(skin, 0, 255).astype(np.uint8), "RGB")
    img.paste(face_img, (0, 0), face_mask.filter(ImageFilter.GaussianBlur(ss)))

    # ---------- 脸的三个结构：颧骨脊、下颌转折、鼻侧阴影（都是硬色块）
    d.polygon([(cx - fw * 0.92, cy + W * 0.010), (cx - fw * 0.58, cy + W * 0.030),
               (cx - fw * 0.62, cy + W * 0.075), (cx - fw * 0.95, cy + W * 0.050)],
              fill=p.skin_shadow + (52,))
    d.line([(cx - fw * 0.80, jaw_y + W * 0.012), (cx - W * 0.052, chin_y - W * 0.012)],
           fill=p.skin_shadow + (72,), width=max(2, int(W * 0.004)))
    d.line([(cx - W * 0.020, cy - W * 0.005), (cx - W * 0.028, cy + W * 0.070)],
           fill=p.skin_shadow + (58,), width=max(3, int(W * 0.006)))

    # ---------- 眼睛
    ey = cy + W * 0.030
    for s in (-1, 1):
        ex = cx + s * W * 0.070
        ew, eh = W * 0.056, W * 0.042
        # 眼白：杏仁形（上弧平、下弧圆）
        top_l, top_r = (ex - ew, ey), (ex + ew, ey - eh * 0.15)
        shape = [(top_l[0], ey - eh * 0.55)]
        shape += _bez((top_l[0], ey - eh * 0.55), (ex, ey - eh * 1.35), (top_r[0], ey - eh * 0.85), 18)
        shape += _bez((top_r[0], ey - eh * 0.85), (ex + ew * 0.75, ey + eh * 0.95),
                      (ex - ew * 0.15, ey + eh * 0.90), 18)
        shape += _bez((ex - ew * 0.15, ey + eh * 0.90), (ex - ew * 0.85, ey + eh * 0.85),
                      (top_l[0], ey - eh * 0.55), 18)
        d.polygon(shape, fill=(251, 249, 246))
        # 虹膜三环
        ir = W * 0.028
        d.ellipse([ex - ir, ey - ir * 1.02, ex + ir, ey + ir * 1.02], fill=_hsv(p.eye_hue, 0.62, p.iris_lit * 0.55))
        d.ellipse([ex - ir * 0.86, ey - ir * 0.88, ex + ir * 0.86, ey + ir * 0.88],
                  fill=_hsv(p.eye_hue, 0.50, p.iris_lit))
        d.ellipse([ex - ir * 0.62, ey + ir * 0.02, ex + ir * 0.62, ey + ir * 0.92],
                  fill=_hsv(p.eye_hue, 0.36, p.iris_lit * 1.45))
        pr = W * 0.011
        d.ellipse([ex - pr, ey - pr * 1.15, ex + pr, ey + pr * 1.15], fill=(24, 30, 34))
        hr = W * 0.010
        a = int(140 + 115 * p.gloss)
        d.ellipse([ex - ir * 0.52 - hr, ey - ir * 0.62 - hr, ex - ir * 0.52 + hr, ey - ir * 0.62 + hr],
                  fill=(255, 255, 255, a))
        # 上睑：一块深色盖住上半（这块决定「眼神」）
        lid = [(ex - ew * 1.05, ey - eh * 0.95)]
        lid += _bez((ex - ew * 1.05, ey - eh * 0.95), (ex, ey - eh * 2.0), (ex + ew * 1.08, ey - eh * 0.30), 18)
        lid += [(ex + ew * 1.08, ey - eh * 1.6), (ex - ew * 1.05, ey - eh * 1.6)]
        d.polygon(lid, fill=_mix(p.skin, (56, 38, 34), 0.55))
        d.line(lid[1:len(lid) - 2] if len(lid) > 3 else lid, fill=(48, 32, 28), width=max(4, int(W * 0.009)))
        d.line([(ex + s * ew * 1.02, ey - eh * 0.35), (ex + s * ew * 1.42, ey - eh * 0.95)],
               fill=(48, 32, 28), width=max(3, int(W * 0.008)))
        d.arc([ex - ew, ey - eh * 0.55, ex + ew, ey + eh * 1.05], 25, 155,
              fill=(132, 96, 86), width=max(1, int(W * 0.003)))
        # 眉
        by = ey - W * 0.062
        d.line([(ex - s * ew * 0.9, by + W * 0.008), (ex + s * ew * 1.1, by - W * 0.010)],
               fill=(70, 46, 34), width=max(3, int(W * 0.0085)))

    # ---------- 鼻、嘴
    d.ellipse([cx - W * 0.004, cy + W * 0.072, cx + W * 0.009, cy + W * 0.084],
              fill=_mix(p.skin, p.skin_shadow, 0.75))
    d.line([(cx - W * 0.020, cy + W * 0.132), (cx + W * 0.020, cy + W * 0.132)],
           fill=(148, 88, 84), width=max(2, int(W * 0.005)))

    # ---------- 侧发（只贴着脸的外缘，不压到脸颊）
    for s in (-1, 1):
        pts = [(cx + s * W * 0.150, cy - W * 0.185)]
        pts += _bez((cx + s * W * 0.150, cy - W * 0.185), (cx + s * W * 0.212, cy - W * 0.02),
                    (cx + s * W * 0.175, cy + W * 0.245), 22)
        pts += [(cx + s * W * 0.132, cy + W * 0.215), (cx + s * W * 0.140, cy - W * 0.09)]
        d.polygon(pts, fill=hair_mid)

    # ---------- 刘海：弯的、盖到 edge，并往额头投一片阴影
    left, right = cx - W * 0.150, cx + W * 0.150
    top = cy - W * 0.235
    edge = cy - W * 0.055 + (W * 0.185) * p.bang
    n = max(4, p.strands)
    # 刘海：一束主发 + 两束辅发（不均分，主流向右侧斜），每片有厚度
    weights = (0.52, 0.30, 0.18)
    cursor = 0.0
    for i, wgt in enumerate(weights):
        x0 = left + (right - left) * (cursor + 0.04)
        x1 = x0 + (right - left) * wgt * 1.02
        cursor += wgt
        tip_y = edge - W * 0.030 + W * 0.070 * i
        tip = ((x0 + x1) / 2 + W * 0.045 * (i + 1), tip_y)
        out_pts = [(x0 - W * 0.015, top - W * 0.015)]
        out_pts += _bez((x0, top), (x0 + W * 0.020, tip_y - W * 0.045), tip, 14)
        out_pts += _bez(tip, (x1 - W * 0.020, tip_y - W * 0.050), (x1, top), 14)
        out_pts += [(x1 + W * 0.015, top - W * 0.015)]
        d.polygon(out_pts, fill=hair_mid)
        in_pts = [(x0 + W * 0.010, top - W * 0.010)]
        in_pts += _bez((x0 + W * 0.018, top), (x0 + W * 0.030, tip_y - W * 0.050),
                       (tip[0] - W * 0.004, tip_y - W * 0.014), 12)
        in_pts += _bez((tip[0] - W * 0.004, tip_y - W * 0.014),
                       (x1 - W * 0.028, tip_y - W * 0.056), (x1 - W * 0.016, top), 12)
        d.polygon(in_pts, fill=_mix(hair_mid, hair_lite, 0.5))
    # 头顶：从发旋往两边分（不是一顶扁帽子）
    crown = (cx, cy - W * 0.252)
    head = _bez((cx - W * 0.168, cy - W * 0.095), (cx - W * 0.175, cy - W * 0.255), crown, 16)
    head += _bez(crown, (cx + W * 0.175, cy - W * 0.255), (cx + W * 0.168, cy - W * 0.095), 16)
    d.polygon([(cx - W * 0.168, cy - W * 0.090)] + head + [(cx + W * 0.168, cy - W * 0.090)],
              fill=hair_mid)
    d.arc([cx - W * 0.150, cy - W * 0.245, cx + W * 0.015, cy - W * 0.100],
          205, 350, fill=hair_lite + (140,), width=max(3, int(W * 0.007)))

    # ---------- 颗粒
    if p.grain > 0.02:
        rng = np.random.default_rng(11)
        arr = np.asarray(img).astype(np.int16)
        arr = np.clip(arr + rng.normal(0, 26 * p.grain, arr.shape[:2])[..., None], 0, 255)
        img = Image.fromarray(arr.astype(np.uint8), "RGB")

    out = img.resize((S, S), Image.LANCZOS)
    ImageDraw.Draw(out).text((12, 8), "望云 · 画像", fill=INK, font=_font(14))
    return out
