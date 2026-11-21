"""生成桌面图标 `assets/wangyun.ico`：五个色相点 + 几条淡淡的线，和一帧场面同构。

只用 Pillow。改配色/布局后重跑：`python tools\\make_icon.py`。
"""

from __future__ import annotations

import colorsys
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZE = 256
BG = (10, 26, 26)
HUES = [0.085, 0.50, 0.78, 0.28, 0.97]  # 橙 / 青 / 紫 / 绿 / 粉
RATIOS = [0.22, 0.17, 0.20, 0.15, 0.13]


def rgb(h: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hls_to_rgb(h, 0.62, 0.55)
    return int(r * 255), int(g * 255), int(b * 255)


def main() -> int:
    img = Image.new("RGBA", (SIZE, SIZE), BG + (255,))

    # 圆角底板
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle([6, 6, SIZE - 6, SIZE - 6], radius=54, fill=255)
    img.putalpha(mask)

    dots = []
    for i, (hue, ratio) in enumerate(zip(HUES, RATIOS)):
        ang = -math.pi / 2 + i * (2 * math.pi / len(HUES))
        c = SIZE / 2
        r = SIZE * 0.28
        dots.append((c + r * math.cos(ang), c + r * math.sin(ang), SIZE * ratio, hue))

    # 淡淡的线（先画，压在点下面）
    lines = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lines)
    order = [0, 3, 1, 4, 2, 0]
    for a, b in zip(order, order[1:]):
        pa, pb = dots[a], dots[b]
        ld.line([pa[0], pa[1], pb[0], pb[1]], fill=(127, 214, 208, 90), width=3)
    img = Image.alpha_composite(img, lines.filter(ImageFilter.GaussianBlur(1.2)))

    draw = ImageDraw.Draw(img, "RGBA")
    for x, y, r, hue in dots:
        col = rgb(hue)
        # 外晕
        glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.ellipse([x - r * 1.5, y - r * 1.5, x + r * 1.5, y + r * 1.5], fill=col + (70,))
        img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(r * 0.55)))
        draw = ImageDraw.Draw(img, "RGBA")
        draw.ellipse([x - r, y - r, x + r, y + r], fill=col + (235,))
        draw.ellipse(
            [x - r * 0.92, y - r * 0.92, x + r * 0.92, y + r * 0.92],
            outline=(255, 255, 255, 28), width=max(1, int(r * 0.10)),
        )

    out = Path(__file__).resolve().parent.parent / "assets" / "wangyun.ico"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(f"写好了：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
