"""视觉系统：只吃群表（和一张 k×k 的群间强度），不读关系网络。

两核之间只有一张群表。渲染器看不见 n² 条关系，也看不见网络长什么样——
它只按 (群, 成员数, 数值, 稳定度, 年龄) 和群与群之间有多紧来画东西。
这就是「减少支出」的真正机制：成本是 k²，不是 n²。

两个模糊通道（DESIGN.md §4），长得必须不一样：

    A 型  还没算到 / 还没稳住   → 点阵、抖动的填充。含义：别信这里。
    B 型  视角太粗，细节低于分辨率 → 干净平滑的轮廓。含义：可信，但这就是这个尺度上的全部信息。

混在一起就是撒谎：用户会分不清「我不知道」和「我知道到这里为止」。

另外，全局视图明确给出**轮廓与洞**——那是缩放后真正留下来的拓扑，不是糊掉的画面。
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core.layout import Shape

BG = (14, 16, 22)
INK = (232, 236, 244)
DIM = (110, 118, 134)
SETTLED = (96, 208, 214)  # 稳住了的群
CHURN = (240, 160, 72)  # 还在动的群
EDGE = (74, 88, 118)  # 群与群之间的关系
HILITE = (250, 214, 110)  # 这一帧刚刚并/分过的群
# 一张几百像素的图里放不下上千个团：超过这个数就不画点阵了，画雾并写明。
# 和 layout 的 SPRING_MAX_K 是同一条口径——这个尺度上没有可读结构。
RENDER_MAX_BLOBS = 300
HOLE = (28, 32, 44)

_FONT_FILES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/Deng.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


@lru_cache(maxsize=16)
def _font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """有中文字体就用，没有就退回位图字体（标题里的汉字会变成方块，但不会崩）。"""
    for path in _FONT_FILES:
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            continue
    return ImageFont.load_default()


# --------------------------------------------------------------------- 几何


def _canvas_box(size: int) -> tuple[float, float, float, float]:
    """画布上真正能用的那块：上面留给标题，下面留给读数。"""
    return 6.0, 34.0, float(size - 6), float(size - 58)


def _radii(shape: Shape, size: int, fill: float = 0.18) -> np.ndarray:
    """每个群的半径：面积正比于成员数。

    一个单位的面积是常数，所以群合并时面积相加、半径按 √k 长——并起来看得见。
    """
    counts = np.array([g.count for g in shape.groups], dtype=np.float64)
    total = max(float(counts.sum()), 1.0)
    l, t, r, b = _canvas_box(size)
    unit_area = fill * (r - l) * (b - t) / total
    return np.sqrt(counts * unit_area / np.pi)


def _separate(pos: np.ndarray, radii: np.ndarray, strength: np.ndarray | None = None) -> np.ndarray:
    """把群摆开，再缩放回画布里。顺序固定，所以还是确定的。

    推开多少看关系强度：关系强的一对允许贴在一起，甚至重叠——那个尺度上它们
    本来就是同一块形；关系弱的必须分开。所以轮廓本身就是关系的读数。

    整个过程一遍矩阵算完：k 可以到几千，逐对 Python 循环跑不动。
    """
    p = np.array(pos, dtype=np.float64, copy=True)
    k = p.shape[0]
    if k < 2:
        return p

    r = np.asarray(radii, dtype=np.float64)
    w = strength
    if w is None or w.size == 0:
        wmax = 1.0
        a = np.zeros((k, k), dtype=np.float64)
    else:
        wmax = float(w.max()) or 1.0
        a = np.clip(np.asarray(w, dtype=np.float64) / wmax, 0.0, 1.0)
    factor = 1.12 - 0.40 * a
    need = (r[:, None] + r[None, :]) * factor
    np.fill_diagonal(need, 0.0)

    eye = np.eye(k, dtype=bool)
    for _ in range(80):
        delta = p[:, None, :] - p[None, :, :]
        dist = np.sqrt((delta * delta).sum(-1)) + 1e-12
        gap = need - dist
        over = (gap > 0) & ~eye
        if not over.any():
            break
        push = 0.5 * gap
        dirs = delta / dist[..., None]
        step = (dirs * push[..., None]) * over[..., None]
        p += step.sum(axis=1)
    p -= p.mean(axis=0)
    span = float(np.abs(p).max())
    if span > 0:
        p = p / span
    return p


def _blobs(shape: Shape, size: int) -> list[tuple[int, int, int, bool]]:
    """把群表摊成一组圆：(cx, cy, r, 是否稳住)。位置来自关系强度，不是 id。"""
    k = len(shape.groups)
    if k == 0:
        return []

    radii = _radii(shape, size)
    l, t, r_, b = _canvas_box(size)
    cx0, cy0 = (l + r_) * 0.5, (t + b) * 0.5
    span = max(min((r_ - l) * 0.5, (b - t) * 0.5) - float(radii.max()) - 3.0, size * 0.06)
    if k == 1:
        return [(int(cx0), int(cy0), int(radii[0]), shape.groups[0].settled)]

    pos = _separate(shape.pos, radii / max(span, 1.0), shape.strength)
    out = []
    for g, (ux, uy), r in zip(shape.groups, pos, radii):
        cx = int(cx0 + ux * span)
        cy = int(cy0 + uy * span)
        out.append((cx, cy, max(int(r), 3), g.settled))
    return out


def _mask_from_blobs(blobs, size: int, ss: int = 2) -> np.ndarray:
    img = Image.new("L", (size * ss, size * ss), 0)
    d = ImageDraw.Draw(img)
    for cx, cy, r, _ in blobs:
        d.ellipse(
            [cx * ss - r * ss, cy * ss - r * ss, cx * ss + r * ss, cy * ss + r * ss],
            fill=255,
        )
    small = img.resize((size, size), Image.LANCZOS)
    return np.asarray(small) > 96


# --------------------------------------------------------------------- 拓扑


def _components(mask: np.ndarray) -> tuple[int, list[int]]:
    """四邻接的连通块。返回 (块数, 每块大小)。"""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            q = deque([(y, x)])
            seen[y, x] = True
            n = 0
            while q:
                cy, cx = q.popleft()
                n += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            sizes.append(n)
    return len(sizes), sizes


def holes(mask: np.ndarray, min_area: int = 24) -> tuple[int, np.ndarray]:
    """洞 = 背景里从画布外面淹不到的那部分。这就是缩放后留下来的拓扑。"""
    bg = ~mask
    outside = np.zeros_like(bg, dtype=bool)
    h, w = mask.shape
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if bg[y, x] and not outside[y, x]:
                outside[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if bg[y, x] and not outside[y, x]:
                outside[y, x] = True
                q.append((y, x))
    while q:
        cy, cx = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and bg[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                q.append((ny, nx))

    inner = bg & ~outside
    count, sizes = _components(inner)
    keep = np.zeros_like(inner)
    # 太小的洞不算洞，算噪点。
    seen = np.zeros_like(inner, dtype=bool)
    kept = 0
    for y in range(h):
        for x in range(w):
            if not inner[y, x] or seen[y, x]:
                continue
            q = deque([(y, x)])
            seen[y, x] = True
            cell = []
            while q:
                cy, cx = q.popleft()
                cell.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and inner[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(cell) >= min_area:
                kept += 1
                for cy, cx in cell:
                    keep[cy, cx] = True
    return kept, keep


def _edge(mask: np.ndarray) -> np.ndarray:
    return mask & ~(
        np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1)
    )


# --------------------------------------------------------------------- 画一张


def _too_fine(size: int, title: str, k: int, need: int) -> tuple[Image.Image, dict]:
    """k 太多，这个尺度上画不出形状——画一层雾，并明说是雾。

    这是 A 型模糊的极端：不是「可信但到这里为止」（B 型），而是「这里根本还没成形状」。
    上千个团在几百像素里，不管怎么排都只是噪点；与其花十几秒画一堆点假装是结构，
    不如老实说「视角太细」，让人去收视角。
    """
    img = Image.new("RGB", (size, size), BG)
    rng = np.random.default_rng(0)
    d = ImageDraw.Draw(img)

    # 雾：均匀的细点，密度随 k 涨，不画任何线——这里没有「谁连着谁」可看。
    area = size * size
    n_dots = int(np.clip(k * 6, 200, area // 40))
    xs = rng.integers(8, size - 8, size=n_dots)
    ys = rng.integers(8, size - 60, size=n_dots)
    for x, y in zip(xs.tolist(), ys.tolist()):
        d.point((x, y), fill=CHURN)

    if title:
        d.text((12, 8), title, fill=INK, font=_font(15))
    # 字压在雾上读不出来，先垫一块底。
    d.rectangle([size // 2 - 118, size // 2 - 34, size // 2 + 118, size // 2 + 36], fill=BG)
    d.text((size // 2, size // 2 - 10), "视角太细", fill=INK, font=_font(20), anchor="mm")
    d.text(
        (size // 2, size // 2 + 18),
        f"{k} 个团，这个尺度上画不出形状",
        fill=DIM,
        font=_font(13),
        anchor="mm",
    )
    d.rectangle([0, size - 50, size, size], fill=BG)
    d.text((12, size - 44), f"k={k}   holes=—", fill=DIM, font=_font(12))
    d.text((12, size - 27), "这里没有读数：先收视角", fill=DIM, font=_font(12))
    d.text((size - 12, size - 27), "绘制降级（k 太大）", fill=CHURN, font=_font(12), anchor="ra")
    return img, {"k": k, "holes": -1, "settled": 0, "blobs": [], "degraded": True}


def draw_shape(
    shape: Shape,
    size: int = 420,
    title: str = "",
    need: int = 8,
    seed: int = 0,
    highlight: list[int] | None = None,
) -> tuple[Image.Image, dict]:
    """按一张视图画一张。返回图片和它自己数出来的拓扑读数（可写进测试）。

    `highlight` 是这一帧要指出来的群下标（并/分刚发生在它们身上）——圈一圈，
    别指望读者自己在两百个点里找哪两个刚合了。
    """
    if shape.k > RENDER_MAX_BLOBS:
        return _too_fine(size, title, shape.k, need)

    img = Image.new("RGB", (size, size), BG)
    rng = np.random.default_rng(seed)
    px = img.load()

    blobs = _blobs(shape, size)
    groups = shape.groups
    draw = ImageDraw.Draw(img)

    # B 型：视角太粗 → 平滑。合并后的整体轮廓由 mask 统一描，不逐群描。
    for g, (cx, cy, r, settled) in zip(groups, blobs):
        if settled:
            # 稳住了：实心。可信，这就是这个尺度上的全部信息。
            for y in range(max(0, cy - r), min(size, cy + r + 1)):
                dy = y - cy
                for x in range(max(0, cx - r), min(size, cx + r + 1)):
                    dx = x - cx
                    if dx * dx + dy * dy <= r * r:
                        px[x, y] = SETTLED
        else:
            # A 型：还没稳住 → 点阵。别信这里。
            density = float(np.clip(g.stable / max(need, 1), 0.10, 0.90))
            for y in range(max(0, cy - r), min(size, cy + r + 1)):
                dy = y - cy
                for x in range(max(0, cx - r), min(size, cx + r + 1)):
                    dx = x - cx
                    if dx * dx + dy * dy > r * r:
                        continue
                    if rng.random() < density:
                        px[x, y] = CHURN

    # 关系画在纹理之上：点阵是虚的，关系线要穿过它才看得见——「谁连着谁」比「谁在哪」更硬。
    if len(groups) > 1:
        wmax = float(shape.strength.max()) or 1.0
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                w = float(shape.strength[i, j])
                if w <= 0.0:
                    continue
                a = w / wmax
                col = tuple(int(BG[c] + (EDGE[c] - BG[c]) * a) for c in range(3))
                draw.line([blobs[i][:2], blobs[j][:2]], fill=col, width=1 + int(a > 0.5))

    mask = _mask_from_blobs(blobs, size)
    n_holes, hole_mask = holes(mask)
    edge = _edge(mask)
    ys, xs = np.nonzero(edge)
    for y, x in zip(ys.tolist(), xs.tolist()):
        px[x, y] = INK
    ys, xs = np.nonzero(hole_mask & _edge(hole_mask))
    for y, x in zip(ys.tolist(), xs.tolist()):
        px[x, y] = DIM

    if highlight:
        # 并/分刚发生在谁身上，就圈谁。图自己会说话的地方别留给文字。
        # 画在描边之后——先画会被轮廓盖掉，圈了等于没圈。
        for gi in highlight:
            if 0 <= gi < len(blobs):
                cx, cy, r, _ = blobs[gi]
                draw.ellipse(
                    [cx - r - 5, cy - r - 5, cx + r + 5, cy + r + 5],
                    outline=HILITE,
                    width=3,
                )

    if title:
        draw.text((12, 8), title, fill=INK, font=_font(15))
    settled_n = sum(1 for g in groups if g.settled)
    draw.rectangle([0, size - 50, size, size], fill=BG)
    draw.text((12, size - 44), f"k={len(groups)}   holes={n_holes}", fill=DIM, font=_font(12))
    draw.text((12, size - 27), f"稳住了 {settled_n}/{len(groups)}", fill=DIM, font=_font(12))
    if shape.degraded:
        # 位置是占位的，不是解出来的：这时候的眼睛只能告诉你「有多少团」，
        # 不能告诉你「谁在谁旁边」。说清楚，别让人拿占位当结构。
        draw.text((size - 12, size - 27), "布局降级（k 太大）", fill=CHURN, font=_font(12), anchor="ra")
    return img, {
        "k": len(groups),
        "holes": int(n_holes),
        "settled": settled_n,
        "blobs": [(int(cx), int(cy), int(r)) for cx, cy, r, _ in blobs],
    }


# --------------------------------------------------------------------- 曲线


def plot_series(
    series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    size: tuple[int, int] = (420, 240),
    title: str = "",
    log: bool = False,
) -> Image.Image:
    """没有 matplotlib，就自己画。够用就行。"""
    w, h = size
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    pad_l, pad_b, pad_t = 52, 28, 46
    if title:
        d.text((12, 8), title, fill=INK, font=_font(15))
    if not series:
        return img

    def tr(ys: np.ndarray) -> np.ndarray:
        ys = np.asarray(ys, dtype=float)
        if log:
            return np.log10(np.maximum(ys, 1e-300))
        return ys

    ys_all = np.concatenate([tr(s[1]) for s in series])
    y0, y1 = float(np.min(ys_all)), float(np.max(ys_all))
    if y1 - y0 < 1e-12:
        y1 = y0 + 1.0

    d.rectangle([pad_l, pad_t, w - 12, h - pad_b], outline=(52, 58, 72))
    for si, (name, ys, color) in enumerate(series):
        raw = np.asarray(ys, dtype=float)
        n = len(raw)
        if n < 2:
            continue
        vals = tr(raw)
        pts = []
        for i, val in enumerate(vals):
            x = pad_l + (w - 12 - pad_l) * i / (n - 1)
            y = h - pad_b - (h - pad_b - pad_t) * (float(val) - y0) / (y1 - y0)
            pts.append((x, y))
        d.line(pts, fill=color, width=2)
        d.text((pad_l + 8, pad_t + 6 + 16 * si), name, fill=color, font=_font(12))
    fmt = (lambda v: f"1e{int(round(v))}") if log else (lambda v: f"{v:.4g}")
    d.text((4, pad_t - 14), fmt(y1), fill=DIM, font=_font(11))
    d.text((4, h - pad_b - 4), fmt(y0), fill=DIM, font=_font(11))
    return img


def sheet(panels: list[Image.Image], cols: int = 3, gap: int = 10) -> Image.Image:
    """拼版。行高按该行最高的那块算，不留黑边。"""
    if not panels:
        raise ValueError("没有面板")
    rows: list[list[Image.Image]] = []
    for i in range(0, len(panels), cols):
        rows.append(panels[i : i + cols])
    col_w = [0] * cols
    for row in rows:
        for c, p in enumerate(row):
            col_w[c] = max(col_w[c], p.width)
    row_h = [max(p.height for p in row) for row in rows]
    W = sum(col_w) + (cols + 1) * gap
    H = sum(row_h) + (len(rows) + 1) * gap
    out = Image.new("RGB", (W, H), (8, 9, 13))
    y = gap
    for r, row in enumerate(rows):
        x = gap
        for c, p in enumerate(row):
            out.paste(p, (x, y))
            x += col_w[c] + gap
        y += row_h[r] + gap
    return out
