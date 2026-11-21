"""一屏看全：把五层读数拼到一张图上。

望云的读数已经分成五层：关系（负荷、寿命）、单位（值、位数、残差、活性）、
团（k、span、事件）、跨尺度（LOD 粗化）、数值表示（float64 / 整数）。
以前每层各一张表，读的人得自己在脑子里对齐。这一张把它们按顺序摆好：

    形 | 关系寿命 | 负荷
    k(t) | 违约读数 | 稳定位数与活性
    读数条

上面三块回答「形状长什么样、谁在干活、谁早死谁晚死」，
中间三块回答「它怎么走到这儿的」，下面一条是把关键数字写清楚。

这些是**读数**，不是结论：图上每一块的边界和口径都写在 DESIGN 里，
拿它下判断之前先回去看那一节说的是什么。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from wy.core.engine import Engine
from wy.core.layout import make_shape
from wy.core.network import Config, Network
from wy.render import (
    BG,
    CHURN,
    DIM,
    EDGE,
    INK,
    SETTLED,
    _font,
    draw_shape,
    plot_series,
)
from wy.core.shape import StabilityMonitor, group_table, k_of, k_curve, plateau, scale_of

PAD = 12


def collect(cfg: Config, ticks: int, window: int = 100, need: int = 4) -> dict:
    """跑一遍并把五层读数收齐。"""
    net = Network(cfg)
    eng = Engine(net)
    mon = StabilityMonitor(net.n, cfg.window, cfg.digits)

    ks: list[int] = []
    defect: list[float] = []
    stable: list[float] = []
    alive: list[float] = []

    frames: list[np.ndarray] = []
    prev_load = np.zeros(net.nbrs.shape, dtype=np.float64)
    prev_v = np.array(eng.v, dtype=np.float64)

    for _ in range(ticks):
        eng.step()
        mon.push(eng.v)
        live = eng.recent(cfg.window)
        fv = eng.v
        sc = scale_of(fv)
        ks.append(k_of(fv, net, 0.02 * sc))
        defect.append(eng.defect())
        stable.append(mon.mean_length(live))
        moved = (fv != prev_v) & live
        alive.append(float(moved.sum() / max(int(live.sum()), 1)))
        prev_v = np.array(fv, dtype=np.float64)

        if len(ks) % window == 0:
            cum = np.array(eng.load, dtype=np.float64)
            frames.append((cum - prev_load).ravel())
            prev_load = cum

    fv = eng.v
    sc = scale_of(fv)
    fracs = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
    curves = k_curve(fv, net, fracs)
    groups = group_table(fv, net, 0.01 * sc, mon, need=need, touched=eng.recent(cfg.window))
    shape = make_shape(groups, net, seed=cfg.seed)

    mod_of = np.arange(net.n) % cfg.modules
    is_inter = (mod_of[:, None] != mod_of[net.nbrs]).ravel()
    ratio = np.asarray(net.ratio, dtype=np.float64)

    return {
        "net": net,
        "eng": eng,
        "cfg": cfg,
        "ticks": ticks,
        "window": window,
        "need": need,
        "series": {"k": np.array(ks), "defect": np.array(defect),
                   "stable": np.array(stable), "alive": np.array(alive)},
        "load_frames": np.stack(frames) if frames else np.zeros((0, net.nbrs.size)),
        "is_inter": is_inter,
        "ratio": ratio,
        "curves": curves,
        "fracs": fracs,
        "plateau": plateau(fracs, curves),
        "groups": groups,
        "shape": shape,
        "scale": float(sc),
    }


def _load_panel(r: dict, size: tuple[int, int]) -> Image.Image:
    """负荷：每窗口跨群份额 + 总量（两张小图并排，各自纵轴）。"""
    M = r["load_frames"]
    w, h = size
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.text((12, 8), "负荷：谁在干活", fill=INK, font=_font(15))
    if M.size == 0:
        d.text((12, 40), "窗口比 tick 还长", fill=DIM, font=_font(12))
        return img

    tot = M.sum(axis=1)
    share = M[:, r["is_inter"]].sum(axis=1) / np.maximum(tot, 1e-300)
    W = len(share)

    # 左：跨群份额（0~100%）
    box_l = (PAD, 56, PAD + (w - 3 * PAD) // 2, h - 34)
    d.rectangle(box_l, outline=(52, 58, 72))
    d.text((box_l[0], 40), "跨群份额（0~100%）", fill=DIM, font=_font(11))
    if W >= 2:
        pts = [(box_l[0] + (box_l[2] - box_l[0]) * i / (W - 1),
                box_l[3] - (box_l[3] - box_l[1]) * float(s)) for i, s in enumerate(share)]
        d.line(pts, fill=SETTLED, width=2)
    d.text((box_l[0] + 4, box_l[3] + 4), f"末窗 {share[-1]:.0%}", fill=SETTLED, font=_font(11))

    # 右：搬运总量（对数）
    box_r = (box_l[2] + PAD, 56, w - PAD, h - 34)
    d.rectangle(box_r, outline=(52, 58, 72))
    d.text((box_r[0], 40), "搬运总量（对数）", fill=DIM, font=_font(11))
    lt = np.log10(np.maximum(tot, 1e-300))
    lo, hi = float(lt.min()), float(lt.max())
    if hi - lo < 1e-9:
        hi = lo + 1.0
    if W >= 2:
        pts = [(box_r[0] + (box_r[2] - box_r[0]) * i / (W - 1),
                box_r[3] - (box_r[3] - box_r[1]) * (float(v) - lo) / (hi - lo))
               for i, v in enumerate(lt)]
        d.line(pts, fill=CHURN, width=2)
    d.text((box_r[2] - 34, box_r[1] + 2), f"1e{int(round(hi))}", fill=DIM, font=_font(10))
    d.text((box_r[2] - 34, box_r[3] - 14), f"1e{int(round(lo))}", fill=DIM, font=_font(10))
    d.text((box_r[0] + 4, box_r[3] + 4), f"末窗 {tot[-1]:.3g}", fill=CHURN, font=_font(11))
    return img


def _life_panel(r: dict, size: tuple[int, int]) -> Image.Image:
    """关系寿命：每条关系的「重心时刻」，群内一条带、群外一条带。"""
    M = r["load_frames"]
    w, h = size
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.text((12, 8), "关系寿命：重心在什么时候", fill=INK, font=_font(15))
    if M.size == 0:
        return img

    W = M.shape[0]
    center_t = (np.arange(W) + 1) * r["window"]
    tot = M.sum(axis=0)
    live = tot > 0
    center = np.where(live, (M * center_t[:, None]).sum(axis=0) / np.maximum(tot, 1e-300), np.nan)

    box = (PAD, 52, w - PAD, h - 34)
    d.rectangle(box, outline=(52, 58, 72))
    x0, x1 = box[0], box[2]
    bands = [(r["is_inter"], CHURN, "群外"), (~r["is_inter"], SETTLED, "群内")]
    rng = np.random.default_rng(3)
    for bi, (mask, color, name) in enumerate(bands):
        y = box[1] + 34 + bi * ((box[3] - box[1] - 46) // 2)
        d.text((x0 + 4, y - 24), name, fill=color, font=_font(12))
        idx = np.flatnonzero(mask & live)
        xs = center[idx] / max(r["ticks"], 1)
        ys = y + rng.uniform(-5, 5, size=xs.size)
        for x, yy in zip(xs, ys):
            px = x0 + (x1 - x0 - 8) * float(x) + 4
            d.ellipse([px - 1.8, yy - 1.8, px + 1.8, yy + 1.8], fill=color)
        if xs.size:
            d.text((x1 - 96, y - 20), f"中位 {np.median(center[idx]):.0f}", fill=DIM, font=_font(11))

    d.text((x0 + 4, box[3] + 4), f"t=0", fill=DIM, font=_font(10))
    d.text((x1 - 70, box[3] + 4), f"t={r['ticks']}", fill=DIM, font=_font(10))
    return img


def _digits_panel(r: dict, size: tuple[int, int]) -> Image.Image:
    """稳定位数走线，活性在底部做一条细带——两条各有各的纵轴，不硬凑在一起。"""
    w, h = size
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.text((12, 8), "稳定位数 / 活性", fill=INK, font=_font(15))
    pad_l, pad_t, pad_b = 52, 46, 52
    box = (pad_l, pad_t, w - 12, h - pad_b)
    d.rectangle(box, outline=(52, 58, 72))

    ys = np.asarray(r["series"]["stable"], dtype=float)
    lo, hi = float(ys.min()), float(ys.max())
    if hi - lo < 1e-9:
        hi = lo + 1.0
    n = len(ys)
    if n >= 2:
        pts = [(box[0] + (box[2] - box[0]) * i / (n - 1),
                box[3] - (box[3] - box[1]) * (float(v) - lo) / (hi - lo)) for i, v in enumerate(ys)]
        d.line(pts, fill=SETTLED, width=2)
    d.text((4, pad_t - 14), f"{hi:.2f}", fill=DIM, font=_font(11))
    d.text((4, box[3] - 4), f"{lo:.2f}", fill=DIM, font=_font(11))
    d.text((box[0] + 8, pad_t + 2), "稳定位数", fill=SETTLED, font=_font(12))

    # 活性：底部一条细带，宽度按 0~100% 填
    band = (box[0], box[3] + 10, box[2] - 120, box[3] + 34)
    d.rectangle(band, outline=(52, 58, 72))
    al = np.asarray(r["series"]["alive"], dtype=float)
    if n >= 2:
        for i, v in enumerate(al):
            x = band[0] + (band[2] - band[0]) * i / (n - 1)
            y1 = band[3] - (band[3] - band[1]) * float(np.clip(v, 0.0, 1.0))
            d.line([(x, band[3]), (x, y1)], fill=(120, 150, 170))
    d.text((band[2] + 8, band[1] + 2), f"活性 {al[-1]:.0%}", fill=(150, 180, 200), font=_font(12))
    return img


def _readout(r: dict, size: tuple[int, int]) -> Image.Image:
    """读数条：把关键数字写清楚，省得去翻 json。"""
    w, h = size
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    cfg = r["cfg"]
    s = r["series"]
    cur = r["curves"]
    pl = r["plateau"]
    spans = sorted(g.span for g in r["groups"])
    sizes = [len(g.members) for g in r["groups"]]
    base = min(sizes) if sizes else 0
    same = sorted(g.span for g in r["groups"] if len(g.members) == base)
    merged = sum(1 for s in sizes if s > base)
    tight = f"{min(same):.2f}" if same else "—"
    loose = f"{max(same):.2f}" if same else "—"
    pl_txt = f"k={pl['k']}（{pl['frac_lo']:g}–{pl['frac_hi']:g}）" if pl else "无平台"
    line1 = (f"n={cfg.n}  modules={cfg.modules}  ticks={r['ticks']}  seed={cfg.seed}  "
             f"kick={cfg.kick:g}  表示={'整数' if cfg.exact else 'float64'}")
    line2 = (f"k: {s['k'][0]} → {s['k'][-1]}    k(r) 八档: {'/'.join(str(int(c)) for c in cur)}   "
             f"平台 {pl_txt}")
    line3 = (f"稳定位数 {s['stable'][0]:.2f} → {s['stable'][-1]:.2f}   "
             f"defect {s['defect'][0]:.3g} → {s['defect'][-1]:.3g}   "
             f"活性 {s['alive'][-1]:.0%}   同规模({base} 人) span {tight}~{loose}   合并群 {merged}")
    M = r["load_frames"]
    if M.size:
        tot = M.sum(axis=1)
        share = float(M[-1][r["is_inter"]].sum() / max(tot[-1], 1e-300))
        flat = M[-1]
        top = np.sort(flat)[-max(1, int(round(0.10 * (flat > 0).sum()))):].sum() / max(flat.sum(), 1e-300)
        line4 = (f"负荷 末窗跨群份额 {share:.0%}   末窗前 10% 关系占 {top:.0%}   "
                 f"total_edges={M.shape[1]}")
    else:
        line4 = "负荷 没有窗口数据"
    for i, line in enumerate((line1, line2, line3, line4)):
        d.text((PAD, 10 + i * 22), line, fill=INK if i < 3 else DIM, font=_font(13))
    return img


def board(cfg: Config, ticks: int, window: int = 100, need: int = 4,
          size: int = 460) -> Image.Image:
    r = collect(cfg, ticks, window=window, need=need)

    # 左上角：形。用场面画（每个群一个物体），而不是环上的团块
    from .scene import SceneParams, draw_scene

    sh = r["shape"]
    shape_img = draw_scene(
        sh.groups, sh.strength, SceneParams(size=size),
        title=f"形（中视角）k={sh.k}",
        note="大小=人数 · 远近=关系强度 · 表面=群内极差 · 色相=关系在色轮上的位置",
        seed=cfg.seed,
    )
    panels = [
        shape_img,
        _life_panel(r, (size, size)),
        _load_panel(r, (size, size)),
    ]
    row2 = [
        plot_series([("k(t) @2%", r["series"]["k"], INK)], (size, 220), "类型数 k(t)", log=False),
        plot_series([("defect", r["series"]["defect"], CHURN)], (size, 220), "违约读数（对数）", log=True),
        _digits_panel(r, (size, 220)),
    ]
    W = 3 * size + 4 * 10
    H = size + 220 + 120 + 4 * 10
    out = Image.new("RGB", (W, H), (8, 9, 13))
    x, y = 10, 10
    for p in panels:
        out.paste(p, (x, y))
        x += size + 10
    y += size + 10
    x = 10
    for p in row2:
        out.paste(p, (x, y))
        x += size + 10
    y += 220 + 10
    out.paste(_readout(r, (W - 20, 100)), (10, y))
    return out
