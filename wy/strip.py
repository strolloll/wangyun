"""诊断带：把划分事件和五层读数放在同一条时间轴上。

`evo` 给的是形状的几帧，`board` 给的是一屏的读数，两张图各说各的。
问题在于「结构什么时候在重排、什么时候在被磨平」这句话里混了两件事：
划分在变（事件）和读数在走（曲线）。这个工具把它们对齐：

    上排：事件发生的那几 tick 的形状
    下排：k(t) / 违约读数 / 跨群负荷 / 稳定位数 / 色相移动 五条曲线
    每条事件画一根竖线，并、分用不同的记号

这样「t=800 之后只并了一次」和「defect 从这儿开始一路缓降」就能对到同一根竖线上。

最下面那条色相移动：颜色不是每帧重猜的（§30），只在有人换群时才动——
所以它是同一批事件在颜色上的投影：非零的格正好是竖线站着的那些 tick。
尖上的小点标出那一刻是并（橙）还是分（青），和上面各层的 ▼/▲ 同色。
峰值不一定在并群：k 小的时候分开走掉一大团，每人挪一格也会很疼（§30.5）。

这些仍然是读数：竖线告诉你哪一 tick 划分变了，不告诉你为什么变。
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from wy.core.color import hue_track, hue_vec
from wy.core.engine import Engine
from wy.core.layout import group_strength, spring
from wy.core.network import Config, Network
from wy.render import BG, CHURN, DIM, EDGE, INK, SETTLED, _font
from wy.scene import SceneParams, draw_scene
from wy.core.shape import (
    Group,
    StabilityMonitor,
    describe_change,
    group_table,
    k_of,
    partition,
    scale_of,
)

LANE_H = 84
PAD = 12
FONT = 12


def _canon(labels: np.ndarray) -> dict[int, tuple[int, ...]]:
    """按根 id 收成员，返回 {根: 成员元组}，用来和上一帧比。"""
    out: dict[int, list[int]] = {}
    for u, g in enumerate(labels.tolist()):
        out.setdefault(int(g), []).append(u)
    return {g: tuple(sorted(m)) for g, m in out.items()}


def _hue_dots(events: list[dict], hue_x) -> dict[int, tuple[int, int, int]]:
    """色相层上的小点：那一刻动了多少 + 是并还是分，颜色跟 ▼/▲ 一致（§18）。"""
    kinds = {int(e["tick"]): e["kind"] for e in events}
    return {int(t): (CHURN if kinds.get(int(t)) == "merge" else SETTLED)
            for t in hue_x if int(t) in kinds}


def _group_hues(prev_hues: dict, groups: list[Group], net: Network
                ) -> tuple[dict, np.ndarray]:
    """接着上一帧的色相，给这一帧的群一人一个色相（§30）——缩略图也按成员记路。"""
    prev_hues, _moved = hue_track(prev_hues, groups, group_strength(groups, net))
    return prev_hues, hue_vec(prev_hues, groups)


def _as_groups(canon: dict[int, tuple[int, ...]], v: np.ndarray) -> list[Group]:
    """把划分的成员表包成群表，只为算颜色用：色相只认成员（§30），松紧/位数不参与。"""
    out: list[Group] = []
    for gid, members in canon.items():
        vals = v[list(members)]
        out.append(Group(gid=int(gid), members=members,
                         value=float(vals.mean()),
                         span=float(vals.max() - vals.min()),
                         stable=0.0, age=0, settled=False))
    return out


def _carry(prev_groups, prev_pos, groups) -> np.ndarray | None:
    """上一帧的坐标按成员重合搬过来：新群贴在跟它最像的旧群旁边，再交给弹簧。

    群数变了就没法一对一搬，但按成员重合搬仍然成立——这才是「位置记路」的意思：
    认得的是谁跟谁在一起，不是第几号群。
    """
    if prev_groups is None or prev_pos is None:
        return None
    prev_sets = [set(np.asarray(g.members).tolist()) for g in prev_groups]
    out = np.zeros((len(groups), 2), dtype=np.float64)
    for i, g in enumerate(groups):
        mine = set(np.asarray(g.members).tolist())
        best, best_j = -1.0, 0
        for j, s in enumerate(prev_sets):
            jac = len(mine & s) / max(len(mine | s), 1)
            if jac > best:
                best, best_j = jac, j
        out[i] = prev_pos[best_j]
    return out


def collect(cfg: Config, ticks: int, frac: float = 3e-2, probe: int = 20,
            load_window: int = 100, need: int = 4) -> dict:
    net = Network(cfg)
    eng = Engine(net)
    mon = StabilityMonitor(net.n, cfg.window, cfg.digits)

    ticks_x: list[int] = []
    ks: list[int] = []
    defect: list[float] = []
    stable: list[float] = []
    alive: list[float] = []

    events: list[dict] = []
    frames: list[dict] = []
    prev_groups: dict[int, tuple[int, ...]] | None = None
    prev_alive: set[int] = set()

    # 色相移动：和事件同一批采样点，量「这些人这一 sampling 间隔里被迫换了多少颜色」
    hue_x: list[int] = []
    hue_moved: list[float] = []
    prev_hues: dict[frozenset, float] = {}

    load_marks: list[tuple[int, float]] = []
    prev_load = np.zeros(net.nbrs.shape, dtype=np.float64)
    mod_of = np.arange(net.n) % cfg.modules
    is_inter = (mod_of[:, None] != mod_of[net.nbrs]).ravel()

    for t in range(ticks):
        eng.step()
        mon.push(eng.v)
        fv = eng.v
        sc = scale_of(fv)
        ticks_x.append(t + 1)
        ks.append(k_of(fv, net, 0.02 * sc))
        defect.append(eng.defect())
        live = eng.recent(cfg.window)
        stable.append(mon.mean_length(live))

        if (t + 1) % load_window == 0:
            cum = np.array(eng.load, dtype=np.float64)
            inc = (cum - prev_load).ravel()
            prev_load = cum
            tot = inc.sum()
            if tot > 0:
                load_marks.append((t + 1, float(inc[is_inter].sum() / tot)))

        if (t + 1) % probe:
            continue
        cur = _canon(partition(fv, net, frac * sc))
        # 颜色记路：和事件同一批采样点，量这些人这一间隔里被迫换了多少（§30）
        gl = _as_groups(cur, fv)
        prev_hues, moved = hue_track(prev_hues, gl, group_strength(gl, net))
        hue_x.append(t + 1)
        hue_moved.append(float(moved))
        if prev_groups is not None:
            keys = set(cur)
            fresh = keys - prev_alive
            gone = prev_alive - keys
            if fresh or gone:
                ch = describe_change(list(prev_groups.values()), list(cur.values()))
                events.append({"tick": t + 1, "kind": ch["kind"], "text": ch["short"],
                               "k_before": ch["k_before"], "k_after": ch["k_after"]})
        prev_groups = cur
        prev_alive = set(cur)

    return {
        "net": net, "cfg": cfg, "ticks": ticks, "frac": frac,
        "x": np.array(ticks_x), "k": np.array(ks), "defect": np.array(defect),
        "stable": np.array(stable), "alive": np.array(alive),
        "events": events, "loads": load_marks, "need": need, "mon": mon,
        "eng": eng, "hue_x": np.array(hue_x), "hue_moved": np.array(hue_moved),
    }


def _lane(x: np.ndarray, ys: np.ndarray, w: int, h: int, *, log: bool,
          color: tuple[int, int, int], title: str, events: list[dict],
          span: tuple[float, float] | None = None,
          dots: dict[int, tuple[int, int, int]] | None = None) -> Image.Image:
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    d.text((PAD, 4), title, fill=INK, font=_font(13))
    pad_l = 58
    box = (pad_l, 26, w - PAD, h - 10)
    d.rectangle(box, outline=(52, 58, 72))

    v = np.log10(np.maximum(ys, 1e-300)) if log else ys.astype(float)
    if span is None:
        lo, hi = float(np.min(v)), float(np.max(v))
    else:
        lo, hi = span
    if hi - lo < 1e-12:
        hi = lo + 1.0
    n = len(v)
    if n >= 2:
        pts = [(box[0] + (box[2] - box[0]) * i / (n - 1),
                box[3] - (box[3] - box[1]) * (float(y) - lo) / (hi - lo)) for i, y in enumerate(v)]
        d.line(pts, fill=color, width=2)

    # 事件竖线：并=下三角，分=上三角
    xmax = float(x[-1]) if n else 1.0
    for e in events:
        px = box[0] + (box[2] - box[0]) * (e["tick"] / xmax)
        d.line([(px, box[1] + 6), (px, box[3])], fill=(58, 64, 82))
        cy = box[1] + 8
        if e["kind"] == "merge":
            d.polygon([(px, cy + 5), (px - 4, cy - 3), (px + 4, cy - 3)], fill=CHURN)
        else:
            d.polygon([(px, cy - 5), (px - 4, cy + 3), (px + 4, cy + 3)], fill=SETTLED)

    # 小点：只在指定 tick 上画，颜色由调用方给（色相层用它标并/分，§18）
    if dots and n:
        at = {int(xi): float(yi) for xi, yi in zip(x.tolist(), v.tolist())}
        for tick, col in dots.items():
            yv = at.get(int(tick))
            if yv is None:
                continue
            px = box[0] + (box[2] - box[0]) * (tick / xmax)
            py = box[3] - (box[3] - box[1]) * (yv - lo) / (hi - lo)
            d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=col)

    d.text((box[0] - 52, box[3] - 14), f"{v.min():.3g}", fill=DIM, font=_font(10))
    d.text((box[0] - 52, box[1] + 2), f"{v.max():.3g}", fill=DIM, font=_font(10))
    return img


def lifetimes(cfg: Config, ticks: int, sample: int = 10, frac: float = 3e-2) -> dict:
    """结构活多久：k(t) 在多长时间里塌下去。

    k(t) 不是单调的：开头几十 tick，随机初值还没分开，这一档会把它们并成几团；
    之后模块长出来，k 又冲高。所以「塌」只能从**最后一次还看得见整套模块**
    （k >= modules）之后算起，否则量到的是开头那段假塌。

    返回的键：
      `peak` k 的最大值、`last_full` 最后一次 k >= modules 的 tick、
      `t4`/`t2` 那之后第一次 k<=4 / k<=2 的 tick、`last_split` 那之后最后一次「分」。
    """
    net = Network(cfg)
    eng = Engine(net)
    tracks: list[tuple[int, int]] = []
    splits: list[int] = []
    prev_k = None
    for t in range(1, ticks + 1):
        eng.step()
        if t % sample:
            continue
        sc = scale_of(eng.v)
        kk = int(len(np.unique(partition(eng.v, net, frac * sc, cfg.strong))))
        tracks.append((t, kk))
        if prev_k is not None and kk > prev_k:
            splits.append(t)
        prev_k = kk

    last_full = 0
    for t, kk in tracks:
        if kk >= cfg.modules:
            last_full = t

    def first_after(thr: int):
        for t, kk in tracks:
            if t > last_full and kk <= thr:
                return t
        return None

    after = [t for t in splits if t > last_full]
    return {"peak": max((kk for _, kk in tracks), default=0),
            "last_full": last_full, "t4": first_after(4), "t2": first_after(2),
            "last_split": after[-1] if after else None,
            "k_end": tracks[-1][1] if tracks else None}


def strip(cfg: Config, ticks: int, frac: float = 3e-2, max_frames: int = 6,
          thumb: int = 220, width: int = 1180, need: int = 4,
          trace: list | None = None) -> Image.Image:
    r = collect(cfg, ticks, frac=frac, need=need)
    ev = r["events"]
    net = r["net"]

    # 挑帧：在事件里按时间均匀取，少于 max_frames 就全取
    if ev:
        idx = np.linspace(0, len(ev) - 1, min(max_frames, len(ev))).round().astype(int)
        picked = [ev[i] for i in dict.fromkeys(idx.tolist())]
    else:
        picked = []
    if picked:  # 缩略图不能超出画布：边长按帧数算
        thumb = min(thumb, (width - PAD * (len(picked) + 1)) // len(picked))

    # 只在挑中的 tick 上重跑一次拿形状（省内存：不做快照，按 tick 重演）
    thumbs: list[Image.Image] = []
    if picked:
        eng = Engine(net)
        mon = StabilityMonitor(net.n, cfg.window, cfg.digits)
        want = {p["tick"]: p for p in picked}
        prev_pos = None
        prev_groups = None
        prev_hues: dict = {}
        for t in range(1, ticks + 1):
            eng.step()
            mon.push(eng.v)
            if t not in want:
                continue
            fv = eng.v
            groups = group_table(fv, net, frac * scale_of(fv), mon, need=need,
                                 touched=eng.recent(cfg.window))
            if not groups:
                continue
            prev_hues, hue_vec = _group_hues(prev_hues, groups, net)
            if trace is not None:  # 缩略图的色相也要能查：老群跨帧一个色，才敢说「颜色接着上一帧」
                trace.append({"t": t, "k": len(groups), "hues": dict(prev_hues)})
            carried = _carry(prev_groups, prev_pos, groups)
            pos = (np.asarray(spring(group_strength(groups, net), seed=cfg.seed,
                                     iters=200, restarts=1, init=carried), dtype=float)
                   if carried is not None else None)
            if pos is not None:
                prev_pos = pos - pos.mean(axis=0)
            prev_groups = groups
            # 缩略图也从「环上的团块」换成场面：大小=人数、远近=关系强度、表面=极差
            # 颜色接着上一帧（§30）：同一批人跨帧一个色，闪的那一格才真是有人换了群。
            img = draw_scene(groups, group_strength(groups, net),
                             SceneParams(size=thumb),
                             title=f"t={t}  {want[t]['text']}",
                             note=f"k={len(groups)}", seed=cfg.seed,
                             pos=(None if pos is None else np.asarray(pos, dtype=float)),
                             hue=hue_vec)
            thumbs.append(img)

    w = width
    top_h = thumb if thumbs else 0
    lanes = [
        _lane(r["x"], r["k"], w, LANE_H + 20, log=False, color=INK, title="类型数 k(t) @2%",
              events=ev),
        _lane(r["x"], r["defect"], w, LANE_H + 20, log=True, color=CHURN,
              title="违约读数（对数）", events=ev),
        _lane(np.array([p[0] for p in r["loads"]]), np.array([p[1] for p in r["loads"]]),
              w, LANE_H + 20, log=False, color=SETTLED, title="跨群负荷份额（每窗口）",
              events=ev, span=(0.0, 1.0)),
        _lane(r["x"], r["stable"], w, LANE_H + 20, log=False, color=(150, 180, 200),
              title="稳定位数（均值，只算最近在算的）", events=ev),
    ]
    if len(r["hue_x"]) >= 2:  # 采样点太少就不画，凑不出一条线
        lanes.append(_lane(r["hue_x"], r["hue_moved"], w, LANE_H + 20, log=False,
                           color=(176, 150, 210),
                           title="色相移动（按人算：换群的人才动色）", events=ev,
                           dots=_hue_dots(ev, r["hue_x"])))

    H = top_h + 10 + sum(l.height + 8 for l in lanes) + 52
    out = Image.new("RGB", (w, H), (8, 9, 13))
    if thumbs:
        x = PAD
        for img in thumbs:
            out.paste(img, (x, 8))
            x += thumb + 8
    y = top_h + 16
    for lane in lanes:
        out.paste(lane, (0, y))
        y += lane.height + 8

    d = ImageDraw.Draw(out)
    # 时间轴刻度（画在最后一条带和脚注之间）
    ax_y = H - 44
    for frac_x, label in ((0.0, "t=0"), (0.5, f"{ticks // 2}"), (1.0, f"{ticks}")):
        xx = 58 + (w - PAD - 58) * frac_x
        d.text((xx - (0 if frac_x == 0 else 12), ax_y), label, fill=DIM, font=_font(10))
    merges = sum(1 for e in ev if e["kind"] == "merge")
    splits = len(ev) - merges
    note = (f"n={cfg.n}  ticks={ticks}  seed={cfg.seed}  frac={frac:g}  "
            f"划分变化 {len(ev)} 次（并 {merges} / 分 {splits}）"
            f"  ▼=并 ▲=分   —— 竖线只说明「这里的划分变了」，不说明为什么")
    d.text((PAD, H - 26), note, fill=DIM, font=_font(12))
    return out
