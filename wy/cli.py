"""跑实验。

    python -m wy run   --n 128 --ticks 600 --seed 7
    python -m wy read  --edges data/deps.txt --ticks 400
    python -m wy law   --ticks 400
    python -m wy evo   --ticks 4000 --frames 8
    python -m wy board --n 128 --ticks 4000
    python -m wy strip --n 128 --ticks 8000

`run`  : 一次完整实验，落盘指标 + 一张对照图（三档 LOD + 三条曲线）
`read` : 读一张真的边表（可读性 + 群表）
`law`  : 扫 kick，验证「稳定前缀的上限由扰动幅度决定」——前缀 ≈ log10(1/kick)
`evo`  : 把形放到时间轴上，看它长出来又散掉
`board`: 一屏看全（五层读数）
`strip`: 诊断带（事件与读数同一条时间轴）

产物默认落在**当前目录**下的 `out/`；要换地方设环境变量 `WY_OUT`。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from . import render
from .core.digits import StabilityMonitor
from .core.engine import Engine
from .core.layout import broadcast, group_strength, make_shape
from .core.gauge import report as gauge_report
from .core.network import Config, Network
from .core.shape import (
    PartitionTracker,
    describe_change,
    group_table,
    k_curve,
    k_of,
    partition,
    plateau,
    scale_of,
    table_signature,
)

OUT = Path(os.environ.get("WY_OUT") or (Path.cwd() / "out"))
FRACS = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1])


def _cfg(args) -> Config:
    return Config(
        n=args.n,
        modules=args.modules,
        seed=args.seed,
        kick=args.kick,
        window=args.window,
        strong=args.strong,
        intra_ratio=(args.intra_lo, args.intra_hi),
        inter_ratio=(args.inter_lo, args.inter_hi),
        alpha_range=(args.alpha_lo, args.alpha_hi),
        shape_feedback=float(getattr(args, "feedback", 0.0)),
        feedback_every=int(getattr(args, "feedback_every", 25)),
        feedback_age=int(getattr(args, "feedback_age", 64)),
        feedback_until=getattr(args, "feedback_until", None),
        feedback_auto=bool(getattr(args, "feedback_auto", False)),
        feedback_tol=float(getattr(args, "feedback_tol", 0.02)),
        feedback_win=int(getattr(args, "feedback_win", 8)),
        inter_cycles=int(getattr(args, "inter_cycles", 1)),
        walk_power=float(getattr(args, "walk_power", 1.0)),
        weak_module=int(getattr(args, "weak_module", -1)),
        weak_intra=(float(getattr(args, "weak_lo", 0.01)), float(getattr(args, "weak_hi", 0.03))),
        weak_modules=tuple(
            int(x) for x in str(getattr(args, "weak_set", "") or "").split(",") if x.strip()
        ),
        inter_shuffle=bool(getattr(args, "inter_shuffle", False)),
        digits=int(getattr(args, "digits", 12)),
        exact=bool(getattr(args, "exact", False)),
        exact_digits=int(getattr(args, "exact_digits", 30)),
    )


def run_once(cfg: Config, ticks: int, frac_trace: float = 2e-2, need: int = 8,
             net=None) -> dict:
    net = Network(cfg) if net is None else net
    exact = bool(getattr(cfg, "exact", False))
    if exact:
        # 整数演算：位数读数的材料是十进制整数串，不受 float64 的 15 位限制（DESIGN §13.2）。
        from .core.exact import ExactEngine, StableDigits

        xeng = ExactEngine(net, digits=int(cfg.exact_digits), cdigits=24)
        eng = xeng
        mon = StableDigits(cfg.n, cfg.window, keep=min(28, int(cfg.exact_digits)))
    else:
        eng = Engine(net)
        mon = StabilityMonitor(cfg.n, cfg.window, cfg.digits)
        mon.push(eng.v)
    pt = PartitionTracker(cfg.window)

    series = {"defect": [], "stable": [], "k": [], "alive": []}
    prev = eng.v if not exact else list(eng.v)
    fb = float(cfg.shape_feedback)
    every = max(1, int(cfg.feedback_every))
    until = cfg.feedback_until
    auto = bool(cfg.feedback_auto)
    tol = float(cfg.feedback_tol)
    released = False
    released_at: int | None = None
    # 自动松手：每轮焊之前记一次违约读数。焊接会把没抹平的差别顶住，
    # 一旦读数不再往下降（甚至回升），继续焊就只是在抱着一个已经冻住的形状。
    probe: list[float] = []
    half = max(4, int(cfg.feedback_win))

    def values() -> np.ndarray:
        """几何和 k(r) 用的浮点视图（每次只算一次，别在循环里重复换算）。"""
        return eng.to_float() if exact else eng.v

    def push_digits() -> None:
        mon.push(eng.texts() if exact else eng.v)

    def length_vec() -> np.ndarray:
        return mon.length

    def mean_len(mask: np.ndarray) -> float:
        return mon.mean_length(mask)

    for t in range(ticks):
        eng.step()
        push_digits()
        fv = values()
        if fb > 0.0 and (t + 1) % every == 0:
            if released:
                pass
            elif until is not None and (t + 1) > until:
                # 「焊到这儿为止」：之后关系冻住，看结构能不能自己活下去。
                released, released_at = True, t + 1
            else:
                d = xeng.defect(fv) if exact else eng.defect()
                probe.append(d)
                stop = False
                if auto and len(probe) >= 2 * half:
                    prev_m = float(np.mean(probe[-2 * half:-half]))
                    now = float(np.mean(probe[-half:]))
                    # 相对改善小于 tol 就算停了：焊枪已经把结构拉住，读数降不动了。
                    stop = now > prev_m * (1.0 - tol)
                if stop:
                    released, released_at = True, t + 1
                else:
                    # 形反作用于关系：只对**已经站住**的群收紧，而且要用细档认群——
                    # 粗档下整张图早早就是「一个群」，那样加固的不是结构，是同质化。
                    rows = group_table(
                        fv, net, cfg.feedback_frac * scale_of(fv), mon, need,
                        touched=eng.touched,
                    )
                    eng.feedback(rows, fb)
        labels = partition(fv, net, frac_trace * scale_of(fv))
        pt.push(labels, t + 1)
        # 只报「最近还在算」的单位的位数：别处的位数没变，是没人动它。
        live = eng.recent(cfg.window)
        series["defect"].append(xeng.defect(fv) if exact else eng.defect())
        series["stable"].append(mean_len(live))
        series["k"].append(int(np.unique(labels).size))
        # 活性：最近在算的单位里，这一 tick 真的变过几个。
        # 位数是「多久没动过」；值冻住（增量掉到可表示精度以下）时它也「没动过」，
        # 两个读数必须一起看，否则能把「算不动了」读成「收敛了」（DESIGN §13.1）。
        if exact:
            n_alive = float(sum(1 for i in range(cfg.n)
                                if live[i] and eng.v[i] != prev[i]))
            prev = list(eng.v)
        else:
            n_alive = float(np.count_nonzero((fv != prev) & live)) if live.any() else 0.0
            prev = fv.copy()
        series["alive"].append(n_alive / max(1.0, float(live.sum())))

    fv = values()
    ks = k_curve(fv, net, FRACS)
    lv = length_vec()
    live = eng.recent(cfg.window)
    return {
        "config": asdict(cfg),
        "ticks": ticks,
        "shape_feedback": {
            "strength": float(cfg.shape_feedback),
            "every": int(cfg.feedback_every),
            "times": eng.feedback_times,
            "edges": eng.feedback_edges,
            "until": cfg.feedback_until,
            "auto": bool(cfg.feedback_auto),
            "released": bool(released),
            "released_at": released_at,
            "relations_are_static": float(cfg.shape_feedback) <= 0.0,
        },
        "series": {k: [float(x) for x in v] for k, v in series.items()},
        "k_curve": {"fracs": FRACS.tolist(), "k": ks.tolist()},
        "plateau": plateau(FRACS, ks),
        "partition_age": pt.age,
        "events": pt.events,
        "final": {
            "defect": xeng.defect(fv) if exact else eng.defect(),
            "stable_mean": mean_len(live),
            "stable_max": int(lv[live].max()) if live.any() else 0,
            "touched": int(eng.touched.sum()),
            "recent": int(live.sum()),
            "at_cap": int((lv[live] >= (cfg.exact_digits if exact else cfg.digits)).sum()),
            "alive": float(series["alive"][-1]) if series["alive"] else 0.0,
            "k_at_trace": int(series["k"][-1]),
            "exact": exact,
        },
        "_engine": eng,
        "_net": net,
        "_mon": mon,
    }


def _write_groups(groups, path: Path) -> None:
    """形交给下游的样子：一行一个类型，数值是参数，稳定度是「能信到第几位」。"""
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("gid,count,value,span,stable,age,settled,members\n")
        for g in groups:
            members = " ".join(str(u) for u in g.members)
            fh.write(
                f"{g.gid},{g.count},{g.value:.6f},{g.span:.6f},"
                f"{g.stable:.3f},{g.age},{int(g.settled)},\"{members}\"\n"
            )


def cmd_run(args) -> None:
    cfg = _cfg(args)
    res = run_once(cfg, args.ticks, args.frac_trace, args.need)
    eng, net, mon = res.pop("_engine"), res.pop("_net"), res.pop("_mon")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 三个 LOD：视角一收，群就并，洞就该是同一批或更少。
    # 布局在**最粗那一档还看得见结构**的视角上算（k ≥ 2），更细的从它广播下来，
    # 更粗的本来就只有一个点、没有形状可言。这样缩放时画面不跳。
    panels = []
    fv = eng.values()
    scale = scale_of(fv)
    fracs = [args.frac_coarse, args.frac_mid, args.frac_fine]
    tables = {
        f: group_table(fv, net, f * scale, mon, args.need, touched=eng.recent(cfg.window))
        for f in fracs
    }
    counts = [len(tables[f]) for f in fracs]

    anchor = next((i for i, c in enumerate(counts) if c >= 2), 2)
    shapes: list = [None] * len(fracs)
    shapes[anchor] = make_shape(tables[fracs[anchor]], net, seed=cfg.seed)
    for i in range(anchor - 1, -1, -1):
        shapes[i] = make_shape(
            tables[fracs[i]], net, seed=cfg.seed,
            pos=np.zeros((counts[i], 2), dtype=np.float64),
        )
    for i in range(anchor + 1, len(fracs)):
        shapes[i] = make_shape(
            tables[fracs[i]], net, seed=cfg.seed, iters=120,
            init=broadcast(shapes[i - 1], tables[fracs[i]], net, seed=cfg.seed + i),
        )

    views = list(zip(fracs, shapes))

    for frac, view in views:
        img, topo = render.draw_shape(
            view, size=380, title=f"view {frac:g}   k={view.k}", need=args.need, seed=cfg.seed
        )
        panels.append(img)
        res.setdefault("lod", []).append(
            {"frac": frac, **topo, "table": table_signature(view.groups)}
        )

    # 真正的交付物：形 = 一张群表，k 行参数，交给建模软件。
    _write_groups(tables[args.frac_mid], OUT / "groups.csv")

    s = res["series"]
    panels.append(
        render.plot_series(
            [("k(t) @ 2%", np.asarray(s["k"]), (232, 236, 244))],
            title=f"类型数 k(t)  ({args.ticks} ticks)",
        )
    )
    panels.append(
        render.plot_series(
            [("稳定位数", np.asarray(s["stable"]), (96, 208, 214))],
            title="稳定前缀长度（均值）",
        )
    )
    panels.append(
        render.plot_series(
            [("defect", np.asarray(s["defect"]), (240, 160, 72))],
            title="违约读数（对数）",
            log=True,
        )
    )
    sheet = render.sheet(panels, cols=3)
    sheet.save(OUT / "sheet.png")
    (OUT / "metrics.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"n={cfg.n} modules={cfg.modules} seed={cfg.seed} kick={cfg.kick:g} ticks={args.ticks}")
    print(f"k: {s['k'][0]:.0f} -> {s['k'][-1]:.0f}   stable: {s['stable'][0]:.2f} -> {s['stable'][-1]:.2f}")
    print(f"defect: {s['defect'][0]:.4g} -> {s['defect'][-1]:.4g}")
    print(f"k(r): {res['k_curve']['k']}")
    print(f"plateau: {res['plateau']}")
    print(f"partition age: {res['partition_age']}  events: {len(res['events'])}")
    print(f"算到过: {res['final']['touched']}/{cfg.n} 个单位（stable 只按算到过的算）")
    print(f"最近在算: {res['final']['recent']}/{cfg.n} 个单位（位数只在这里有定义）")
    alive = float(res["final"]["alive"])
    print(f"活性: 最近在算的里面 {alive * 100:.0f}% 这一 tick 真的变了（太低说明是冻住，不是收敛）")
    cap = int(res["final"]["at_cap"])
    if cap:
        # 量尺到头了。这时候再说「稳了 N 位」是把尺子的长度当成世界的精度。
        print(
            f"  注意：{cap} 个单位的稳定位数撞到量尺上限（digits={cfg.digits}）——"
            f"要更长的读数用 --digits"
        )
    fb = res["shape_feedback"]
    if fb["strength"] > 0.0:
        tail = ""
        if fb.get("until") is not None:
            tail = f"；焊到 t={fb['until']} 就停手，之后关系冻住"
        if fb.get("auto") and fb.get("released"):
            tail += f"；自动松手于 t={fb['released_at']}（违约读数不再下降）"
        print(
            f"形反作用: 每 {fb['every']} tick 收紧一次，共 {fb['times']} 次，"
            f"改动 {fb['edges']} 条有向关系{tail}（关系已不是静态数据）"
        )
    else:
        print("形反作用: 关（关系是静态数据）")
    for row in res.get("lod", []):
        print(f"  view {row['frac']:g}: k={row['k']} holes={row['holes']} settled={row['settled']}")
    print(f"wrote {OUT / 'metrics.json'} and {OUT / 'sheet.png'}")


class _FrozenMonitor:
    """记忆下来的那一份稳定度档案。只给 group_table 看的三个字段，不用重跑引擎。"""

    def __init__(self, length: np.ndarray, age: np.ndarray) -> None:
        self.length = length
        self.age = age

    def settled_mask(self, need: int) -> np.ndarray:
        return self.length >= need


def _align(prev_groups, prev_pos, groups, pos) -> np.ndarray:
    """把这一帧转回上一帧的朝向。

    弹簧布局只定「谁离谁近」，整体转个角度对它来说是同一个解；但连着看的时候，
    圈每帧自己转一点，读的人会以为结构在动。所以按成员对应关系解一个最贴合的旋转。
    """
    if prev_pos is None or not len(prev_groups):
        return pos
    src, dst = [], []
    for i, g in enumerate(groups):
        mine = set(g.members)
        best_j, best_ov = -1, 0
        for j, pg in enumerate(prev_groups):
            ov = len(mine & set(pg.members))
            if ov > best_ov:
                best_j, best_ov = j, ov
        if best_j >= 0:
            src.append(pos[i])
            dst.append(prev_pos[best_j])
    if len(src) < 2:
        return pos
    c = np.array(src)
    p = np.array(dst)
    theta = float(np.arctan2((c[:, 0] * p[:, 1] - c[:, 1] * p[:, 0]).sum(),
                             (c[:, 0] * p[:, 0] + c[:, 1] * p[:, 1]).sum()))
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return pos @ rot.T


def _carry(prev_groups, prev_pos, groups, strength) -> np.ndarray | None:
    """上一帧的位置搬到这一帧：按成员重叠加权。

    这么做的理由不是好看，是**读得懂**：如果每一帧都重新布局，画面会跳，
    看的人会以为结构变了；位置接得上，才看得出「这两团并成了一团」。
    新出现、跟上一帧没有任何重叠的群，贴在它关系最强的那一团旁边，交给弹簧推开。
    """
    if prev_pos is None or not len(prev_groups):
        return None
    out = np.zeros((len(groups), 2), dtype=np.float64)
    placed = np.zeros(len(groups), dtype=bool)
    for i, g in enumerate(groups):
        mine = set(g.members)
        acc = np.zeros(2)
        tot = 0
        for j, pg in enumerate(prev_groups):
            ov = len(mine & set(pg.members))
            if ov:
                acc += prev_pos[j] * ov
                tot += ov
        if tot:
            out[i] = acc / tot
            placed[i] = True

    for i in range(len(groups)):  # 新群：贴住已落位里关系最强的那一团
        if placed[i]:
            continue
        row = strength[i].copy()
        row[~placed] = 0.0
        row[i] = 0.0
        if row.max() > 0:
            j = int(np.argmax(row))
            out[i] = out[j] + 0.08 * np.array([np.cos(i), np.sin(i)])
        placed[i] = True
    return out


def _members(labels: np.ndarray) -> list[tuple[int, ...]]:
    return [tuple(int(i) for i in np.flatnonzero(labels == r).tolist())
            for r in sorted(set(labels.tolist()))]


def _pick_frac(snaps_v, net, manual, want_stable: bool) -> tuple[float, str]:
    """看哪一档：挑「整段时间里划分最稳」的那一档，而不是终态有平台的那一档。

    平台只说明末了那一档有结构；带子横跨整段时间，用平台档会看到阈值在抖——
    那不是结构在变，是读数在颤。
    """
    if manual is not None:
        return float(manual), "手动指定"
    best, best_cost = None, None
    for f in FRACS:
        kk = [k_of(v, net, f * scale_of(v)) for v in snaps_v]
        cost = float(np.sum(np.abs(np.diff(kk)))) + 0.05 * float(np.mean(kk))
        if min(kk) >= 2 and (best_cost is None or cost < best_cost):
            best, best_cost = float(f), cost
    if best is None:
        return 0.03, "没有稳定档，退到 3%"
    return best, "这段时间划分最稳的一档"


def cmd_evo(args) -> None:
    """把形放到时间轴上：同一份关系，在不同 tick 上的形状连成一条带子。

    这是望云最初要的东西——「要能看到有形的过程」。不是动画，是采样：
    每一帧都是那一刻的群表，位置接着上一帧，所以并、分、散都看得出来。

    两种采法：
      `--mode events`  只在划分真的变了的那一 tick 采一帧，标题写清并了谁 / 分了谁；
      `--mode even`    等间隔采，看整体节奏。
    """
    cfg = _cfg(args)
    frames = max(2, args.frames)
    ticks = args.ticks

    def make_run():
        net = Network(cfg)
        eng = Engine(net)
        mon = StabilityMonitor(cfg.n, cfg.window, cfg.digits)
        mon.push(eng.v)
        return net, eng, mon

    # 第一遍：只为了定「看哪一档」。三遍跑的是同一份确定性演算，逐位相同。
    net, eng, _mon = make_run()
    probe = [i for i in range(max(2, ticks // 40), ticks + 1, max(1, ticks // 40))]
    snap_v = []
    for t in range(1, ticks + 1):
        eng.step()
        if t in set(probe):
            snap_v.append(eng.v.copy())
    frac, why = _pick_frac(snap_v, net, args.frac, args.mode == "events")

    # 第二遍：只记事件（便宜）。
    net, eng, mon = make_run()
    events: list[dict] = []
    prev_members = None
    for t in range(1, ticks + 1):
        eng.step()
        mon.push(eng.v)
        members = _members(partition(eng.v, net, frac * scale_of(eng.v)))
        if prev_members is not None and members != prev_members:
            info = describe_change(prev_members, members)
            info["tick"] = t
            events.append(info)
        prev_members = members

    if args.mode == "events":
        if not events:
            print("这段时间划分一次都没变——只有一个形状，没有过程可看。")
            return
        # 只在「已经长出结构」的阶段里挑：一开始 k 上百，几十个小球互相并，
        # 圈出来是一团糊，那不是过程，是还没成形。
        big = [e for e in events if max(e["k_before"], e["k_after"]) <= args.max_k]
        pool = big if len(big) >= 2 else events
        idx = np.unique(np.linspace(0, len(pool) - 1, frames).round().astype(int))
        chosen = [pool[int(i)] for i in idx]
    else:
        marks = sorted({max(1, round(ticks * (i + 1) / frames)) for i in range(frames)})
        chosen = [{"tick": m, "text": "", "cur_members": [], "kind": "even"} for m in marks]
    want = {e["tick"] for e in chosen}

    # 第三遍：在选中的那些 tick 上取快照。
    net, eng, mon = make_run()
    snaps: dict[int, tuple] = {}
    for t in range(1, ticks + 1):
        eng.step()
        mon.push(eng.v)
        if t in want:
            snaps[t] = (eng.v.copy(), eng.defect(), mon.length.copy(),
                        mon.age.copy(), eng.touched.copy())

    panels = []
    hits: list[int] = []
    frame_ks: list[int] = []
    prev_groups, prev_pos = [], None
    for ev in chosen:
        t = ev["tick"]
        v, _defect, length, age, touched = snaps[t]
        rows = group_table(
            v, net, frac * scale_of(v), _FrozenMonitor(length, age), args.need, touched=touched
        )
        init = _carry(prev_groups, prev_pos, rows, group_strength(rows, net))
        shape = make_shape(
            rows, net, seed=cfg.seed, init=init, iters=180 if init is not None else 420
        )
        pos = _align(prev_groups, prev_pos, rows, shape.pos)
        shape = replace(shape, pos=pos)
        title = (
            f"t={t}  {ev['short']}  k {ev['k_before']}→{ev['k_after']}"
            if ev.get("short")
            else f"t={t}  k={shape.k}"
        )
        # 圈「刚并进来的那一团 / 刚裂开的那几块」——按成员认，不按下标认。
        idx_of = {tuple(sorted(g.members)): i for i, g in enumerate(rows)}
        hit = [idx_of[m] for m in ev.get("cur_members", []) if m in idx_of]
        img, _ = render.draw_shape(
            shape,
            size=args.frame,
            title=title,
            need=args.need,
            seed=cfg.seed + t,
            highlight=hit or None,
        )
        hits.append(len(hit))
        frame_ks.append(shape.k)
        panels.append(img)
        prev_groups, prev_pos = rows, shape.pos

    OUT.mkdir(parents=True, exist_ok=True)
    strip = render.sheet(panels, cols=args.cols)
    strip.save(OUT / "evolution.png")
    print(f"看的档: frac={frac:g}（{why}）")
    if args.mode == "events":
        print(f"一共 {len(events)} 次划分变化；带子取了 {len(panels)} 帧。")
    else:
        print(f"等间隔取了 {len(panels)} 帧。")
    print(f"wrote {OUT / 'evolution.png'}   {args.ticks} ticks")
    for ev, nhit, kk in zip(chosen, hits, frame_ks):
        defect = snaps[ev["tick"]][1]
        what = ev["text"] if ev.get("text") else ""
        kk_txt = (
            f"k {ev['k_before']}→{ev['k_after']}"
            if ev.get("k_before") is not None
            else f"k={kk}"
        )
        print(f"  t={ev['tick']:<6} {kk_txt:<12} {what:<20} 圈{nhit}  defect={defect:.3e}")


def cmd_law(args) -> None:
    print(f"{'kick':>10} {'log10(1/kick)':>14} {'stable(mean)':>13} {'stable(max)':>12} {'k':>4}")
    rows = []
    kicks = [10.0 ** -e for e in range(1, 10)]
    for kick in kicks:
        cfg = Config(
            n=args.n, modules=args.modules, seed=args.seed, kick=kick, window=args.window,
            inter_ratio=(args.inter_lo, args.inter_hi), alpha_range=(args.alpha_lo, args.alpha_hi),
        )
        res = run_once(cfg, args.ticks)
        s = res["series"]["stable"]
        tail = s[-max(1, len(s) // 4) :]
        mean = float(np.mean(tail))
        rows.append((kick, float(np.log10(1.0 / kick)), mean))
        print(
            f"{kick:10.0e} {rows[-1][1]:14.2f} {mean:13.2f} "
            f"{res['final']['stable_max']:12d} {res['final']['k_at_trace']:4d}"
        )

    # 两段：kick 小的时候被结构残差顶住（平），kick 大的时候由扰动定死（线性）。
    floor = float(np.mean([r[2] for r in rows if r[0] <= 1e-6]))
    strong = [r for r in rows if r[0] >= 1e-2]
    print(f"\n结构底（kick ≤ 1e-6 那一段）: {floor:.2f} 位")
    if len(strong) >= 2:
        x = np.array([r[1] for r in strong])
        y = np.array([r[2] for r in strong])
        slope, intercept = np.polyfit(x, y, 1)
        print(f"扰动段（kick ≥ 1e-2）: stable ≈ {slope:.3f} * log10(1/kick) + {intercept:.3f}")
        print("slope ≈ 1 的那一段说明：数能信到第几位，是被「还有多少东西在动」定死的。")
        print(f"两段合起来： stable ≈ min({floor:.2f}, 扰动决定的位数)。这就是用户 1.01 / 1.001 那个例子。")


def cmd_board(args) -> None:
    """一屏看全：把五层读数拼到一张图上。"""
    from wy.board import board as make_board
    from wy.core.network import Config as Cfg

    cfg = Cfg(n=args.n, modules=args.modules, seed=args.seed, kick=args.kick,
              inter_ratio=(args.inter_lo, args.inter_hi),
              alpha_range=(args.alpha_lo, args.alpha_hi), window=args.window)
    img = make_board(cfg, args.ticks, window=args.load_window, need=args.need, size=args.size)
    path = OUT / (args.out or "board.png")
    img.save(path)
    print(f"一屏看全: {path}  ({img.width}x{img.height})")


def cmd_strip(args) -> None:
    """诊断带：划分事件和读数放在同一条时间轴上。"""
    from wy.core.network import Config as Cfg
    from wy.strip import strip as make_strip

    cfg = Cfg(n=args.n, modules=args.modules, seed=args.seed,
              inter_ratio=(args.inter_lo, args.inter_hi),
              alpha_range=(args.alpha_lo, args.alpha_hi), window=args.window,
              inter_cycles=args.inter_cycles)
    img = make_strip(cfg, args.ticks, frac=args.frac, max_frames=args.max_frames,
                     thumb=args.thumb, need=args.need)
    path = OUT / (args.out or "strip.png")
    img.save(path)
    print(f"诊断带: {path}  ({img.width}x{img.height})")


def cmd_read(args) -> None:
    """读一张真图：边表 → 关系组 → 跑一遍 → 报读数。

    跟 `run` 只差一件事：图不是生成的，是读进来的。所以才需要先报清
    「这张图长什么样」（点数、边数、度数分布、孤立点），否则后面那些读数
    没法归因——是结构是这样，还是我把它读成了这样。
    """
    from .core.load import load_graph

    cfg = _cfg(args)
    weighted = None if args.weighted == "auto" else (args.weighted == "yes")
    net = load_graph(args.edges, cfg, base=args.base, weighted=weighted)
    cfg = net.cfg

    res = run_once(cfg, args.ticks, args.frac_trace, args.need, net=net)
    eng, mon = res.pop("_engine"), res.pop("_mon")
    res.pop("_net", None)
    sig = net.signature()
    res["graph"] = {k: v for k, v in sig.items() if k not in ("seed", "kick", "window")}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "read.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fv = eng.values()
    scale = scale_of(fv)
    fracs = [args.frac_coarse, args.frac_mid, args.frac_fine]
    touched = eng.recent(cfg.window)
    tables = {f: group_table(fv, net, f * scale, mon, args.need, touched=touched) for f in fracs}
    counts = [len(tables[f]) for f in fracs]
    anchor = next((i for i, c in enumerate(counts) if c >= 2), len(fracs) - 1)
    shapes: list = [None] * len(fracs)
    shapes[anchor] = make_shape(tables[fracs[anchor]], net, seed=cfg.seed)
    for i in range(anchor - 1, -1, -1):
        shapes[i] = make_shape(tables[fracs[i]], net, seed=cfg.seed,
                               pos=np.zeros((counts[i], 2), dtype=np.float64))
    for i in range(anchor + 1, len(fracs)):
        shapes[i] = make_shape(tables[fracs[i]], net, seed=cfg.seed, iters=120,
                               init=broadcast(shapes[i - 1], tables[fracs[i]], net,
                                              seed=cfg.seed + i))

    panels = []
    for frac, view in zip(fracs, shapes):
        img, topo = render.draw_shape(view, size=380, title=f"view {frac:g}   k={view.k}",
                                      need=args.need, seed=cfg.seed)
        panels.append(img)
        res.setdefault("lod", []).append({"frac": frac, **topo,
                                          "table": table_signature(view.groups)})
    _write_groups(tables[args.frac_mid], OUT / "groups.csv")

    s = res["series"]
    panels.append(render.plot_series(
        [("k(t)", np.asarray(s["k"]), (232, 236, 244))],
        title=f"类型数 k(t)  ({args.ticks} ticks)"))
    panels.append(render.plot_series(
        [("稳定位数", np.asarray(s["stable"]), (96, 208, 214))],
        title="稳定前缀长度（均值）"))
    panels.append(render.plot_series(
        [("defect", np.asarray(s["defect"]), (240, 160, 72))],
        title="违约读数（对数）", log=True))
    sheet = render.sheet(panels, cols=3)
    out = OUT / (args.out or "read.png")
    sheet.save(out)

    deg = net.degree_vec
    print(f"图: {args.edges}  点 {net.n}  关系 {int(net.mask.sum())}  "
          f"平均度 {deg.mean():.2f}  最大度 {int(deg.max())}  孤立点 {int(net.isolated.sum())}")
    # 这张图能不能读：先量瓶颈和松紧差，再让人看读数（§24）。
    # 量的是中档视角的划分——它就是平时拿来下结论的那一档。
    mid_labels = partition(fv, net, args.frac_mid * scale)
    gauge = gauge_report(mid_labels, fv, net)
    res["gauge"] = gauge
    print(f"可读性: contrast {gauge['contrast']:.3f}  q_excess {gauge['q_excess']:+.3f}"
          f"  dispersion {gauge['dispersion']:.1f}  -> {gauge['verdict']}")
    print(f"跑 {args.ticks} tick, seed={cfg.seed}, kick={cfg.kick:g}, walk_power={cfg.walk_power:g}")
    print(f"k: {s['k'][0]:.0f} -> {s['k'][-1]:.0f}   stable: {s['stable'][0]:.2f} -> {s['stable'][-1]:.2f}")
    print(f"defect: {s['defect'][0]:.4g} -> {s['defect'][-1]:.4g}")
    print(f"k(r): {res['k_curve']['k']}")
    print(f"plateau: {res['plateau']}")
    print(f"最近在算: {res['final']['recent']}/{net.n} 个单位")
    print(f"图: {out}")
    print("按度数分组看形（度数不同的群本来就该长得不一样）：")
    for g in sorted(tables[args.frac_mid], key=lambda g: -len(g.members))[:8]:
        d = deg[list(g.members)]
        print(f"  {len(g.members):3d} 人  span {g.span:8.3f}  位数 {g.stable:4.2f}  "
              f"{'稳' if g.settled else '动'}  平均度 {d.mean():.2f}")



def cmd_paint(args) -> None:
    """把一次演算的读数画成一张图：参数从读数来，画面是参数的函数。"""
    from .paint import draw_character, params_from_engine

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (args.out or "paint.png")
    meta = {"brief": getattr(args, "brief", ""), "png": str(path)}

    if getattr(args, "brief", ""):
        from .brief import choose, parse

        b = parse(args.brief)
        pick = choose(b, seed=args.seed, n=min(64, args.n))
        p = replace(pick.params, size=int(args.size or pick.params.size))
        print(f"意图: 眼 {b.eye} / 花纹 {b.swirl} / 星 {b.stars} / 神态 {b.mood}")
        print(f"挑中: {pick.candidate.note}  k={pick.readings.get('k')} "
              f"defect={pick.readings.get('defect', 0):.2e}  偏差={pick.score:.2f}")
        if not pick.ok():
            print("注意: 这个目标在望云能长出来的范围边缘，图只是「最接近的那一档」。")
        meta.update({
            "readings": {k: (float(v) if isinstance(v, (int, float)) else v)
                         for k, v in pick.readings.items()},
            "candidate": pick.candidate.note,
            "score": float(pick.score),
            "report": [[float(s), note] for s, note in pick.report],
        })
    else:
        cfg = _cfg(args)
        res = run_once(cfg, args.ticks, frac_trace=args.frac_mid, need=args.need)
        eng, net, mon = res["_engine"], res["_net"], res["_mon"]
        fv = eng.values() if hasattr(eng, "values") else eng.v
        scale = scale_of(fv)
        groups = group_table(fv, net, args.frac_mid * scale, mon, args.need,
                             touched=eng.recent(cfg.window))
        spans = [g.span for g in groups if len(g.members) > 1]
        defect = float(res["final"]["defect"])
        stable = float(res["final"]["stable_mean"])
        k = len(groups)
        p = params_from_engine(k, spans, stable, defect)
        if args.size:
            p = replace(p, size=int(args.size))
        print(f"读数: k={k}  stable={stable:.2f}  defect={defect:.3e}")
        meta["readings"] = {"k": k, "spans": [float(s) for s in spans],
                            "stable": stable, "defect": defect, "ticks": args.ticks}

    img = draw_character(p, ss=2)
    img.save(path)
    meta["params"] = asdict(p)
    (OUT / "paint.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"参数: 眼睛高 {p.eye_h:.2f}R  花纹 {p.swirl:.2f}  辉光 {p.glow:.2f}  星 {p.stars:.2f}")
    print(f"图: {path}")


def cmd_scene(args) -> None:
    """把一张图的读数画成场面。

    和 `paint` 的区别：`paint` 画的是一张脸（参数化肖像），`scene` 画的是
    **这张图长什么样**——每个群是一个物体，关系强的离得近、长得大，
    还在动的群是点阵的，稳住的群是实心的。没有「构图」这一步：位置来自
    关系的弹簧布局，远近来自每个群的平均关系强度，大小来自它的人数。
    """
    from .core.layout import group_strength
    from .scene import SceneParams, draw_scene

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (args.out or "scene.png")

    if args.edges:
        from .core.load import load_graph

        weighted = None if args.weighted == "auto" else (args.weighted == "yes")
        net = load_graph(args.edges, _cfg(args), base=args.base, weighted=weighted)
        cfg = net.cfg
    else:
        cfg = _cfg(args)
        net = Network(cfg)

    res = run_once(cfg, args.ticks, args.frac_trace, args.need, net=net)
    eng, mon = res.pop("_engine"), res.pop("_mon")
    res.pop("_net", None)
    fv = eng.values()
    scale = scale_of(fv)
    touched = eng.recent(cfg.window)
    groups = None
    if args.frac_mid > 0:
        cand = [args.frac_mid]
    else:  # 自动挑一档：让画面里有 6~30 个物体，太多太少都读不动
        cand = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
    best_k = None
    for f in cand:
        gs = group_table(fv, net, f * scale, mon, args.need, touched=touched)
        if len(gs) < 2:
            continue
        if best_k is None or abs(len(gs) - args.target_k) < abs(best_k - args.target_k):
            best_k, groups = len(gs), gs
    if groups is None:
        groups = group_table(fv, net, args.frac_mid * scale, mon, args.need, touched=touched)
    strength = group_strength(groups, net)

    note = (f"{args.edges or 'synthetic'}  n={net.n}  k={len(groups)}  "
            f"defect={eng.defect():.2e}  ticks={args.ticks}")
    if args.edges:  # 真图：把「这次读数值不值得信」也写在图上
        from .core.gauge import report

        labels = np.zeros(net.n, dtype=np.int64)
        for i, g in enumerate(groups):
            labels[list(g.members)] = i
        gauge = report(labels, fv, net)
        note += f"  ·  contrast {gauge['contrast']:.3f}  q {gauge['q_excess']:+.3f}  → {gauge['verdict']}"
    img = draw_scene(groups, strength, SceneParams(size=args.size),
                     title=f"望云 · {len(groups)} 个群", note=note, seed=cfg.seed)
    img.save(path)

    meta = {
        "png": str(path),
        "readings": {k: (float(v) if isinstance(v, (int, float)) else v)
                     for k, v in res.items() if not isinstance(v, (list, dict))},
        "groups": [{"gid": g.gid, "count": len(g.members), "value": float(g.value),
                    "span": float(g.span), "stable": float(g.stable),
                    "settled": bool(getattr(g, "settled", False))} for g in groups],
    }
    (OUT / "scene.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    print(f"n={net.n}  k={len(groups)}  defect={eng.defect():.3e}")
    print(f"图: {path}")


def cmd_anim(args) -> None:
    """让场面动起来：采样若干帧，拼成带子（和 GIF）。"""
    from .anim import filmstrip, frames, save_gif

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(args)
    trace: list = []
    imgs = frames(cfg, args.ticks, count=args.frames, frac=args.frac_mid,
                  need=args.need, size=args.size, trace=trace)
    if not imgs:
        print("没采到帧（是不是 tick 太少，结构还没分开？）")
        return
    strip = filmstrip(imgs, cols=min(args.cols, len(imgs)))
    strip.save(OUT / (args.out or "anim.png"))
    if args.gif:
        save_gif(imgs, OUT / args.gif, duration=args.duration)
    print(f"帧数 {len(imgs)}  size {args.size}  ticks {args.ticks}")
    if trace:
        print("色相移动 " + "  ".join(f"t={r['t']}:{r['moved']:.3f}" for r in trace))
    print(f"图: {OUT / (args.out or 'anim.png')}")
    if args.gif:
        print(f"GIF: {OUT / args.gif}")


def cmd_drill(args) -> None:
    """点开群：把每个群内部的关系摆开，看它里面的人长什么样。"""
    from .drill import sheet as drill_sheet

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(args)
    net = Network(cfg)
    res = run_once(cfg, args.ticks, args.frac_trace, args.need, net=net)
    eng, mon = res.pop("_engine"), res.pop("_mon")
    res.pop("_net", None)
    fv = eng.values()
    scale = scale_of(fv)
    groups = group_table(fv, net, args.frac_mid * scale, mon, args.need,
                         touched=eng.recent(cfg.window))
    if not groups:
        print("这一档没有群，换个 frac 或多跑一点")
        return
    if args.focus >= 0:
        from .drill import inset
        from .core.layout import group_strength

        img = inset(groups, net, fv, args.focus, group_strength(groups, net),
                    size=args.size * 3, seed=cfg.seed)
        path = OUT / (args.out or "inset.png")
        img.save(path)
        print(f"连续放大：群 {args.focus} 的内部贴在它自己在图上的位置")
        print(f"图: {path}")
        return

    img = drill_sheet(groups, net, fv, top=args.top, cols=args.cols,
                      size=args.size, seed=cfg.seed)
    path = OUT / (args.out or "drill.png")
    img.save(path)
    print(f"群 {len(groups)} 个，画了最大的 {min(args.top, len(groups))} 个")
    print(f"图: {path}")


def cmd_zoom(args) -> None:
    """第三层：盯住一个单位，看它直接连着谁。"""
    from .drill import zoom as drill_zoom

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(args)
    net = Network(cfg)
    res = run_once(cfg, args.ticks, args.frac_trace, args.need, net=net)
    eng = res.pop("_engine")
    fv = eng.values()
    img = drill_zoom(net, fv, args.member, size=args.size, seed=cfg.seed, hops=args.hops)
    path = OUT / (args.out or "zoom.png")
    img.save(path)
    print(f"单位 {args.member}：一跳邻居 {img.size[0]}px 的图已出")
    print(f"图: {path}")


def cmd_serve(args) -> None:
    """起本地观察台：页面和别的软件吃同一份 bundle（wy.core.observe）。"""
    from .server import serve

    serve(args.host, args.port, args.open)


def cmd_scan(args) -> None:
    """真图上最要紧的是「在哪个 tick 看」：把几个 tick 的读数摆成一张表。"""
    from .core.load import load_graph
    from .core.scan import best, scan

    cfg = _cfg(args)
    weighted = None if args.weighted == "auto" else (args.weighted == "yes")
    net = load_graph(args.edges, cfg, base=args.base, weighted=weighted)
    ticks = [int(x) for x in args.ticks_list.split(",") if x.strip()]
    target = args.target_k or None
    rows = scan(net, ticks, frac=args.frac_mid, need=args.need, target_k=target)

    print(f"{'tick':>6} {'k':>5} {'defect':>9} {'contrast':>9} {'q':>7}  {'frac':>7}  判语")
    for r in rows:
        print(f"{r['tick']:>6} {r['k']:>5} {r['defect']:>9.2e} {r['contrast']:>9.3f} "
              f"{r['q']:>+7.3f}  {r['frac']:>7g}  {r['verdict']}")
    b = best(rows)
    if b:
        print(f"\n最值得看的一格：tick={b['tick']}  k={b['k']}  q={b['q']:+.3f}  → {b['verdict']}")
        print(f"接着画这一格： python -m wy scene --edges {args.edges} --ticks {b['tick']} "
              f"--frac-mid {b['frac']:g}")



def cmd_portrait(args) -> None:
    """画像：从一次演算的读数长出一张肖像。"""
    from .portrait import draw_portrait, params_from_readings

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(args)
    net = Network(cfg)
    res = run_once(cfg, args.ticks, args.frac_trace, args.need, net=net)
    eng = res.pop("_engine")
    res.pop("_mon", None)
    res.pop("_net", None)
    fv = eng.values()
    scale = scale_of(fv)
    groups = group_table(fv, net, args.frac_mid * scale, None, args.need,
                         touched=eng.recent(cfg.window))
    spans = [float(g.span) for g in groups] or [0.0]
    p = params_from_readings(len(groups), float(np.mean(spans)),
                             float(res.get("final", {}).get("stable", 0.0)),
                             float(res.get("final", {}).get("defect", 0.0)))
    img = draw_portrait(p.__class__(**{**p.__dict__, "size": args.size}))
    path = OUT / (args.out or "portrait.png")
    img.save(path)
    print(f"画像参数: 刘海 {p.bang:.2f}  发束 {p.strands}  眼神 {p.gloss:.2f}  颗粒 {p.grain:.2f}")
    print(f"图: {path}")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    p = argparse.ArgumentParser(prog="wy", description="望云：看形从关系里长出来的观测器")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--n", type=int, default=128)
        sp.add_argument("--modules", type=int, default=8)
        sp.add_argument("--seed", type=int, default=7)
        sp.add_argument("--ticks", type=int, default=600)
        sp.add_argument("--window", type=int, default=8)
        sp.add_argument("--digits", type=int, default=12,
                        help="稳定前缀最多数到几位（数到 digits+1 就说明量尺到头了，不是精度到头了）")
        sp.add_argument("--exact", action="store_true",
                        help="换整数精确演算（wy/exact.py）：位数不再受 float64 的 15 位限制，代价是慢约一倍")
        sp.add_argument("--exact-digits", type=int, default=30,
                        help="整数演算的值定标位数（默认 30）")
        sp.add_argument("--strong", type=float, default=0.0)
        sp.add_argument("--need", type=int, default=4, help="算「稳住了」要几位")
        sp.add_argument("--intra-lo", type=float, default=0.70)
        sp.add_argument("--intra-hi", type=float, default=1.00)
        sp.add_argument("--inter-lo", type=float, default=0.002)
        sp.add_argument("--inter-hi", type=float, default=0.006)
        sp.add_argument("--alpha-lo", type=float, default=0.40)
        sp.add_argument("--alpha-hi", type=float, default=0.90)
        sp.add_argument("--inter-cycles", type=int, default=1, dest="inter_cycles",
                        help="模块之间的外连连成几个独立的环（1=一大圈，模块数与模块内结构不变）")
        sp.add_argument("--weak-module", type=int, default=-1, dest="weak_module",
                        help="让这个模块内部变松（-1=不做手脚）：拓扑不变，只是它内部没那么紧")
        sp.add_argument("--weak-lo", type=float, default=0.01, dest="weak_lo")
        sp.add_argument("--weak-hi", type=float, default=0.03, dest="weak_hi")
        sp.add_argument("--walk-power", type=float, default=1.0, dest="walk_power",
                        help="走动规则里比例的指数：1=原规则（弱关系靠大差值抢活干），"
                             "2≈按条数分摊（结构活得久约 2.6 倍，见 DESIGN §20）")
        sp.add_argument("--weak-set", type=str, default="", dest="weak_set",
                        help="一次弱化好几个模块，逗号隔开（如 4,5,6,7）：连通块数不变，只是密度分两档")
        sp.add_argument("--inter-shuffle", action="store_true", dest="inter_shuffle",
                        help="打乱模块分组的成员（环的个数不变，谁跟谁一圈变了）")

    r = sub.add_parser("run", help="跑一次完整实验")
    common(r)
    r.add_argument("--kick", type=float, default=1e-6)
    r.add_argument("--feedback", type=float, default=0.0,
                   help="形反作用强度：每次对稳住的那几团收紧多少（0=关，关系保持静态数据）")
    r.add_argument("--feedback-every", type=int, default=25, dest="feedback_every",
                   help="每多少 tick 反馈一次")
    r.add_argument("--feedback-age", type=int, default=64, dest="feedback_age",
                   help="结构年龄到多少才算「已经站住」，才有资格改关系")
    r.add_argument("--feedback-until", type=int, default=None, dest="feedback_until",
                   help="只焊到第 N 个 tick，之后关系冻住继续跑（看结构能不能自持）")
    r.add_argument("--feedback-auto", action="store_true", dest="feedback_auto",
                   help="自动松手：违约读数不再下降时停手，不用手填 N")
    r.add_argument("--feedback-tol", type=float, default=0.02, dest="feedback_tol",
                   help="自动松手的判据：相对改善小于这个比例就算「降不动了」")
    r.add_argument("--feedback-win", type=int, default=8, dest="feedback_win",
                   help="自动松手看最近几轮违约读数（前后各一半比较）")
    r.add_argument("--frac-trace", type=float, default=2e-2)
    r.add_argument("--frac-coarse", type=float, default=1e-1)
    r.add_argument("--frac-mid", type=float, default=1e-2)
    r.add_argument("--frac-fine", type=float, default=1e-3)
    r.set_defaults(func=cmd_run)

    l = sub.add_parser("law", help="扫 kick，验证前缀长度的上限")
    common(l)
    l.set_defaults(func=cmd_law)

    e = sub.add_parser("evo", help="把形放到时间轴上，看它长出来又散掉")
    common(e)
    e.add_argument("--kick", type=float, default=1e-6)
    e.add_argument("--frames", type=int, default=8, help="采几帧")
    e.add_argument("--cols", type=int, default=4)
    e.add_argument("--frame", type=int, default=300, help="单帧像素")
    e.add_argument("--frac-mid", type=float, default=1e-2)
    e.add_argument("--frac", type=float, default=None, help="手动指定看哪一档分辨率")
    e.add_argument("--mode", choices=["events", "even"], default="events",
                   help="events=只在划分变了的那一 tick 采一帧；even=等间隔采")
    e.add_argument("--max-k", type=int, default=24, dest="max_k",
                   help="events 模式只挑 k 已降到这以下的那些事件（太早的时候还没成形）")
    e.add_argument("--min-k", type=int, default=3, help="平台至少要几个群才算结构")
    e.set_defaults(func=cmd_evo)

    b = sub.add_parser("board", help="一屏看全：五层读数拼到一张图上")
    common(b)
    b.add_argument("--kick", type=float, default=1e-6)
    b.add_argument("--load-window", type=int, default=100, dest="load_window",
                   help="负荷按多少个 tick 一个窗口记")
    b.add_argument("--size", type=int, default=460, help="形状面板边长（像素）")
    b.add_argument("--out", default="board.png", help="落在 out/ 下的文件名")
    b.set_defaults(func=cmd_board)

    s = sub.add_parser("strip", help="诊断带：事件和读数放在同一条时间轴上")
    common(s)
    s.add_argument("--frac", type=float, default=3e-2, help="在哪一档分辨率上判划分变化")
    s.add_argument("--max-frames", type=int, default=6, dest="max_frames")
    s.add_argument("--thumb", type=int, default=220, help="缩略图边长（像素）")
    s.add_argument("--out", default="strip.png", help="落在 out/ 下的文件名")
    s.set_defaults(func=cmd_strip)

    g = sub.add_parser("read", help="读一张真图：把边表当关系组，跑望云")
    common(g)
    g.add_argument("--edges", required=True, help="一行一条关系的边表（u v [w]，# 开头是注释）")
    g.add_argument("--base", type=int, default=0, help="id 从几开始（0 或 1）")
    g.add_argument("--weighted", choices=["auto", "yes", "no"], default="auto",
                   help="有第三列就按权重的比例算；no = 每条一样强")
    g.add_argument("--kick", type=float, default=1e-6)
    g.add_argument("--frac-trace", type=float, default=2e-2)
    g.add_argument("--frac-coarse", type=float, default=1e-1)
    g.add_argument("--frac-mid", type=float, default=1e-2)
    g.add_argument("--frac-fine", type=float, default=1e-3)
    g.add_argument("--out", default="read.png", help="落在 out/ 下的图名")
    g.set_defaults(func=cmd_read)

    pt = sub.add_parser("paint", help="把读数画成一张图（参数驱动）")
    common(pt)
    pt.add_argument("--kick", type=float, default=1e-6)
    pt.add_argument("--frac-mid", type=float, default=1e-2, dest="frac_mid")
    pt.add_argument("--size", type=int, default=560)
    pt.add_argument("--brief", default="", help="想要的样子，如「大眼睛，少花纹，星空」")
    pt.add_argument("--out", default="paint.png", help="落在 out/ 下的图名")
    pt.set_defaults(func=cmd_paint)

    sn = sub.add_parser("scene", help="把一张图的读数画成场面：每个群一个物体，关系决定远近和大小")
    common(sn)
    sn.add_argument("--edges", default="", help="给了边表就读真图；不给就用合成的模块图")
    sn.add_argument("--base", type=int, default=0)
    sn.add_argument("--weighted", choices=["auto", "yes", "no"], default="auto")
    sn.add_argument("--kick", type=float, default=1e-6)
    sn.add_argument("--frac-trace", type=float, default=2e-2)
    sn.add_argument("--frac-mid", type=float, default=0.0,
                    help="用哪一档认群；0 = 自动挑一档，让画面里有 6~30 个物体")
    sn.add_argument("--target-k", type=int, default=14, help="自动挑档时想要的物体数")
    sn.add_argument("--size", type=int, default=900)
    sn.add_argument("--out", default="scene.png", help="落在 out/ 下的图名")
    sn.set_defaults(func=cmd_scene)

    an = sub.add_parser("anim", help="让场面动起来：采样若干帧，拼成带子和 GIF")
    common(an)
    an.add_argument("--frames", type=int, default=8, help="采几帧")
    an.add_argument("--kick", type=float, default=1e-6)
    an.add_argument("--cols", type=int, default=4)
    an.add_argument("--frac-mid", type=float, default=3e-2, dest="frac_mid")
    an.add_argument("--size", type=int, default=360)
    an.add_argument("--duration", type=int, default=450, help="GIF 每帧毫秒")
    an.add_argument("--gif", default="", help="GIF 文件名（留空就不存）")
    an.add_argument("--out", default="anim.png", help="带子文件名")
    an.set_defaults(func=cmd_anim)

    dr = sub.add_parser("drill", help="点开群：看一个群内部的人长什么样")
    common(dr)
    dr.add_argument("--kick", type=float, default=1e-6)
    dr.add_argument("--frac-trace", type=float, default=2e-2)
    dr.add_argument("--frac-mid", type=float, default=1e-3, dest="frac_mid",
                    help="用这一档认群（细一点才看得到内部）")
    dr.add_argument("--top", type=int, default=6, help="画最大的几个群")
    dr.add_argument("--focus", type=int, default=-1,
                    help="连续放大：只放大这一个群，贴在它自己在图上的位置")
    dr.add_argument("--cols", type=int, default=3)
    dr.add_argument("--size", type=int, default=300)
    dr.add_argument("--out", default="drill.png")
    dr.set_defaults(func=cmd_drill)

    zm = sub.add_parser("zoom", help="再往里一层：盯住一个单位，看它连着谁")
    common(zm)
    zm.add_argument("--kick", type=float, default=1e-6)
    zm.add_argument("--frac-trace", type=float, default=2e-2)
    zm.add_argument("--member", type=int, default=0, help="看哪个单位")
    zm.add_argument("--hops", type=int, default=1, help="往外走几跳")
    zm.add_argument("--size", type=int, default=420)
    zm.add_argument("--out", default="zoom.png")
    zm.set_defaults(func=cmd_zoom)

    sv = sub.add_parser("serve", help="起本地观察台（网页端，零新依赖）")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--open", action="store_true", help="起来以后自动开浏览器")
    sv.set_defaults(func=cmd_serve)

    sc = sub.add_parser("scan", help="真图上「该在哪个 tick 看」：几个 tick 的读数摆成一张表")
    common(sc)
    sc.add_argument("--edges", required=True)
    sc.add_argument("--base", type=int, default=0)
    sc.add_argument("--weighted", choices=["auto", "yes", "no"], default="auto")
    sc.add_argument("--kick", type=float, default=1e-6)
    sc.add_argument("--ticks-list", default="100,200,400,800,1600", dest="ticks_list")
    sc.add_argument("--frac-mid", type=float, default=1e-2, dest="frac_mid")
    sc.add_argument("--target-k", type=int, default=12, dest="target_k")
    sc.set_defaults(func=cmd_scan)

    pt = sub.add_parser("portrait", help="画像：从读数长出一张肖像")
    common(pt)
    pt.add_argument("--kick", type=float, default=1e-6)
    pt.add_argument("--frac-trace", type=float, default=2e-2)
    pt.add_argument("--frac-mid", type=float, default=1e-3, dest="frac_mid")
    pt.add_argument("--size", type=int, default=600)
    pt.add_argument("--out", default="portrait.png")
    pt.set_defaults(func=cmd_portrait)


    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
