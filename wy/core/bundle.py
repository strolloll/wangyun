"""一次观察的全部读数，做成可以直接落盘的 JSON。

这是「核心」对外的交付面：别的软件（网页端、桌面壳、别人的工具）不该自己
跑 Engine、算群、摆位置——只该拿这份 bundle，然后按自己的方式画。

里面有什么（都是参数定出来的，没有外部渲染后端）：

    meta      这次跑的确定参数（n / seed / window / ticks / frac / need）
    readings  每个 tick 的读数：k(t)、违约读数（对数）、稳定位数
    events    划分变化的时刻：并 / 分 / 挪动，带前后群数
    frames    采样帧：群表 + 群间强度 + 坐标 + 色相，色相接上一帧记路（§30）

纯演算，不 import Pillow、不碰文件；`dumps()` 负责转成 JSON 文本。
"""

from __future__ import annotations

import json
import math

import numpy as np

from .color import hue_track, hue_vec
from .digits import StabilityMonitor
from .engine import Engine
from .layout import carry_pos, make_shape
from .network import Config, Network
from .shape import describe_change, group_table, k_of, partition, scale_of

__all__ = ["observe", "dumps"]


def _num(x: float, nd: int = 6):
    """浮点转 JSON：非有限值写 null，别偷偷塞个 NaN 让下游瞎猜。"""
    v = float(x)
    return round(v, nd) if math.isfinite(v) else None


def _canon(labels: np.ndarray) -> dict[int, tuple[int, ...]]:
    """按根收成员：{根: 成员元组}。和带子里的写法一致，用来和上一帧比。"""
    out: dict[int, list[int]] = {}
    for u, g in enumerate(labels.tolist()):
        out.setdefault(int(g), []).append(u)
    return {g: tuple(sorted(m)) for g, m in out.items()}


def observe(
    cfg: Config,
    ticks: int,
    *,
    frac: float = 3e-2,
    probe: int = 20,
    need: int = 4,
    seed: int = 0,
    iters: int = 280,
    members: bool = False,
) -> dict:
    """跑一遍望云，把每个 tick 的读数和采样帧收成 JSON-able 的 dict。

    `probe` 是采样间隔：读数每 tick 都记，群表/坐标/色相每 `probe` 个 tick 记一帧。
    `members=True` 时帧里带成员下标（网页上想按人 hover 用）；默认只给人数。
    """
    if ticks <= 0:
        raise ValueError("ticks 要是正数")
    if probe <= 0:
        raise ValueError("probe 要是正数")

    net = Network(cfg)
    eng = Engine(net)
    mon = StabilityMonitor(net.n, cfg.window, cfg.digits)
    win = cfg.window

    t_x: list[int] = []
    xs_k: list[int] = []
    xs_defect: list[float] = []
    xs_stable: list[float] = []

    frames: list[dict] = []
    events: list[dict] = []
    prev_hues: dict[frozenset, float] = {}
    prev_groups = None
    prev_pos = np.zeros((0, 2), dtype=np.float64)
    prev_keys: set[int] = set()
    prev_members: dict[int, tuple[int, ...]] = {}

    for t in range(1, ticks + 1):
        eng.step()
        mon.push(eng.v)
        v = eng.v
        scale = scale_of(v)

        t_x.append(t)
        xs_k.append(k_of(v, net, frac * scale))
        xs_defect.append(float(eng.defect()))
        xs_stable.append(float(mon.mean_length(eng.recent(win))))

        if t % probe:
            continue

        r = frac * scale
        labels = partition(v, net, r)
        groups = group_table(v, net, r, mon, need=need,
                             touched=eng.recent(win), labels=labels)
        canon = _canon(labels)

        if prev_groups is not None:
            keys = set(canon)
            if keys != prev_keys:
                ch = describe_change(list(prev_members.values()), list(canon.values()))
                events.append({
                    "tick": t,
                    "kind": ch["kind"],
                    "text": ch["text"],
                    "k_before": ch["k_before"],
                    "k_after": ch["k_after"],
                })

        init = None
        if prev_groups is not None and len(prev_groups) > 0 and prev_pos.size == len(prev_groups) * 2:
            init = carry_pos(prev_groups, prev_pos, groups)
        shape = make_shape(groups, net, seed=seed, init=init, iters=iters)

        prev_hues, moved = hue_track(prev_hues, groups, shape.strength)
        hues = hue_vec(prev_hues, groups)

        frame = {
            "t": t,
            "k": len(groups),
            "hue_moved": _num(moved),
            "groups": [
                {
                    "gid": int(g.gid),
                    "count": g.count,
                    "value": _num(g.value),
                    "span": _num(g.span),
                    "stable": _num(g.stable),
                    "settled": bool(g.settled),
                    "hue": _num(float(hues[i]), 5),
                    **({"members": list(g.members)} if members else {}),
                }
                for i, g in enumerate(groups)
            ],
            "strength": [[_num(shape.strength[i, j], 5) for j in range(len(groups))]
                         for i in range(len(groups))],
            "pos": [[_num(shape.pos[i, 0], 5), _num(shape.pos[i, 1], 5)]
                    for i in range(len(groups))],
        }
        frames.append(frame)

        prev_groups = groups
        prev_pos = np.asarray(shape.pos, dtype=np.float64)
        prev_keys = set(canon)
        prev_members = canon

    return {
        "meta": {
            "n": cfg.n,
            "modules": cfg.modules,
            "seed": cfg.seed,
            "window": cfg.window,
            "digits": cfg.digits,
            "ticks": ticks,
            "frac": frac,
            "probe": probe,
            "need": need,
            "exact": bool(cfg.exact),
        },
        "readings": {
            "t": t_x,
            "k": xs_k,
            "defect": [_num(x) for x in xs_defect],
            "stable": [_num(x) for x in xs_stable],
        },
        "events": events,
        "frames": frames,
    }


def dumps(bundle: dict) -> str:
    """bundle → JSON 文本。中文不转义，读起来还是人话。"""
    return json.dumps(bundle, ensure_ascii=False, allow_nan=False)
