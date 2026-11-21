"""像什么：把一句人话变成一次演算的配置。

望云只能画「读数长出来的样子」。所以「我要大眼睛、少花纹、星空」这件事，
必须先变成**读数目标**，再变成**网络配置**，最后才能画。

两步都在这里：
1. `parse(text)`     人话 → `Brief`（目标参数：眼睛大小、花纹多少、星多少、色调）
2. `choose(brief)`   目标参数 → 一组候选配置，各跑一次短演算，按「画出来离目标多近」挑一个

这一步不假装能画任意东西。它只会在望云本来就长得出的范围内挑，
挑不到就直接说「这个目标在范围内找不到」，而不是硬画一张不像的。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from .core.network import Config
from .paint import Params, params_from_engine

# ---------------------------------------------------------------- 人话 → 目标


@dataclass(frozen=True)
class Brief:
    """想要的样子：三个尺度上的意图 + 一句原话。"""

    text: str = ""
    eye: str = "medium"     # small | medium | large
    swirl: str = "some"     # none | some | lots
    stars: str = "night"    # calm | night | dense
    mood: str = "calm"      # calm | dazed | bright

    def target(self) -> Params:
        """意图 → 目标参数（区间中点）。"""
        eye_h = {"small": 0.24, "medium": 0.30, "large": 0.37}[self.eye]
        swirl = {"none": 0.30, "some": 0.55, "lots": 0.85}[self.swirl]
        stars = {"calm": 0.22, "night": 0.45, "dense": 0.85}[self.stars]
        glow = {"calm": 0.42, "dazed": 0.30, "bright": 0.80}[self.mood]
        hue = {"calm": 0.470, "dazed": 0.500, "bright": 0.462}[self.mood]
        return replace(Params(), eye_h=eye_h, swirl=swirl, stars=stars, glow=glow, hue_body=hue)

    def tolerance(self) -> dict:
        """每个意图允许的误差。范围给得宽，是因为它本来就不是精确仪器。"""
        return {"eye_h": 0.035, "swirl": 0.14, "stars": 0.20, "hue_body": 0.03}


_WORDS = {
    "eye": [("large", ("大眼", "眼睛大", "大眼睛", "呆", "萌", "大头")),
            ("small", ("小眼", "眼睛小", "精", "凶", "细"))],
    "swirl": [("none", ("少花纹", "花纹少", "没花纹", "干净", "素气", "光溜", "简单")),
              ("lots", ("花纹多", "花纹", "纹多", "复杂", "花哨"))],
    "stars": [("dense", ("星空", "星多", "满天", "热闹")),
              ("calm", ("安静", "少星", "素净", "夜静", "空一点"))],
    "mood": [("bright", ("亮", "精神", "清醒", "有神")),
             ("dazed", ("困", "睡", "迷糊", "没睡醒"))],
}


def parse(text: str) -> Brief:
    """从一句话里读出意图。读不出来就用默认（不猜，也不报错）。"""
    t = (text or "").strip()
    got = {}
    for key, rules in _WORDS.items():
        for value, words in rules:
            if any(w in t for w in words):
                got[key] = value
                break
    return Brief(text=t, **got)


# ---------------------------------------------------------------- 目标参数 → 配置


@dataclass(frozen=True)
class Candidate:
    """一组候选：模块数、跨模块强度、走路规则指数、tick 数、扰动幅度。"""

    modules: int
    inter: tuple
    walk_power: float
    ticks: int
    note: str = ""
    kick: float = 1e-6


def candidates() -> list[Candidate]:
    """候选不是乱撒的：每个都对应一个已知会改变读数方向的旋钮。"""
    out = []
    for modules in (2, 3, 4, 6, 8):
        out.append(Candidate(modules, (0.002, 0.006), 1.0, 600,
                             f"{modules} 个模块：块越多，花纹越多"))
    out.append(Candidate(8, (0.004, 0.010), 2.0, 600, "弱关系打折：结构更耐，眼睛更容易长大"))
    out.append(Candidate(2, (0.002, 0.006), 1.0, 200, "跑得短：结构还没分开，背景更闹"))
    out.append(Candidate(3, (0.001, 0.003), 1.5, 1200, "弱耦合 + 长跑：块更清楚，星更少"))
    # 星星要亮：得让违约读数真的高——那是「还没和解」的意思，靠扰动和短跑能拿到
    out.append(Candidate(3, (0.002, 0.006), 1.0, 150, "扰动大：背景星多", kick=2e-3))
    out.append(Candidate(4, (0.004, 0.010), 1.0, 300, "中扰动：星稍多", kick=5e-4))
    out.append(Candidate(6, (0.002, 0.006), 1.0, 900, "扰动极小：几乎没星", kick=1e-9))
    # 要「安静」：得让它真的收敛到底——长跑 + 极小扰动，defect 才掉到 1e-7 以下
    out.append(Candidate(6, (0.002, 0.006), 1.0, 6000, "长跑：收干净了，星几乎没有", kick=1e-9))
    out.append(Candidate(8, (0.002, 0.006), 2.0, 4000, "长跑 + 弱关系打折：块清楚又安静", kick=1e-8))
    return out


def _cfg_of(c: Candidate, seed: int, n: int = 64) -> Config:
    return Config(n=n, modules=c.modules, seed=seed, kick=c.kick,
                  inter_ratio=c.inter, walk_power=c.walk_power)


def _score(p: Params, target: Params, tol: dict) -> float:
    """离目标多远：每项按容差归一，越接近 0 越好。"""
    parts = []
    for name, t in tol.items():
        d = abs(float(getattr(p, name)) - float(getattr(target, name))) / max(t, 1e-9)
        parts.append(min(d, 4.0) ** 2)
    return float(math.sqrt(sum(parts) / len(parts)))


def evaluate(c: Candidate, brief: Brief, seed: int = 7, n: int = 64) -> tuple[float, Params, dict]:
    """跑一次短演算，把读数映射成参数，量它离目标多远。"""
    from .core.digits import StabilityMonitor
    from .core.engine import Engine
    from .core.network import Network
    from .core.shape import group_table, scale_of

    # 每个模块 12 个单位：保证 n 能被 modules 整除，且模块大到放得下 intra=3（环+对径）
    cfg = _cfg_of(c, seed, c.modules * 12)
    net = Network(cfg)
    eng = Engine(net)
    mon = StabilityMonitor(net.n, cfg.window, cfg.digits)
    for _ in range(c.ticks):
        eng.step()
        mon.push(eng.v)
    scale = scale_of(eng.v)
    groups = group_table(eng.v, net, 1e-2 * scale, mon, 4, touched=eng.recent(cfg.window))
    spans = [g.span for g in groups if len(g.members) > 1]
    p = params_from_engine(len(groups), spans, mon.mean_length(eng.recent(cfg.window)), eng.defect())
    return _score(p, brief.target(), brief.tolerance()), p, {
        "k": len(groups), "defect": float(eng.defect()),
        "stable": float(mon.mean_length(eng.recent(cfg.window))),
    }


@dataclass
class Choice:
    brief: Brief
    candidate: Candidate
    params: Params
    score: float
    readings: dict
    report: list = field(default_factory=list)

    def ok(self, limit: float = 1.6) -> bool:
        return self.score <= limit


def choose(brief: Brief, seed: int = 7, n: int = 64) -> Choice:
    """挑一组最接近目标的配置；挑不到就照实说。"""
    rows = []
    for c in candidates():
        try:
            s, p, r = evaluate(c, brief, seed, n)
        except Exception as exc:  # 某个候选构造成本太高就跳过，不拖垮整次挑选
            rows.append((9.9, c, None, {"error": str(exc)}))
            continue
        rows.append((s, c, p, r))
    rows.sort(key=lambda row: (row[0], row[1].modules))
    s, c, p, r = rows[0]
    return Choice(brief=brief, candidate=c, params=p, score=s, readings=r,
                  report=[(row[0], row[1].note) for row in rows])
