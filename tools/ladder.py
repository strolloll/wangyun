r"""r 能一直往下调吗：k(r) 的分辨率地板在哪，换数值表示之后地板动了没有。

用法：
    & $py -u tools\ladder.py --ticks 4000 20000 --exact
    & $py -u tools\ladder.py --ticks 4000 20000

它问的是一件具体的事：现在的 k(r) 阶梯最细只到 1e-4（相对）。原因不是关系不平滑，
而是**读数本身有噪声底**（违约读数）。把 r 调到噪声底以下，k(r) 会一路涨到 n——
每个单位自己一群，那不是「更细的结构」，是「分不清了」。

两套数值表示各跑一遍，看噪声底是不是跟着表示走。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.network import Config, Network  # noqa: E402
from wy.shape import k_of, scale_of  # noqa: E402

LADDER = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7,
                   3e-8, 1e-8, 3e-9, 1e-9, 1e-11, 1e-13, 1e-15, 1e-17])


def build(cfg: Config):
    net = Network(cfg)
    if getattr(cfg, "exact", False):
        from wy.exact import ExactEngine

        eng = ExactEngine(net, digits=int(cfg.exact_digits), cdigits=24)
    else:
        from wy.engine import Engine

        eng = Engine(net)
    return net, eng


def values(eng):
    return eng.to_float() if hasattr(eng, "to_float") else eng.v


def defect(eng, fv):
    """float 引擎的 defect() 收的是残差数组，exact 引擎收的是浮点视图——别混。"""
    if hasattr(eng, "texts"):  # ExactEngine
        return float(eng.defect(fv))
    return float(eng.defect())


def main() -> int:
    ap = argparse.ArgumentParser(prog="ladder")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--modules", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--kick", type=float, default=1e-6)
    ap.add_argument("--ticks", type=int, nargs="+", default=[4000, 20000])
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--exact-digits", type=int, default=30)
    a = ap.parse_args()

    for ticks in a.ticks:
        cfg = Config(n=a.n, modules=a.modules, seed=a.seed, kick=a.kick,
                     exact=a.exact, exact_digits=a.exact_digits)
        net, eng = build(cfg)
        for _ in range(ticks):
            eng.step()
        fv = values(eng)
        sc = scale_of(fv)
        parts = []
        for f in LADDER:
            parts.append(k_of(fv, net, f * sc))
        tag = "exact " if a.exact else "float64"
        d = defect(eng, fv)
        print(f"\n{tag}  ticks={ticks}  scale={sc:.1f}  defect={d:.3e}"
              f"  （噪声底 = defect 那一栏）")
        print("  frac :  " + " ".join(f"{f:7.0e}" for f in LADDER))
        print("  k(r) :  " + " ".join(f"{k:7d}" for k in parts))
        # 第一次到 n（每个单位自己一群）的档位，就是这条读数的分辨率地板。
        first_n = next((f for f, k in zip(LADDER, parts) if k >= net.n), None)
        # 噪声底换算成相对量：defect * scale 是绝对差量级。
        floor = d
        print(f"  第一次 k>=n 的档: {first_n if first_n is not None else '没有'}"
              f"   相对噪声底 ~ {floor:.1e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
