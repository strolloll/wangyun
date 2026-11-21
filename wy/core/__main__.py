"""`python -m wy.core`：跑一份读数，把 bundle 打成 JSON。

给「不装 Pillow、不装命令行」的使用方用——别的语言、脚本、网页后端、DSH 插件。
只依赖 numpy：

    python -m wy.core --n 64 --ticks 400 > bundle.json
    python -m wy.core --n 128 --ticks 1200 --members --out out/bundle.json
"""

from __future__ import annotations

import argparse
import sys

from .bundle import dumps, observe
from .network import Config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m wy.core",
        description="跑一份望云读数，输出 bundle JSON（纯 numpy，不需要 Pillow）",
    )
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--modules", type=int, default=8)
    ap.add_argument("--ticks", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--frac", type=float, default=3e-2, help="划分用的关系比例阈值")
    ap.add_argument("--probe", type=int, default=20, help="每几 tick 记一个采样帧")
    ap.add_argument("--need", type=int, default=4, help="稳定位数门槛")
    ap.add_argument("--iters", type=int, default=280, help="布局步数上限")
    ap.add_argument("--walk-power", type=float, default=1.0)
    ap.add_argument("--members", action="store_true", help="每个群带上成员下标")
    ap.add_argument("--out", default=None, help="写到文件；不给就打到 stdout")
    args = ap.parse_args(argv)

    cfg = Config(
        n=args.n,
        modules=args.modules,
        seed=args.seed,
        walk_power=args.walk_power,
    )
    bundle = observe(
        cfg,
        args.ticks,
        frac=args.frac,
        probe=args.probe,
        need=args.need,
        seed=args.seed,
        iters=args.iters,
        members=args.members,
    )
    text = dumps(bundle)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n")
    else:
        sys.stdout.write(text)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
