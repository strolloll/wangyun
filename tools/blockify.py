"""对照实验：把真图「做出瓶颈」，看读数能不能认回来。

做法很简单——**图不变、权重不变**，只做一件事：跨组的那些关系乘一个系数 α（默认 0.02）。

如果做了瓶颈之后望云就能读出那 11 个组，那说明问题不在读数，而在**真图的社区
不是瓶颈**：真实网络里「同一个组」是按语义分的，不是按「切断它要花多少代价」分的。
反过来，如果做了瓶颈还是读不出，那就是读数自己的问题。

    python tools/blockify.py data/miserables.txt data/miserables_labels.txt --alpha 0.02
输出 data/miserables_block.txt（带一条注释说明 α）。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def read_labels(path: str | Path) -> dict[int, str]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        out[int(parts[0])] = parts[-1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("edges")
    ap.add_argument("labels")
    ap.add_argument("--alpha", type=float, default=0.02, help="跨组关系的权重乘数")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = read_labels(args.labels)
    src = Path(args.edges)
    dst = Path(args.out) if args.out else src.with_name(src.stem + "_block.txt")
    kept, cut = 0, 0
    with dst.open("w", encoding="utf-8") as f:
        f.write(f"# {src.name} 的对照组：跨组关系 × {args.alpha}，其余不变\n")
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            u, v, w = line.split()[:3]
            same = ref.get(int(u)) == ref.get(int(v))
            if same:
                kept += 1
                f.write(f"{u} {v} {w}\n")
            else:
                cut += 1
                f.write(f"{u} {v} {float(w) * args.alpha:.6f}\n")
    print(f"{dst}  组内 {kept} 条、跨组 {cut} 条（×{args.alpha}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
