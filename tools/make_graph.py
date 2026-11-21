"""把一张真图转成望云吃的边表。

现在只认 vega-datasets 那套 {"nodes":[{name,group,index}], "links":[{source,target,value}]}。
输出两份文件：

    data/<name>.txt          u v w        （一行一条关系，0 起编号）
    data/<name>_labels.txt   u 原始名字 参照分组

参照分组只用来事后对账，**不进入演算**——望云看不见它。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def convert(src: str | Path, out_dir: str | Path = "data") -> tuple[Path, Path]:
    raw = json.loads(Path(src).read_text(encoding="utf-8"))
    nodes = raw["nodes"]
    links = raw["links"]
    idx = {n["index"]: i for i, n in enumerate(sorted(nodes, key=lambda x: x["index"]))}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    edges = out / f"{stem}.txt"
    labels = out / f"{stem}_labels.txt"
    with edges.open("w", encoding="utf-8") as f:
        f.write(f"# {stem}: {len(nodes)} 点 {len(links)} 条关系（u v 权重）\n")
        for e in links:
            u, v = idx[e["source"]], idx[e["target"]]
            w = e.get("value", 1)
            f.write(f"{u} {v} {w}\n")
    with labels.open("w", encoding="utf-8") as f:
        f.write("# u 名字 参照分组（不进入演算，只用于事后对账）\n")
        for i, n in enumerate(sorted(nodes, key=lambda x: x["index"])):
            f.write(f"{i} {n['name']} {n.get('group', -1)}\n")
    return edges, labels


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/miserables.json"
    e, l = convert(src)
    print(f"{e}  {l}")
