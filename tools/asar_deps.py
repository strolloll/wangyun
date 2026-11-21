"""从一份 asar 归档里抠出 npm 依赖图，当「真有瓶颈的真图」用。

为什么找它：望云前面所有正面结论都在人造图上。要检验「有瓶颈时它真的有用」，
得找一张**结构不由我决定、但我知道正确答案该长什么样**的图。
本机这台机器的 Electron 归档里有一整棵 node_modules——包与包谁依赖谁是客观的，
而且「哪些包属于一家人」（@babel/*、eslint-*、rollup-* …）是我可以事后对账的参照。

    python tools\\asar_deps.py "D:\\deepseek\\resources\\app.asar" --out data\\deps.txt

做法：asar 头是一段 pickle 包着的 JSON，文件内容在归档里按 offset 存。
只读 `package.json`，取 `name` 与 `dependencies`，把「父包 → 子包」当一条有向关系。
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path


def read_header(f) -> tuple[dict, int]:
    """asar 头：pickle 套 pickle。

    实测这份归档的头是：
        [0:4]   = 4            （pickle 头长度）
        [4:8]   = 头部总长
        [8:12]  = 字符串 pickle 长度
        [12:16] = JSON 长度
        [16:16+JSON长度] = JSON
    文件内容的 offset 相对于 `8 + 头部总长`。
    """
    head = f.read(16)
    if len(head) < 16:
        raise ValueError("文件太短，不是 asar")
    size, header_size, str_len, json_size = struct.unpack("<IIII", head[:16])
    if not (0 < json_size < 100_000_000):
        raise ValueError(f"JSON 长度不合理：{json_size}")
    for base_off, length in ((16, json_size), (12, json_size), (16, str_len - 7)):
        if length <= 0:
            continue
        f.seek(base_off)
        raw = f.read(length)
        try:
            tree = json.loads(raw.decode("utf-8"))
        except Exception:
            continue
        if "files" in tree:
            return tree, 8 + header_size
    raise ValueError("读不出 asar 头")


def walk(node: dict, prefix: str, out: dict[str, dict]) -> None:
    for name, item in node.get("files", {}).items():
        path = f"{prefix}{name}"
        if "files" in item:
            walk(item, path + "/", out)
        else:
            out[path] = item


def main() -> int:
    ap = sys.argv
    if len(ap) < 2:
        print(__doc__)
        return 2
    src = Path(ap[1])
    out_path = Path("data/deps.txt")
    for i, a in enumerate(ap):
        if a == "--out" and i + 1 < len(ap):
            out_path = Path(ap[i + 1])

    with src.open("rb") as f:
        tree, base = read_header(f)
        files: dict[str, dict] = {}
        walk(tree, "", files)
        print(f"归档里 {len(files)} 个文件")

        pkgs = {p: m for p, m in files.items() if p.endswith("package.json")}
        print(f"其中 package.json {len(pkgs)} 个")

        meta: dict[str, dict] = {}
        for path, item in pkgs.items():
            off = int(item.get("offset", 0))
            size = int(item.get("size", 0))
            if size <= 0 or size > 2_000_000:
                continue
            f.seek(base + off)
            raw = f.read(size)
            try:
                data = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                continue
            name = data.get("name")
            if not isinstance(name, str):
                continue
            meta[name] = {"deps": data.get("dependencies") or {},
                          "path": path}

    print(f"能读出名字的包 {len(meta)} 个")
    edges = []
    for name, info in meta.items():
        for dep in info["deps"]:
            if dep in meta:
                edges.append((name, dep))
    ids = {name: i for i, name in enumerate(sorted(meta))}
    print(f"包间依赖 {len(edges)} 条")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# npm dependency graph from asar\n")
        for a, b in edges:
            f.write(f"{ids[a]} {ids[b]} 1\n")
    labels = out_path.with_name(out_path.stem + "_labels.txt")
    with labels.open("w", encoding="utf-8") as f:
        f.write("# name per node\n")
        for name, i in sorted(ids.items(), key=lambda kv: kv[1]):
            f.write(f"{i} {name}\n")
    print(f"写出 {out_path} 与 {labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
