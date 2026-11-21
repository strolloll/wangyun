"""把望云读出来的群，跟一张已知分组对一次账。

不参与演算，只在事后问：它读出来的划分，跟外面的人已经知道的划分，像不像。
两个数：

- 纯度（purity）：每个读出来的群里人数最多的那个参照组占它多少；然后再按人数加权。
  越接近 1 越像「一锅端对了」，但它看不见「把一个组拆成三份」这种错。
- 调整兰德指数（ARI）：把「同一组 / 不同组」这一对关系拿出来，数一致和不一致的对数，
  再扣掉随机就能撞上的那部分。1 = 完全相同，0 = 跟瞎猜一样，负 = 比瞎猜还差。
  它对「拆」和「并」都敏感，所以是主判据。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def read_groups_csv(path: str | Path) -> dict[int, int]:
    """群表 → {单位: 群号}。"""
    out: dict[int, int] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        gid = int(line.split(",")[0])
        members = line.rsplit('"', 2)[1] if '"' in line else ""
        for u in members.split():
            out[int(u)] = gid
    return out


def read_labels(path: str | Path) -> dict[int, str]:
    """`u 名字 参照分组` → {单位: 分组}。"""
    out: dict[int, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        out[int(parts[0])] = parts[-1]
    return out


def purity(pred: dict[int, int], ref: dict[int, str]) -> float:
    keys = sorted(set(pred) & set(ref))
    if not keys:
        return 0.0
    tot = 0
    for g in set(pred[u] for u in keys):
        members = [u for u in keys if pred[u] == g]
        counts: dict[str, int] = {}
        for u in members:
            counts[ref[u]] = counts.get(ref[u], 0) + 1
        tot += max(counts.values())
    return tot / len(keys)


def adjusted_rand(pred: dict[int, int], ref: dict[int, str]) -> float:
    keys = sorted(set(pred) & set(ref))
    n = len(keys)
    if n < 2:
        return 0.0
    a = np.asarray([pred[u] for u in keys])
    b = np.asarray([ref[u] for u in keys])
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)

    def comb2(x):
        x = np.asarray(x, dtype=np.float64)
        return x * (x - 1.0) / 2.0

    sum_ij = comb2(table).sum()
    sum_i = comb2(table.sum(axis=1)).sum()
    sum_j = comb2(table.sum(axis=0)).sum()
    total = comb2(n)
    expected = sum_i * sum_j / total if total else 0.0
    mx = (sum_i + sum_j) / 2.0
    if mx == expected:
        return 1.0 if sum_ij == expected else 0.0
    return float((sum_ij - expected) / (mx - expected))


def score(groups_csv: str | Path, labels: str | Path) -> dict:
    pred = read_groups_csv(groups_csv)
    ref = read_labels(labels)
    return {
        "groups": len(set(pred.values())),
        "purity": purity(pred, ref),
        "ari": adjusted_rand(pred, ref),
        "pred": pred,
        "ref": ref,
    }


if __name__ == "__main__":
    g = sys.argv[1] if len(sys.argv) > 1 else "out/groups.csv"
    l = sys.argv[2] if len(sys.argv) > 2 else "data/miserables_labels.txt"
    s = score(g, l)
    print(f"群数 {s['groups']}  纯度 {s['purity']:.3f}  ARI {s['ari']:.3f}")
