r"""量诊断带色相层：峰值到底落在并群还是分开上。

§30.5 的说法要能被重复量出来，别只在文档里写死。

    & $py -u tools\probe_hue_peak.py
"""

from __future__ import annotations

import time

from wy.network import Config
from wy.strip import collect


def run(n: int = 128, seed: int = 7, ticks: int = 4000) -> None:
    t0 = time.time()
    r = collect(Config(n=n, seed=seed), ticks)
    kinds = {int(e["tick"]): e["kind"] for e in r["events"]}
    moves = {int(t): float(m) for t, m in zip(r["hue_x"], r["hue_moved"])}
    peak = max(moves, key=moves.get)
    merge_max = max((moves[t] for t in moves if kinds.get(t) == "merge"), default=0.0)
    split_max = max((moves[t] for t in moves if kinds.get(t) == "split"), default=0.0)
    print(f"n={n} seed={seed} ticks={ticks}  {time.time() - t0:.1f}s")
    print(f"事件 {len(kinds)} 次（并 {sum(v == 'merge' for v in kinds.values())}"
          f" / 分 {sum(v == 'split' for v in kinds.values())}），"
          f"采样点 {len(moves)} 个，其中非零 {sum(1 for m in moves.values() if m > 0)} 个")
    print(f"峰值 t={peak}（{kinds.get(peak)}）moved={moves[peak]:.3f}"
          f"  并最大 {merge_max:.3f}  分最大 {split_max:.3f}")
    print("峰值落在分开上" if kinds.get(peak) == "split" else "峰值落在并群上")


if __name__ == "__main__":
    run()
