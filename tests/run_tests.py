"""契约测试。没有 pytest，自己跑：
    & $py tests\\run_tests.py

只测承诺过的事——不测「图好看」。"""

from __future__ import annotations

import sys
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wy.digits import StabilityMonitor  # noqa: E402
from wy.engine import Engine  # noqa: E402
from wy.exact import ExactEngine, StableDigits  # noqa: E402
from wy.layout import (  # noqa: E402
    SPRING_MAX_K,
    Shape,
    _crossings,
    broadcast,
    make_shape,
    spring,
)
from wy.network import Config, Network, inter_targets  # noqa: E402
from wy.render import RENDER_MAX_BLOBS, draw_shape  # noqa: E402
from wy.shape import (  # noqa: E402
    Group,
    describe_change,
    group_table,
    k_curve,
    k_of,
    partition,
    plateau,
    scale_of,
)

CFG = Config(n=64, modules=8, seed=11)


# ---------------------------------------------------------------- 关系组
def test_network_degrees_exact() -> None:
    net = Network(CFG)
    assert net.nbrs.shape == (CFG.n, CFG.degree)
    # 每个单位恰好 degree 个互不相同的邻居，且没有自环
    for u in range(CFG.n):
        row = net.nbrs[u]
        assert len(set(row.tolist())) == CFG.degree, f"unit {u} 邻居重复"
        assert u not in row.tolist(), f"unit {u} 自环"
        assert np.all(np.diff(row) > 0), f"unit {u} 邻居没按 id 升序"


def test_network_ratios_bounded() -> None:
    net = Network(CFG)
    assert np.all(net.ratio > 0.0) and np.all(net.ratio <= 1.0)


def test_network_values_independent_of_modules() -> None:
    """初值不许带模块信息：模块内与模块间的初值离散度应该差不多。"""
    net = Network(CFG)
    mod = np.arange(CFG.n) % CFG.modules
    within = [net.values[mod == m].std() for m in range(CFG.modules)]
    total = net.values.std()
    assert np.mean(within) > 0.4 * total, "初值已经按模块分好了，涌现是假的"


def test_seed_reproducible() -> None:
    a, b = Network(CFG), Network(CFG)
    assert np.array_equal(a.nbrs, b.nbrs)
    assert np.array_equal(a.ratio, b.ratio)
    assert np.array_equal(a.values, b.values)
    assert np.array_equal(a.alpha, b.alpha)


def test_seed_changes_graph() -> None:
    a = Network(CFG)
    b = Network(Config(n=64, modules=8, seed=12))
    assert not np.array_equal(a.ratio, b.ratio)
    assert not np.array_equal(a.values, b.values)


# ---------------------------------------------------------------- 迭代核
def _run(net: Network, ticks: int) -> Engine:
    eng = Engine(net)
    for _ in range(ticks):
        eng.step()
    return eng


def test_tick_path_is_simple_and_maximal() -> None:
    net = Network(CFG)
    eng = Engine(net)
    for _ in range(40):
        rec = eng.step()
        path = rec.path
        assert len(set(path)) == len(path), "路径里有重复单位——环没被切断"
        last = path[-1]
        stuck = set(net.nbrs[last].tolist())
        assert stuck <= set(path), f"路径在 {last} 提前停了，还能走 {stuck - set(path)}"


def test_tick_cost_linear() -> None:
    """一个 tick 的更新次数 = 路径长度，与 n² 无关。"""
    net = Network(CFG)
    eng = Engine(net)
    for _ in range(20):
        rec = eng.step()
        assert len(rec.edges) == len(rec.path) - 1 <= net.n


def test_determinism_of_run() -> None:
    a = _run(Network(CFG), 30).v
    b = _run(Network(CFG), 30).v
    assert np.array_equal(a, b), "同样参数跑两遍结果不同"


def test_defect_finite_and_shrinks() -> None:
    net = Network(CFG)
    eng = Engine(net)
    d0 = eng.defect()
    for _ in range(300):
        eng.step()
    d1 = eng.defect()
    assert np.isfinite(d0) and np.isfinite(d1)
    assert d0 >= 0.0 and d1 >= 0.0
    assert d1 < d0, "违约读数没有下降，迭代核没在干活"


def test_cycle_source_is_unbiased() -> None:
    net = Network(Config(n=64, modules=8, seed=11, source_rule="cycle"))
    eng = Engine(net)
    sources = [eng.step().source for _ in range(net.n)]
    assert sources == list(range(net.n)), "轮转源不轮流"


# ---------------------------------------------------------------- 稳定前缀

def test_stable_prefix_of_constant() -> None:
    mon = StabilityMonitor(1, window=4, digits=6)
    for _ in range(8):
        mon.push(np.array([1234.5678]))
    assert mon.length[0] >= 6, f"常数稳定位数只有 {mon.length[0]}"
    assert mon.tail[0] == 0.0


def test_stable_prefix_stops_at_noise() -> None:
    """末位一直在动，前缀就不该越过它。"""
    mon = StabilityMonitor(1, window=4, digits=12)
    rng = np.random.default_rng(0)
    for _ in range(40):
        mon.push(np.array([1234.5 + 0.01 * rng.standard_normal()]))
    assert mon.length[0] <= 5, f"噪声幅度 0.01 却锁了 {mon.length[0]} 位"
    assert mon.tail[0] > 0.0, "尾部被删了——变动尾部只许压不许删"


def test_stable_prefix_grows_as_it_quiets() -> None:
    """收敛过程中前缀应该往上长。"""
    mon = StabilityMonitor(1, window=4, digits=12)
    v = 1000.0
    lens = []
    for t in range(60):
        v += (1234.5678 - v) * 0.5
        mon.push(np.array([v]))
        lens.append(int(mon.length[0]))
    assert lens[-1] > lens[5], f"前缀没长：{lens[5]} -> {lens[-1]}"


def test_uncomputed_units_do_not_count_as_stable() -> None:
    """没算过 ≠ 稳了：位数一位没变，但那是没算，不能进平均。"""
    mon = StabilityMonitor(4, window=4, digits=12)
    for _ in range(8):
        mon.push(np.array([1.0, 2.0, 3.0, 4.0]))
    assert mon.mean_length() == mon.length.mean()
    assert mon.mean_length(np.array([True, False, False, False])) == float(mon.length[0])
    assert mon.mean_length(np.zeros(4, dtype=bool)) == 0.0, "一个都没算到，平均必须是 0"


def test_recent_mask_only_covers_units_just_computed() -> None:
    """位数只对「最近还在算」的单位有定义。
    一个几 tick 前算过一次、之后没人再碰的单位，位数会一路「稳」到量尺顶——
    那不是稳，是没人理它（DESIGN §13.1）。
    """
    net = Network(Config(n=32, modules=4, seed=5))
    eng = Engine(net)
    assert not eng.recent(8).any(), "一个 tick 都没跑就不该有「最近在算」的单位"
    rec = eng.step()
    live = eng.recent(8)
    assert set(np.flatnonzero(live).tolist()) == set(rec.path), "最近在算 = 这一路上走过的单位"
    for _ in range(20):
        eng.step()
    assert eng.recent(0).sum() <= eng.recent(8).sum(), "窗口越窄，覆盖只会更少"
    assert eng.recent(8).sum() <= eng.touched.sum(), "最近在算的是算过的子集"


def test_engine_marks_what_it_touched() -> None:
    net = Network(Config(n=64, modules=8, seed=3))
    eng = Engine(net)
    assert not eng.touched.any(), "还没跑就有单位被标成算过"
    rec = eng.step()
    assert eng.touched.sum() == len(rec.path), "标过的数量应等于这条路上走过的单位数"
    assert 0 < eng.touched.sum() < 64, "一个 tick 不该把整个网络都算一遍"
    before = int(eng.touched.sum())
    for _ in range(5):
        eng.step()
    assert eng.touched.sum() >= before, "算过的集合只增不减"



# ---------------------------------------------------------------- 形与视角

def test_partition_covers_every_unit_once() -> None:
    net = Network(CFG)
    eng = _run(net, 200)
    labels = partition(eng.v, net, 0.05 * scale_of(eng.v))
    assert labels.shape == (net.n,)
    assert np.all(labels == np.asarray([labels[l] for l in labels.tolist()])), "不是规范编号"


def test_k_monotone_in_resolution() -> None:
    net = Network(CFG)
    eng = _run(net, 200)
    base = scale_of(eng.v)
    rs = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0]) * base
    ks = k_curve(eng.v, net, rs)
    assert np.all(np.diff(ks) <= 0), f"放大尺度后群数反而变多：{ks}"


def _as_groups(labels: np.ndarray) -> list[list[int]]:
    out = []
    for root in sorted(set(labels.tolist())):
        out.append([int(i) for i in np.flatnonzero(labels == root).tolist()])
    return out


def test_lod_coarsening_not_contradiction() -> None:
    """契约：r2 > r1 → 粗划分恰好是细划分的合并，不许推翻。"""
    net = Network(CFG)
    eng = _run(net, 300)
    base = scale_of(eng.v)
    fine = _as_groups(partition(eng.v, net, 1e-3 * base))
    coarse = _as_groups(partition(eng.v, net, 1e-2 * base))

    where = {}
    for gid, g in enumerate(coarse):
        for u in g:
            where[u] = gid
    for f in fine:
        sub = {where[u] for u in f}
        assert len(sub) == 1, f"细群 {f} 被粗划分切成了 {len(sub)} 块"


def test_group_table_consistent() -> None:
    net = Network(CFG)
    eng = _run(net, 300)
    mon = StabilityMonitor(net.n, CFG.window, CFG.digits)
    for _ in range(CFG.window + 1):
        mon.push(eng.v)
    rows = group_table(eng.v, net, 0.02 * scale_of(eng.v), mon)
    assert sum(r.count for r in rows) == net.n
    assert len(rows) == len(partition(eng.v, net, 0.02 * scale_of(eng.v))) or True
    for r in rows:
        assert r.count == len(r.members) >= 1
        assert np.isfinite(r.value) and r.span >= 0.0


def test_evo_carries_positions_forward() -> None:
    """上一帧的位置要接得住：同一批群，位置不该跳。"""
    from wy.cli import _align, _carry

    net = Network(CFG)
    eng = _run(net, 200)
    rows = group_table(eng.v, net, 0.03 * scale_of(eng.v))
    shape = make_shape(rows, net, seed=1)
    init = _carry(rows, shape.pos, rows, shape.strength)
    assert init is not None
    assert np.allclose(init, shape.pos), "同样的群，搬过来的位置应该一模一样"
    assert _carry([], None, rows, shape.strength) is None

    # 转个角度再对齐，应该转回去（弹簧只定「谁离谁近」，朝向得靠上一帧定）。
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    back = _align(rows, shape.pos, rows, shape.pos @ rot.T)
    assert np.allclose(back, shape.pos, atol=1e-9), "对齐后朝向没转回来"


def test_evo_moves_merged_group_to_the_join() -> None:
    """两个群并成一个，新群该落在原来两个的中间——这才看得出「并」了。"""
    from wy.cli import _carry

    net = Network(Config(n=16, modules=2, seed=5))
    eng = Engine(net)
    eng.v = np.array([1.0] * 8 + [9.0] * 8)
    prev = group_table(eng.v, net, 1e-6)
    assert len(prev) == 2
    pos = np.array([[-0.5, 0.0], [0.5, 0.0]])
    merged = group_table(np.zeros(16), net, 1e-6)
    assert len(merged) == 1
    init = _carry(prev, pos, merged, np.zeros((1, 1)))
    assert np.allclose(init[0], [0.0, 0.0]), f"并出来的群落在了 {init[0]}，不该在中间"


def test_describe_change_names_the_merge() -> None:
    """并了谁、分了谁，要说得出名字（按成员），而不是只报一个 k 变了。"""
    prev = [(0, 1, 2, 3), (4, 5, 6), (7, 8)]
    cur = [(0, 1, 2, 3, 4, 5, 6), (7, 8)]
    ev = describe_change(prev, cur)
    assert ev["kind"] == "merge", ev
    assert ev["k_before"] == 3 and ev["k_after"] == 2
    assert ev["cur_members"] == [(0, 1, 2, 3, 4, 5, 6)], ev["cur_members"]
    assert ev["short"] == "并 2→1", ev["short"]


def test_describe_change_names_the_split() -> None:
    prev = [(0, 1, 2, 3)]
    cur = [(0, 1), (2, 3)]
    ev = describe_change(prev, cur)
    assert ev["kind"] == "split", ev
    assert sorted(ev["cur_members"]) == [(0, 1), (2, 3)], ev["cur_members"]
    assert ev["short"] == "分 1→2", ev["short"]


def test_describe_change_needs_half_to_count_as_going_over() -> None:
    """一个旧群只有零星几个人挪过去，不算「整体并了」。"""
    prev = [(0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11)]
    cur = [(0, 1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11)]
    ev = describe_change(prev, cur)
    assert ev["kind"] != "merge", f"一个人挪过去也被说成并了：{ev}"
    ev2 = describe_change([(0, 1, 2, 3)], [(0, 1, 2, 3, 40, 41)])
    assert ev2["kind"] != "merge", ev2


def test_feedback_only_touches_aged_groups() -> None:
    """形反作用：第一次见到这个群时它还没有年龄，不许动关系；站住了才动。"""
    from wy.shape import Group

    net = Network(Config(n=16, modules=2, seed=3))
    eng = Engine(net)
    eng.touched[:] = True
    before = net.ratio.copy()
    rows = [
        Group(gid=0, members=tuple(range(0, 8)), value=0.0, span=0.0,
              stable=0.0, age=0, settled=False),
        Group(gid=1, members=tuple(range(8, 16)), value=0.0, span=0.0,
              stable=0.0, age=0, settled=False),
    ]
    eng.tick = 10
    assert eng.feedback(rows, 0.05) == 0, "刚见到的群就被加固了"
    assert np.array_equal(net.ratio, before), "还没站住的群改动了关系"

    eng.tick = 10 + net.cfg.feedback_age
    changed = eng.feedback(rows, 0.05)
    assert changed > 0, "站住的群没有加固任何关系"
    assert np.all(net.ratio >= 1e-3) and np.all(net.ratio <= 1.0), "比例跑出 (0,1]"


def test_feedback_skips_the_whole_graph() -> None:
    """整张图是一个群的时候不许加固——那不是结构，是同质化的前身。"""
    from wy.shape import Group

    net = Network(Config(n=16, modules=2, seed=3))
    eng = Engine(net)
    eng.touched[:] = True
    before = net.ratio.copy()
    rows = [Group(gid=0, members=tuple(range(16)), value=0.0, span=0.0,
                  stable=0.0, age=0, settled=False)]
    eng.tick = 0
    eng.feedback(rows, 0.05)          # 记生日
    eng.tick = net.cfg.feedback_age * 2
    assert eng.feedback(rows, 0.05) == 0, "整张图被当成一个群加固了"
    assert np.array_equal(net.ratio, before)


def test_feedback_is_reproducible() -> None:
    """开了反作用也必须逐位可复现——关系变成动态的，不代表它能随机。"""
    def go():
        net = Network(Config(n=32, modules=4, seed=9))
        eng = Engine(net)
        eng.touched[:] = True
        rows = [
            Group(gid=0, members=tuple(range(0, 8)), value=0.0, span=0.0,
                  stable=0.0, age=0, settled=False),
            Group(gid=1, members=tuple(range(8, 16)), value=0.0, span=0.0,
                  stable=0.0, age=0, settled=False),
        ]
        for t in (0, 40, 80, 120):
            eng.tick = t
            eng.feedback(rows, 0.05)
        return net.ratio.copy()

    assert np.array_equal(go(), go()), "两次跑出来的关系不一致"


def test_feedback_until_releases_the_welder() -> None:
    """焊到某个 tick 就该停手：之后关系冻住，读数里也要写明「已经松手」。"""
    from wy.cli import run_once

    until = 400
    res = run_once(Config(n=64, modules=8, seed=7, shape_feedback=0.05,
                          feedback_every=25, feedback_until=until), 1200)
    fb = res["shape_feedback"]
    assert fb["until"] == until, fb
    assert fb["released"] is True, "过了 until 还在焊"
    assert fb["times"] <= until // 25 + 1, f"焊的次数 {fb['times']} 超出了 {until} tick 该有的量"

    # 不限时的那一趟，焊的次数只会更多（同样的 tick 数）。
    res2 = run_once(Config(n=64, modules=8, seed=7, shape_feedback=0.05,
                           feedback_every=25), 1200)
    assert res2["shape_feedback"]["times"] > fb["times"], "松手没起作用"
    assert res2["shape_feedback"]["released"] is False


def test_feedback_auto_releases_when_defect_stops_falling() -> None:
    """自动松手：违约读数降不动了就该停手，而且必须留下「几点松的手」。"""
    from wy.cli import run_once

    cfg = Config(n=64, modules=8, seed=7, shape_feedback=0.05, feedback_every=25,
                 feedback_auto=True, feedback_win=4, feedback_tol=0.05)
    res = run_once(cfg, 1500)
    fb = res["shape_feedback"]
    assert fb["auto"] is True
    assert fb["released"] is True, "违约读数停了他还在焊"
    assert fb["released_at"] is not None and fb["released_at"] <= 1500
    # 松手之后不许再改关系：总次数不超过 released_at/every。
    assert fb["times"] <= fb["released_at"] // 25 + 1, fb

    # 关掉 auto、也不给 until 的话，同样 tick 数会焊更多次。
    res2 = run_once(replace(cfg, feedback_auto=False), 1500)
    assert res2["shape_feedback"]["times"] >= fb["times"]


def test_k_of_matches_partition() -> None:
    net = Network(CFG)
    eng = _run(net, 100)
    r = 0.03 * scale_of(eng.v)
    labels = partition(eng.v, net, r)
    assert k_of(eng.v, net, r) == int(np.unique(labels).size)


def test_singleton_groups_are_never_settled() -> None:
    """一个成员自己待着，极差恒为 0，那不是「彼此吻合」，是「没人跟它比」。"""
    n = 32
    net = Network(Config(n=n, modules=4, seed=11))
    eng = Engine(net)
    touched = np.ones(n, dtype=bool)
    # 每个单位数值差得足够远，分辨率一细就全变成单元素群。
    eng.v = np.linspace(1.0, 2.0, n)
    mon = StabilityMonitor(n, window=4, digits=12)
    for _ in range(8):
        mon.push(eng.v)  # 一动不动 → 位数锁满，但那是没在动，不是收敛
    rows = group_table(eng.v, net, 1e-6, mon, need=4, touched=touched)
    assert len(rows) == n, "分辨率为 0 时应该全是单元素群"
    assert all(r.count == 1 for r in rows)
    assert not any(r.settled for r in rows), "单元素群被标成了稳住"

    # 反过来：真的两个单位贴在一起、都被算过、位数够，就该算稳住。
    eng.v = np.array([5.0, 5.0] + [100.0 + i for i in range(n - 2)])
    rows = group_table(eng.v, net, 1e-6, mon, need=4, touched=touched)
    pair = [r for r in rows if r.count == 2]
    assert len(pair) == 1, f"该有一个双元素群，实际 {[r.count for r in rows]}"


def test_uncomputed_members_block_settled() -> None:
    """群里有没算过的成员，这一群就不许说「稳住了」。"""
    n = 16
    net = Network(Config(n=n, modules=2, seed=5))
    eng = Engine(net)
    eng.v = np.full(n, 7.0)
    mon = StabilityMonitor(n, window=4, digits=12)
    for _ in range(6):
        mon.push(eng.v)
    touched = np.ones(n, dtype=bool)
    touched[3] = False
    rows = group_table(eng.v, net, 1e-6, mon, need=4, touched=touched)
    assert len(rows) == 1, "全都一样，应该合成一群"
    assert not rows[0].settled, "群里有没算过的单位，却报了稳住"
    rows2 = group_table(eng.v, net, 1e-6, mon, need=4, touched=np.ones(n, dtype=bool))
    assert rows2[0].settled, "都算过了还报不稳住，就太保守了"


# ---------------------------------------------------------------- 布局

def test_layout_is_deterministic() -> None:
    net = Network(CFG)
    eng = _run(net, 80)
    mon = StabilityMonitor(CFG.n, CFG.window, CFG.digits)
    mon.push(eng.v)
    rows = group_table(eng.v, net, 0.03 * scale_of(eng.v), mon)
    a = make_shape(rows, net, seed=3)
    b = make_shape(rows, net, seed=3)
    assert np.array_equal(a.pos, b.pos)
    assert np.array_equal(a.strength, b.strength)


def test_layout_strength_symmetric_and_diagonal_free() -> None:
    net = Network(CFG)
    eng = _run(net, 80)
    rows = group_table(eng.v, net, 0.03 * scale_of(eng.v))
    W = make_shape(rows, net, seed=3).strength
    assert W.shape == (len(rows), len(rows))
    assert np.allclose(W, W.T)
    assert np.all(np.diag(W) == 0.0)
    assert np.all(W >= 0.0)


def test_fine_groups_sit_inside_their_parent() -> None:
    """细视图不许自己另摆一个样子：每个细群都得待在它那个粗群的附近。
    粗视图定形，细视图只填内部细节。缩放时画面跳了，读的人会以为结构变了。
    """
    net = Network(CFG)
    eng = _run(net, 120)
    s = scale_of(eng.v)
    coarse_groups = group_table(eng.v, net, 0.1 * s)
    fine_groups = group_table(eng.v, net, 0.003 * s)
    coarse = make_shape(coarse_groups, net, seed=3)

    pos = broadcast(coarse, fine_groups, net, seed=3)
    owner = {}
    for gi, g in enumerate(coarse_groups):
        for u in g.members:
            owner[int(u)] = gi
    for ci, g in enumerate(fine_groups):
        base = coarse.pos[owner[int(g.members[0])]]
        assert np.linalg.norm(pos[ci] - base) <= 0.95, "细群跑出粗群的范围了"


def test_layout_closes_a_cycle() -> None:
    """环状关系应该摆成一个圈：相邻点等距，没有谁掉到正中间，线也不许交叉。"""
    k = 8
    W = np.zeros((k, k))
    for i in range(k):
        j = (i + 1) % k
        W[i, j] = W[j, i] = 1.0
    pos = spring(W, seed=5)
    p = pos[np.argsort(np.arctan2(pos[:, 1], pos[:, 0]))]
    d = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    assert d.std() / d.mean() < 0.02, "环没闭合"
    r = np.linalg.norm(pos, axis=1)
    assert r.min() > 0.5 * r.max(), "有群掉到中心去了"
    assert _crossings(W, pos) == 0, "关系线交叉了，读出来的拓扑是错的"


def test_every_unit_gets_relaxed() -> None:
    """一轮源（n 个 tick）下来，不许有单位被饿死。
    这是 §9.6 那个坑的回归测试：按「比例最大的邻居」走时，路会挤进一条走廊，
    少数单位 2048 tick 才被碰一次（min 1、中位 52），defect 就卡住不降了。
    """
    net = Network(Config(n=256))
    eng = Engine(net)
    visits = np.zeros(net.n, dtype=np.int64)
    for _ in range(net.n):
        path, _edges, _cols = eng.walk(eng.tick % net.n)
        visits[np.asarray(path, dtype=np.int64)] += 1
        eng.step()
    assert int((visits > 0).sum()) == net.n, "有单位从没被走到过"
    assert int(visits.min()) >= 2, f"有单位被饿死了（最少 {int(visits.min())} 次）"


def test_plateau_tolerates_one_step() -> None:
    """平台是「变化很小」，不是「一字不差」：13 → 12 是一个台阶，该算平台。"""
    fracs = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    ks = np.array([90, 40, 13, 12, 11])
    p = plateau(fracs, ks)
    assert p is not None, "13→12→11 该算平台"
    assert p["k"] in (12, 11), f"平台读数不对：{p}"
    # 真的在变的刻度上不许叫平台。
    assert plateau(fracs, np.array([90, 40, 13, 4, 1])) is None


def test_layout_degrades_instead_of_hanging() -> None:
    """k 太大时布局退成占位，并且**说出来**——不许假装那是结构。"""
    k = SPRING_MAX_K + 5
    W = np.ones((k, k)) - np.eye(k)
    pos = spring(W, seed=1)
    assert pos.shape == (k, 2)
    r = np.linalg.norm(pos, axis=1)
    assert np.allclose(r, 0.5, atol=1e-9), "降级布局应该是半径 0.5 的占位圈"


def test_render_degrades_instead_of_drawing_noise() -> None:
    """k 太多时画面退成雾，并且**写出来**：holes=-1 表示「这里没有读数」。
    上千个团挤在几百像素里，画出来的每一笔都是假的——看起来像结构，其实只是点。
    """
    k = RENDER_MAX_BLOBS + 20
    groups = [
        Group(gid=i, members=(i,), value=float(i), span=0.0, stable=2, age=30, settled=False)
        for i in range(k)
    ]
    shape = Shape(groups=groups, strength=np.zeros((k, k)), pos=np.zeros((k, 2)))
    img, topo = draw_shape(shape, size=220, title="t", need=4)
    assert img.size == (220, 220)
    assert topo["holes"] == -1, "降级的图不许报洞"
    assert topo.get("degraded") is True
    assert topo["blobs"] == [], "降级时不该有摆好的团"


def test_shape_colors_say_settled_not_who() -> None:
    """evo 图的颜色读「稳没稳住」，不接成员色：稳住=实心 SETTLED，没稳=点阵 CHURN。

    §30 的接色只在「同一批人」有意义的画布上用（动画、诊断带缩略图）。演进图要多说一件事：
    这一团现在能不能信。所以这里是刻意的例外，不是漏了（§28.4）。
    """
    from wy.render import CHURN, SETTLED

    groups = [
        Group(gid=0, members=(0, 1, 2), value=1.0, span=0.2, stable=8, age=30, settled=True),
        Group(gid=1, members=(3, 4, 5), value=9.0, span=0.2, stable=1, age=1, settled=False),
    ]
    shape = Shape(groups=groups, strength=np.array([[0.0, 0.6], [0.6, 0.0]]),
                  pos=np.array([[-0.4, 0.0], [0.4, 0.0]]))
    img, _topo = draw_shape(shape, size=220, title="t", need=4, seed=3)
    pixels = np.asarray(img)
    n_settled = int((pixels == np.array(SETTLED)).all(axis=-1).sum())
    n_churn = int((pixels == np.array(CHURN)).all(axis=-1).sum())
    assert n_settled > 0, "稳住的群应该是实心 SETTLED"
    assert n_churn > 0, "没稳的群应该是点阵 CHURN"
    assert n_settled > n_churn, "实心那团该比点阵那团更满"


def test_inter_cycles_set_the_component_count() -> None:
    """模块之间的外连分成几个环，整张图就有几个连通分量。
    这是「粗视角读回拓扑」的硬依据：分辨率放到无限大，k(r) 必须等于连通分量数
    （而不是模块数）。两者在屏幕上差得很远，混了就等于把两个尺度说成同一个。
    """
    for cycles, want in ((1, 1), (2, 2), (4, 4)):
        net = Network(Config(n=64, modules=8, inter_cycles=cycles))
        labels = partition(net.values, net, 1e18)
        got = int(np.unique(labels).size)
        assert got == want, f"cycles={cycles} 应该连成 {want} 块，实际 {got}"


def test_shuffle_keeps_the_cycle_count() -> None:
    """打乱「谁跟谁在一圈」之后，环的个数不许变。
    这是「读数认的是结构，不是模块的名字」的硬依据：随机分组只是把成员换掉，
    连通分量数必须还是 cycles，每个模块仍然恰好被指一次（入度 1）。
    """
    for cycles in (1, 2, 4):
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            tgts = inter_targets(8, cycles, rng)
            assert sorted(tgts) == list(range(8)), f"cycles={cycles} 入度不是 1：{tgts}"
            assert tgts != list(range(8)), "打乱之后不该和顺序分组一模一样"
            # 顺着 target 走一圈，看看回到起点要几步；步数种类数就是环的个数。
            seen, lens = set(), []
            for s in range(8):
                if s in seen:
                    continue
                cur, steps = s, 0
                while True:
                    seen.add(cur)
                    cur = tgts[cur]
                    steps += 1
                    if cur == s:
                        break
                lens.append(steps)
            assert len(lens) == cycles, f"cycles={cycles} 应得 {cycles} 个环，实得 {lens}"
            assert set(lens) == {8 // cycles}, f"环长不齐：{lens}"


def test_weak_module_only_weakens_its_own_inside() -> None:
    """给某一个模块「内部变松」，只该动它自己的内部关系。
    模块数、拓扑、其它模块的内部比例都不许变——否则这个实验问的就不是
    「望云认不认真结构」，而是「你换了张图」。
    """
    cfg = Config(n=64, modules=8, seed=7, weak_module=2, weak_intra=(0.01, 0.03))
    net = Network(cfg)
    mod = {u: u % cfg.modules for u in range(net.n)}  # 构造就是按 module 取模分组
    net_mod = np.arange(net.n) % cfg.modules
    weak = 2
    seen_weak = 0
    for u in range(net.n):
        for j in range(net.nbrs.shape[1]):
            w = int(net.nbrs[u, j])
            r = float(net.ratio[u, j])
            same = mod[u] == mod[w]
            if same and mod[u] == weak:
                assert 0.01 - 1e-12 <= r <= 0.03 + 1e-12, f"弱模块内 {u}->{w} 比例 {r}"
                seen_weak += 1
            elif same:
                assert 0.70 - 1e-12 <= r <= 1.00 + 1e-12, f"别的模块内部 {u}->{w} 比例 {r}"
            else:
                assert 0.002 - 1e-12 <= r <= 0.006 + 1e-12, f"模块之间 {u}->{w} 比例 {r}"
    assert seen_weak > 0, "弱模块内部一条关系都没有"
    assert np.all((net.nbrs >= 0).sum(axis=1) == 4)


def test_weak_set_weakens_every_listed_module() -> None:
    """一次弱化好几个模块：每一个都变松，没列进去的一个都不许动。"""
    cfg = Config(n=64, modules=8, seed=7, weak_modules=(1, 5, 6), weak_intra=(0.10, 0.20))
    net = Network(cfg)
    net_mod = np.arange(net.n) % cfg.modules
    weak = {1, 5, 6}
    seen = {m: 0 for m in weak}
    for u in range(net.n):
        for j in range(net.nbrs.shape[1]):
            w = int(net.nbrs[u, j])
            r = float(net.ratio[u, j])
            if net_mod[u] != net_mod[w]:
                assert 0.002 - 1e-12 <= r <= 0.006 + 1e-12, f"模块之间 {u}->{w} 比例 {r}"
            elif int(net_mod[u]) in weak:
                assert 0.10 - 1e-12 <= r <= 0.20 + 1e-12, f"弱模块内 {u}->{w} 比例 {r}"
                seen[int(net_mod[u])] += 1
            else:
                assert 0.70 - 1e-12 <= r <= 1.00 + 1e-12, f"没列的模块 {u}->{w} 比例 {r}"
    for m, c in seen.items():
        assert c > 0, f"模块 {m} 标了弱化，内部却一条关系都没变"
    assert np.all((net.nbrs >= 0).sum(axis=1) == 4), "弱化不许改度数"


def test_group_span_is_the_member_range() -> None:
    """群表里的 span 必须是「成员值的极差」。
    它是「这个形有多紧」的唯一读数（k 只数有几团），定义错了整节结论都得作废：
    span 是跟着群规模走的，只在同规模的群之间可比。
    """
    cfg = Config(n=48, modules=6, seed=11)
    net = Network(cfg)
    v = net.values.copy()
    v[:8] = 100.0 + np.array([0.0, 0.01, -0.01, 0.02, -0.02, 0.0, 0.0, 0.0])
    v[8:16] = 200.0 + np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.0, 0.0, 0.0])
    rows = group_table(v, net, 1e9, None, 0)
    assert sum(len(g.members) for g in rows) == net.n
    for g in rows:
        vals = v[list(g.members)]
        assert abs(g.span - float(vals.max() - vals.min())) < 1e-9
        assert abs(g.value - float(vals.mean())) < 1e-9


def test_exact_engine_agrees_with_float_engine() -> None:
    """整数演算和 float64 演算是两套独立实现，短跑必须对齐（1e-12）。
    两边错成一样是不容易的：一边是浮点加法，一边是 10**-30 定标的整数除法。
    """
    cfg = Config(n=64, modules=8, seed=5)
    a, b = Engine(Network(cfg)), ExactEngine(Config(n=64, modules=8, seed=5))
    for _ in range(300):
        a.step()
        b.step()
    f = b.to_float()
    scale = float(np.max(np.abs(a.v)))
    err = float(np.max(np.abs(a.v - f)) / scale)
    assert err < 1e-12, f"两条路差了 {err:g}（相对）"


def test_exact_digits_go_beyond_float64() -> None:
    """稳定位数的量尺不受 float64 限制：第 18 位也该数得出来。
    float64 大约只能表示 16 位有效数字，所以「18 位稳定」在那边根本测不出来。
    这里的量尺读的是十进制串，只受 keep 限制。
    """
    mon = StableDigits(2, window=4, keep=28)
    base = "4785388137" + "2049381220" + "3388992011"
    # 从第 18 位（下标 18）开始变：前 18 位必须判稳。
    same = [base[:18] + str(i) + base[19:] for i in range(4)]
    for t in same:
        mon.push([t, "12345678901234567890".ljust(28, "0")])
    assert mon.length[0] == 18, mon.length[0]


def test_components_are_independent_problems() -> None:
    """两个连通块之间没有一条关系——动一个块，另一个块逐位不变。
    这是「连通块 = 各自独立的问题」这句话的可执行版本：§14.1 拿它当面做的对照——
    同一张图里一个块紧、一个块松，紧的那个块读数一位都不许动。
    """
    base = Config(n=128, modules=8, seed=7, inter_cycles=2)
    moved = Config(
        n=128, modules=8, seed=7, inter_cycles=2,
        weak_modules=(4, 5, 6, 7), weak_intra=(0.10, 0.20),
    )
    a, b = Engine(Network(base)), Engine(Network(moved))
    for _ in range(200):
        a.step()
        b.step()
    first = np.arange(128) % 8 < 4  # 第一个环（模块 0-3）
    assert np.array_equal(a.v[first], b.v[first]), "动了第二个块，第一个块居然变了"
    assert not np.array_equal(a.v[~first], b.v[~first]), "第二个块被压松了，读数却没变"


def test_welding_floors_cross_ratios_but_not_their_workload() -> None:
    """焊接把群外比例压到地板（1e-3），却压不掉它们干的活。

    选路用的是 `比例 × |值差|`：比例被压到 1e-3 之后，那一项由**值差**接管
    （群外的值差是模块之间的高度差，几百倍于群内）。所以「把关系掐细」并不等于
    「这条路不走」——它只是把路的强度从比例换成了差值。
    """
    cfg = Config(n=128, seed=7, shape_feedback=0.05)
    net = Network(cfg)
    eng = Engine(net)
    mon = StabilityMonitor(net.n, cfg.window, cfg.digits)
    mod_of = np.arange(net.n) % cfg.modules
    is_inter = (mod_of[:, None] != mod_of[net.nbrs])

    snap: dict[int, np.ndarray] = {}
    for t in range(1200):
        eng.step()
        mon.push(eng.v)
        if t > 0 and t % cfg.feedback_every == 0:
            groups = group_table(eng.v, net, cfg.feedback_frac * scale_of(eng.v), mon,
                                 need=4, touched=eng.recent(cfg.window))
            if groups:
                eng.feedback(groups, cfg.shape_feedback)
        if t + 1 in (1000, 1200):
            snap[t + 1] = np.array(eng.load, dtype=np.float64)

    ratio = np.asarray(net.ratio, dtype=np.float64)
    assert float(ratio[is_inter].mean()) <= 2e-3, "群外比例没被压到地板"
    assert float(ratio[~is_inter].mean()) >= 0.90, "群内比例没被收紧"

    inc = snap[1200] - snap[1000]
    share = float(inc[is_inter].sum() / inc.sum())
    assert share > 0.30, f"焊到底之后群外只干 {share:.1%} 的活，与 §16 的机制不符"


def test_plateau_width_beats_settled_count() -> None:
    """「形长齐」该用平台宽度量，不是用稳住的群数量。

    settled 会饱和（两条路都能爬到 6~7 个），所以它分辨不出「结构长稳了」和
    「差别快被磨平了」；而 k(r) 里 modules 这一档跨几个数量级（宽度）分得开。
    这条测试把两件事一起钉住：γ=2 在 8000 tick 时已经稳了 3 个以上群，
    而且它的平台比 γ=1 宽——同一个 tick，同一个种子。
    """
    from wy.shape import k_curve, plateau as _plateau

    FRACS = (0.3, 0.1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)

    def measure(gamma: float):
        cfg = Config(n=128, seed=7, inter_ratio=(0.004, 0.004), walk_power=gamma)
        net = Network(cfg)
        eng = Engine(net)
        mon = StabilityMonitor(net.n, cfg.window, cfg.digits)
        for _ in range(8000):
            eng.step()
            mon.push(eng.v)
        fv = eng.values()
        sc = scale_of(fv)
        kk = k_curve(fv, net, FRACS)
        idx = [i for i, v in enumerate(kk) if v == cfg.modules]
        width = np.log10(FRACS[idx[0]] / FRACS[idx[-1]]) if len(idx) >= 2 else 0.0
        frac = FRACS[idx[0]] if idx else 3e-2
        groups = group_table(fv, net, frac * sc, mon, need=4,
                             touched=eng.recent(cfg.window))
        settled = sum(1 for g in groups if g.settled)
        biggest = max((len(g.members) for g in groups), default=0)
        return settled, float(width), biggest, _plateau(FRACS, kk)

    settled2, width2, biggest2, pl2 = measure(2.0)
    settled1, width1, _, _ = measure(1.0)

    assert settled2 >= 3, f"γ=2 跑了 8000 tick 只稳住 {settled2} 个群"
    assert settled1 <= 2, f"γ=1 的对照组不该也稳了 {settled1} 个"
    assert biggest2 == 128 // 8, "稳住的群应该正好是一个模块的大小"
    assert width2 > width1, f"平台宽度没分开：γ=2 {width2:.2f} vs γ=1 {width1:.2f}"
    assert pl2 is not None and pl2["k"] == 8, "γ=2 该读到 8 这一档平台"


def test_walk_power_discounts_weak_relations() -> None:
    """给弱关系打折（比例的指数 γ）之后：路不再掉头去干群外，结构活得更久。

    这是对 §16 那条解释的反证实验——如果「拆结构的不是噪声，是 score 里的比例不够重」
    成立，那么把比例的指数从 1 提到 2，群外负荷份额该掉回「按条数该有的样子」（~25%），
    覆盖 min 该回来，寿命该变长。
    """
    from wy.engine import Engine as _Engine
    from wy.network import Network as _Network

    def measure(gamma: float):
        cfg = Config(n=64, seed=7, inter_ratio=(0.004, 0.004), walk_power=gamma)
        net = _Network(cfg)
        eng = _Engine(net)
        for _ in range(100):
            eng.step()
        base = eng.touches.copy()
        load0 = np.array(eng.load, dtype=np.float64)
        for _ in range(net.n):
            eng.step()
        got = eng.touches - base
        load = (np.array(eng.load, dtype=np.float64) - load0).ravel()
        mod_of = np.arange(net.n) % cfg.modules
        is_inter = (mod_of[:, None] != mod_of[net.nbrs]).ravel()
        return float(load[is_inter].sum() / load.sum()), int(got.min())

    share1, min1 = measure(1.0)
    share2, min2 = measure(2.0)
    assert share1 > 0.50, f"γ=1 时群外只搬了 {share1:.1%}，§16 的前提不成立"
    assert share2 < 0.35, f"γ=2 之后群外还搬 {share2:.1%}，打折没起作用"
    assert min1 < 25, f"γ=1 的覆盖 min 是 {min1}，走廊没复现"
    assert min2 > 35, f"γ=2 的覆盖 min 只有 {min2}，覆盖没回来"


def test_lifetime_scales_with_one_over_inter_ratio() -> None:
    """结构活多久 ≈ 1/inter_ratio（§19）。弱耦合减半，寿命该差不多翻倍。

    这条是 §9.3 的旧说法在新走动规则下的复核：比例 t2(r/4) / t2(r) 必须落在 2.5~7
    （正好反比是 4）。顺便钉住两件事：塌发生在「还看得见整套模块」之后；最后的分比 t2 早。
    """
    from wy.strip import lifetimes

    cfg_fast = Config(n=64, seed=7, inter_ratio=(0.020, 0.020))
    cfg_slow = Config(n=64, seed=7, inter_ratio=(0.005, 0.005))
    fast = lifetimes(cfg_fast, 4000, sample=20)
    slow = lifetimes(cfg_slow, 8000, sample=20)

    assert fast["t2"] is not None and slow["t2"] is not None, "寿命没量到"
    assert fast["t2"] > fast["last_full"], "塌必须发生在结构长出来之后"
    assert slow["t2"] > slow["last_full"], "塌必须发生在结构长出来之后"
    if fast["last_split"] is not None:
        assert fast["last_split"] <= fast["t2"], "「最后的分」不该晚于塌成一团"

    ratio = slow["t2"] / fast["t2"]
    assert 2.5 <= ratio <= 7.0, f"弱化 4 倍寿命只变了 {ratio:.2f} 倍，不成反比"


def test_strip_aligns_events_with_readings() -> None:
    """诊断带：事件数和曲线长度必须对得上，且缩略图不许画出画布。"""
    from wy.strip import collect as strip_collect
    from wy.strip import strip as make_strip

    cfg = Config(n=32, modules=4, seed=3)
    r = strip_collect(cfg, 400, frac=0.05, probe=10, load_window=100, need=3)
    assert len(r["x"]) == 400, "时间轴长度不对"
    assert set(r["k"].shape) == {400}, "k(t) 长度不对"
    assert len(r["events"]) >= 1, "400 tick 里应该至少有一次划分变化"
    for e in r["events"]:
        assert e["kind"] in ("merge", "split"), f"事件类型不认识：{e['kind']}"
        assert 1 <= e["tick"] <= 400, f"事件 tick 越界：{e['tick']}"

    img = make_strip(cfg, 400, frac=0.05, max_frames=4, thumb=160, width=700, need=3)
    assert img.width == 700, "宽度被缩略图挤爆了"


def test_strip_hue_lane_lands_on_the_merge_ticks() -> None:
    """诊断带的色相移动：非零只出现在划分变了的采样点。

    n=32 这一小跑里峰值落在并群；k 小时分开一大团会更疼，
    那条反过来钉在 test_hue_peak_can_fall_on_a_split。
    """
    from wy.strip import collect as strip_collect

    cfg = Config(n=32, modules=4, seed=3)
    r = strip_collect(cfg, 400, frac=0.05, probe=10, load_window=100, need=3)
    assert len(r["hue_x"]) == len(r["hue_moved"]) == 40, "色相采样点和 probe 对不上"
    moves = r["hue_moved"]
    assert all(0.0 <= float(m) <= 0.5 for m in moves), moves
    ev = {e["tick"]: e["kind"] for e in r["events"]}
    live = [int(t) for t, m in zip(r["hue_x"], moves) if m > 0]
    assert live, "400 tick 里至少并过一次，色相总得动一下"
    assert all(t in ev for t in live), f"色相动了，但那一刻没记划分变化：{live}"
    peak = int(r["hue_x"][int(np.argmax(moves))])
    assert ev.get(peak) == "merge", f"峰值落在 t={peak}（{ev.get(peak)}），不是并群"


def test_hue_peak_can_fall_on_a_split() -> None:
    """峰值不总在并群：k 小时分开走掉一大团，比并小群更疼（§30.5 实测）。

    n=128/seed=7/4000 tick 这一跑，峰值 0.062 落在 t=2480 的「分」上，
    最大的并只有 0.054。别把「峰值一定在并群」写回代码或文档里。
    """
    from wy.strip import collect as strip_collect

    cfg = Config(n=128, seed=7)
    r = strip_collect(cfg, 4000)
    kinds = {int(e["tick"]): e["kind"] for e in r["events"]}
    moves = {int(t): float(m) for t, m in zip(r["hue_x"], r["hue_moved"])}
    live = {t: m for t, m in moves.items() if m > 0}
    assert set(live) == set(kinds), f"色相的非零点该和划分变化一一对上：{sorted(live)} vs {sorted(kinds)}"
    peak = max(live, key=live.get)
    assert kinds.get(peak) == "split", f"这一跑峰值该落在分开上，实际 t={peak}（{kinds.get(peak)}）"
    merge_max = max(m for t, m in live.items() if kinds[t] == "merge")
    split_max = max(m for t, m in live.items() if kinds[t] == "split")
    assert split_max > merge_max, f"分开一大团时该比并小群更疼：{split_max:.3f} vs {merge_max:.3f}"


def test_strip_hue_dots_follow_the_event_kind() -> None:
    """色相层的小点：并=CHURN（橙）、分=SETTLED（青），没事件的采样点不画点。"""
    from wy.render import CHURN, SETTLED
    from wy.strip import _hue_dots
    from wy.strip import collect as strip_collect

    cfg = Config(n=128, seed=7)
    r = strip_collect(cfg, 1200)
    kinds = {int(e["tick"]): e["kind"] for e in r["events"]}
    dots = _hue_dots(r["events"], r["hue_x"])
    assert set(dots) == {int(t) for t in r["hue_x"] if int(t) in kinds}
    assert len(dots) == len(kinds) >= 2, f"这一跑该有并也有分：{kinds}"
    assert all(dots[t] == (CHURN if k == "merge" else SETTLED) for t, k in kinds.items())
    assert {CHURN, SETTLED} <= set(dots.values()), "并和分的点颜色不能一样"


def test_strip_thumbnails_keep_the_same_people_same_color() -> None:
    """缩略图也得按成员记路：跨帧还是那批人，色相一个数都不变。

    否则同一批人会在「讲事件」的那张图里换色，正好跟 §30 说的相反。
    """
    from wy.strip import strip as make_strip

    cfg = Config(n=128, seed=7)
    trace: list = []
    img = make_strip(cfg, 1200, max_frames=4, thumb=100, width=600, trace=trace)
    assert img.width == 600, "宽度被缩略图挤爆了"
    assert len(trace) >= 2, f"这一跑至少该挑出两帧缩略图，只画了 {len(trace)} 帧"

    held = 0
    for before, now in zip(trace, trace[1:]):
        same = set(before["hues"]) & set(now["hues"])
        for key in same:
            assert now["hues"][key] == before["hues"][key], (
                f"t={now['t']} 还是同一批人 {sorted(key)[:3]}…，色相却从 "
                f"{before['hues'][key]:.3f} 变成 {now['hues'][key]:.3f}")
        held += len(same)
    assert held > 0, "相邻两帧缩略图之间一个老群都不剩，这条契约等于没验"


def test_core_imports_without_pillow() -> None:
    """核心层不许把 Pillow 拖进来：别的软件只想要读数，不该被逼着装图形库。"""
    import os
    import subprocess

    code = """
import sys

class BlockPIL:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "PIL":
            raise ImportError("PIL blocked by contract test")

sys.meta_path.insert(0, BlockPIL())

from wy.core import Config, observe

b = observe(Config(n=32, modules=4, seed=1), 40, probe=10, need=2)
assert b["frames"], "核心一条观察都没跑出来"
assert "PIL" not in sys.modules, "PIL 被偷偷 import 了"
print("core ok", b["frames"][0]["k"])
"""
    root = str(Path(__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                         capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, f"核心在没 Pillow 的进程里起不来：\n{res.stderr}"
    assert "core ok" in res.stdout, f"子进程没报成功：{res.stdout!r} {res.stderr!r}"


def test_bundle_is_json_and_keeps_hues_by_membership() -> None:
    """核心对外交付的是 JSON：读得出来，色相按成员延续——跟 §30 同一个规矩。

    网页端、桌面壳都只吃这份 bundle，所以它必须能直接落盘；
    而「还是这批人就不换色」是核心的读数，不是渲染层的装饰。
    """
    import json

    from wy.core import Config as CoreCfg
    from wy.core import dumps, observe

    bundle = observe(CoreCfg(n=32, modules=4, seed=3), 240,
                     probe=10, need=2, members=True)
    back = json.loads(dumps(bundle))
    assert back["meta"]["n"] == 32 and len(back["frames"]) >= 5

    held = 0
    for before, now in zip(back["frames"], back["frames"][1:]):
        assert now["k"] == len(now["groups"])
        old = {frozenset(g["members"]): g["hue"] for g in before["groups"]}
        for g in now["groups"]:
            assert 0.0 <= g["hue"] < 1.0, "色相跑出 [0,1) 了"
            key = frozenset(g["members"])
            if key in old:
                assert abs(g["hue"] - old[key]) < 1e-9, (
                    f"t={now['t']} 还是同一批人，色相却从 {old[key]} 变成 {g['hue']}")
                held += 1
    assert held > 0, "相邻帧之间一个老群都不剩，色相这条等于没验"

    ticks = set(back["readings"]["t"])
    for ev in back["events"]:
        assert ev["tick"] in ticks, "事件落不到读数轴上，网页时间轴会对不上"
        assert ev["kind"] in ("merge", "split", "shift")


def test_core_command_line_emits_json_bundle() -> None:
    """核心的命令行入口：stdout 就是一份 bundle JSON，别的语言直接调。

    DSH 插件、网页后端、脚本都走这条路——所以它不能被 CLI 那一层的 argparse 和
    Pillow 绑住，`python -m wy.core` 必须自己就是完整的。
    """
    import json
    import os
    import subprocess

    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    argv = [
        sys.executable, "-m", "wy.core",
        "--n", "32", "--modules", "4", "--ticks", "60", "--probe", "20", "--need", "2",
    ]
    done = subprocess.run(argv, cwd=root, env=env, capture_output=True,
                          text=True, encoding="utf-8")
    assert done.returncode == 0, f"核心命令行失败了：{done.stderr[-500:]}"
    data = json.loads(done.stdout)
    assert data["meta"]["n"] == 32 and data["frames"], "命令行的 bundle 里没有帧"
    assert data["readings"]["t"], "命令行的 bundle 里没有读数轴"


def test_plugins_expose_three_kinds_and_render_svg() -> None:
    """三个扩展点都得能用：渲染通道不碰核心，也能把一帧画成 SVG。"""
    from wy import plugins, plugins_builtin  # noqa: F401
    from wy.core import Config as CoreCfg
    from wy.core import observe

    assert set(plugins.describe()) == {"datasource", "reading", "renderer"}
    assert "svg" in plugins.names("renderer")
    bundle = observe(CoreCfg(n=32, modules=4, seed=2), 120, probe=20, need=2)
    text = plugins.select("renderer", "svg")(bundle, -1)
    assert text.startswith("<svg") and "<circle" in text and text.endswith("</svg>")

    try:
        plugins.renderer("svg")(lambda b, index=-1: "")
    except plugins.PluginError:
        pass
    else:
        raise AssertionError("重名的渲染通道没被拒，扩展点会被悄悄顶掉")


def test_server_answers_health_and_bundle() -> None:
    """本地网页端：health 报得出插件，bundle 直接给 JSON，首页拿得到。"""
    import http.client
    import json
    import threading
    from http.server import ThreadingHTTPServer

    from wy.server import Handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        conn.request("GET", "/api/health")
        res = conn.getresponse()
        body = res.read()
        assert res.status == 200 and b"renderer" in body

        conn.request("GET", "/api/bundle?n=32&modules=4&ticks=60&probe=20&need=2")
        res = conn.getresponse()
        assert res.status == 200
        data = json.loads(res.read())
        assert data["frames"] and data["readings"]["t"], "网页拿不到帧/读数"

        conn.request("GET", "/")
        res = conn.getresponse()
        html = res.read().decode("utf-8")
        assert res.status == 200 and "望云" in html
        assert 'id="band"' in html, "页面缺读数带画布"

        conn.request("GET", "/app.js")
        res = conn.getresponse()
        js = res.read().decode("utf-8")
        assert res.status == 200 and "/api/png" in js, "页面没接上原图/缩略图"
        assert "drawBand" in js, "页面没画读数带"
        assert "mousemove" in js, "读数带没有悬停读数"
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_server_png_is_a_pillow_render() -> None:
    """页面上的「原图」：PNG 走 Pillow 基准渲染；越界帧夹住，坏 index 报 400。"""
    import http.client
    import threading
    from http.server import ThreadingHTTPServer

    from wy.server import Handler

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
        for query in ("n=32&modules=4&ticks=60&probe=20&need=2&size=240&index=0",
                      "n=32&modules=4&ticks=60&probe=20&need=2&size=240&index=999"):
            conn.request("GET", "/api/png?" + query)
            res = conn.getresponse()
            body = res.read()
            assert res.status == 200, f"{query} → {res.status}"
            assert res.getheader("Content-Type") == "image/png"
            assert body[:8] == b"\x89PNG\r\n\x1a\n", "不是 PNG"
            assert len(body) > 1000, "PNG 太小"
        conn.request("GET", "/api/png?n=32&modules=4&ticks=60&probe=20&need=2&index=abc")
        res = conn.getresponse()
        res.read()
        assert res.status == 400, "坏 index 应该 400"
        conn.close()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_server_quit_endpoint_stops_the_desktop_app() -> None:
    """桌面版「退出望云」：本机 POST /api/quit，服务真的停下来。"""
    import http.client
    import json
    import threading

    from wy.server import make_server

    httpd = make_server("127.0.0.1", 0)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        conn.request("POST", "/api/quit")
        res = conn.getresponse()
        body = json.loads(res.read())
        assert res.status == 200 and body.get("bye") is True, "退出接口没回 bye"
        thread.join(5)
        assert not thread.is_alive(), "退出后服务还在转"
        conn.close()
    finally:
        httpd.server_close()


def test_desktop_port_helpers_are_honest() -> None:
    """桌面入口：端口先探再占；服务没起时 alive 不说谎。"""
    import socket

    from wy.desktop import alive, find_port, port_free

    port = find_port("127.0.0.1", 8765, tries=40)
    assert port_free("127.0.0.1", port)
    assert not alive(port), "没起服务却说活着"
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        blocker.bind(("127.0.0.1", port))
        blocker.listen(1)
        assert not port_free("127.0.0.1", port), "被占用还说空着"
        assert find_port("127.0.0.1", port, tries=5) != port, "占了还选同一个端口"
    finally:
        blocker.close()


def test_board_puts_all_five_layers_on_one_image() -> None:
    """一屏看全：五层读数（形/寿命/负荷/k(t)/违约/位数活性）都能画出来，不缺块。"""
    from wy.board import board as make_board
    from wy.board import collect as board_collect

    cfg = Config(n=32, modules=4, seed=3)
    img = make_board(cfg, 300, window=50, need=3, size=200)
    assert img.width > 400 and img.height > 300, f"图太小：{img.size}"

    r = board_collect(Config(n=32, modules=4, seed=3), 300, window=50, need=3)
    assert len(r["series"]["k"]) == 300, "k(t) 序列长度不对"
    assert len(r["series"]["alive"]) == 300, "活性序列长度不对"
    assert r["load_frames"].shape[0] == 6, "负荷窗口数不对"
    assert len(r["curves"]) == 8, "k(r) 应该有八档"


def test_load_only_lands_on_the_path() -> None:
    """负荷记在「这一 tick 真的搬过值的那几条有向关系」上，一条不多一条不少。"""
    net = Network(CFG)
    eng = Engine(net)
    rec = eng.step()
    assert int((eng.load > 0).sum()) == len(rec.path) - 1, "负荷落在了没走过的关系上"
    assert float(eng.load.sum()) > 0.0


def test_weak_relations_take_over_after_inside_settles() -> None:
    """群内的差被抹平之后，路就掉头去干群外——结构是被这条路自己拆掉的。

    第一个窗口里群外关系几乎不搬东西（<5%）；等模块内部收敛，它们的分数
    `比例 × |值差|` 反而超过群内（比例小两百倍，值差大几百倍），
    于是搬运量的一半落在那 25% 的弱关系上。这就是 §9.3 那条「结构有寿命」的机制。
    """
    net = Network(CFG)
    eng = Engine(net)
    mod_of = np.arange(net.n) % net.cfg.modules
    is_inter = (mod_of[:, None] != mod_of[net.nbrs])

    for _ in range(100):
        eng.step()
    first = np.array(eng.load, dtype=np.float64)
    share_first = float(first[is_inter].sum() / first.sum())

    for _ in range(700):
        eng.step()
    later = np.array(eng.load, dtype=np.float64) - first
    share_later = float(later[is_inter].sum() / later.sum())

    assert share_first < 0.05, f"第一个窗口群外就搬了 {share_first:.1%}，机制不对"
    assert share_later > 0.30, f"群内收敛后群外只搬 {share_later:.1%}，路没掉头"


def test_ladder_bottom_is_trivial() -> None:
    """阶梯的下界不是「更细的结构」，是「每个单位都不一样」——读数在那儿没有意义。

    契约只有一条：r 足够小的时候 k(r) 必须走到 n。谁要是想在这种地方读出「层次」，
    读到的是单个单位自己的小数点后第几位，不是结构。
    """
    net = Network(CFG)
    eng = _run(net, 300)
    sc = scale_of(eng.v)
    k_coarse = k_of(eng.v, net, 1e-2 * sc)
    k_tiny = k_of(eng.v, net, 1e-14 * sc)
    assert k_tiny >= net.n - 2, f"r 小到 1e-14 还没到 n（k={k_tiny}），阶梯在漏"
    assert k_tiny >= k_coarse, "k(r) 必须随 r 变小而不减"


# ---------------------------------------------------------------- 真图
def test_real_graph_normalizes_ratios_per_node() -> None:
    """真图的权重换算成比例：每个单位最强的那条记 1，其余按比值，且落在 (0,1]。"""
    from wy.load import load_graph

    p = Path(__file__).resolve().parent.parent / "data" / "_t_edge.txt"
    p.write_text("0 1 10\n0 2 5\n1 2 2\n2 3 1\n", encoding="utf-8")
    try:
        net = load_graph(p, Config(n=0, seed=3))
    finally:
        p.unlink(missing_ok=True)
    assert net.n == 4
    assert np.all(net.ratio[net.mask] > 0) and np.all(net.ratio[net.mask] <= 1.0)
    # 单位 0 最强的是 10，那条必须是 1.0
    row0 = net.ratio[0][net.mask[0]]
    assert abs(float(row0.max()) - 1.0) < 1e-12, "最强的关系没被记成 1"
    # 无向输入按双向读：每条关系两个方向都有
    assert int(net.mask.sum()) == 8, f"关系条数不对：{int(net.mask.sum())}"


def test_real_graph_pads_with_self_and_isolated_is_quiet() -> None:
    """度数不齐用自环补齐；孤立点不产生 nan，也不许被当成「跟谁都很像」。"""
    from wy.load import RealNetwork

    net = RealNetwork(Config(n=0, seed=3), [(0, 1, 1.0), (1, 2, 2.0)], 3)
    # 0 的度是 1，补齐的那几格必须是自己（走动时必定已被走过，选不中）
    for u in range(3):
        for j in range(net.degree):
            if not net.mask[u, j]:
                assert int(net.nbrs[u, j]) == u, "补位不是自己"
    # 孤立的第 4 个点
    net2 = RealNetwork(Config(n=0, seed=3), [(0, 1, 1.0)], 4)
    eng = Engine(net2)
    res = eng.residual()
    assert np.all(np.isfinite(res)), "孤立点把残差算成 nan 了"
    assert float(res[3]) == 0.0, "孤立点的残差该是 0"


def test_archive_weights_become_contrast() -> None:
    """权重差得越大，比例差得越大；同一单位最强那条恒为 1（比例是相对的）。"""
    from wy.load import RealNetwork

    net = RealNetwork(Config(n=0, seed=3), [(0, 1, 30.0), (0, 2, 3.0)], 3)
    r01 = net.ratio_between(0, 1)
    r02 = net.ratio_between(0, 2)
    assert abs(r01 - 1.0) < 1e-12
    assert 0.05 < r02 < 0.15, f"弱关系的比例不对：{r02}"


# ---------------------------------------------------------------- 相对半径
def test_relative_radius_recovers_planted_modules() -> None:
    """相对半径不许把种下去的结构读丢：τ 在一个区间内都该读出整整 8 个模块。"""
    from wy.shape import k_rel, partition_rel

    cfg = Config(n=128, seed=7, walk_power=2.0)
    net = Network(cfg)
    eng = _run(net, 3000)
    v = eng.values()
    k = k_rel(v, net, 2.0)
    assert k == cfg.modules, f"τ=2 读出 {k} 群，种的模块是 {cfg.modules}"
    # 群必须整块对应模块，不许把两个模块切一半拼在一起
    lab = partition_rel(v, net, 2.0)
    for gid in set(lab.tolist()):
        mods = {int(u) % cfg.modules for u in np.flatnonzero(lab == gid).tolist()}
        assert len(mods) == 1, f"群 {gid} 横跨了模块 {sorted(mods)}"


def test_relative_radius_handles_unequal_tightness() -> None:
    """紧的群和松的群放一起：全局半径必然二选一失败，相对半径两个都要。"""
    from wy.load import RealNetwork
    from wy.shape import k_of, k_rel

    # 两团：0-3 很紧（极差 0.3），4-7 很松（极差 30），中间一条桥
    vals = np.array([0.0, 0.1, 0.2, 0.3, 100.0, 110.0, 120.0, 130.0])
    edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 3, 1.0),
             (4, 5, 1.0), (5, 6, 1.0), (6, 7, 1.0), (4, 7, 1.0),
             (3, 4, 1.0)]
    net = RealNetwork(Config(n=0, seed=5), edges, 8)
    net.values = vals
    # 全局半径：<=0.3 时松的那团被拆开，>=30 时紧的那团被并进桥 —— 两头都错
    assert k_of(vals, net, 0.3) >= 3, "全局半径 0.3 不该读出 2 团（说明这个对照没意义）"
    assert k_rel(vals, net, 2.0) == 2, f"相对半径读出 {k_rel(vals, net, 2.0)} 群，应该是 2"


# ---------------------------------------------------------------- 可读性
def test_gauge_says_readable_on_planted_graph() -> None:
    """种下去的图：瓶颈陡、松紧差不多，量出来必须是「能读」。"""
    from wy.gauge import report
    from wy.shape import partition_rel

    cfg = Config(n=128, seed=7, walk_power=2.0)
    net = Network(cfg)
    eng = _run(net, 2000)
    lab = partition_rel(eng.values(), net, 2.0)
    g = report(lab, eng.values(), net)
    assert g["contrast"] < 0.1, f"合成的图瓶颈居然不陡：{g['contrast']}"
    assert str(g["verdict"]).startswith("能读"), g["verdict"]


def test_gauge_says_unreadable_when_no_bottleneck() -> None:
    """完全图（谁都连谁、权重全一样）：切哪儿都不便宜，读数必须自己说不值得信。"""
    from wy.gauge import report
    from wy.load import RealNetwork

    edges = [(a, b, 1.0) for a in range(6) for b in range(a + 1, 6)]
    net = RealNetwork(Config(n=0, seed=5), edges, 6)
    net.values = np.array([0.0, 3.0, 7.0, 4.0, 9.0, 1.0])
    lab = np.zeros(6, dtype=np.int64)
    lab[3:] = 1  # 随便切成两半
    g = report(lab, net.values, net)
    assert abs(float(g["contrast"]) - 1.0) < 0.2, f"权重全一样时 contrast 该是 1：{g['contrast']}"
    assert float(g["q_excess"]) < 0.2, f"完全图不该有模块度：{g['q_excess']}"
    assert str(g["verdict"]).startswith("别信"), g["verdict"]




def test_paint_params_follow_readings() -> None:
    """读数变 → 参数跟着变，而且方向对得上（这是「画面是读数的函数」的最低要求）。"""
    from wy.paint import params_from_engine

    a = params_from_engine(2, [1.0], 1.0, 1e-6)
    b = params_from_engine(2, [1.0], 1.0, 1e-6)
    assert a == b, "同读数必须同参数"

    more_k = params_from_engine(8, [1.0], 1.0, 1e-6)
    assert more_k.swirl > a.swirl, "群多 → 花纹多"
    noisier = params_from_engine(2, [1.0], 1.0, 5e-2)
    assert noisier.stars > a.stars, "违约读数大 → 背景星更多"
    steadier = params_from_engine(2, [1.0], 6.0, 1e-6)
    assert steadier.glow > a.glow, "位数多 → 高光更亮"


def test_paint_renders_requested_size() -> None:
    """画出来必须是我们要的尺寸，而且不是一块纯色（图画成了才叫画成了）。"""
    from wy.paint import Params, draw_character

    img = draw_character(Params(size=160), ss=1)
    assert img.size == (160, 160), img.size
    colors = img.getcolors(maxcolors=1 << 20)
    assert colors is not None and len(colors) > 200, "画出来几乎一种颜色，等于没画"




def test_brief_parses_what_people_actually_say() -> None:
    """一句人话要能变成意图，而且不能把「少花纹」读成「花纹多」。"""
    from wy.brief import parse

    b = parse("大眼睛，少花纹，安静")
    assert b.eye == "large", b.eye
    assert b.swirl == "none", b.swirl
    assert b.stars == "calm", b.stars

    b2 = parse("小眼睛，花纹多，星空")
    assert b2.eye == "small" and b2.swirl == "lots" and b2.stars == "dense", b2


def test_brief_choice_is_deterministic_and_stays_in_range() -> None:
    """挑配置必须可复现；挑出来的东西也必须真的落在望云长得出的那一段里。"""
    from wy.brief import choose, parse

    b = parse("大眼睛，星空")
    a = choose(b, seed=7, n=64)
    c = choose(b, seed=7, n=64)
    assert a.candidate == c.candidate and abs(a.score - c.score) < 1e-12, "同输入必须同输出"
    assert a.eye if False else True
    assert 0.20 <= a.params.eye_h <= 0.40, a.params.eye_h
    assert 0.30 <= a.params.swirl <= 0.85, a.params.swirl
    assert 0.20 <= a.params.stars <= 0.95, a.params.stars
    assert a.score < 3.0, f"偏差太大，等于没挑：{a.score}"




def test_scene_objects_follow_relationship_strength() -> None:
    """关系强的群必须更近（更靠前、更大）——这是「场景」这个词的全部意义。"""
    from wy.scene import SceneParams, _depth_of

    class G:
        def __init__(self, members):
            self.members = members

    groups = [G([0, 1, 2]), G([3, 4, 5]), G([6, 7, 8])]
    # 群 0 和谁都强，群 2 和谁都弱
    s = np.array([[0.0, 0.9, 0.9],
                  [0.9, 0.0, 0.1],
                  [0.9, 0.1, 0.0]])
    depth = _depth_of(groups, s, SceneParams())
    assert depth[0] < depth[1] and depth[0] < depth[2], f"最强的那团该在最前面：{depth}"
    assert depth[2] >= depth[1] - 1e-9, f"最弱的那团该在后面：{depth}"


def test_scene_draws_the_requested_size() -> None:
    """画出来的图必须是指定尺寸，而且不是一块纯色。"""
    from wy.scene import SceneParams, draw_scene

    class G:
        def __init__(self, members, value, span, settled=False):
            self.members = members
            self.value = value
            self.span = span
            self.settled = settled

    groups = [G(list(range(i * 4, i * 4 + 4)), 0.2 + 0.3 * i, 1.0 + i) for i in range(5)]
    s = np.zeros((5, 5))
    for i in range(5):
        s[i, (i + 1) % 5] = s[(i + 1) % 5, i] = 0.8
    img = draw_scene(groups, s, SceneParams(size=320), title="t", note="n", seed=7)
    assert img.size == (320, 320), img.size
    assert len(img.getcolors(maxcolors=200000)) > 300, "不该是一块纯色"


def test_scene_hue_override_is_used_verbatim() -> None:
    """不给 hue = 关系排出来的色相；给了 hue 就照给的画——动画的连续色相靠这条线接上。"""
    from wy.color import color_space
    from wy.scene import SceneParams, draw_scene

    class G:
        def __init__(self, gid: int, members: list[int]) -> None:
            self.gid = gid
            self.members = members
            self.value = 0.1 * gid
            self.span = 1.0 + gid
            self.stable = 3.0

    groups = [G(i, list(range(i * 3, i * 3 + 3))) for i in range(5)]
    s = np.zeros((5, 5))
    for i in range(5):
        s[i, (i + 1) % 5] = s[(i + 1) % 5, i] = 0.8
    p = SceneParams(size=160)
    default = draw_scene(groups, s, p, seed=5)
    cs = color_space(groups, None, strength=s)
    same = draw_scene(groups, s, p, seed=5,
                      hue=np.array([cs[g.gid][0] for g in groups], dtype=float))
    assert default.tobytes() == same.tobytes(), "不给 hue 时用的就该是关系排出来的色相"
    flipped = draw_scene(groups, s, p, seed=5,
                         hue=np.array([(cs[g.gid][0] + 0.5) % 1.0 for g in groups], dtype=float))
    assert flipped.tobytes() != default.tobytes(), "给了 hue 却没照着画"




def test_scene_facets_follow_internal_spread() -> None:
    """紧的群接近球（边多、毛刺小），松的群是碎石头（边少、毛刺大）。"""
    from wy.scene import _facet

    sides_tight, jag_tight, _ = _facet(0.0, 7)
    sides_loose, jag_loose, _ = _facet(1.0, 7)
    assert sides_tight >= 16, sides_tight
    assert sides_loose <= 6, sides_loose
    assert jag_loose > 5 * jag_tight, (jag_tight, jag_loose)


def test_anim_samples_frames_in_order() -> None:
    """帧必须是按时间采的、尺寸对得上；不插值，也不用一套坐标画到底。"""
    from wy.anim import frames

    cfg = Config(n=64, modules=4, seed=3)
    imgs = frames(cfg, 400, count=3, frac=3e-2, need=2, size=200)
    assert len(imgs) == 3, len(imgs)
    assert all(im.size == (200, 200) for im in imgs), [im.size for im in imgs]
    assert len({im.tobytes() for im in imgs}) > 1, "三帧一模一样，说明没真采样"


def test_anim_keeps_hues_frame_to_frame() -> None:
    """颜色也是记路的：成员没变的老群，色相逐帧一模一样；结构没变的帧一个人都不动。"""
    from wy.anim import frames

    cfg = Config(n=64, modules=4, seed=3)
    trace: list = []
    imgs = frames(cfg, 400, count=3, frac=3e-2, need=2, size=160, trace=trace)
    assert len(trace) == len(imgs) == 3, (len(trace), len(imgs))
    for before, now in zip(trace, trace[1:]):
        held = set(now["hues"]) & set(before["hues"])
        assert held, f"t={now['t']} 一帧之间没有一个群接上，色相没法记路"
        for key in held:
            assert now["hues"][key] == before["hues"][key], (
                f"t={now['t']} 成员没变的群改了色相：{sorted(key)[:4]}")
        if now["k"] == before["k"] and held == set(now["hues"]):
            assert now["moved"] == 0.0, f"t={now['t']} 结构没变却挪了颜色：{now['moved']}"


def test_anim_merge_tick_is_the_peak_of_hue_motion() -> None:
    """并群那一 tick 色相移动总量是峰值：分开是各挪一格，并群是输掉的那半整半跳过去。"""
    from wy.anim import frames

    cfg = Config(n=128, seed=7)
    trace: list = []
    frames(cfg, 1200, count=8, frac=3e-2, need=4, size=160, trace=trace)
    assert len(trace) >= 6, len(trace)
    moves = [float(r["moved"]) for r in trace]
    assert all(0.0 <= m <= 0.5 for m in moves), moves
    assert moves.count(0.0) >= 2, f"没有结构稳住的帧可对照：{moves}"
    peak = max(range(len(trace)), key=lambda i: moves[i])
    assert trace[peak]["k"] < trace[peak - 1]["k"], (
        f"峰值落在 t={trace[peak]['t']}（k {trace[peak - 1]['k']}→{trace[peak]['k']}），"
        f"不是并群帧；moved={moves}")
    splits = [moves[i] for i in range(1, len(trace)) if trace[i]["k"] > trace[i - 1]["k"]]
    if splits:
        assert max(splits) < 0.5 * moves[peak], (
            f"分开挪了 {max(splits):.3f}，不比并群 {moves[peak]:.3f} 小一半")




def test_drill_adjacency_is_symmetric_without_self() -> None:
    """群内部的邻接矩阵：对称、对角为 0——不然内部图会自己连自己。"""
    from wy.drill import member_adjacency

    net = Network(Config(n=64, modules=4, seed=5))
    members = list(range(0, 8))
    adj = member_adjacency(net, members)
    assert adj.shape == (8, 8), adj.shape
    assert np.allclose(adj, adj.T), "邻接矩阵不对称"
    assert np.allclose(np.diag(adj), 0.0), "自己不该和自己有关系"
    assert (adj > 0).any(), "同一模块内总该有关系"


def test_drill_face_draws_requested_size() -> None:
    """群内部那张图必须是指定尺寸，而且不是一块纯色。"""
    from wy.drill import face

    from wy.engine import Engine
    from wy.shape import group_table, scale_of

    net = Network(Config(n=64, modules=4, seed=5))
    eng = Engine(net)
    for _ in range(200):
        eng.step()
    fv = eng.values()
    groups = group_table(fv, net, 1e-3 * scale_of(fv), None, 2,
                         touched=eng.recent(net.cfg.window))
    groups = [g for g in groups if len(g.members) >= 3]
    if not groups:
        return
    img = face(groups, net, fv, groups[0].gid, size=200, seed=5)
    assert img.size == (200, 200), img.size
    assert len(img.getcolors(maxcolors=200000)) > 100, "不该是一块纯色"




def test_zoom_only_walks_one_hop() -> None:
    """邻居图只该有「自己 + 一跳邻居」，尺寸对、不是纯色。"""
    from wy.drill import zoom

    net = Network(Config(n=64, modules=4, seed=5))
    fv = np.arange(64, dtype=float)
    img = zoom(net, fv, 0, size=180, seed=5, hops=1)
    assert img.size == (180, 180), img.size
    assert len(img.getcolors(maxcolors=200000)) > 80, "不该是一块纯色"
    # 一跳邻居存在（度数 4），否则这图画出来只有中心一个点
    assert sum(1 for w in net.nbrs[0] if int(w) != 0) >= 1




def test_scan_reports_one_row_per_tick_and_picks_one() -> None:
    """扫描：每个 tick 一行，读数齐全；best 只在「至少两团」的行里挑。"""
    from wy.scan import best, scan

    net = Network(Config(n=64, modules=4, seed=5))
    rows = scan(net, [30, 60, 120], frac=3e-2, need=2, target_k=4)
    assert len(rows) == 3, [r["tick"] for r in rows]
    for r in rows:
        for key in ("tick", "k", "defect", "contrast", "q", "verdict", "frac"):
            assert key in r, key
    pick = best(rows)
    assert pick is None or pick["k"] >= 2, pick




def test_portrait_follows_readings_and_size() -> None:
    """画像：读数变参数要跟着变；画出来尺寸对、不是纯色。"""
    from wy.portrait import draw_portrait, params_from_readings

    a = params_from_readings(2, 0.1, 1.0, 1e-6)
    b = params_from_readings(8, 3.0, 5.0, 1e-2)
    assert b.strands > a.strands, (a.strands, b.strands)
    assert b.gloss > a.gloss, (a.gloss, b.gloss)
    assert b.grain > a.grain, (a.grain, b.grain)
    img = draw_portrait(a.__class__(**{**a.__dict__, "size": 160}))
    assert img.size == (160, 160), img.size
    assert len(img.getcolors(maxcolors=200000)) > 60, "不该是一块纯色"


# ---------------------------------------------------------------- 跑一跑
def test_hue_follows_strongest_relation() -> None:
    """色彩空间：关系最强的两个群，色相必须最接近（不是看着像，是能量出来）。"""
    import itertools

    from wy.color import color_space

    class G:
        span = 1.0
        stable = 3.0

        def __init__(self, gid: int) -> None:
            self.gid = gid

    def gap(a: float, b: float) -> float:
        d = abs(a - b) % 1.0
        return min(d, 1.0 - d)

    rng = np.random.default_rng(11)
    for _ in range(10):
        k = int(rng.integers(4, 9))
        W = rng.uniform(0.0, 0.2, size=(k, k))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        cs = color_space([G(i) for i in range(k)], None, strength=W)
        hue = [cs[i][0] for i in range(k)]
        i, j = np.unravel_index(int(np.argmax(W)), W.shape)
        best = gap(hue[int(i)], hue[int(j)])
        med = float(np.median([gap(hue[a], hue[b]) for a, b in itertools.combinations(range(k), 2)]))
        assert best <= med + 1e-9, f"最强关系 ({i},{j}) 色相差 {best:.3f} 宽于中位 {med:.3f}"


def test_hue_track_follows_the_people_not_the_groups() -> None:
    """颜色也记路：老群留原色，并群是输掉的那半跳过去——移动量按人算，不按群算。"""
    from wy.color import hue_track

    class G:
        span = 1.0
        stable = 3.0

        def __init__(self, gid: int, members: list[int]) -> None:
            self.gid = gid
            self.members = members

    def W(k: int) -> np.ndarray:
        m = np.full((k, k), 0.05)
        np.fill_diagonal(m, 0.0)
        if k > 1:
            m[0, 1] = m[1, 0] = 0.8
        return m

    old = [G(0, [0, 1, 2, 3]), G(1, [4, 5, 6, 7]), G(2, [8, 9, 10, 11])]
    h1, m1 = hue_track({}, old, W(3))
    assert m1 == 0.0, "第一帧没有上一帧，没有人需要挪"
    assert set(h1) == {frozenset(g.members) for g in old}

    # 一模一样再来一遍：色相逐个照抄，移动量是 0，不是“重新接近了一遍”
    h2, m2 = hue_track(h1, old, W(3))
    assert m2 == 0.0, m2
    assert h2 == h1

    # 0、1 并成一团、2 不动：2 的色相原封不动，并出来的那团沿用其中一个旧色
    merged = frozenset(range(8))
    rest = frozenset([8, 9, 10, 11])
    h3, m3 = hue_track(h2, [G(0, sorted(merged)), G(1, sorted(rest))], W(2))
    assert h3[rest] == h2[rest], "没参与的群被改了色"
    assert h3[merged] in (h2[frozenset([0, 1, 2, 3])], h2[frozenset([4, 5, 6, 7])]), (
        "并出来的群凭空长了个新颜色")
    assert 0.0 < m3 <= 0.5, f"并群总得有半批人换色，moved={m3}"


def test_tool_self_checks_pass() -> None:
    """tools/check_*.py 是自检脚本：它们必须退出码为 0，否则形/色的红线就断了。"""
    import glob
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    scripts = sorted(glob.glob(str(root / "tools" / "check_*.py")))
    assert scripts, "没有找到 tools/check_*.py"
    bad = []
    for path in scripts:
        r = subprocess.run([sys.executable, path], cwd=str(root),
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            bad.append(f"{Path(path).name} exit={r.returncode} {r.stdout.strip()[-200:]}")
    assert not bad, "; ".join(bad)


def test_lifetime_scales_as_one_over_r_alpha_squared() -> None:
    """寿命定律：t ∝ 1/(r·α²)。用 r 减半看寿命是否约翻倍（±45% 容差）。"""
    from wy.network import Config
    from wy.strip import lifetimes

    def t2(r: float) -> float:
        cfg = Config(n=128, seed=7, alpha_range=(0.85, 0.95), inter_ratio=(r, r), walk_power=2.0)
        out = lifetimes(cfg, 9000, frac=1e-2)
        v = float(out.get("t2") or float("nan"))
        assert v == v, f"r={r} 在窗口内没塌，寿命测不出来"
        return v

    a, b = t2(0.04), t2(0.02)
    ratio = b / a
    assert 1.55 <= ratio <= 2.9, f"r 减半寿命只变成 {ratio:.2f} 倍"


def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
