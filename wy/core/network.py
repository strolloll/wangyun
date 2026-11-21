"""关系组：单位之间的比例。无形的那半边——不渲染，只提供不变量与校验。

结构约定：

- 单位只认识少数邻居（有界度数 `degree`）。这是它能永远跑下去又不爆资源的唯一办法。
- 关系**双向存在、比例各自独立**：i 对 j 的比例和 j 对 i 的比例不是同一个数。
  关系是单方面的量，不是一条对称的边。
- 每个单位恰好有一个模块内的小圈子（`intra` 个邻居，模块内 3-正则环）和
  `inter` 个模块间邻居（相邻模块同号配对）。这样每个单位的入度都 ≥ 1，
  不会有「永远传不到」的单位——这个坑踩过一次，见 DESIGN.md §9。
- 强关系在模块内（0.7~1.0），弱关系跨模块（默认 0.1~0.35，可调到 0.004）。
- **模块划分不进入初值**。初值完全随机，模块结构只能从关系比例里长出来。
- 每个单位有自己的耦合系数 alpha（异质）。单位全同、规则全对称，跑一万年只会是一张格子布。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    """一次实验的全部确定性参数。改任何一个，图都该变，所以要一起记下来。"""

    n: int = 128
    modules: int = 8
    intra: int = 3
    inter: int = 1
    intra_ratio: tuple[float, float] = (0.70, 1.00)
    # 跨模块的弱关系。这个数决定「涌现出来的结构能活多久」：
    # 大约 1/inter_ratio 个 tick 之后模块就开始并成一片（实测见 DESIGN.md §9）。
    inter_ratio: tuple[float, float] = (0.002, 0.006)
    alpha_range: tuple[float, float] = (0.40, 0.90)
    kick: float = 1.0e-6
    window: int = 8
    # 成团时要多强的关系才算数。0 = 所有关系都算，这样 k(r) 的台阶完全由
    # 数值收敛的程度决定（平台的宽度 = 结构对比度），不是被阈值硬切出来的。
    strong: float = 0.0
    seed: int = 7
    v_lo: float = 1.0
    v_hi: float = 1.0e4
    digits: int = 12
    # 源规则：谁当 tick 开头那个「变量」。
    #   cycle    = 轮转（tick % n），确定、可复现、不偏（默认）
    #   residual = 每 tick 挑残差最大的单位（听起来聪明，实测是反扩散的，见 DESIGN.md §9）
    source_rule: str = "cycle"
    # 路径规则：每步走哪条关系。
    #   violation = 欠得最多的那条（比例 × |值差|）最大（默认）
    #   ratio     = 比例最大的那条（静态，会把路挤进固定走廊，少数单位饿死，见 DESIGN.md §9）
    walk_rule: str = "violation"
    # 数值表示（DESIGN.md §13.2）：float64 的位数天花板约 15 位，
    # 换成分数定标的整数演算（wy/exact.py）之后只受定标位数和时间限制。
    # 关 = 走 float64（默认，快）；开 = 走整数演算（慢约一倍，位数可到 20+）。
    exact: bool = False
    exact_digits: int = 30
    # 形反作用于关系（DESIGN.md §7 第 4 条的定案）：每 feedback_every 个 tick，
    # 对**已经稳住**的群做一次收紧——群内的关系比例往 1 推、群外往 0 推。
    # 0 = 关闭（关系是静态数据，默认）。开了以后关系就不是静态数据了，
    # 读数里必须写明（见 DESIGN.md §10）。
    shape_feedback: float = 0.0
    feedback_every: int = 25
    # 多大的结构年龄才算「已经站住」——同一个成员集合连续这么多个 tick 没变过。
    feedback_age: int = 64
    # 拿哪一档分辨率来认「群」。必须用细的：用粗档时整张图早早就是「一个群」，
    # 于是全部关系一起被加固，等于给同质化踩油门（实测 4000 tick 就并成 2 团）。
    feedback_frac: float = 1e-3
    # 「焊到这儿为止」：只在前 feedback_until 个 tick 里收紧，之后关系冻住、继续跑。
    # 用来分清两件事——结构是**靠焊枪撑着**，还是焊过一次之后能**自己活下去**。
    feedback_until: int | None = None
    # 自动松手：不用手填 N。每轮焊之前记一次违约读数，一旦最近几轮不再下降
    # （相对改善 < feedback_tol），就当「焊枪已经把差别顶住」停手。
    feedback_auto: bool = False
    feedback_tol: float = 0.02
    feedback_win: int = 8
    # 模块之间的那些外连，连成几个独立的环。1=所有模块一大圈（默认）；
    # 2=分成两组各一个圈；数字变大就是把「谁跟谁在一圈里」拆得更碎。
    # 模块数和模块内结构完全不变，变的只有这个——用来问「形状读不读得出间接结构」。
    inter_cycles: int = 1
    # 让某一个模块「内部不紧」：它的内部关系改用这一段比例，拓扑和模块数都不变。
    # 用来问：望云认的是真结构，还是生成器给的标签。−1 = 不做手脚。
    weak_module: int = -1
    weak_intra: tuple[float, float] = (0.01, 0.03)
    # 一次让好几个模块内部不紧（可以为空）。跟弱化一半的模块配合，就能造出
    # 「同一个连通块里住着两种形」——模块数、连通块数都不变，只是密度分了两档。
    weak_modules: tuple[int, ...] = ()
    # 打乱「谁跟谁在一圈里」：环的个数不变（还是 inter_cycles 个），成员变了。
    # 用来验一件事——读数认的是结构，还是模块的名字。
    inter_shuffle: bool = False

    # 走动规则里比例的指数：score = 比例**walk_power × |值差|
    # 1.0 = 现在这条（§9.6）；调大更看重强关系（走廊风险回来）；调小更看重差值本身。
    walk_power: float = 1.0

    @property
    def degree(self) -> int:
        return self.intra + self.inter


class Network:
    """关系组。构造完之后完全只读。"""

    def __init__(self, cfg: Config) -> None:
        if cfg.n % cfg.modules != 0:
            raise ValueError("n 必须能被 modules 整除")
        size = cfg.n // cfg.modules
        if size < 2 * cfg.intra + 2:
            raise ValueError("每个模块太小，凑不出 intra 个邻居还不重样")

        self.cfg = cfg
        self.n = cfg.n
        # 三条互相独立的随机流：改网络不会牵动初值，改初值不会牵动 kick。
        rng_net = np.random.default_rng([cfg.seed, 0])
        rng_val = np.random.default_rng([cfg.seed, 1])

        members = [np.arange(m, cfg.n, cfg.modules) for m in range(cfg.modules)]
        # 第四条随机流：只用来打乱「模块之间谁跟谁一圈」，不牵动比例和初值。
        shuffle_rng = np.random.default_rng([cfg.seed, 2]) if cfg.inter_shuffle else None
        edges = _relation_edges(
            members,
            size,
            cfg.intra,
            cfg.inter,
            cfg.inter_cycles,
            cfg.weak_module,
            shuffle_rng,
            tuple(cfg.weak_modules),
        )

        nbrs = np.zeros((cfg.n, cfg.degree), dtype=np.int64)
        ratios = np.zeros((cfg.n, cfg.degree), dtype=np.float64)
        deg = np.zeros(cfg.n, dtype=np.int64)
        edge_r: dict[tuple[int, int], float] = {}

        for kind, a, b in edges:
            if kind == "intra":
                span = cfg.intra_ratio
            elif kind == "intra_weak":
                span = cfg.weak_intra
            else:
                span = cfg.inter_ratio
            r = float(rng_net.uniform(*span))
            if deg[a] >= cfg.degree:
                raise ValueError("度数超了，检查模块规模")
            nbrs[a, deg[a]] = b
            ratios[a, deg[a]] = r
            deg[a] += 1
            lo, hi = (a, b) if a <= b else (b, a)
            edge_r[(lo, hi)] = max(edge_r.get((lo, hi), 0.0), r)

        # 行内按邻居 id 升序：argmax 撞平手时天然取最小 id，确定性的来源之一。
        for u in range(cfg.n):
            order = np.argsort(nbrs[u], kind="stable")
            nbrs[u] = nbrs[u][order]
            ratios[u] = ratios[u][order]
        if not np.all(deg == cfg.degree):
            raise ValueError("有单位的度数不对")

        self.nbrs = nbrs
        self.ratio = ratios
        # 邻接表的 Python 版本：走动那一步只看 4 个邻居，用元组比用 numpy 便宜一个数量级。
        # 比例会被反馈就地改，所以这里只缓存**邻居 id**（静态），比例仍从 self.ratio 现取。
        self.nbrs_list: list[tuple[int, ...]] = [tuple(int(x) for x in row) for row in nbrs]
        self.alpha = rng_net.uniform(*cfg.alpha_range, size=cfg.n)
        self.values = rng_val.uniform(cfg.v_lo, cfg.v_hi, size=cfg.n)
        self.i_edge = np.asarray([k[0] for k in edge_r], dtype=np.int64)
        self.j_edge = np.asarray([k[1] for k in edge_r], dtype=np.int64)
        self.r_edge = np.asarray(list(edge_r.values()), dtype=np.float64)

    # ---------------------------------------------------------------- 查询

    def ratio_between(self, u: int, w: int) -> float:
        """u → w 的那条关系比例，没有就 0。"""
        hit = np.flatnonzero(self.nbrs[u] == w)
        if hit.size == 0:
            return 0.0
        return float(self.ratio[u, hit[0]])

    def kick_delta(self, tick: int) -> float:
        """第 tick 次注入的扰动量。只由 (seed, tick) 决定，与机器快慢无关。"""
        rng = np.random.default_rng([self.cfg.seed, 2, tick])
        mag = self.cfg.kick * (1.0 + 0.5 * float(rng.random()))
        sign = 1.0 if float(rng.random()) < 0.5 else -1.0
        return sign * mag

    def signature(self) -> dict:
        """可复现结果里该被记下来的那部分。"""
        return {
            "n": self.cfg.n,
            "modules": self.cfg.modules,
            "degree": self.cfg.degree,
            "intra_ratio": list(self.cfg.intra_ratio),
            "inter_ratio": list(self.cfg.inter_ratio),
            "alpha_range": list(self.cfg.alpha_range),
            "seed": self.cfg.seed,
            "kick": self.cfg.kick,
            "window": self.cfg.window,
            "strong": self.cfg.strong,
            "source_rule": self.cfg.source_rule,
        }


def _relation_edges(
    members,
    size: int,
    intra: int,
    inter: int,
    cycles: int = 1,
    weak_module: int = -1,
    shuffle_rng=None,
    weak_modules: tuple[int, ...] = (),
):
    """有向关系表。

    模块内：环 + 对径，双向都连（`intra`=3 时每单位模块内 3 出 3 入）。
    模块间：只连「前一个模块 → 后一个模块」同号配对，于是每单位 1 出 1 入，
    八个模块连成一个环。出度 = 入度 = intra + inter。

    `weak_module >= 0` 时，那一个模块的内部关系改用另一段（低得多的）比例，
    标记成 `intra_weak`。模块数和拓扑都不变，只是它内部「没那么紧」——
    用来问一句：望云认的是真结构，还是生成器给的标签。
    """
    out = []
    seen = set()
    weak = set(weak_modules) | ({weak_module} if weak_module >= 0 else set())

    def add(kind: str, a: int, b: int) -> None:
        if a == b:
            return
        key = (kind, a, b)
        if key in seen:
            return
        seen.add(key)
        out.append((kind, a, b))

    if intra != 3:
        raise ValueError("目前只实现了 intra=3（环 + 对径）")
    half = size // 2
    for m, mem in enumerate(members):
        kind = "intra_weak" if m in weak else "intra"
        for t in range(size):
            u, v = int(mem[t]), int(mem[(t + 1) % size])
            add(kind, u, v)
            add(kind, v, u)
            w = int(mem[(t + half) % size])
            add(kind, u, w)
            add(kind, w, u)

    if inter:
        targets = inter_targets(len(members), cycles, shuffle_rng)
        for m in range(len(members)):
            tgt = targets[m]
            a_mem, b_mem = members[m], members[tgt]
            for t in range(size):
                add("inter", int(a_mem[t]), int(b_mem[t]))
    return out


def inter_targets(modules: int, cycles: int, rng=None) -> list[int]:
    """每个模块的那一条外连指向谁。

    默认按顺序分组（0..per-1 一个环、per..2per-1 一个环……），`rng` 给了就
    先把模块顺序打乱再分组——**环的个数不变，谁跟谁在一圈里变了**。
    两种情况每个模块仍恰好被指一次，所以入度依旧是 1。
    """
    cycles = max(1, min(cycles, modules))
    per = modules // cycles
    if per * cycles != modules:  # 除不尽就退回一个大环，别悄悄丢掉模块
        return [(m + 1) % modules for m in range(modules)]
    order = list(range(modules))
    if rng is not None:
        rng.shuffle(order)
    targets = [0] * modules
    for g in range(cycles):
        grp = order[g * per:(g + 1) * per]
        for i, m in enumerate(grp):
            targets[m] = grp[(i + 1) % per]
    return targets
