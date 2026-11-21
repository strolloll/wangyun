"""望云的核心：只依赖 numpy，不碰 Pillow、不碰命令行、不碰文件系统。

其他软件（网页端、桌面壳、别人的工具）只该从这一层拿读数；
出图是另一层（`wy.render` / `wy.scene`），吃同样的参数画。

分层规矩（`tests/run_tests.py` 里有契约钉着）：

    wy.core.*    纯演算与读数：network / engine / digits / shape / layout / color /
                 gauge / exact / scan / load / bundle
    wy.render    出图：Pillow 两型模糊、场景、动画、诊断带、一屏看全
    wy.server    本地 Web（stdlib HTTP，零新依赖）
    wy.plugins   扩展点：数据源 / 读数通道 / 渲染通道
"""

from .network import Config, Network, inter_targets
from .engine import Engine, TickRecord
from .digits import StabilityMonitor, Tail
from .shape import (
    Group,
    PartitionTracker,
    describe_change,
    group_table,
    k_curve,
    k_of,
    local_scale,
    partition,
    partition_rel,
    plateau,
    scale_of,
    table_signature,
)
from .layout import Shape, broadcast, group_strength, make_shape, spring
from .color import color_space, hue_track, hue_vec
from .gauge import contrast_and_dispersion, modularity, q_excess, verdict
from .exact import ExactEngine, ExactTrace, StableDigits
from .load import RealNetwork, load_graph, read_edgelist
from .scan import best as best_tick
from .scan import scan
from .bundle import dumps, observe

__all__ = [
    "Config", "Network", "inter_targets",
    "Engine", "TickRecord",
    "StabilityMonitor", "Tail",
    "Group", "PartitionTracker", "describe_change", "group_table", "k_curve", "k_of",
    "local_scale", "partition", "partition_rel", "plateau", "scale_of", "table_signature",
    "Shape", "broadcast", "group_strength", "make_shape", "spring",
    "color_space", "hue_track", "hue_vec",
    "contrast_and_dispersion", "modularity", "q_excess", "verdict",
    "ExactEngine", "ExactTrace", "StableDigits",
    "RealNetwork", "load_graph", "read_edgelist",
    "scan", "best_tick",
    "observe", "dumps",
]
