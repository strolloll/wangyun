"""扩展点：数据源 / 读数通道 / 渲染通道。

核心里只有参数和读数（`wy.core`）。「读什么图、多记一条什么读数、画成什么样」
都从这三个口子接进来——插件不该改核心的判据，只该换输入输出。

用法：

    from wy import plugins

    @plugins.renderer("svg")
    def svg(bundle, index=-1):
        ...

    plugins.select("renderer", "svg")(bundle)

装成包以后也能被发现（entry point 组 `wangyun.plugins`，指向一个
`register(registry)` 函数）：`plugins.discover()`。
"""

from __future__ import annotations

from importlib import metadata
from typing import Any, Callable

KINDS = ("datasource", "reading", "renderer")

#: 每种扩展点是干什么的，报错和 `describe()` 都用这句话，别猜。
KIND_DOC = {
    "datasource": "数据源：(path | dict, ...) → 能给 Engine 吃的图",
    "reading": "读数通道：(bundle, ...) → 你自己的一条序列/结论文本",
    "renderer": "渲染通道：(bundle, index=-1) → 一帧的画面（SVG 文本 / PNG bytes / ...）",
}


class PluginError(Exception):
    """插件名重复、种类不认识、找不到——都算这一类。"""


class Registry:
    """三个口子的登记处。核心不 import 它，只有壳（CLI / Web）用。"""

    def __init__(self) -> None:
        self._table: dict[str, dict[str, dict[str, Any]]] = {k: {} for k in KINDS}

    def register(self, kind: str, name: str, fn: Callable, doc: str = "") -> Callable:
        if kind not in self._table:
            raise PluginError(f"不认识的扩展点 {kind!r}，只有 {KINDS}")
        if not callable(fn):
            raise PluginError(f"{kind}/{name} 得是个能调的东西，给的是 {type(fn).__name__}")
        if name in self._table[kind]:
            raise PluginError(f"{kind}/{name} 已经有人占了")
        self._table[kind][name] = {
            "fn": fn,
            "doc": (doc or (getattr(fn, "__doc__", "") or "")).strip(),
        }
        return fn

    def select(self, kind: str, name: str) -> Callable:
        try:
            return self._table[kind][name]["fn"]
        except KeyError:
            have = "、".join(sorted(self._table.get(kind, {}))) or "（空）"
            raise PluginError(f"没有 {kind}/{name}；现成的：{have}") from None

    def names(self, kind: str) -> list[str]:
        return sorted(self._table.get(kind, {}))

    def describe(self) -> dict[str, dict[str, str]]:
        return {kind: {n: v["doc"] for n, v in sorted(rows.items())}
                for kind, rows in self._table.items()}


REGISTRY = Registry()


def datasource(name: str, doc: str = "") -> Callable[[Callable], Callable]:
    return _deco("datasource", name, doc)


def reading(name: str, doc: str = "") -> Callable[[Callable], Callable]:
    return _deco("reading", name, doc)


def renderer(name: str, doc: str = "") -> Callable[[Callable], Callable]:
    return _deco("renderer", name, doc)


def _deco(kind: str, name: str, doc: str):
    def wrap(fn: Callable) -> Callable:
        REGISTRY.register(kind, name, fn, doc)
        return fn

    return wrap


def select(kind: str, name: str) -> Callable:
    return REGISTRY.select(kind, name)


def names(kind: str) -> list[str]:
    return REGISTRY.names(kind)


def describe() -> dict[str, dict[str, str]]:
    return REGISTRY.describe()


def discover(group: str = "wangyun.plugins") -> list[str]:
    """装上来的插件包（entry point 组 `wangyun.plugins`，目标是 `register(registry)`）。"""
    loaded: list[str] = []
    for ep in metadata.entry_points(group=group):
        fn = ep.load()
        fn(REGISTRY)
        loaded.append(ep.name)
    return loaded
