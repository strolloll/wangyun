"""兼容口：真正的家在 `wy.core.layout`。新代码从那边进。"""

from importlib import import_module as _import_module

_core = _import_module("wy.core.layout")


def __getattr__(name: str):
    return getattr(_core, name)
