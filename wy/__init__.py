"""望云：看形从关系里长出来的观测器。

分层（与 DESIGN.md §31 对应）：

    core/       纯演算：关系、迭代、位数、形、布局、颜色、诊断、真图、bundle
    render.py   出图层（Pillow）：只吃群表，A/B 两个模糊通道
    server.py   本地网页端；web/ 是页面
    desktop.py  桌面入口：双击图标 → 起服务 → 开窗口
    cli.py      命令行（十四条）

`wy.network` 等旧路径是 PEP 562 转发到 `wy.core.*` 的 shim，兼容老脚本。
"""

__version__ = "4.3.0"
