"""望云：看形从关系里长出来的观测器。

模块分工（与 DESIGN.md 对应）：

    network.py  关系组（无形那半边）：比例、邻居、异质耦合
    engine.py   迭代核（编辑器）：一个 tick = 一次不重复的传播
    digits.py   稳定前缀：按时间轴判断「这几位多久没动过」
    shape.py    形（输出端）：类型划分、群表、k(r) 曲线
    render.py   视觉系统：只吃群表，Pillow 出图，A/B 两个模糊通道
    cli.py      跑实验
"""

__all__ = ["network", "engine", "digits", "shape", "render"]

__version__ = "4.2.0"
