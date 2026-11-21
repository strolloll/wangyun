# 望云读数 / wangyun-tools

望云（`D:\望云`）是「观察者，不是解题者」：一张关系网被推着走，它只报自己看见的——分成几群、什么时候换群、色相怎么跟着人走。核心已经抽出来（`wy.core`，纯 numpy），这份插件只是把核心的命令行接进 DSH：模型多两个工具——`wy_observe` 看总览，`wy_frame` 追问某一帧，不画图也能看数。

## 工具

| 工具 | 干嘛 |
| --- | --- |
| `wy_observe` | 跑一次读数，返回有界的摘要：划分 k 的走向、违约读数、稳定位数、划分变化（并/分）的时刻、末帧每个群几个人和什么颜色 |
| `wy_frame` | 看某一 tick 附近的那一帧：每群几个人、色相、稳定位数、稳没稳，以及群与群之间最紧的几对。先 `wy_observe` 看事件，再拿它追问「那一刻是谁和谁」；要具体是谁就带 `members=true`，每人下标会附在群下面（超过 12 人截断） |

它不生成图。图在另外三条路上：命令行 `python -m wy ...`、网页端 `python -m wy serve`、别的软件直接吃 bundle JSON。

## 装

把这个目录当 bundle 装：

```
plugin_manager install_bundle target=D:\望云\dsh-plugin
```

装完看返回里的 `application` 和 `warnings`——那两栏才决定它真起来没有。装完重启一半（新会话）工具就在了。

## 配

都写在 `cordis.patch.yml` 的 `config` 里，缺的字段用默认值。

| 键 | 默认 | 意思 |
| --- | --- | --- |
| `python` | 本机 runtime 的 python | 跑 `python -m wy.core` 的解释器。换机器/换 venv 改这里 |
| `workspace` | `D:\望云` | 望云仓库根；要在那儿能 import 到 `wy` |
| `timeoutMs` | `120000` | 一次读数最多跑多久（毫秒） |

## 为什么这么薄

- 核心是纯 numpy 的，插件只递参数、收 JSON、压成一段字。渲染、网页、别的语言都吃同一份 bundle（`wy.core.observe` / `python -m wy.core`），插件不复制任何算法。
- 没有守护进程、没有轮询：不调用就不动。
- 错误翻成人话：python 路径不对、仓库不对，返回的是「检查哪个字段」，不是一串栈。

## 往上长

- 想多一个读数：核心加 `observe` 的字段（要先有契约），插件的 `summarize()` 多读一段。
- 想给别的软件用：不用碰插件，`python -m wy.core --n 64 --ticks 400 --out bundle.json` 就是交付面。
- 想看动画/诊断带：命令行 `wy anim` / `wy strip`，或 `wy serve` 起网页端。
