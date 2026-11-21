# 望云的分层：核心、出图、网页、插件

这份文件写给「想拿望云做别的东西」的人——网页端、桌面壳、别人的工具。
它只讲接法和边界，不讲读数怎么来（那是 `DESIGN.md` 的事）。

## 1. 为什么分成两层

望云原来是一整块：跑实验的代码、出图的代码、读数的代码混在一起，想拿它的读数
就得连 Pillow 一起装。要把它做成软件、接到网页上时，这个混在一起的结构就成了
阻碍：网页不该为了拿一条 k(t) 曲线去装图形库。

分完以后只有一条硬规矩：

    wy.core.*    纯演算，只依赖 numpy；不 import Pillow、不碰命令行、不写文件
    其余         出图、命令、网页、插件——吃核心的读数，加自己的表现

用 `tests/run_tests.py` 钉着：`test_core_imports_without_pillow` 在一个「PIL
被拦住」的子进程里把核心跑一遍，核心要是偷偷把 Pillow 拖进来，这条就红。

## 2. 目录

```text
wy/
  core/             核心（纯）
    network.py      Config / Network：关系组怎么长出来
    engine.py       Engine：走一步、违约读数、反馈
    digits.py       StabilityMonitor：前缀长度、稳定位数
    shape.py        Group / group_table / partition / describe_change
    layout.py       Shape / group_strength / spring / make_shape / carry_pos
    color.py        color_space / hue_track / hue_vec（色相按成员记路）
    gauge.py        可读性读数：modularity / q_excess / contrast
    exact.py        整数定标演算（位数天花板）
    scan.py         真图上挑 tick
    load.py         边表 → 真图
    bundle.py       ★ 对外交付面：observe() → JSON-able dict
  render.py         出图：Pillow 两型模糊、字、读数曲线
  scene.py          场面：群 → 物体（位置/远近/大小/色相）
  anim.py           动画：采样若干帧 + 色相接路
  strip.py          诊断带：事件和读数同一条时间轴
  board.py / drill.py / paint.py / portrait.py / brief.py   其余视图
  server.py         本地网页端（stdlib http.server，零新依赖）
  plugins.py        扩展点登记处：datasource / reading / renderer
  plugins_builtin.py 自带插件（svg 渲染、hue-moved 读数、边表数据源）
  web/              网页：index.html + app.js + style.css
  cli.py            命令行（14 个命令）
dsh-plugin/         DSH 插件包（@local/wangyun-tools）：wy_observe / wy_frame → python -m wy.core
```

旧的 `wy.network` / `wy.engine` / `wy.shape` / `wy.color` … 这些路径还留着，
是兼容口，转给 `wy.core.*`。新代码请直接从 `wy.core` 进。

## 3. 核心对外的两件东西

**一件是「怎么跑」**：`Config` 是所有确定性参数，`observe(cfg, ticks, ...)`
跑完把结果收成 dict。不需要知道 Engine、群表、弹簧这些细节。

```python
from wy.core import Config, observe, dumps

bundle = observe(Config(n=128, seed=7), 1200, probe=20, need=4)
open("bundle.json", "w", encoding="utf-8").write(dumps(bundle))
```

**另一件是 bundle**，它的结构就是交付契约：

```text
meta      n / modules / seed / window / digits / ticks / frac / probe / need / exact
readings  t、k(t)、违约读数（对数）、稳定位数——每个 tick 一个点
events    划分变化的时刻：tick / kind(merge|split|shift) / text / k_before / k_after
frames    采样帧：t / k / hue_moved / groups / strength / pos
           groups[i] = gid / count / value / span / stable / settled / hue
           （members=True 时每组多一个 members 下标数组）
```

帧里的色相是**接着上一帧按成员记路**的（`hue_track`，见 DESIGN §30）：
还是这批人，色相一个数都不变；换群才换色。网页上拖动时间轴时颜色不会乱跳，
就是因为它来自这份 bundle，而不是每个画面各算一套。

## 4. 网页端

```bash
wy serve                       # 默认 http://127.0.0.1:8765/
wy serve --port 9000 --open    # 换端口并自动开浏览器
```

页面只有三样东西：参数（n / 模块 / ticks / 种子 / 采样 / 位数门 / 视角）、
画布（群的位置和大小来自 bundle，色相来自 bundle）、时间轴（帧滑杆、事件列表、
播放）。画布是原生 canvas，不引前端框架，也不下载任何外部资源。

接口（都只看参数，不吃状态）：

| 路径 | 给什么 |
| --- | --- |
| `GET /` | 页面 |
| `GET /api/bundle?n=&ticks=&seed=&...` | 一份完整 bundle（JSON） |
| `GET /api/svg?...&index=-1` | 当前帧的 SVG（走插件 `renderer/svg`） |
| `GET /api/png?...&index=-1&size=760` | 当前帧的 PNG（Pillow 基准渲染，和命令行同一套 `draw_scene`） |
| `GET /api/plugins` | 三个扩展点上都有什么 |
| `GET /api/health` | 探活 + 插件表 |

页面和别的软件吃的是同一份 bundle——不是「网页一套、命令行一套」。

## 5. 插件

三个口子，都在 `wy/plugins.py`：

```python
from wy import plugins

@plugins.datasource("my-graph", "从我们的数据仓库拿图")
def my_graph(path, cfg=None):
    ...

@plugins.reading("attention", "再记一条读数")
def attention(bundle):
    ...

@plugins.renderer("svg", "换成我们自己的画法")
def svg(bundle, index=-1):
    ...
```

- 名字重复会直接报错（不许悄悄顶掉别人的通道）。
- 装成包以后能被发现：entry point 组 `wangyun.plugins`，目标是一个
  `register(registry)` 函数，`plugins.discover()` 会把它们都叫一遍。
- 插件不许改核心判据：读数的规矩在 `wy.core` 里，插件只换输入输出。

自带的三个例子在 `wy/plugins_builtin.py`：`renderer/svg`（零依赖，把一帧画成
SVG）、`reading/hue-moved`、`datasource/edgelist`。写自己的插件时照它们的形状来。

## 6. 装与跑

```bash
pip install -e .               # 装成 wangyun 命令
wangyun run --n 128 --seed 7   # 原来的命令行一个没少
wangyun serve --open           # 网页端
```

只想要读数、不想要 Pillow 的环境：装 numpy，把 `wy` 目录拷过去，直接
`from wy.core import Config, observe`——核心这一层不会把 Pillow 拖进来
（有契约测试钉着）。

打包本身也自检：`tools/check_package.py` 查构建后端、元数据、包列全没列全、
`web/` 里每个文件在不在 package-data 里、命令行入口能不能 import、版本号两处一不一致。
真装一遍的验法（本机做过的）：`python -m venv --system-site-packages out\_venv`，
`out\_venv\Scripts\python -m pip wheel . --no-deps -w out\_wheel` 出 wheel，
`pip install --no-deps --target out\_site out\_wheel\wangyun-4.2.0-py3-none-any.whl`，
然后从仓库外的目录 `wangyun --help`、`importlib.resources.files("wy").joinpath("web/app.js")`
都通。本机 site-packages 写入探测被文件锁挡过一次（WinError 32），`--target` 绕开即可。

## 7. 接进 DSH（`dsh-plugin/`）

望云的整个软件，可以直接挂在 DeepSeek Harness 里当一个工具：

- 包在 `dsh-plugin/`，名字 `@local/wangyun-tools`，`type: module`，没有客户端半边。
- 装：`plugin_manager install_bundle target=D:\望云\dsh-plugin`（写进本机 profile，
  重启一半就在）。
- 工具两个：`wy_observe` 看总览（参数和命令行一样 `n/modules/ticks/seed/frac/probe/need`，
  可带 `members`、`out`），压成有界的一段字：k 走向、事件、末帧每个群几个人和什么颜色；
  `wy_frame` 追问某一帧（多一个必填 `tick`，可带 `members=true` 展开成员下标），报每群的
  色相/稳定位数/稳没稳，以及群与群之间最紧的几对。两个都内部跑 `python -m wy.core`，
  不画图——图在 `wy render/serve` 那边。
- 配置在 `dsh-plugin/cordis.patch.yml` 的 `config`：`python`（本机 runtime 的
  python）、`workspace`（望云仓库根）、`timeoutMs`。
- `tools/check_plugin.py` 管装之前的那几件事：`files` 罩不罩得住实物、`exports`
  齐不齐、locale/图标/patch id、`node --check` 过得去。它跟别的 `tools/check_*.py`
  一起被 `test_tool_self_checks_pass` 调到。

## 8. 测试

```bash
python tests/run_tests.py      # 96 条，自写 runner，无 pytest
```

跟这一层结构直接相关的几条：

- `test_core_imports_without_pillow`：核心在没有 Pillow 的进程里能跑。
- `test_bundle_is_json_and_keeps_hues_by_membership`：bundle 能 JSON 落盘，
  色相按成员延续。
- `test_core_command_line_emits_json_bundle`：`python -m wy.core` 的 stdout 就是一份
  bundle（DSH 插件走的就是这条路）。
- `test_plugins_expose_three_kinds_and_render_svg`：三个扩展点都在，重名会被拒。
- `test_server_answers_health_and_bundle`：网页端的接口真的能返回数据。
- `test_server_png_is_a_pillow_render`：网页端「原图」发的是真 PNG，帧号越界会夹住。
- `tools/check_docs.py`：文档里引用的契约名、条数、§号必须和代码对得上。
- `tools/check_plugin.py`：插件包的清单、说明、图标、接线、语法。
- `tools/check_package.py`：打包清单、package-data 罩不罩得住 `web/`、入口与版本号。
