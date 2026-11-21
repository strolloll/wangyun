# 望云 · 交接单（新对话从这里开始）

工作区：`D:\望云`。所有命令在这里跑：

```powershell
$py = "C:\Users\CS TAO\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONPATH = "D:\望云"; Set-Location "D:\望云"
& $py tests\run_tests.py          # 96/96
& $py -m wy --help                # 十四条命令
```

## 它是什么

一个**观测器**，不是求解器：给一张关系网络，看「形」怎么从关系里长出来、长成几团、活多久、能信几位、什么颜色。

一条铁律：**颜色/形状都必须由参数自己长出来**，不允许外部渲染代劳（用户明确否决过「换渲染后端」）。

## 现在的状态

- 测试 **96/96**（`tests/run_tests.py`，自写 runner，无 pytest；`test_*` 自动发现）
- 文档：`DESIGN.md`（实验日志，§0–§31）、`ARCHITECTURE.md`（分层与接法）、`GUIDE.md`（八节判据表）、`README.md`（上手）
- 命令：`run / law / evo / board / strip / read / paint / scene / anim / drill / zoom / scan / portrait / serve`
- 环境：numpy + Pillow；**核心只依赖 numpy**（Pillow 只服务出图层）；没有 matplotlib、没有 pytest；不要用 PowerShell 改文本文件（会变 mojibake，只能用 write/edit）
- 自检：`tools/check_*.py`，`test_tool_self_checks_pass` 会跑：形/色红线 + `check_docs.py`（文档里引用的契约名、自称测试条数、§N 必须真的存在）+ `check_plugin.py`（插件清单/说明/图标/接线/`node --check`）+ `check_package.py`（构建后端、包列全没列全、package-data 罩不罩得住 `web/`、入口能 import、版本号两处一致）
- 分层（v4.0）：`wy.core` 纯演算 + `bundle.py`（`observe()` → JSON，交付面）；出图仍 Pillow；网页端 `wy/server.py` + `wy/web/`；插件 `wy/plugins.py` + `plugins_builtin.py`；接 DSH 的包在 `dsh-plugin/`（已 `install_bundle`，工具 `wy_observe`）

## 钉住的结论（别重做）

- 寿命定律 `t ≈ 33/(r·α²)`（§19 修订），契约 `test_lifetime_scales_as_one_over_r_alpha_squared`
- 色相跟随最强关系（`test_hue_follows_strongest_relation`）；动画里色相按成员记路，换群才换色（§30.1–§30.4，四条契约）；「并群最疼」只在 k 大时成立，k 小了分开一大团也能更疼（§30.5 修正，`test_hue_peak_can_fall_on_a_split`）；诊断带第五层就是它（四条 strip 契约）
- `evo` 的颜色是另一种读法：实心/点阵读「稳没稳住」，**不**接成员色（§28.4，`test_shape_colors_say_settled_not_who`）
- 松紧**没有**跨 n 的读数（§14.2，四条路都堵上）
- 位数天花板由 kick 定（§13）、量尺会伪装成天花板（§13.1）
- 真图读取要先 `scan` 挑 tick，可读性看 `contrast / q_excess`（§24/§25）
- 分层规矩（§31）：核心不碰 Pillow/CLI/文件（`test_core_imports_without_pillow`）；bundle 能落 JSON 且跨进程保色（`test_bundle_is_json_and_keeps_hues_by_membership`）；核心命令行 stdout 就是一份 bundle（`test_core_command_line_emits_json_bundle`）；插件三通道重名拒绝（`test_plugins_expose_three_kinds_and_render_svg`）；网页端接口真的给数据（`test_server_answers_health_and_bundle`）；网页端「原图」发的是真 PNG 且帧号越界会夹住（`test_server_png_is_a_pillow_render`）
- 接进 DSH 只走一条路：`dsh-plugin/` 的 `wy_observe` 调 `python -m wy.core`，不复制算法、不画图（`tools/check_plugin.py` 钉清单和接线）

## 上一件（已完成，别重做）

v4.0 产品化：核心抽出 + 本地网页端 + 插件 + 文档。

- 十个模块搬进 `wy/core/`（内容没动）：network / engine / digits / shape / layout / color / gauge / exact / load / scan；根模块的 import 全部改成 `.core.*`
- 旧路径 `wy/network.py` … 十个 shim 留着兼容，用 PEP 562 `__getattr__` 按名转发。踩过两次：`from .core.X import *` 会丢 `_crossings` 这类私有名（测试会 import）；`from .core import scan` 拿到的是**函数**不是模块，转发必须用 `importlib` 绝对路径
- `wy/core/bundle.py`：`observe(cfg, ticks, *, frac, probe, need, seed, iters, members)` 与 `dumps()`；旧路径的 `scene.carry` 逻辑提成 `layout.carry_pos()`；§30.5 三处各写一遍的色相向量收成 `color.hue_vec()`
- 输出面：`frames` 每群带 gid/count/value/span/stable/settled/hue（`members=True` 加成员下标）；`readings` 每 tick t/k/defect/stable；`events` 带 kind 和 k 前后。帧色相来自 `hue_track`，网页拖时间轴不乱跳
- `wy/plugins.py` 三通道（datasource/reading/renderer，重名报错，entry point 组 `wangyun.plugins`）；`plugins_builtin.py` 自带 svg / hue-moved / edgelist；`wy/server.py`（stdlib，`/api/bundle`、`/api/svg`、`/api/plugins`、`/api/health`，bundle 缓存 8 份）；`wy/web/` 三件静态（原生 canvas，无外部资源）；`wy serve --port --open`
- `pyproject.toml` 4.2.0：`packages = ["wy", "wy.core"]`，package-data 带 `web/*.{html,css,js}`
- 文档：`ARCHITECTURE.md` 新写；`DESIGN.md` §31 + 目录行；README 命令/文件树/「做成软件」一节；GUIDE 不动
- 新增四条契约，v4.1 加命令行契约，v4.2 加 PNG 契约：现在共 96 条（名字见上面「钉住的结论」最后两条）
- 坑：n=16/modules=4 会 `每个模块太小，凑不出 intra 个邻居还不重样`，测试一律用 n=32；整套跑约 110 秒，用 `& $py -u tests\run_tests.py *> out\suite.log` 再 tail 日志，别用管道干等（会像死机）

v4.1 接 DSH：

- 核心多了 `python -m wy.core`（`wy/core/__main__.py`，纯 numpy）：`--n/--modules/--ticks/--seed/--frac/--probe/--need/--iters/--walk-power/--members/--out`，stdout 是一份 bundle JSON；契约 `test_core_command_line_emits_json_bundle`
- `dsh-plugin/`（`@local/wangyun-tools`，Host-only bundle）：`index.js` 只干三件事——读 patch 里的 config、`execFile` 跑核心、把 bundle 压成有界摘要；工具 `wy_observe`（n/modules/ticks/seed/frac/probe/need/members/out）
- 已装：`plugin_manager install_bundle target=D:\望云\dsh-plugin` → `application: applied`；重启一半后工具在工具表里。实测 n=64/240 tick 出数正常，`/api/*` 没动
- `tools/check_plugin.py`：清单（`files` 罩不罩得住实物）、说明（locale 两段 meta）、图标（≤256 KiB）、接线（patch id/name/config + index.js 有 apply/tools.register/wy_observe）、语法（`node --check`）；`test_tool_self_checks_pass` 会自动跑到
- 文档：`ARCHITECTURE.md` 加第 7 节「接进 DSH」；README「做成软件」一节补一句；本条交接单

v4.2 网页端原图：

- `/api/png?...&index=-1&size=760`：用 bundle 里已算好的帧搭 `_FrameGroup`（人数当 `members` 长度），交给 Pillow 的 `draw_scene`——和命令行同一套基准渲染；`index` 越界夹住、`size` 夹在 320–1100；没有采样帧时报 400
- 页面侧栏加「原图（Pillow）」链接（`app.js` 跟着当前帧号走）；canvas 仍负责拖时间轴
- 契约 `test_server_png_is_a_pillow_render`（PNG 魔数、越界夹住、坏 index 400）；`ARCHITECTURE.md` 接口表与 `DESIGN.md` §31.4 都加了一行
- 现在共 96 条

v4.3 打包收口：

- `tools/check_package.py`（六组：构建 / 元数据 / 包 / 打包数据 / 入口 / 版本）：包列全没列全、`web/` 每个文件有没有被 package-data 罩住、`wangyun = wy.cli:main` 能不能 import 到可调用对象、`pyproject` 与 `wy/__init__.py` 的版本号一致；`test_tool_self_checks_pass` 自动跑到
- 版本：`pyproject.toml` 与 `wy/__init__.py`（`__version__`）都是 **4.2.0**
- 真装了一遍（本机验过）：`python -m venv --system-site-packages out\_venv` → `pip wheel . --no-deps` 出 `wangyun-4.2.0-py3-none-any.whl`（142145 bytes）→ `pip install --no-deps --target out\_site` 成功；从仓库外目录跑 `wangyun --help` 列出 14 条命令、`importlib.metadata.version("wangyun")=4.2.0`、`importlib.resources.files("wy").joinpath("web/app.js")` 为真、`python -m wy.core --out` 落出 JSON（n=32 → 3 帧）
- 坑：本机 site-packages 写入探测被文件锁挡（`WinError 32` 删不掉 accesstest 临时文件），`--target` 绕开；PowerShell `>` 重定向写 UTF-16，JSON 一律用 `--out` 落盘
- 文档：`ARCHITECTURE.md` §6 加「打包自检 + 真装一遍的验法」、契约清单补 `check_package.py`；README tools 行同步

v4.4 事件缩略图：

- 事件列表每个事件带一张 320px 的 Pillow 缩略图（`<a class="shot">` 指 `/api/png?...&index=<最近帧>&size=320`，`loading="lazy"`，点开原图、不打断「点事件跳帧」）；canvas 仍是拖时间轴用的轻量示意
- 样式：`#events li` 改两列（`.ev-row` flex），缩略图 56px 圆角、悬停变冷色边
- 手工验过：`node --check wy/web/app.js` 通过；起服务后 `/`=200 含望云、`/app.js`=200 含 shot、缩略图 URL 200 `image/png` 31630 bytes、PNG 魔数对
- 契约没变（96 条），`/api/png` 测试已覆盖这条数据源；文档 DESIGN §31.4 与 README「做成软件」各补一句

v4.5 下载 bundle + 原图尺寸 + 第二个工具：

- 网页端侧栏：`下载 bundle`（`<a id="bundle" download>`，href 指向当前参数那份 `/api/bundle`，文件名 `wangyun-n<...>-seed<...>.json`）；`原图尺寸` 滑杆 320–1100 步长 40，`draw()` 把它拼进 `#png` 的 href（缩略图仍固定 320）
- 插件加第二个工具 `wy_frame`（必填 `tick`）：取最接近的采样帧，报每群人数/色相/稳定位数/稳没稳 + 群与群之间最紧的几对；`apply()` 里两个工具逐个 `tools.register`；`check_plugin.py` 的「接线」组要求两个名字都在
- 插件重装：先 `remove_bundle @local/wangyun-tools` 再 `install_bundle target=D:\望云\dsh-plugin`（1.0.0→1.1.0）；
  `wy_frame` 已真的出现在工具表，并空跑过一次 n=64/ticks=240/tick=200 → k=8、8 群各 8 人、色相 232/277/322/7/52/97/142/187°
- 手工验过：`node --check` 两个 js 都过；起服务 `/` 200 含 `id="size"`/`id="bundle"`、`/app.js` 200 含 `/api/bundle` 与 `sizeval`、`/api/png?size=1100` 200 106233 bytes、`size=5000` 被夹到同样的 106233、bundle 3851 bytes
- 这一版没动的：`wy_frame` 的成员名单默认不展开（要就带 `members=true`，超过 12 人截断）

v4.6 缩略图滑杆 + 成员名单：

- 网页端侧栏加第二条滑杆「缩略图」160–320（默认 320）：事件缩略图的 href 用它，改动后 250 ms 防抖只重设 `img.src`，不重跑 bundle
- 插件 1.2.0：`wy_frame` 加可选 `members=true`（核心 `--members`），每群下面附成员下标，超过 12 人截断；实测 n=32/modules=4/t=40 → 四群各 8 人，成员 0 4 8 12… / 1 5 9 13…
- 验过：`node --check` 两个 js、`smoke.mjs`（含 members 一次）、活工具 `wy_frame` 带 members 真调通过；`check_plugin` 5/5

v4.7 网页端读数带：

- `.stage` 里 `<canvas id="band" width="760" height="120">`：上一截画 k(t) 折线（按最高 k 归一），下一截画每帧的色相移动柱（按最大移动归一），事件画成竖线（并=暖橙、分=冷青），橘色播放头跟 `state.index` 的帧走
- `draw()` 末尾调 `drawBand()`，所以拖时间轴 / 播放 / 改参数都会重画；数据全来自 bundle 的 `readings.t/k` 与帧的 `hue_moved`，不另算
- 契约：`test_server_answers_health_and_bundle` 现在也断言页面有 `id="band"`、`/app.js` 里有 `drawBand`（条数仍 96）
- 手工验过：`node --check wy/web/app.js`；起服务 `/` 200 含 `id="band"`、`/app.js` 200 含 `drawBand`、bundle 20 帧/400 个读数点

v4.8 读数带悬停：

- `#band` 加 `mousemove`/`mouseleave`：按 CSS 缩放反算画布 x → tick，找最近的读数点（k）、最近的帧（hue_moved）、8px 内的事件；结果放 `state.hover`，`drawBand()` 画虚线 + 顶部小框
- 框里是 `t=… k=… 色相移动=…`，落在事件上时补 `· 并/分 k_before→k_after`；`run()` 会把 hover 清掉
- 契约：服务测试再断言 `/app.js` 里有 `mousemove`（条数仍 96）
- 手工验过：`node --check wy/web/app.js` 通过；结构上页面/脚本/数据都已在服务里取到（无头环境看不到鼠标悬停的像素，逻辑靠单测与勾稽）

## 下一件事（候选，等用户挑）

1. 探针保留（用户定的，当开发记录）：`tools/probe_anim_hue.py`、`tools/probe_hue_peak.py`
2. 界面小活：缩略图尺寸也接滑杆（现在写死 320）、`wy_frame` 可选展开成员名单
3. 要保持的边界：`_as_groups()` 的 `stable=0.0` 是占位；evo 的配色是刻意的另一种读法

## 别踩的坑

- `Engine(net)` 只收一个参数；`group_table` 要传 `StabilityMonitor` 实例
- 测试里的假群对象只有 `value/span`，`gid/stable` 要用 `getattr` 兜底
- 含中文的正文一律用 write/edit，别用 `python -c` 内联（引号会被吃掉）
- `wy.core` 导出的 `scan` 是函数，shim 转发要用 `importlib.import_module("wy.core.scan")`
- `_as_groups()` 造的 `Group` 里 `stable=0.0` 是占位，不是读数
- `evo` 的颜色是刻意读法（§28.4），别顺手「统一配色」
