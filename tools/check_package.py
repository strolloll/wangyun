"""打包自检：`pip install` 之前能查的那几件事。

本机没有一套干净的 pip 环境时，装包这件事只能靠读 `pyproject.toml` 猜。
这个脚本把「装出去会缺什么」挨个列出来：包没列全、`web/` 里新加的文件没进
package-data、命令行入口指不到、版本号两处不一致——都是装完才发现的问题。

这不是在验「能不能装上」，只验「清单、事实、入口、版本四样对得上」。
"""

from __future__ import annotations

import fnmatch
import importlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
WEB_DIR = ROOT / "wy" / "web"


def load() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def check_build(cfg: dict) -> list[str]:
    """构建后端只用 setuptools，别引一堆没装的构建工具。"""
    bad: list[str] = []
    build = cfg.get("build-system") or {}
    requires = [str(x) for x in (build.get("requires") or [])]
    if not any(r.startswith("setuptools") for r in requires):
        bad.append("build-system.requires 里没有 setuptools")
    if str(build.get("build-backend", "")) != "setuptools.build_meta":
        bad.append(f"build-backend 是 {build.get('build-backend')!r}，不是 setuptools.build_meta")
    return bad


def check_metadata(cfg: dict) -> list[str]:
    """该有的字段一个不能少；依赖里必须有 numpy（核心）和 Pillow（出图）。"""
    bad: list[str] = []
    project = cfg.get("project") or {}
    for key in ("name", "version", "description", "readme", "requires-python"):
        if not project.get(key):
            bad.append(f"project.{key} 是空的")
    deps = [str(d).lower() for d in (project.get("dependencies") or [])]
    for mod in ("numpy", "pillow"):
        if not any(d.startswith(mod) for d in deps):
            bad.append(f"dependencies 里缺 {mod}")
    scripts = project.get("scripts") or {}
    if "wangyun" not in scripts:
        bad.append("[project.scripts] 里没有 wangyun")
    return bad


def check_packages(cfg: dict) -> list[str]:
    """列出来的包得真的存在；盘上每个子包也得被列进去。"""
    bad: list[str] = []
    listed = [str(p) for p in ((cfg.get("tool") or {}).get("setuptools") or {}).get("packages") or []]
    for pkg in listed:
        d = ROOT.joinpath(*pkg.split("."))
        if not (d / "__init__.py").is_file():
            bad.append(f"列了 {pkg}，但 {d} 里没有 __init__.py")
    on_disk = set()
    for init in ROOT.glob("wy/**/__init__.py"):
        rel = init.parent.relative_to(ROOT)
        on_disk.add(".".join(rel.parts))
    missing = sorted(on_disk - set(listed))
    if missing:
        bad.append("盘上有但没列进 packages：" + ", ".join(missing))
    return bad


def check_package_data(cfg: dict) -> list[str]:
    """`web/` 里的实物得被 package-data 罩住，不然装完页面打不开。"""
    bad: list[str] = []
    data = ((cfg.get("tool") or {}).get("setuptools") or {}).get("package-data") or {}
    patterns = [str(p) for p in (data.get("wy") or [])]
    if not patterns:
        bad.append("缺 [tool.setuptools.package-data] wy")
    files = sorted(p for p in WEB_DIR.iterdir() if p.is_file())
    for f in files:
        rel = f.relative_to(ROOT / "wy").as_posix()
        if not any(fnmatch.fnmatch(rel, pat) for pat in patterns):
            bad.append(f"web/{f.name} 没被任何 package-data 模式罩住")
    for pat in patterns:
        hit = any(fnmatch.fnmatch(f.relative_to(ROOT / "wy").as_posix(), pat) for f in files)
        if not hit:
            bad.append(f"package-data 模式 {pat} 一个文件都没匹配上")
    return bad


def check_entry(cfg: dict) -> list[str]:
    """命令行入口得能 import 到、拿到可调用的函数；`python -m wy` 也得能走。"""
    bad: list[str] = []
    target = str(((cfg.get("project") or {}).get("scripts") or {}).get("wangyun", ""))
    if ":" not in target:
        bad.append(f"wangyun 的入口 {target!r} 不是 module:function 形式")
    else:
        mod_name, func_name = target.split(":", 1)
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001 - 自检要看得见原因
            bad.append(f"import {mod_name} 失败：{exc!r}")
        else:
            if not callable(getattr(mod, func_name, None)):
                bad.append(f"{mod_name} 里没有可调用的 {func_name}")
    if not (ROOT / "wy" / "__main__.py").is_file():
        bad.append("缺 wy/__main__.py，`python -m wy` 走不通")
    return bad


def check_version(cfg: dict) -> list[str]:
    """pyproject 和包里各写一遍的版本号必须一样，别让装出来的版本骗人。"""
    bad: list[str] = []
    declared = str(((cfg.get("project") or {}).get("version") or ""))
    init = (ROOT / "wy" / "__init__.py").read_text(encoding="utf-8")
    line = next((ln for ln in init.splitlines() if ln.startswith("__version__")), "")
    if not line:
        bad.append("wy/__init__.py 里没有 __version__")
    else:
        value = line.split("=", 1)[1].strip().strip('"').strip("'")
        if value != declared:
            bad.append(f"pyproject 是 {declared}，wy/__init__.py 是 {value}")
    return bad


def main() -> int:
    cfg = load()
    groups = [
        ("构建", check_build(cfg)),
        ("元数据", check_metadata(cfg)),
        ("包", check_packages(cfg)),
        ("打包数据", check_package_data(cfg)),
        ("入口", check_entry(cfg)),
        ("版本", check_version(cfg)),
    ]
    failed = 0
    for title, problems in groups:
        if problems:
            failed += 1
            print(f"{title}: {len(problems)} 处对不上")
            for line in problems[:20]:
                print("  " + line)
        else:
            print(f"{title}: 对得上")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
