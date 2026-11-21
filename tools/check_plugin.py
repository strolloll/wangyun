"""插件自检：DSH 插件包在装之前就能查的那几件事。

装一次改一处配置，最容易错的不是代码，是「清单和实物对不上」：`files` 漏写、
`exports` 少一条、locale 少一段、图标超了大小、patch 里的 id 和包名不一致。
这些都能在装之前查出来，不用等商城预检。

这不是在验「插件好不好用」，只验「装得上、说得清、指得到」。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "dsh-plugin"
NODE_HINTS = (
    Path(r"D:\deepseek\resources\runtime\primary-runtime\dependencies\node\bin\node.exe"),
)
ICON_LIMIT = 256 * 1024


def files_covered() -> list[str]:
    """`files` 得把实物都罩住——漏一个，装出去就少一块。"""
    import fnmatch

    pkg = json.loads((PLUGIN / "package.json").read_text(encoding="utf-8"))
    patterns = [str(p) for p in (pkg.get("files") or [])]
    must = [
        "index.js",
        "cordis.patch.yml",
        "icon.svg",
        "locale/en.json",
        "locale/zh.json",
        "docs/DETAIL.md",
    ]

    def covered(name: str) -> bool:
        for pat in patterns:
            if fnmatch.fnmatch(name, pat):
                return True
            if "/" not in pat and name.startswith(pat + "/"):
                return True
        return False

    bad: list[str] = []
    for name in must:
        if not (PLUGIN / name).exists():
            bad.append(f"实物缺：{name}")
        elif not covered(name):
            bad.append(f"files 里没有罩住 {name}")
    if not any(pat.startswith("docs") for pat in patterns):
        bad.append("files 里没有 docs/")
    return bad


def check_manifest() -> list[str]:
    bad: list[str] = []
    pkg = json.loads((PLUGIN / "package.json").read_text(encoding="utf-8"))
    if not pkg.get("name"):
        bad.append("package.json 没有 name")
    bundle = (pkg.get("dsh") or {}).get("bundle") or {}
    patch = bundle.get("patch")
    if not patch or not (PLUGIN / patch).exists():
        bad.append("dsh.bundle.patch 指向的 patch 文件不存在")
    exports = pkg.get("exports") or {}
    for key in (".", "./icon", "./locale/*.json"):
        if key not in exports:
            bad.append(f"exports 少一条：{key}")
    if not pkg.get("private"):
        bad.append("本地包应标 private（别被误发出去）")
    return bad + files_covered()


def check_locale() -> list[str]:
    bad: list[str] = []
    for lang in ("en", "zh"):
        path = PLUGIN / "locale" / f"{lang}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta") or {}
        for key in ("title", "description"):
            text = str(meta.get(key) or "").strip()
            if not text:
                bad.append(f"locale/{lang}.json 的 meta.{key} 是空的")
            elif lang == "en":
                if not text[0].isupper():
                    bad.append(f"en/{key} 不是 sentence case 开头：{text[:20]}")
                if text.endswith("."):
                    bad.append(f"en/{key} 别拿句号收尾：{text[:20]}")
    detail = (PLUGIN / "docs" / "DETAIL.md")
    if not detail.exists() or detail.stat().st_size < 200:
        bad.append("docs/DETAIL.md 太短或不存在（详情页没有内容）")
    return bad


def check_icon() -> list[str]:
    icon = PLUGIN / "icon.svg"
    if not icon.exists():
        return ["icon.svg 不存在"]
    bad: list[str] = []
    head = icon.read_text(encoding="utf-8", errors="replace")[:200]
    if "<svg" not in head:
        bad.append("icon.svg 没有 <svg 根节点")
    if icon.stat().st_size > ICON_LIMIT:
        bad.append(f"icon.svg 超过 {ICON_LIMIT // 1024} KiB")
    return bad


def check_patch() -> list[str]:
    text = (PLUGIN / "cordis.patch.yml").read_text(encoding="utf-8")
    bad: list[str] = []
    if "id: wangyun-tools" not in text:
        bad.append("patch 里没有 id: wangyun-tools")
    if "name: '@local/wangyun-tools'" not in text:
        bad.append("patch 里的 name 和包名对不上")
    for key in ("python:", "workspace:", "timeoutMs:"):
        if key not in text:
            bad.append(f"patch 的 config 里缺 {key}")
    return bad


def check_entry() -> list[str]:
    text = (PLUGIN / "index.js").read_text(encoding="utf-8")
    bad: list[str] = []
    if "export function apply" not in text:
        bad.append("index.js 没有 export function apply")
    if ".tools.register(" not in text:
        bad.append("index.js 没有注册工具")
    for name in ("wy_observe", "wy_frame"):
        if name not in text:
            bad.append(f"index.js 里找不到工具名 {name}")
    return bad


def check_syntax() -> tuple[list[str], str]:
    """有 node 就真解析一遍（ESM 的 node --check 认）；没有就只留一句说明。"""
    node = next((p for p in NODE_HINTS if p.exists()), None) or shutil.which("node")
    if not node:
        return [], "跳过（这台机器上没有 node）"
    r = subprocess.run([str(node), "--check", str(PLUGIN / "index.js")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return [f"node --check 失败：{(r.stderr or r.stdout).strip()[-300:]}"], ""
    return [], f"node {Path(node).name} --check 通过"


def main() -> int:
    syntax_bad, syntax_note = check_syntax()
    groups = [
        ("清单", check_manifest()),
        ("说明", check_locale()),
        ("图标", check_icon()),
        ("接线", check_patch() + check_entry()),
        ("语法", syntax_bad),
    ]
    failed = 0
    for title, problems in groups:
        if problems:
            failed += 1
            print(f"{title}: {len(problems)} 处对不上")
            for line in problems[:20]:
                print("  " + line)
        else:
            extra = f"（{syntax_note}）" if title == "语法" and syntax_note else ""
            print(f"{title}: 对得上{extra}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
