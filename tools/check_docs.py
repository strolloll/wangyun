"""文档引用自检：写进文档的契约名和节号，必须真的存在。

三件事，任何一件断了就退出码 1：

1. 文档里引用的 `test_xxx` 必须在 `tests/run_tests.py` 里定义过——改名之后最容易漏的就是这里；
2. `DESIGN.md` 目录表里的节号和正文标题逐个对齐、顺序一致——加节时忘了补目录会被抓出来；
3. 文档里写的 `§N` 必须真有第 N 节——防止引用一句没写过的结论。

这不是在验「写得对不对」，只验「指得到」。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ("DESIGN.md", "HANDOFF.md", "README.md", "GUIDE.md")


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def check_test_names() -> list[str]:
    """文档里提到的 test_xxx 都得在 tests/run_tests.py 里有定义。"""
    defined = set(re.findall(r"^def (test_[A-Za-z0-9_]+)\(", read("tests/run_tests.py"), re.M))
    bad: list[str] = []
    for name in DOCS:
        text = read(name)
        for m in re.finditer(r"\btest_[A-Za-z0-9_]+\b", text):
            word = m.group(0)
            if word not in defined:
                line = text.count("\n", 0, m.start()) + 1
                bad.append(f"{name}:{line} 引用 {word}，但 tests 里没这个契约")
    return bad


def headings(text: str) -> list[tuple[int, str]]:
    return [(int(n), title.strip()) for n, title in
            re.findall(r"^## (\d+)\. (.+)$", text, re.M)]


def check_toc(design: str) -> list[str]:
    """目录表里的 §N 必须和正文 §N 标题数量、顺序一致。"""
    heads = headings(design)
    if not heads:
        return ["DESIGN.md 里一个 ## N. 标题都没有"]

    start = design.find("## 目录")
    if start < 0:
        return ["DESIGN.md 里找不到「## 目录」"]
    rest = design.find("\n## ", start + 1)
    toc = design[start:rest if rest > 0 else len(design)]
    rows = [int(n) for n in re.findall(r"^\| §(\d+) \|", toc, re.M)]

    bad: list[str] = []
    if rows != [n for n, _ in heads]:
        bad.append(f"目录节号 {rows} 与正文标题 {[n for n, _ in heads]} 对不上")
    for n, title in heads:
        if not title:
            bad.append(f"第 {n} 节没有标题")
    if len({n for n, _ in heads}) != len(heads):
        bad.append("正文里有重复的节号")
    return bad


def check_test_counts() -> list[str]:
    """文档里自称的测试条数必须等于 tests 里真正定义的条数（这种数最容易放久了自己长歪）。"""
    defined = re.findall(r"^def (test_[A-Za-z0-9_]+)\(", read("tests/run_tests.py"), re.M)
    total = len(defined)
    bad: list[str] = []

    readme = read("README.md")
    for m in re.finditer(r"契约测试[（(，,]?\s*(\d+)\s*条", readme):
        if int(m.group(1)) != total:
            line = readme.count("\n", 0, m.start()) + 1
            bad.append(f"README.md:{line} 说 {m.group(1)} 条，实际 {total} 条")

    handoff = read("HANDOFF.md")
    for pattern, label in ((r"\*\*(\d+)/(\d+)\*\*", "测试行"),
                           (r"#\s*(\d+)/(\d+)\s*$", "命令行")):
        for m in re.finditer(pattern, handoff, re.M):
            if (int(m.group(1)), int(m.group(2))) != (total, total):
                line = handoff.count("\n", 0, m.start()) + 1
                bad.append(f"HANDOFF.md:{line} {label}写 {m.group(1)}/{m.group(2)}，实际 {total}/{total}")
    return bad


def check_section_refs() -> list[str]:
    """文档里每个 §N 都要真有第 N 节。"""
    known = {n for n, _ in headings(read("DESIGN.md"))}
    bad: list[str] = []
    for name in DOCS:
        text = read(name)
        for m in re.finditer(r"§(\d+)", text):
            n = int(m.group(1))
            if n not in known:
                line = text.count("\n", 0, m.start()) + 1
                bad.append(f"{name}:{line} 引用 §{n}，但 DESIGN.md 没有这一节")
    return bad


def main() -> int:
    design = read("DESIGN.md")
    groups = [
        ("契约名", check_test_names()),
        ("条数", check_test_counts()),
        ("目录", check_toc(design)),
        ("节号引用", check_section_refs()),
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
    sys.exit(main())
