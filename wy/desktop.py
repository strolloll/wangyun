"""桌面入口：双击图标 → 起本地观察台 → 开一个应用窗口。

用 `pythonw -m wy.desktop` 启动时没有控制台，所以：

- 日志写到 `%LOCALAPPDATA%\\wangyun\\desktop.log`
- 已经在跑（端口上能读到 `/api/health`）就只开窗口，不重复起服务
- 窗口优先用 Edge 的 `--app=` 模式（像个独立软件），没有 Edge 就退回系统浏览器
- 页面右侧有「退出望云」，点了会 POST `/api/quit`，服务停掉
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

APP = "wangyun"
HOST = "127.0.0.1"
PORT = 8765


def data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    d = Path(base) / APP
    d.mkdir(parents=True, exist_ok=True)
    return d


def log(msg: str) -> None:
    """没有控制台时也要留下痕迹；日志超过 1 MB 就重来。"""
    try:
        path = data_dir() / "desktop.log"
        if path.exists() and path.stat().st_size > 1024 * 1024:
            path.unlink()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] {msg}\n")
    except OSError:
        pass


def state_path() -> Path:
    return data_dir() / "state.json"


def read_state() -> dict:
    try:
        return json.loads(state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_state(port: int) -> None:
    try:
        state_path().write_text(
            json.dumps({"port": port, "pid": os.getpid(), "at": time.time()}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def clear_state() -> None:
    try:
        state_path().unlink()
    except OSError:
        pass


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def find_port(host: str, start: int, tries: int = 20) -> int:
    for port in range(start, start + tries):
        if port_free(host, port):
            return port
    raise RuntimeError(f"从 {start} 起 {tries} 个端口都被占了")


def alive(port: int, timeout: float = 0.8) -> bool:
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/api/health", timeout=timeout) as resp:
            return resp.status == 200
    except OSError:
        return False


def edge_path() -> str | None:
    cand = [
        shutil.which("msedge"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for path in cand:
        if path and Path(path).is_file():
            return path
    return None


def open_window(url: str) -> None:
    """开一个像软件一样的窗口：Edge app 模式；都不行就交给系统浏览器。"""
    edge = edge_path()
    if edge:
        try:
            subprocess.Popen(
                [edge, f"--app={url}", "--window-size=1360,940"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            log(f"opened edge app window {url}")
            return
        except OSError as exc:
            log(f"edge launch failed: {exc!r}")
    webbrowser.open(url)
    log(f"opened default browser {url}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="wy.desktop", description="望云桌面入口")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-open", action="store_true", help="只起服务，不开窗口")
    args = ap.parse_args(argv)

    log("desktop start")

    # 已经在跑：直接开窗口，别起第二个服务
    state = read_state()
    for port in ([int(state.get("port", 0))] if state.get("port") else []) + [args.port]:
        if port and alive(port):
            url = f"http://{HOST}:{port}/"
            log(f"reusing running server at {url}")
            if not args.no_open:
                open_window(url)
            return 0

    from .server import make_server

    port = find_port(args.host, args.port)
    httpd = make_server(args.host, port)
    url = f"http://{args.host}:{port}/"
    write_state(port)
    log(f"serving at {url}")

    if not args.no_open:
        threading.Timer(0.5, lambda: open_window(url)).start()

    try:
        httpd.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        clear_state()
        log("desktop stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
