"""本地 Web：`wy serve` 起的那个观察台。

只用标准库（http.server + json），不新增依赖。页面本身在 `wy/web/`，
数据一律走 `wy.core.observe()`（纯演算），SVG 走插件 `renderer/svg`，
PNG 走 Pillow 的 `scene`（和命令行同一套基准渲染），
所以「页面」和「别的软件」吃的是同一份 bundle。
"""

from __future__ import annotations

import io
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import plugins, plugins_builtin  # noqa: F401  导入即注册自带插件
from .core import Config
from .core.bundle import dumps, observe
from .scene import SceneParams, draw_scene

WEB = Path(__file__).with_name("web")

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}

#: 参数白名单与范围。超范围直接夹住，不报错——页面上滑杆本来就有边界。
RULES: dict[str, tuple[str, float, float]] = {
    "n": ("int", 8, 2000),
    "modules": ("int", 1, 24),
    "ticks": ("int", 10, 20000),
    "seed": ("int", 0, 100000),
    "probe": ("int", 2, 500),
    "need": ("int", 1, 64),
    "frac": ("float", 1e-4, 0.5),
    "walk_power": ("float", 0.1, 8.0),
}

DEFAULTS = {"n": 128, "modules": 8, "ticks": 1200, "seed": 7,
            "probe": 20, "need": 4, "frac": 3e-2, "walk_power": 1.0}

_CACHE: dict[str, str] = {}
_CACHE_LOCK = threading.Lock()


def parse_params(query: str) -> dict:
    """URL 查询串 → 参数 dict；坏值回落到默认，越界夹住。"""
    raw = parse_qs(query)
    out: dict = {}
    for key, (kind, lo, hi) in RULES.items():
        if key not in raw:
            out[key] = DEFAULTS[key]
            continue
        try:
            val = int(raw[key][0]) if kind == "int" else float(raw[key][0])
        except (TypeError, ValueError):
            raise ValueError(f"{key} 得是{'整数' if kind == 'int' else '小数'}")
        out[key] = max(lo, min(hi, val))
    return out


def build_config(p: dict) -> Config:
    cfg = Config(n=int(p["n"]), modules=int(p["modules"]), seed=int(p["seed"]))
    if float(p["walk_power"]) != 1.0:
        cfg = Config(**{**cfg.__dict__, "walk_power": float(p["walk_power"])})
    return cfg


def make_bundle(p: dict) -> dict:
    return observe(build_config(p), int(p["ticks"]), frac=float(p["frac"]),
                   probe=int(p["probe"]), need=int(p["need"]))


def bundle_json(p: dict) -> str:
    key = json.dumps(p, sort_keys=True)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit is not None:
        return hit
    text = dumps(make_bundle(p))
    with _CACHE_LOCK:
        if len(_CACHE) >= 8:
            _CACHE.clear()
        _CACHE[key] = text
    return text


class _FrameGroup:
    """bundle 里的群 → `draw_scene` 要的对象（人数够它算大小，别的字段照传）。"""

    def __init__(self, g: dict) -> None:
        self.gid = int(g["gid"])
        self.members = range(int(g["count"]))
        self.value = float(g["value"])
        self.span = float(g["span"])
        self.stable = float(g["stable"])
        self.settled = bool(g["settled"])


def png_bytes(p: dict, index: int = -1, size: int = 760) -> bytes:
    """与页面同源的帧 → Pillow 基准渲染的 PNG。`index` 越界就夹住。"""
    bundle = json.loads(bundle_json(p))
    frames = bundle.get("frames") or []
    if not frames:
        raise ValueError("这一跑没有采样帧（ticks 比 probe 小）")
    i = int(index)
    i = len(frames) - 1 if i < 0 else min(i, len(frames) - 1)
    frame = frames[i]
    groups = [_FrameGroup(g) for g in frame["groups"]]
    img = draw_scene(
        groups,
        np.asarray(frame["strength"], dtype=float),
        SceneParams(size=max(96, min(1100, int(size)))),
        title=f"t={frame['t']}  k={frame['k']}",
        note=f"望云 · n={bundle['meta']['n']} seed={bundle['meta']['seed']} · Pillow 基准渲染",
        seed=int(bundle["meta"]["seed"]),
        pos=np.asarray(frame["pos"], dtype=float),
        hue=np.asarray([g["hue"] for g in frame["groups"]], dtype=float),
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    server_version = "wangyun/0.1"

    def log_message(self, fmt: str, *args) -> None:  # 安静点，别刷终端
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _static(self, rel: str) -> None:
        path = (WEB / rel.lstrip("/")).resolve()
        if WEB.resolve() not in path.parents and path != WEB.resolve():
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
            return
        if not path.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
            return
        ctype = MIME.get(path.suffix, "application/octet-stream")
        self._send(HTTPStatus.OK, path.read_bytes(), ctype)

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                self._static("index.html")
                return
            if url.path in ("/app.js", "/style.css"):
                self._static(url.path)
                return
            if url.path == "/api/plugins":
                self._json(HTTPStatus.OK, plugins.describe())
                return
            if url.path == "/api/bundle":
                p = parse_params(url.query)
                self._send(HTTPStatus.OK, bundle_json(p).encode("utf-8"),
                           "application/json; charset=utf-8")
                return
            if url.path == "/api/svg":
                p = parse_params(url.query)
                index = int(parse_qs(url.query).get("index", ["-1"])[0])
                render = plugins.select("renderer", "svg")
                body = render(json.loads(bundle_json(p)), index).encode("utf-8")
                self._send(HTTPStatus.OK, body, "image/svg+xml; charset=utf-8")
                return
            if url.path == "/api/png":
                p = parse_params(url.query)
                raw = parse_qs(url.query)
                index = int(raw.get("index", ["-1"])[0])
                size = int(raw.get("size", ["760"])[0])
                self._send(HTTPStatus.OK, png_bytes(p, index, size), "image/png")
                return
            if url.path == "/api/health":
                self._json(HTTPStatus.OK, {"ok": True, "plugins": plugins.describe()})
                return
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - 本地服务，出错要看得见
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": repr(exc)})


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False) -> None:
    """起服务。默认只听本机；`open_browser` 为真时顺手开一下页面。"""
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{httpd.server_address[1]}/"
    print(f"望云观察台：{url}")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n收工。")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="wy.server", description="望云本地观察台")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="起来以后自动开浏览器")
    args = ap.parse_args(argv)
    serve(args.host, args.port, args.open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
