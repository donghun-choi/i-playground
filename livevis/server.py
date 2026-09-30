"""localhost 실시간 시각화 서버 (표준 라이브러리만 사용).

브라우저는 /events 를 Server-Sent Events 로 구독한다. 접속하자마자 지금까지의
기록 전체(init)를 받고, 이후에는 log() 호출마다 새 값(log)을 받는다.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import queue
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_INDEX = Path(__file__).with_name("static") / "index.html"
_QUEUE_LIMIT = 10_000
_PING_SECONDS = 10


class _Hub:
    """지표 기록을 보관하고 SSE 구독자에게 뿌린다."""

    def __init__(self, title: str, max_points: int):
        self.title = title
        self.max_points = max_points
        self._series: dict[str, deque] = {}
        self._subs: set[queue.Queue] = set()
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> dict:
        return {
            "title": self.title,
            "max_points": self.max_points,
            "series": {name: list(points) for name, points in self._series.items()},
        }

    def subscribe(self) -> tuple[queue.Queue, dict]:
        # 구독 등록과 스냅샷을 한 락 안에서 해야 그 사이의 값이 빠지거나 겹치지 않는다.
        q: queue.Queue = queue.Queue(maxsize=_QUEUE_LIMIT)
        with self._lock:
            self._subs.add(q)
            return q, self._snapshot()

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def publish(self, step, metrics: dict) -> None:
        event = {"step": step, "metrics": metrics}
        with self._lock:
            for name, value in metrics.items():
                points = self._series.setdefault(name, deque(maxlen=self.max_points))
                points.append((step, value))
            for q in list(self._subs):
                try:
                    q.put_nowait(event)
                except queue.Full:
                    # 못 따라오는 클라이언트는 끊는다. 브라우저가 재연결하면 init 으로 다시 맞춰진다.
                    self._subs.discard(q)
                    self._kick(q)

    def close(self) -> None:
        with self._lock:
            for q in self._subs:
                self._kick(q)
            self._subs.clear()

    @staticmethod
    def _kick(q: queue.Queue) -> None:
        with contextlib.suppress(queue.Empty):
            q.get_nowait()
        q.put_nowait(None)


class _Handler(BaseHTTPRequestHandler):
    hub: _Hub

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", _INDEX.read_bytes())
        elif path == "/snapshot":
            self._send(200, "application/json", json.dumps(self.hub.snapshot()).encode())
        elif path == "/events":
            self._stream()
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        q, snapshot = self.hub.subscribe()
        try:
            self._event("init", snapshot)
            while True:
                try:
                    event = q.get(timeout=_PING_SECONDS)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                if event is None:
                    break
                self._event("log", event)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.hub.unsubscribe(q)

    def _event(self, name: str, data: dict) -> None:
        payload = json.dumps(data, separators=(",", ":"))
        self.wfile.write(f"event: {name}\ndata: {payload}\n\n".encode())
        self.wfile.flush()

    def log_message(self, format, *args):
        pass


class LiveVis:
    """실험 코드에서 지표를 찍으면 브라우저에 실시간으로 그려준다.

        viz = LiveVis("my experiment").start()
        for step in range(100):
            viz.log(step, loss=..., acc=...)
        viz.wait()

    step 은 증가하는 순서로 찍는다. 생략하면 직전 step + 1 이 된다.
    """

    def __init__(
        self,
        title: str = "i-playground",
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
        max_points: int = 5000,
    ):
        if port is None:
            port = int(os.environ.get("LIVEVIS_PORT", "8000"))
        self._hub = _Hub(title, max_points)
        self._host = host
        self._port = port
        self._server: ThreadingHTTPServer | None = None
        self._next_step = 0

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2] if self._server else (self._host, self._port)
        if host in ("127.0.0.1", "0.0.0.0", "::", ""):
            host = "localhost"
        return f"http://{host}:{port}/"

    def start(self, open_browser: bool = False) -> LiveVis:
        if self._server:
            return self
        handler = type("Handler", (_Handler,), {"hub": self._hub})
        try:
            self._server = ThreadingHTTPServer((self._host, self._port), handler)
        except OSError as e:
            raise OSError(
                f"{self._host}:{self._port} 에 서버를 띄울 수 없습니다 ({e}). "
                "port= 인자나 LIVEVIS_PORT 로 다른 포트를 지정하세요."
            ) from e
        threading.Thread(target=self._server.serve_forever, name="livevis", daemon=True).start()
        print(f"[livevis] {self.url} 에서 실시간으로 확인하세요", flush=True)
        if open_browser:
            webbrowser.open(self.url)
        return self

    def log(self, step=None, **metrics) -> None:
        if step is None:
            step = self._next_step
        self._next_step = step + 1
        self._hub.publish(step, {name: _finite_or_none(value) for name, value in metrics.items()})

    def stop(self) -> None:
        if not self._server:
            return
        self._hub.close()
        self._server.shutdown()
        self._server.server_close()
        self._server = None

    def wait(self) -> None:
        """Ctrl+C 전까지 서버를 유지한다. 스크립트가 끝난 뒤에도 차트를 보고 싶을 때 쓴다."""
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def __enter__(self) -> LiveVis:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _finite_or_none(value) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None
