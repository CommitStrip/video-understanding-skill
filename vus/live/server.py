#!/usr/bin/env python3
"""
server.py - 实时理解状态服务（stdlib SSE，零新依赖）
====================================================
给"机器人的实时视觉"和监控面板的查询/订阅入口：

  GET /state     当前 SessionState 快照（JSON，四层状态 + 分层滞后遥测）
  GET /events    text/event-stream：先推一条 state 快照，再持续推增量事件
  GET /healthz   探活 + 最小滞后信息（供看门狗/负载均衡）

设计要点：
  - 每个 /events 连接独立一条 EventBus 订阅（有界，慢客户端丢最旧不影响别人）；
  - 客户端断开（BrokenPipe）直接结束该连接线程，不影响服务；
  - ThreadingHTTPServer 每连接一线程；事件循环以 1s 超时轮询停机标志，
    stop() 时所有 SSE 连接在 1s 内收线。
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class LiveServer:
    """实时理解只读服务。start()/stop() 包住 ThreadingHTTPServer 生命周期。"""

    def __init__(self, bus, state, host="127.0.0.1", port=8600):
        self.bus = bus
        self.state = state
        self.host = host
        self.port = int(port)
        self._httpd = None
        self._thread = None
        self._stop_evt = threading.Event()

    def start(self):
        """绑定端口并后台服务；端口被占等 OSError 原样抛出（显式失败）。"""
        server = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/state":
                    self._reply_json(server.state.snapshot())
                elif path == "/healthz":
                    snap = server.state.snapshot()
                    self._reply_json({
                        "status": "ok",
                        "t_now": snap["session"]["t_now"],
                        "lag": snap["telemetry"].get("lag"),
                    })
                elif path == "/events":
                    self._serve_events()
                else:
                    self.send_error(404)

            def _reply_json(self, obj, status=200):
                body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _serve_events(self):
                """SSE 流：先发 state 快照，再持续推总线增量事件。"""
                sub = server.bus.subscribe(maxsize=1024)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                try:
                    self.wfile.write(b"retry: 3000\n\n")
                    self._send_event("state", server.state.snapshot())
                    while not server._stop_evt.is_set():
                        ev = sub.get(timeout=1.0)
                        if ev is not None:
                            self._send_event(ev.get("type", "event"), ev)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass  # 客户端断开：正常结束该连接
                finally:
                    self.close_connection = True

            def _send_event(self, name, obj):
                data = json.dumps(obj, ensure_ascii=False)
                self.wfile.write(f"event: {name}\ndata: {data}\n\n".encode("utf-8"))
                self.wfile.flush()

            def log_message(self, *args):
                pass  # 静默访问日志（长直播进程不刷屏）

        self._httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
        self.port = self._httpd.server_address[1]  # port=0 时回读实际端口
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True, name="vus-live-server")
        self._thread.start()

    @property
    def url(self):
        return f"http://{self.host}:{self.port}"

    def stop(self):
        self._stop_evt.set()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
