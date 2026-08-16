#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判决结果外发。**用户已经写好了往 local IP 推的那部分**，这里只提供接线口。

唯一的硬性要求：**绝不能阻塞 DSP 循环**。
一次阻塞的 socket 写（对端慢、TCP 窗口满、网络抖动）就能吃掉整个 20ms 预算，
进而让采集侧溢出。所以这里的做法是：

- 单独一个发送线程
- 队列满了就**丢最旧的**，而不是阻塞生产者
- 只发判决结果（几十字节），不发整个谱

把你现有的推送函数传给 `sink` 即可：

    pub = Publisher(sink=my_push_to_ip)
    pub.start()
    ...
    pub.send({"present": True, "doppler_hz": 281.2})
"""

import json
import socket
import threading
from queue import Empty, Full, Queue


class Publisher:
    """非阻塞外发。sink 是你自己的推送函数，签名 sink(payload: dict) -> None。
    sink 里抛异常不会影响 DSP，只会被计到 self.errors。"""

    def __init__(self, sink=None, maxsize: int = 8):
        self.sink = sink
        self._q: "Queue[dict]" = Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = None
        self.sent = 0
        self.dropped = 0
        self.errors = 0

    def start(self) -> "Publisher":
        if self.sink is None:
            return self
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def send(self, payload: dict) -> None:
        """DSP 线程调这个。永不阻塞：队列满就丢最旧的一条。"""
        if self.sink is None:
            return
        try:
            self._q.put_nowait(payload)
        except Full:
            try:
                self._q.get_nowait()      # 丢最旧的，保证发出去的是最新状态
                self._q.put_nowait(payload)
            except (Empty, Full):
                pass
            self.dropped += 1

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except Empty:
                continue
            try:
                self.sink(item)
                self.sent += 1
            except Exception:
                self.errors += 1

    def status(self) -> str:
        if self.sink is None:
            return "外发: 未接"
        return f"外发: sent={self.sent} dropped={self.dropped} errors={self.errors}"


def stdout_sink(payload: dict) -> None:
    """调试用的 sink：把判决打到 stdout。换成你自己的 IP 推送即可。"""
    print(json.dumps(payload, ensure_ascii=False), flush=True)


class UdpSink:
    """把判决打成 UDP JSON 数据报，发给 isac_bridge.py。

    ## 为什么隔一层 UDP 而不是直接 import isac_server

    **两个解释器互斥**：`/usr/bin/python3` 有 uhd 没 fastapi，
    miniconda 的 python3 有 fastapi 没 uhd（装 uhd 会撞 GLIBCXX）。
    所以 web 那半边只能跑在另一个进程里。

    这反倒是对的架构：uvicorn 的 asyncio 事件循环不会挤进实时进程抢 GIL，
    而且 **UDP 发送不会阻塞**——对端没起、崩了、网线拔了，sendto 都立即返回。
    换成 TCP 的话，对端不 accept 就会卡住发送线程；虽然发送在 Publisher 的
    独立线程里不至于打穿 DSP，但状态会静默堆积在队列里。UDP 没这个问题：
    判决是**状态**不是事件流，丢一两个包无所谓，下一个心跳就补上了。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9099):
        self.addr = (host, port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def __call__(self, payload: dict) -> None:
        self._sock.sendto(json.dumps(payload).encode("utf-8"), self.addr)

    def __repr__(self) -> str:
        return f"UdpSink({self.addr[0]}:{self.addr[1]})"
