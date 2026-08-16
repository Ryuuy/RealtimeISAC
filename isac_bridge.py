#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ISAC 桥接：收 rt_main 发来的 UDP 判决，转成 isac_server 的 shadowing/normal。

## ⚠️ 这个脚本用 miniconda 的 python3 跑，不是 /usr/bin/python3

    python3 RealtimeISAC/isac_bridge.py          # 注意：不写 /usr/bin/

和整个项目其它脚本相反。原因是两个解释器的依赖互斥：

| | uhd | fastapi/uvicorn |
|---|---|---|
| `/usr/bin/python3` | ✅ | ❌ |
| miniconda `python3` | ❌ (撞 GLIBCXX) | ✅ |

所以 `rt_main.py` 只能在前者跑、`isac_server` 只能在后者跑，中间用 UDP 连起来。

## 跑法（两个终端）

    # 终端 A：起桥接 + web server
    python3 RealtimeISAC/isac_bridge.py

    # 终端 B：起实时检测，把判决打到桥接
    cd RealtimeISAC/realtime
    /usr/bin/python3 rt_main.py --duration 600 --isac

网页连 `ws://<本机IP>:8765/isac`，或者 `http://<本机IP>:8765/isac/state` 看当前状态。

## 两个健壮性设计

1. **心跳**：rt_main 除了状态翻转时发，还每 2 秒重发一次当前状态。
   这样桥接后启动、或丢了一个 UDP 包，最多 2 秒就同步回来。
   `push_result()` 本身对相同状态直接 return，重发不会刷屏也不占带宽。
2. **失效超时**：超过 stale_sec 收不到任何包（rt_main 崩了 / 被 Ctrl+C），
   自动回落到 normal。否则网页会永远停在最后那个 shadowing 上，
   而实际上早就没人在测了——**宁可漏报也不要挂着一个假状态**。
"""

import argparse
import json
import os
import socket
import sys
import time

DEFAULT_SERVER_DIR = "/home/usrpb210/DASH/DASH_html_old-main/isac_server"


def main() -> int:
    p = argparse.ArgumentParser(description="rt_main 判决 -> isac_server")
    p.add_argument("--udp-port", type=int, default=9099, help="监听 rt_main 的端口")
    p.add_argument("--udp-host", type=str, default="127.0.0.1")
    p.add_argument("--isac-port", type=int, default=8765, help="isac_server 的端口")
    p.add_argument("--server-dir", type=str, default=DEFAULT_SERVER_DIR,
                   help="isac_server.py 所在目录")
    p.add_argument("--stale-sec", type=float, default=6.0,
                   help="这么久没收到判决就回落到 normal（0 = 不回落）")
    p.add_argument("--dry", action="store_true",
                   help="不起 isac_server，只把收到的判决打出来（联调用）")
    a = p.parse_args()

    push_result = None
    if not a.dry:
        if not os.path.isdir(a.server_dir):
            print(f"找不到 isac_server 目录: {a.server_dir}", file=sys.stderr)
            return 1
        sys.path.insert(0, a.server_dir)
        try:
            from isac_server import push_result, start_server_in_background
        except ImportError as e:
            print(f"import isac_server 失败: {e}\n"
                  f"这个脚本要用 **miniconda 的 python3** 跑（有 fastapi/uvicorn），"
                  f"不是 /usr/bin/python3。", file=sys.stderr)
            return 1
        start_server_in_background(port=a.isac_port)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((a.udp_host, a.udp_port))
    sock.settimeout(0.5)
    print(f"[bridge] 监听 udp://{a.udp_host}:{a.udp_port}"
          f"{'  (dry 模式，不起 isac_server)' if a.dry else ''}")
    print("[bridge] 等 rt_main 的判决... Ctrl+C 退出")

    mode = "normal"
    last_rx = 0.0
    n_rx = 0

    def emit(new_mode: str, why: str) -> None:
        nonlocal mode
        if new_mode == mode:
            return
        mode = new_mode
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] -> {new_mode:9s} ({why})", flush=True)
        if push_result is not None:
            push_result(new_mode)

    try:
        while True:
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                if a.stale_sec and last_rx and time.time() - last_rx > a.stale_sec:
                    emit("normal", f"超过 {a.stale_sec:.0f}s 没收到判决，回落")
                    last_rx = 0.0
                continue
            try:
                msg = json.loads(data.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue                     # 不是我们的包，忽略
            if "present" not in msg:
                continue
            last_rx = time.time()
            n_rx += 1
            why = (f"doppler={msg.get('doppler_hz', 0):+.0f}Hz "
                   f"E={msg.get('energy_db', 0):+.0f}dB "
                   f"score={msg.get('score', '?')}")
            emit("shadowing" if msg["present"] else "normal", why)
    except KeyboardInterrupt:
        print(f"\n[bridge] 退出（共收到 {n_rx} 条判决）")
        if push_result is not None:
            push_result("normal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
