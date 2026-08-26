#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ISAC 桥接：收 rt_main 发来的 UDP 判决，转成 isac_server 的 shadowing/normal。

20260826 新增：同一个 UDP 端口上，也转发 rt_main 发来的另外两路连续流：
  - dBFS 包（{"dBFS": ..., "dBFS_channels": [...], "wallClock": ...}）——
    上行/reference channel，每步都发，调 isac_server 的 push_dBFS()
  - power 包（{"power": ..., "wallClock": ...}）——下行/sensing channel
    （Master 方向）判遮挡用，TDD 锁定时才发，调 isac_server 的 push_power()
两路都跟 present/shadowing 那支完全独立——不做心跳/去抖/失效回落，
rt_main 发多快这边转多快，见下面 emit() 之后紧跟着的两个分支。

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
    push_dBFS = None
    push_power = None
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
        # 20260826 新增：push_dBFS 单独 try——a.server_dir 指向的那份 isac_server.py
        # 可能还没同步这次新加的函数（比如这台机器的部署还是旧版本），单独 import
        # 失败不该拖累整个 bridge 起不来，present 判决照常转发，只是先不广播 dBFS，
        # 打个警告方便发现"忘记同步 isac_server.py"这种情况。
        try:
            from isac_server import push_dBFS
        except ImportError:
            print(f"[bridge] 警告：{a.server_dir} 里的 isac_server.py 还没有 push_dBFS"
                  f"（版本较旧，缺 20260826 的 dBFS 广播支持）——present 判决照常转发，"
                  f"dBFS 判决会被丢弃。同步一下 isac_server.py 就好。", file=sys.stderr)
        # 20260826 新增：push_power 同上，单独 try，缺了只警告不拖累整个 bridge
        try:
            from isac_server import push_power
        except ImportError:
            print(f"[bridge] 警告：{a.server_dir} 里的 isac_server.py 还没有 push_power"
                  f"（版本较旧，缺下行功率判遮挡的广播支持）——present 判决照常转发，"
                  f"下行 power 判决会被丢弃。同步一下 isac_server.py 就好。", file=sys.stderr)
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
    n_rx_dbfs = 0
    last_dbfs_print = 0.0
    n_rx_power = 0
    last_power_print = 0.0

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

            # 20260826 新增：dBFS 是独立的一路连续流（rt_main.py 每步都发一次，
            # ~50Hz，不跟 present 的心跳/去抖挂钩），跟下面 present 那支分开处理，
            # 不影响 stale_sec 回落逻辑——那个逻辑只关心"多久没收到 present 判决"，
            # dBFS 包不算数，避免两路互相干扰对方的语义。
            if "dBFS" in msg:
                n_rx_dbfs += 1
                dbfs_now = time.time()
                if push_dBFS is not None:
                    push_dBFS(msg["dBFS"], msg.get("wallClock"))
                # ~50Hz 量级，全打印会刷屏；节流到最多每秒一行，只为确认这条链路在动
                if dbfs_now - last_dbfs_print >= 1.0:
                    print(f"[bridge] dBFS={msg['dBFS']:+.1f} "
                          f"channels={msg.get('dBFS_channels')} (共收到 {n_rx_dbfs} 条)",
                          flush=True)
                    last_dbfs_print = dbfs_now
                continue

            # 20260826 新增：power 是下行（Master 方向）功率判遮挡那一路，
            # 跟上面 dBFS（上行/reference channel）是完全独立的通道——两个都
            # 可能出现在同一个 UDP 端口上，靠字段名区分，互不影响对方的处理。
            if "power" in msg:
                n_rx_power += 1
                power_now = time.time()
                if push_power is not None:
                    push_power(msg["power"], msg.get("wallClock"))
                if power_now - last_power_print >= 1.0:
                    print(f"[bridge] power(downlink)={msg['power']:+.1f}dBFS "
                          f"(共收到 {n_rx_power} 条)", flush=True)
                    last_power_print = power_now
                continue

            if "present" not in msg:
                continue
            last_rx = time.time()
            n_rx += 1
            why = (f"doppler={msg.get('doppler_hz', 0):+.0f}Hz "
                   f"E={msg.get('energy_db', 0):+.0f}dB "
                   f"score={msg.get('score', '?')}")
            emit("shadowing" if msg["present"] else "normal", why)
    except KeyboardInterrupt:
        print(f"\n[bridge] 退出（共收到 {n_rx} 条判决，{n_rx_dbfs} 条 dBFS，{n_rx_power} 条 power）")
        if push_result is not None:
            push_result("normal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
