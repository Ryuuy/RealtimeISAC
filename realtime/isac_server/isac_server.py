"""
isac_server.py
20260812 新增

给"另一个 Python 算法程序"用的 ISAC -> 网页 通讯模块。

用法（在你现有的算法循环里）：

    from isac_server import start_server_in_background, push_result

    start_server_in_background(port=8765)   # 只需要在程序开头调一次

    while True:
        mode = run_my_isac_algorithm()      # 你已有的算法，返回 "shadowing" 或 "normal"
        push_result(mode)                   # 交给这个模块处理，剩下的不用管
        time.sleep(0.1)

设计对应关系（按现在讨论定下来的四点）：
    1. 连接建立时立即发当前 state      -> isac_ws() 里 accept() 之后马上 send 一次
    2. 网页关闭后 server 正确删除连接   -> try/finally 里 _clients.discard(websocket)
    3. 断线后 JS 自动 reconnect        -> 这个是前端 isac_ws_client.js 里做的，这边不用管
    4. 只在 state 改变时才 push        -> push_result() 里先比较，没变就直接 return

调试用：浏览器直接访问 http://<这台机器的局域网IP>:8765/isac/state
可以看到当前 state 的 JSON，不用等 WebSocket 也能确认 server 是不是活的。
"""

# from starlette.websockets import WebSocket   # 被下面 fastapi 的 WebSocket 覆盖，没意义，注释掉


# from starlette.datastructures import State   # 只用于下面那行没有实际类型检查意义的 set[WebSocket[State]]()，注释掉


import asyncio
import json
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

app = FastAPI()

# _clients: set[WebSocket] = set[WebSocket[State]]()   # 能跑但 WebSocket[State] 没有类型意义，简化成下面这行
_clients: set[WebSocket] = set()
_current_mode = "normal"
_state_lock = threading.Lock()   # 保护 _current_mode：算法线程和 asyncio 事件循环线程都会碰它
_loop: asyncio.AbstractEventLoop | None = None


@app.get("/isac/state")
async def get_state():
    return {"mode": _current_mode, "ts": time.time()}


def has_clients() -> bool:
    """给测试脚本用：判断有没有网页已经连上了，避免掐好的时间点在没人看的时候就已经推送完了。"""
    return len(_clients) > 0


@app.websocket("/isac")
async def isac_ws(websocket: WebSocket):
    await websocket.accept()
    _clients.add(websocket)
    try:
        # 要点 1：连接一建立就把"当前"state 发一次。
        # 这样网页刷新、或者断线重连回来的那一刻，不用干等下一次 shadowing 触发，
        # 立刻就能同步到服务端现在到底是 normal 还是 shadowing。
        with _state_lock:
            mode_now = _current_mode
        await websocket.send_text(json.dumps({"mode": mode_now, "ts": time.time()}))

        # 之后这个连接不需要网页发东西过来，这里只是靠 receive 挂起来监听"对方断开了没"
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        # 要点 2：网页关闭 / 断网时，把这个连接从活跃集合里删掉，
        # 不然 _clients 里会越攒越多已经死掉的连接，广播时对着死连接 send 还会报错。
        _clients.discard(websocket)


async def _broadcast(mode: str):
    payload = json.dumps({"mode": mode, "ts": time.time()})
    dead = []
    for ws in list(_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


def push_result(mode: str):
    """
    从算法主循环里同步调用即可，不用管 asyncio。

    对应你画的那张图：
        Algorithm -> push_result(mode) -> 和旧 state 比较
            相同 -> 什么都不做
            不同 -> 更新 state，广播给所有连着的浏览器
    """
    global _current_mode

    with _state_lock:
        # 要点 4：state 没变化就直接返回，不广播、不占带宽、不刷网页日志
        if mode == _current_mode:
            return
        _current_mode = mode

    if _loop is not None:
        # push_result 大概率是算法线程（同步代码）调用的，
        # 而 _broadcast 是要跑在 uvicorn 那个 asyncio 事件循环里的协程，
        # 用 run_coroutine_threadsafe 把它安全地丢过去执行，避免跨线程直接碰 asyncio 对象。
        asyncio.run_coroutine_threadsafe(_broadcast(mode), _loop)


def _run_server(host: str, port: int):
    global _loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _loop = loop
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())


def start_server_in_background(host: str = "0.0.0.0", port: int = 8765):
    """
    host 用 0.0.0.0：不是"服务器的地址是 0.0.0.0"，而是"这台机器上所有网卡的地址都能接受连接"，
    局域网里的另一台机器就能用这台机器的 192.168.x.x 连进来，不用改代码。
    """
    t = threading.Thread(target=_run_server, args=(host, port), daemon=True)
    t.start()

    # 等 uvicorn 的事件循环真的建好了再返回，
    # 不然如果算法循环起手就调 push_result()，可能 _loop 还是 None。
    while _loop is None:
        time.sleep(0.01)

    print(f"[isac_server] listening on ws://{host}:{port}/isac "
          f"(局域网内用这台机器的 192.168.x.x 地址连)")


if __name__ == "__main__":
    # 单独跑这个文件时，用假数据模拟算法结果，方便你在没接上真实算法之前先联调前端。
    start_server_in_background(port=8765)
    print("模拟模式：每 5 秒在 normal / shadowing 之间切换一次，用于测试前端连接。Ctrl+C 退出。")
    demo_modes = ["normal", "shadowing"]
    i = 0
    try:
        while True:
            push_result(demo_modes[i % 2])
            i += 1
            time.sleep(5)
    except KeyboardInterrupt:
        pass
