DASH_20261025c_requestlog.zip —— 3 个文件：dash-js/telemetry.js、dash-js/DASHttp.js、dash-js/isac_ws_client.js
（覆盖在已装好 DASH_20261025_code.zip 的 DASH_html_old-main 上；不需要重启 server；浏览器 Ctrl+F5）
注意：这个包里的 DASHttp.js 已经包含 8 s 超时 / 0.2 s 重试（和你手改的那三行一致），并多了请求日志，所以直接覆盖就行，不用再手改。

加了什么（只加记录，不改播放/选码率逻辑）：
  快照（Save Full Snapshot）里 telemetry 多了三个数组：
    requests   页面每一次分片请求（含失败/超时/重试）：url、attempt(第几次尝试)、startWall/endWall(浏览器 Date.now)、
               state(ok|timeout|error)、httpStatus、bytes、timeoutMs、nextRetryMs、representationId
    ntgStates  NTG 网页那六个状态（装置/無線/認証/監視信号/イーサ 自局·対向局），由 Ubuntu 的 isac_bridge 推来（1 Hz 变化+5 s 心跳）
    dataplane  Ubuntu→视频 server 的 ping 通不通（翻转+5 s 心跳，带 rtt）
  后两个需要 Ubuntu 侧重启 isac_bridge.py（新版带 link_monitor）才会有；没重启的话它们是空数组，requests 照常有。
检查：F12 控制台输入  dashTelemetry.requests.length  应该 >0；  Object.keys(dashTelemetry.getSnapshot())  里有 requests/ntgStates/dataplane。
